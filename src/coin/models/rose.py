"""Relative Order Sequence Estimator (ROSE) for permutation problems.

This proof-of-concept learns both exact job positions and signed pairwise
distances from the selected cohort.  TemplateROSE reuses HBSA's WT position
punching policy and parent/child replacement semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from coin.core.random_state import SplitMix64
from .hbsa import sample_template_positions

RollMode = Literal["fixed", "random", "all"]
RoseSamplingMode = Literal["pairwise_window", "mean_anchor"]
ReferenceSelection = Literal["mean", "uniform", "confidence"]


@dataclass(frozen=True, slots=True)
class RoseConfig:
    problem_size: int
    population_size: int = 100
    selection_ratio: int = 20
    sampling_mode: RoseSamplingMode = "pairwise_window"
    roll_mode: RollMode = "random"
    fixed_roll: int = 3
    max_roll: int = 5
    node_weight: float = 0.25
    temperature: float = 1.0
    smoothing: float = 1.0
    template_enabled: bool = False
    template_sample_ratio: int = 50
    reference_selection: ReferenceSelection = "mean"
    objective: Literal["min", "max"] = "min"

    def __post_init__(self) -> None:
        if self.problem_size < 2: raise ValueError("problem_size must be at least 2")
        if self.population_size < 1: raise ValueError("population_size must be positive")
        if not 1 <= self.selection_ratio <= 100: raise ValueError("selection_ratio must be between 1 and 100")
        if self.sampling_mode not in ("pairwise_window", "mean_anchor"): raise ValueError("invalid sampling_mode")
        if self.roll_mode not in ("fixed", "random", "all"): raise ValueError("invalid roll_mode")
        if self.fixed_roll < 1 or self.max_roll < 1: raise ValueError("roll sizes must be positive")
        if not 0 <= self.node_weight <= 1: raise ValueError("node_weight must be between 0 and 1")
        if self.temperature <= 0: raise ValueError("temperature must be positive")
        if self.smoothing <= 0: raise ValueError("smoothing must be positive")
        if not 0 <= self.template_sample_ratio <= 100: raise ValueError("template_sample_ratio must be between 0 and 100")
        if self.reference_selection not in ("mean", "uniform", "confidence"): raise ValueError("invalid reference_selection")


@dataclass(slots=True)
class RoseStatistics:
    exact: np.ndarray
    relative: np.ndarray
    relative_counts: np.ndarray
    selected: np.ndarray
    population: np.ndarray
    fitness: np.ndarray


class ROSE:
    def __init__(self, config: RoseConfig, *, seed: int = 0):
        self.config = config
        self._rng = SplitMix64(seed)
        self.exact: np.ndarray | None = None
        self.relative: np.ndarray | None = None
        self.relative_counts: np.ndarray | None = None
        self.templates: np.ndarray | None = None
        self.template_fitness: np.ndarray | None = None
        self._pending_parents: list[np.ndarray] = []
        self._pending_parent_fitness: list[float] = []
        self.effective_population: np.ndarray | None = None
        self.effective_fitness: np.ndarray | None = None
        self._fallback_count = 0
        self._roll_sizes: list[int] = []
        self._offspring_total = 0
        self._offspring_unique = 0
        self._improved = 0
        self._template_fixed = 0
        self._template_resampled = 0
        self._relative_samples = 0
        self._exact_fallbacks = 0
        self._uniform_fallbacks = 0
        self._reference_counts = np.zeros(config.problem_size, dtype=np.int64)
        self._sampled_distances: list[int] = []
        self._fixed_reference_count = 0
        self._new_reference_count = 0

    def _shuffle(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.int16).copy()
        for index in range(len(values) - 1, 0, -1):
            other = self._rng.randbelow(index + 1)
            values[index], values[other] = values[other], values[index]
        return values

    def _random_permutation(self) -> np.ndarray:
        return self._shuffle(np.arange(self.config.problem_size, dtype=np.int16))

    def _weighted_choice(self, candidates: np.ndarray, weights: np.ndarray) -> int:
        weights = np.asarray(weights, dtype=float)
        total = float(weights.sum())
        if total <= 0 or not np.isfinite(total) or not np.all(np.isfinite(weights)):
            self._fallback_count += 1
            self._uniform_fallbacks += 1
            return int(candidates[self._rng.randbelow(len(candidates))])
        threshold = (self._rng.randbelow(1 << 53) / float(1 << 53)) * total
        cumulative = 0.0
        for candidate, weight in zip(candidates, weights):
            cumulative += float(weight)
            if threshold < cumulative: return int(candidate)
        return int(candidates[-1])

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        scaled = np.asarray(scores, dtype=float) / self.config.temperature
        if not np.all(np.isfinite(scaled)):
            return np.ones(len(scaled), dtype=float)
        shifted = scaled - np.max(scaled)
        weights = np.exp(shifted)
        return weights if np.isfinite(weights).all() and weights.sum() > 0 else np.ones(len(scores))

    def _roll_size(self, placed_count: int) -> int:
        if placed_count <= 0: return 0
        if self.config.roll_mode == "all": size = placed_count
        elif self.config.roll_mode == "fixed": size = min(self.config.fixed_roll, placed_count)
        else: size = min(1 + self._rng.randbelow(self.config.max_roll), placed_count)
        self._roll_sizes.append(size)
        return size

    def _position_weights(self, job: int, free: np.ndarray, history: list[int],
                          positions: np.ndarray) -> np.ndarray:
        eps = np.finfo(float).tiny
        node = np.log(np.maximum(self.exact[job, free], eps))
        if not history:
            return self._softmax(node)
        window = history[-self._roll_size(len(history)):]
        if self.config.reference_selection != "mean":
            reference = self._select_reference(job, window)
            self._reference_counts[reference] += 1
            relative_scores = np.empty(len(free), dtype=float)
            offset = self.config.problem_size - 1
            distribution = self.relative[reference, job]
            informative = self.relative_counts is not None and self.relative_counts[reference, job].sum() > 0
            if not informative:
                self._exact_fallbacks += 1
                return self._softmax(node)
            for free_index, target in enumerate(free):
                delta = int(target - positions[reference])
                tensor_index = delta + offset
                relative_scores[free_index] = np.log(max(float(distribution[tensor_index]), eps)) if delta and 0 <= tensor_index < len(distribution) else np.log(eps)
            self._relative_samples += 1
            score = self.config.node_weight * node + (1.0 - self.config.node_weight) * relative_scores
            return self._softmax(score)
        relative_scores = np.empty(len(free), dtype=float)
        offset = self.config.problem_size - 1
        if self.config.sampling_mode == "mean_anchor":
            anchor = float(np.mean([positions[item] for item in window]))
        for free_index, target in enumerate(free):
            evidence = []
            for prior in window:
                delta = int(target - positions[prior]) if self.config.sampling_mode == "pairwise_window" else int(round(target - anchor))
                tensor_index = delta + offset
                probability = self.relative[prior, job, tensor_index] if delta and 0 <= tensor_index < self.relative.shape[2] else eps
                evidence.append(np.log(max(float(probability), eps)))
            relative_scores[free_index] = float(np.mean(evidence))
        score = self.config.node_weight * node + (1.0 - self.config.node_weight) * relative_scores
        return self._softmax(score)

    def _select_reference(self, job: int, window: list[int]) -> int:
        candidates = np.asarray(window, dtype=np.int16)
        if self.config.reference_selection == "uniform" or len(candidates) == 1:
            return int(candidates[self._rng.randbelow(len(candidates))])
        entropies = np.asarray([self._distribution_entropy(self.relative[int(reference), job]) for reference in candidates])
        confidence = 1.0 / np.maximum(entropies, 1e-9)
        median = float(np.median(confidence))
        if median > 0:
            confidence = np.minimum(confidence, median * 10.0)
        return self._weighted_choice(candidates, confidence)

    @staticmethod
    def _distribution_entropy(distribution: np.ndarray) -> float:
        positive = np.asarray(distribution, dtype=float)
        positive = positive[positive > 0]
        return float(-np.sum(positive * np.log(positive)))

    def _construct(self, template: np.ndarray | None = None,
                   sampled_positions: np.ndarray | None = None) -> np.ndarray:
        n = self.config.problem_size
        result = np.full(n, -1, dtype=np.int16)
        positions = np.full(n, -1, dtype=np.int16)
        if template is None:
            free = np.arange(n, dtype=np.int16)
            jobs = np.arange(n, dtype=np.int16)
            history: list[int] = []
            fixed_jobs_set: set[int] = set()
        else:
            free = np.asarray(sampled_positions, dtype=np.int16)
            fixed_mask = np.ones(n, dtype=bool); fixed_mask[free] = False
            result[fixed_mask] = template[fixed_mask]
            fixed_jobs = result[fixed_mask].astype(int)
            fixed_positions = np.flatnonzero(fixed_mask)
            positions[fixed_jobs] = fixed_positions
            jobs = template[free]
            centre = float(free.mean()) if len(free) else 0.0
            history = [int(result[position]) for position in sorted(fixed_positions, key=lambda p: -abs(float(p) - centre))]
            fixed_jobs_set = set(history)
            self._template_fixed += int(fixed_mask.sum())
            self._template_resampled += len(free)
        construction_order = self._shuffle(jobs)
        for job in construction_order:
            before = self._reference_counts.copy() if self.config.reference_selection != "mean" and history else None
            weights = self._position_weights(int(job), free, history, positions)
            position = self._weighted_choice(free, weights)
            if before is not None:
                chosen = np.flatnonzero(self._reference_counts > before)
                if len(chosen):
                    if int(chosen[0]) in fixed_jobs_set: self._fixed_reference_count += 1
                    else: self._new_reference_count += 1
                    self._sampled_distances.append(int(position - positions[int(chosen[0])]))
            result[position] = job; positions[int(job)] = position; history.append(int(job))
            free = free[free != position]
        if sorted(result.tolist()) != list(range(n)):
            raise RuntimeError("ROSE construction did not produce a valid permutation")
        return result

    def generate_population(self) -> np.ndarray:
        self._pending_parents = []; self._pending_parent_fitness = []
        if self.exact is None:
            population = np.vstack([self._random_permutation() for _ in range(self.config.population_size)])
        else:
            population = []
            order = self._shuffle(np.arange(len(self.templates), dtype=np.int16)) if self.config.template_enabled and self.templates is not None else None
            for candidate_index in range(self.config.population_size):
                if order is not None:
                    template_index = int(order[candidate_index % len(order)])
                    template = self.templates[template_index]
                    ratio = self.config.template_sample_ratio
                    if ratio == 0:
                        sampled = np.asarray([], dtype=np.int16)
                    else:
                        sampled = sample_template_positions(self._rng, self.config.problem_size, ratio)
                    self._pending_parents.append(template.copy())
                    self._pending_parent_fitness.append(float(self.template_fitness[template_index]))
                    population.append(template.copy() if not len(sampled) else self._construct(template, sampled))
                else:
                    population.append(self._construct())
            population = np.asarray(population, dtype=np.int16)
        self._offspring_total += len(population)
        self._offspring_unique += len({tuple(row) for row in population.tolist()})
        return population

    def _fit_estimators(self, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.config.problem_size
        exact = np.full((n, n), self.config.smoothing, dtype=float)
        relative = np.full((n, n, 2 * n - 1), self.config.smoothing, dtype=float)
        relative_counts = np.zeros((n, n, 2 * n - 1), dtype=np.int64)
        relative[:, :, n - 1] = 0.0
        for permutation in selected:
            position = np.empty(n, dtype=np.int64)
            position[permutation.astype(int)] = np.arange(n)
            exact[permutation.astype(int), np.arange(n)] += 1.0
            for first in range(n):
                for second in range(n):
                    if first != second:
                        index = position[second] - position[first] + n - 1
                        relative[first, second, index] += 1.0
                        relative_counts[first, second, index] += 1
        exact /= exact.sum(axis=1, keepdims=True)
        totals = relative.sum(axis=2, keepdims=True)
        relative = np.divide(relative, totals, out=np.zeros_like(relative), where=totals > 0)
        return exact, relative, relative_counts

    def statistics(self, population: np.ndarray, fitness: np.ndarray) -> tuple[RoseStatistics, None]:
        survivors = np.asarray(population, dtype=np.int16).copy()
        survivor_fitness = np.asarray(fitness, dtype=float).copy()
        if self.config.template_enabled and self._pending_parents:
            parents = np.asarray(self._pending_parents, dtype=np.int16)
            parent_fitness = np.asarray(self._pending_parent_fitness, dtype=float)
            improved = survivor_fitness < parent_fitness if self.config.objective == "min" else survivor_fitness > parent_fitness
            self._improved += int(improved.sum())
            survivors[~improved] = parents[~improved]; survivor_fitness[~improved] = parent_fitness[~improved]
        order = np.argsort(survivor_fitness)
        if self.config.objective == "max": order = order[::-1]
        count = max(1, round(len(survivors) * self.config.selection_ratio / 100))
        selected = survivors[order[:count]]
        exact, relative, relative_counts = self._fit_estimators(selected)
        self.effective_population = survivors; self.effective_fitness = survivor_fitness
        return RoseStatistics(exact, relative, relative_counts, selected.copy(), survivors, survivor_fitness), None

    def update(self, statistics: RoseStatistics, _unused: None) -> None:
        self.exact = statistics.exact; self.relative = statistics.relative; self.relative_counts = statistics.relative_counts
        self.templates = statistics.population.copy(); self.template_fitness = statistics.fitness.copy()

    @staticmethod
    def _entropy(distribution: np.ndarray, axis: int) -> float:
        terms = np.zeros_like(distribution)
        positive = distribution > 0
        terms[positive] = distribution[positive] * np.log(distribution[positive])
        return float(np.mean(-terms.sum(axis=axis)))

    def diagnostics(self) -> dict[str, object]:
        total = max(1, self._offspring_total)
        return {
            "exact_position_entropy": None if self.exact is None else self._entropy(self.exact, 1),
            "relative_position_entropy": None if self.relative is None else self._entropy(self.relative, 2),
            "unique_offspring_count": self._offspring_unique,
            "duplicate_rate": 1.0 - self._offspring_unique / total,
            "offspring_improvement_rate": self._improved / total,
            "sampling_fallback_count": self._fallback_count,
            "relative_sampling_count": self._relative_samples,
            "exact_position_fallback_count": self._exact_fallbacks,
            "uniform_fallback_count": self._uniform_fallbacks,
            "relative_fallback_rate": self._exact_fallbacks / max(1, self._relative_samples + self._exact_fallbacks),
            "reference_selection_frequency": {str(i): int(value) for i, value in enumerate(self._reference_counts) if value},
            "reference_entropy": self._distribution_entropy(self._reference_counts / max(1, self._reference_counts.sum())),
            "sampled_distance_mean": float(np.mean(self._sampled_distances)) if self._sampled_distances else 0.0,
            "sampled_distance_sd": float(np.std(self._sampled_distances)) if self._sampled_distances else 0.0,
            "mean_roll_size": float(np.mean(self._roll_sizes)) if self._roll_sizes else 0.0,
            "roll_size_distribution": {str(size): self._roll_sizes.count(size) for size in sorted(set(self._roll_sizes))},
            "template_fixed_positions": self._template_fixed,
            "template_resampled_positions": self._template_resampled,
            "template_fixed_reference_count": self._fixed_reference_count,
            "newly_placed_reference_count": self._new_reference_count,
        }

    def pair_statistics(self, reference: int, job: int) -> dict[str, float | int | None]:
        """Descriptive diagnostics over observed signed distances for one pair."""
        if self.relative_counts is None:
            raise RuntimeError("ROSE estimators have not been fitted")
        counts = self.relative_counts[reference, job]
        offset = self.config.problem_size - 1
        indices = np.flatnonzero(counts)
        count = int(counts.sum())
        if not count:
            return {"count": 0, "mean": None, "minimum": None, "maximum": None,
                    "standard_deviation": None, "entropy": None, "mode": None,
                    "observed_distance_values": 0}
        distances = indices - offset
        probabilities = counts[indices] / count
        mean = float(np.sum(distances * probabilities))
        variance = float(np.sum(((distances - mean) ** 2) * probabilities))
        return {
            "count": count, "mean": mean, "minimum": int(distances.min()),
            "maximum": int(distances.max()), "standard_deviation": float(np.sqrt(variance)),
            "entropy": self._distribution_entropy(probabilities),
            "mode": int(distances[int(np.argmax(counts[indices]))]),
            "observed_distance_values": int(len(indices)),
        }


class TemplateROSE(ROSE):
    def __init__(self, config: RoseConfig, *, seed: int = 0):
        if not config.template_enabled:
            from dataclasses import replace
            config = replace(config, template_enabled=True)
        super().__init__(config, seed=seed)


class ROSESingleRef(ROSE):
    """ROSE v2: sample one real reference from the active candidate window."""
    def __init__(self, config: RoseConfig, *, seed: int = 0):
        if config.reference_selection == "mean":
            from dataclasses import replace
            config = replace(config, reference_selection="uniform")
        super().__init__(config, seed=seed)


class TemplateROSESingleRef(TemplateROSE):
    """Template ROSE v2 with one real reference per regenerated job."""
    def __init__(self, config: RoseConfig, *, seed: int = 0):
        if config.reference_selection == "mean":
            from dataclasses import replace
            config = replace(config, reference_selection="uniform")
        super().__init__(config, seed=seed)
