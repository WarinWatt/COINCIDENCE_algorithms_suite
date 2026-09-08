"""Empirical-histogram SingleRef variants retained as a research baseline."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .rose import ROSESingleRef, RoseConfig


class ROSESingleRefHistogram(ROSESingleRef):
    """Select one reference and sample from its empirical distance histogram.

    This deliberately dense baseline uses ``O(n^3)`` memory. It is kept
    separately from :class:`ROSESingleRef`, whose range estimator is ``O(n^2)``.
    """

    def __init__(self, config: RoseConfig, *, seed: int = 0):
        super().__init__(config, seed=seed)
        self.relative: np.ndarray | None = None
        self.relative_counts: np.ndarray | None = None
        self._pending_relative: np.ndarray | None = None
        self._pending_relative_counts: np.ndarray | None = None

    def _fit_estimators(self, selected: np.ndarray) -> tuple[np.ndarray, ...]:
        compact = super()._fit_estimators(selected)
        n = self.config.problem_size
        counts = np.zeros((n, n, 2 * n - 1), dtype=np.int32)
        for permutation in selected:
            positions = np.empty(n, dtype=np.int32)
            positions[permutation.astype(int)] = np.arange(n)
            distances = positions[None, :] - positions[:, None]
            first, second = np.where(~np.eye(n, dtype=bool))
            counts[first, second, distances[first, second] + n - 1] += 1
        probabilities = counts.astype(np.float64) + self.config.smoothing
        probabilities[:, :, n - 1] = 0.0
        totals = probabilities.sum(axis=2, keepdims=True)
        probabilities = np.divide(probabilities, totals, out=np.zeros_like(probabilities), where=totals > 0)
        self._pending_relative = probabilities
        self._pending_relative_counts = counts
        return compact

    def update(self, statistics, unused) -> None:
        super().update(statistics, unused)
        self.relative = self._pending_relative
        self.relative_counts = self._pending_relative_counts

    def _position_weights(self, job: int, free: np.ndarray, history: list[int],
                          positions: np.ndarray) -> np.ndarray:
        eps = np.finfo(float).tiny
        node = np.log(np.maximum(self.exact[job, free], eps))
        if not history or self.relative is None:
            return self._softmax(node)
        window = history[-self._roll_size(len(history)):]
        reference = self._select_reference(job, window)
        self._reference_counts[reference] += 1
        counts = self.relative_counts[reference, job]
        if counts.sum() == 0:
            self._exact_fallbacks += 1
            return self._softmax(node)
        offset = self.config.problem_size - 1
        scores = np.full(len(free), np.log(eps), dtype=float)
        for index, target in enumerate(free):
            delta = int(target - positions[reference])
            tensor_index = delta + offset
            if delta != 0 and 0 <= tensor_index < self.relative.shape[2]:
                scores[index] = np.log(max(float(self.relative[reference, job, tensor_index]), eps))
        self._relative_samples += 1
        return self._softmax(self.config.node_weight * node + (1.0 - self.config.node_weight) * scores)

    def _select_reference(self, job: int, window: list[int]) -> int:
        candidates = np.asarray(window, dtype=np.int16)
        if self.config.reference_selection == "uniform" or len(candidates) == 1 or self.relative is None:
            return int(candidates[self._rng.randbelow(len(candidates))])
        entropies = np.asarray([
            self._distribution_entropy(self.relative[int(reference), job]) for reference in candidates
        ])
        confidence = 1.0 / np.maximum(entropies, 1e-9)
        median = float(np.median(confidence))
        if median > 0:
            confidence = np.minimum(confidence, median * 10.0)
        return self._weighted_choice(candidates, confidence)

    def diagnostics(self) -> dict[str, object]:
        result = super().diagnostics()
        if self.relative is not None:
            result.update({
                "estimator_family": "empirical_histogram",
                "relative_histogram_entropy": self._entropy(self.relative, 2),
                "histogram_memory_bytes": int(self.relative.nbytes + self.relative_counts.nbytes),
                "estimator_memory_bytes": int(result["estimator_memory_bytes"] + self.relative.nbytes + self.relative_counts.nbytes),
            })
        return result

    def pair_statistics(self, reference: int, job: int) -> dict[str, float | int | None]:
        if self.relative_counts is None:
            raise RuntimeError("ROSE histogram estimator has not been fitted")
        counts = self.relative_counts[reference, job]
        indices = np.flatnonzero(counts)
        count = int(counts.sum())
        if not count:
            return {"count": 0, "mean": None, "minimum": None, "maximum": None,
                    "standard_deviation": None, "entropy": None, "mode": None,
                    "observed_distance_values": 0}
        distances = indices - (self.config.problem_size - 1)
        probabilities = counts[indices] / count
        mean = float(np.sum(distances * probabilities))
        variance = float(np.sum((distances - mean) ** 2 * probabilities))
        return {
            "count": count,
            "mean": mean,
            "minimum": int(distances.min()),
            "maximum": int(distances.max()),
            "standard_deviation": float(np.sqrt(variance)),
            "entropy": self._distribution_entropy(probabilities),
            "mode": int(distances[int(np.argmax(counts[indices]))]),
            "observed_distance_values": int(len(indices)),
        }


class TemplateROSESingleRefHistogram(ROSESingleRefHistogram):
    """Punched-template form of the empirical-histogram SingleRef model."""

    def __init__(self, config: RoseConfig, *, seed: int = 0):
        if not config.template_enabled:
            config = replace(config, template_enabled=True)
        super().__init__(config, seed=seed)
