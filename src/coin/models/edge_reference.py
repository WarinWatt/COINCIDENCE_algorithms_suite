"""Readable, loop-oriented reference implementation of Edge COIN."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from coin.core.random_state import SplitMix64
from coin.learning.reward_punishment import update_edge_weights
from .edge import EdgeConfig, GenerationRecord, initial_edge_weights


def _selected_indices(size: int, ratio: int, objective: str, reward: bool) -> range:
    loop = round(size * ratio / 100)
    take_top = (objective == "min") == reward
    if take_top:
        return range(0, loop)
    # Delphi rows population-loop..population map to these zero-based indices.
    return range(max(0, size - loop - 1), size)


class ReferenceEdgeCoin:
    def __init__(self, config: EdgeConfig, *, seed: int = 0):
        self.config = config
        self.weights = initial_edge_weights(config)
        self._rng = SplitMix64(seed)
        self.generation = 0
        self.history: list[GenerationRecord] = []

    def generate_population(self) -> np.ndarray:
        n = self.config.problem_size
        population = np.empty((self.config.population_size, n), dtype=np.int16)
        for candidate_index in range(self.config.population_size):
            used = [False] * n
            prior = self._rng.randbelow(n)
            cursor = self._rng.randbelow(n)
            population[candidate_index, 0] = prior
            used[prior] = True
            for position in range(1, n):
                total = sum(
                    int(self.weights[prior, node])
                    for node in range(n)
                    if not used[node] and self.weights[prior, node] > 0
                )
                if total <= 0:
                    raise RuntimeError("no positive unused outgoing edge remains")
                threshold = self._rng.randbelow(total)
                cumulative = 0
                selected = -1
                for offset in range(n):
                    node = (cursor + offset) % n
                    if not used[node] and self.weights[prior, node] > 0:
                        cumulative += int(self.weights[prior, node])
                        if threshold < cumulative:
                            selected = node
                            cursor = (node + 1) % n
                            break
                if selected < 0:
                    raise RuntimeError("roulette selection failed")
                population[candidate_index, position] = selected
                used[selected] = True
                prior = selected
        return population

    def statistics(self, population: np.ndarray, fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = sorted(range(len(fitness)), key=lambda index: int(fitness[index]))
        ranked = population[np.asarray(order)]
        rewards = np.zeros_like(self.weights)
        punishments = np.zeros_like(self.weights)
        if self.config.rewards_enabled:
            indices = _selected_indices(
                len(ranked), self.config.reward_ratio, self.config.objective, True
            )
            self._count_cycles(ranked, indices, rewards)
        if self.config.punishments_enabled:
            indices = _selected_indices(
                len(ranked), self.config.punishment_ratio, self.config.objective, False
            )
            self._count_cycles(ranked, indices, punishments)
        return rewards, punishments

    @staticmethod
    def _count_cycles(population: np.ndarray, indices: range, output: np.ndarray) -> None:
        for candidate_index in indices:
            candidate = population[candidate_index]
            for position in range(candidate.size):
                prior = int(candidate[position])
                following = int(candidate[(position + 1) % candidate.size])
                output[prior, following] += 1

    def update(self, rewards: np.ndarray, punishments: np.ndarray) -> None:
        update_edge_weights(self.weights, self.config, rewards, punishments)

    def step(self, evaluator: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        population = self.generate_population()
        fitness = np.asarray(evaluator(population), dtype=np.int16)
        if fitness.shape != (self.config.population_size,):
            raise ValueError("evaluator returned an invalid fitness shape")
        rewards, punishments = self.statistics(population, fitness)
        self.update(rewards, punishments)
        self.generation += 1
        best = int(fitness.max() if self.config.objective == "max" else fitness.min())
        self.history.append(GenerationRecord(self.generation, best, float(fitness.mean())))
        return population, fitness

    def run(
        self,
        generations: int,
        evaluator: Callable[[np.ndarray], np.ndarray],
        *,
        target_score: int | None = None,
    ) -> list[GenerationRecord]:
        for _ in range(generations):
            self.step(evaluator)
            if target_score is not None:
                best = self.history[-1].best
                reached = best >= target_score if self.config.objective == "max" else best <= target_score
                if reached:
                    break
        return self.history
