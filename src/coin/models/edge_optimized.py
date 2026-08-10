"""NumPy/Numba implementation behaviorally equivalent to the reference model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numba import njit

from .edge import EdgeConfig, GenerationRecord, initial_edge_weights, validate_weight_matrix

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_INCREMENT = np.uint64(0x9E3779B97F4A7C15)


@njit(cache=False)
def _randbelow(state: np.uint64, bound: int) -> tuple[np.uint64, int]:
    if bound <= 0:
        return state, np.int64(0)
    state = state + _INCREMENT
    value = state
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    value = value ^ (value >> np.uint64(31))
    return state, np.int64(value % np.uint64(bound))


@njit(cache=False)
def _generate(weights: np.ndarray, population_size: int, state: np.uint64) -> tuple[np.ndarray, np.uint64]:
    n = weights.shape[0]
    population = np.empty((population_size, n), dtype=np.int16)
    used = np.empty(n, dtype=np.uint8)
    for candidate_index in range(population_size):
        used[:] = 0
        state, prior = _randbelow(state, n)
        state, cursor = _randbelow(state, n)
        population[candidate_index, 0] = prior
        used[prior] = 1
        for position in range(1, n):
            total = 0
            for node in range(n):
                if used[node] == 0 and weights[prior, node] > 0:
                    total += weights[prior, node]
            if total <= 0:
                raise RuntimeError("no positive unused outgoing edge remains")
            state, threshold = _randbelow(state, total)
            cumulative = 0
            selected = -1
            for offset in range(n):
                node = (cursor + offset) % n
                if used[node] == 0 and weights[prior, node] > 0:
                    cumulative += weights[prior, node]
                    if threshold < cumulative:
                        selected = node
                        cursor = (node + 1) % n
                        break
            if selected < 0:
                raise RuntimeError("roulette selection failed")
            population[candidate_index, position] = selected
            used[selected] = 1
            prior = selected
    return population, state


@njit(cache=False)
def _count_selected_cycles(ranked: np.ndarray, low: int, high: int) -> np.ndarray:
    counts = np.zeros((ranked.shape[1], ranked.shape[1]), dtype=np.int64)
    for candidate_index in range(low, high):
        for position in range(ranked.shape[1]):
            prior = ranked[candidate_index, position]
            following = ranked[candidate_index, (position + 1) % ranked.shape[1]]
            counts[prior, following] += 1
    return counts


@njit(cache=False)
def _conservative_update(weights: np.ndarray, rewards: np.ndarray, punishments: np.ndarray) -> None:
    n = weights.shape[0]
    for following in range(n):
        for prior in range(n):
            for _ in range(punishments[prior, following]):
                if weights[prior, following] > n:
                    for competitor in range(n):
                        if competitor != prior:
                            weights[prior, competitor] += 1
                    weights[prior, following] -= n - 1
    for following in range(n):
        for prior in range(n):
            for _ in range(rewards[prior, following]):
                for competitor in range(n):
                    if weights[prior, competitor] > n:
                        weights[prior, following] += 1
                        weights[prior, competitor] -= 1


@njit(cache=False)
def _reconstruct(weights: np.ndarray, rewards: np.ndarray) -> None:
    n = weights.shape[0]
    for prior in range(n):
        for following in range(n):
            if prior != following:
                weights[prior, following] = rewards[prior, following] * 10 + 1


def _selection_bounds(size: int, ratio: int, objective: str, reward: bool) -> tuple[int, int]:
    loop = round(size * ratio / 100)
    take_top = (objective == "min") == reward
    return (0, loop) if take_top else (max(0, size - loop - 1), size)


class OptimizedEdgeCoin:
    def __init__(self, config: EdgeConfig, *, seed: int = 0):
        self.config = config
        self.weights = initial_edge_weights(config)
        self._state = np.uint64(seed)
        self.generation = 0
        self.history: list[GenerationRecord] = []

    def generate_population(self) -> np.ndarray:
        population, state = _generate(
            self.weights, self.config.population_size, self._state
        )
        # Numba boxes uint64 scalars as Python ints. Convert back explicitly so
        # subsequent calls compile/use the same unsigned RNG state signature.
        self._state = np.uint64(int(state) & 0xFFFFFFFFFFFFFFFF)
        return population

    def statistics(self, population: np.ndarray, fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(fitness, kind="stable")
        ranked = np.ascontiguousarray(population[order], dtype=np.int16)
        shape = self.weights.shape
        rewards = np.zeros(shape, dtype=np.int64)
        punishments = np.zeros(shape, dtype=np.int64)
        if self.config.rewards_enabled:
            low, high = _selection_bounds(
                len(ranked), self.config.reward_ratio, self.config.objective, True
            )
            rewards = _count_selected_cycles(ranked, low, high)
        if self.config.punishments_enabled:
            low, high = _selection_bounds(
                len(ranked), self.config.punishment_ratio, self.config.objective, False
            )
            punishments = _count_selected_cycles(ranked, low, high)
        return rewards, punishments

    def update(self, rewards: np.ndarray, punishments: np.ndarray) -> None:
        if self.config.learning_mode == "reconstruction":
            _reconstruct(self.weights, rewards)
        else:
            _conservative_update(self.weights, rewards, punishments)
        validate_weight_matrix(self.weights)

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
