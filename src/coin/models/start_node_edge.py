"""Start-Node Edge COIN with independent first-job and edge models."""

from __future__ import annotations

import numpy as np

from coin.core.random_state import SplitMix64
from coin.learning.statistics import count_edge_cycles, ranked_population, selected_bounds
from coin.learning.reward_punishment import update_edge_weights
from .edge import EdgeConfig, initial_edge_weights, validate_weight_matrix


class StartNodeEdgeCoin:
    variant_name = "start_node_edge"

    def __init__(self, config: EdgeConfig, *, seed: int = 0):
        self.config = config
        initial = round(config.problem_size * 100 / config.training_rate)
        self.start_node_weights = np.full(config.problem_size, initial, dtype=np.int64)
        self.edge_weights = initial_edge_weights(config)
        self._rng = SplitMix64(seed)

    def _sample(self, weights: np.ndarray, available: np.ndarray, cursor: int = 0) -> tuple[int, int]:
        total = int(weights[available].sum())
        if total <= 0:
            raise RuntimeError("no positive available weight remains")
        threshold = self._rng.randbelow(total)
        cumulative = 0
        n = weights.size
        for offset in range(n):
            job = (cursor + offset) % n
            if available[job] and weights[job] > 0:
                cumulative += int(weights[job])
                if threshold < cumulative:
                    return job, (job + 1) % n
        raise RuntimeError("roulette selection failed")

    def generate_population(self) -> np.ndarray:
        n = self.config.problem_size
        population = np.empty((self.config.population_size, n), dtype=np.int16)
        for candidate_index in range(self.config.population_size):
            available = np.ones(n, dtype=bool)
            first, cursor = self._sample(self.start_node_weights, available)
            population[candidate_index, 0] = first
            available[first] = False
            current = first
            for position in range(1, n):
                following, cursor = self._sample(self.edge_weights[current], available, cursor)
                population[candidate_index, position] = following
                available[following] = False
                current = following
        return population

    def statistics(self, population: np.ndarray, fitness: np.ndarray):
        ranked = ranked_population(population, fitness)
        start_reward = np.zeros_like(self.start_node_weights)
        start_punishment = np.zeros_like(self.start_node_weights)
        edge_reward = np.zeros_like(self.edge_weights)
        edge_punishment = np.zeros_like(self.edge_weights)
        if self.config.rewards_enabled:
            low, high = selected_bounds(len(ranked), self.config.reward_ratio, self.config.objective, True)
            for candidate in ranked[low:high]:
                start_reward[int(candidate[0])] += 1
            edge_reward = count_edge_cycles(ranked, low, high, self.config.problem_size)
        if self.config.punishments_enabled:
            low, high = selected_bounds(len(ranked), self.config.punishment_ratio, self.config.objective, False)
            for candidate in ranked[low:high]:
                start_punishment[int(candidate[0])] += 1
            edge_punishment = count_edge_cycles(ranked, low, high, self.config.problem_size)
        return (start_reward, edge_reward), (start_punishment, edge_punishment)

    def update(self, rewards, punishments) -> None:
        start_rewards, edge_rewards = rewards
        start_punishments, edge_punishments = punishments
        n = self.config.problem_size
        if self.config.learning_mode == "reconstruction":
            self.start_node_weights[:] = start_rewards * 10 + 1
            for prior in range(n):
                for following in range(n):
                    if prior != following:
                        self.edge_weights[prior, following] = edge_rewards[prior, following] * 10 + 1
            validate_weight_matrix(self.edge_weights)
            return
        for job in range(n):
            for _ in range(int(start_punishments[job])):
                if self.start_node_weights[job] > n:
                    self.start_node_weights[:] += 1
                    self.start_node_weights[job] -= n
        for job in range(n):
            for _ in range(int(start_rewards[job])):
                for donor in range(n):
                    if self.start_node_weights[donor] > n:
                        self.start_node_weights[job] += 1
                        self.start_node_weights[donor] -= 1

        update_edge_weights(self.edge_weights, self.config, edge_rewards, edge_punishments)


# Descriptive compatibility alias matching the specification's suggested name.
StartNodeEdgeCoinModel = StartNodeEdgeCoin
