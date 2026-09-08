"""Chain-wise Node/Edge Hybrid COIN."""

from __future__ import annotations

import numpy as np

from coin.core.random_state import SplitMix64
from .edge import EdgeConfig
from .edge_reference import ReferenceEdgeCoin
from .position import PositionCoin


class HybridChainCoin:
    """Start from Node[position=0], then choose Node or Edge at each link."""

    variant_name = "hybrid_chain"

    def __init__(self, config: EdgeConfig, *, seed: int = 0):
        self.config = config
        self.edge = ReferenceEdgeCoin(config, seed=seed)
        self.position = PositionCoin(config, seed=seed)
        self._rng = SplitMix64(seed ^ 0x434841494E)
        self.last_source_masks = np.zeros(
            (config.population_size, config.problem_size), dtype=bool
        )

    @property
    def edge_weights(self) -> np.ndarray:
        return self.edge.weights

    @property
    def position_weights(self) -> np.ndarray:
        return self.position.weights

    def _sample(self, weights: np.ndarray, used: np.ndarray, cursor: int) -> tuple[int, int]:
        size = len(used)
        total = sum(int(weights[node]) for node in range(size) if not used[node] and weights[node] > 0)
        if total <= 0:
            candidates = np.flatnonzero(~used)
            selected = int(candidates[self._rng.randbelow(len(candidates))])
            return selected, (selected + 1) % size
        threshold = self._rng.randbelow(total)
        cumulative = 0
        for offset in range(size):
            node = (cursor + offset) % size
            if not used[node] and weights[node] > 0:
                cumulative += int(weights[node])
                if threshold < cumulative:
                    return node, (node + 1) % size
        raise RuntimeError("hybrid chain roulette selection failed")

    def generate_population(self) -> np.ndarray:
        size = self.config.problem_size
        population = np.empty((self.config.population_size, size), dtype=np.int16)
        for row in range(self.config.population_size):
            used = np.zeros(size, dtype=bool)
            cursor = 0
            selected, cursor = self._sample(self.position.weights[0], used, cursor)
            population[row, 0] = selected
            used[selected] = True
            self.last_source_masks[row, 0] = True  # True means Node/Position.
            prior = selected
            for position in range(1, size):
                use_node = self._rng.randbelow(2) == 0
                weights = self.position.weights[position] if use_node else self.edge.weights[prior]
                selected, cursor = self._sample(weights, used, cursor)
                population[row, position] = selected
                used[selected] = True
                self.last_source_masks[row, position] = use_node
                prior = selected
        return population

    def statistics(self, population: np.ndarray, fitness: np.ndarray):
        return (
            self.edge.statistics(population, fitness),
            self.position.statistics(population, fitness),
        )

    def update(self, rewards, punishments) -> None:
        edge_stats, position_stats = rewards, punishments
        self.edge.update(*edge_stats)
        self.position.update(*position_stats)
