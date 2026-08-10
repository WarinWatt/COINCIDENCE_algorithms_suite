"""Node-template / Edge-completion Hybrid COIN."""

from __future__ import annotations

import numpy as np

from coin.core.random_state import SplitMix64
from .edge import EdgeConfig
from .edge_reference import ReferenceEdgeCoin
from .position import PositionCoin


class HybridCoin:
    """Punch a Node/Position template, then complete it with Edge COIN.

    Position zero is always retained.  For every candidate, a uniformly drawn
    30--70 percent of positions are retained and scattered across the
    permutation. Edge probabilities fill the holes while reserving values
    needed by future template positions. Both models learn from the completed
    population.
    """

    variant_name = "hybrid_coin"
    template_min_percent = 30
    template_max_percent = 70

    def __init__(self, config: EdgeConfig, *, seed: int = 0):
        self.config = config
        self.edge = ReferenceEdgeCoin(config, seed=seed)
        self.position = PositionCoin(config, seed=seed)
        self._rng = SplitMix64(seed ^ 0x485942524944)
        self.last_template_masks = np.zeros(
            (config.population_size, config.problem_size), dtype=bool
        )

    @property
    def edge_weights(self) -> np.ndarray:
        return self.edge.weights

    @property
    def position_weights(self) -> np.ndarray:
        return self.position.weights

    def _template_mask(self, size: int) -> np.ndarray:
        ratio = self.template_min_percent + self._rng.randbelow(
            self.template_max_percent - self.template_min_percent + 1
        )
        retained = max(1, min(size, round(size * ratio / 100)))
        positions = list(range(1, size))
        for index in range(len(positions) - 1, 0, -1):
            other = self._rng.randbelow(index + 1)
            positions[index], positions[other] = positions[other], positions[index]
        mask = np.zeros(size, dtype=bool)
        mask[0] = True
        for position in positions[: retained - 1]:
            mask[position] = True
        return mask

    def _edge_complete(self, template: np.ndarray, mask: np.ndarray) -> np.ndarray:
        size = len(template)
        result = np.full(size, -1, dtype=np.int16)
        used = np.zeros(size, dtype=bool)
        reserved = np.zeros(size, dtype=bool)
        reserved[template[mask]] = True
        cursor = 0
        prior = -1

        for position in range(size):
            if mask[position]:
                selected = int(template[position])
                reserved[selected] = False
            else:
                allowed = ~used & ~reserved
                if prior < 0:
                    weights = np.ones(size, dtype=np.int64)
                else:
                    weights = self.edge.weights[prior]
                total = sum(
                    int(weights[node]) for node in range(size)
                    if allowed[node] and weights[node] > 0
                )
                if total <= 0:
                    candidates = np.flatnonzero(allowed)
                    selected = int(candidates[self._rng.randbelow(len(candidates))])
                else:
                    threshold = self._rng.randbelow(total)
                    cumulative = 0
                    selected = -1
                    for offset in range(size):
                        node = (cursor + offset) % size
                        if allowed[node] and weights[node] > 0:
                            cumulative += int(weights[node])
                            if threshold < cumulative:
                                selected = node
                                cursor = (node + 1) % size
                                break
                    if selected < 0:
                        raise RuntimeError("hybrid edge completion failed")
            result[position] = selected
            used[selected] = True
            prior = selected
        return result

    def generate_population(self) -> np.ndarray:
        templates = self.position.generate_population()
        population = np.empty_like(templates)
        for row, template in enumerate(templates):
            mask = self._template_mask(self.config.problem_size)
            self.last_template_masks[row] = mask
            population[row] = self._edge_complete(template, mask)
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
