"""Readable Node/Position COIN migrated from the Delphi model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coin.core.random_state import SplitMix64
from coin.learning.statistics import count_positions, ranked_population, selected_bounds
from .edge import EdgeConfig

PositionConfig = EdgeConfig


def initial_position_weights(config: PositionConfig) -> np.ndarray:
    initial = round(config.problem_size * 100 / config.training_rate)
    return np.full((config.problem_size, config.problem_size), initial, dtype=np.int64)


class PositionCoin:
    """Learns `weights[position, job]` and samples without replacement."""

    def __init__(self, config: PositionConfig, *, seed: int = 0):
        self.config = config
        self.weights = initial_position_weights(config)
        self._rng = SplitMix64(seed)

    def _sample_value(self, position: int, used: np.ndarray, cursor: int) -> tuple[int, int]:
        n = self.config.problem_size
        total = int(self.weights[position, ~used].sum())
        if total <= 0: raise RuntimeError("no positive unused position weight remains")
        threshold=self._rng.randbelow(total); cumulative=0
        for offset in range(n):
            job=(cursor+offset)%n
            if not used[job] and self.weights[position,job]>0:
                cumulative+=int(self.weights[position,job])
                if threshold<cumulative: return job,(job+1)%n
        raise RuntimeError("position roulette selection failed")

    def generate_population(self) -> np.ndarray:
        n=self.config.problem_size; population=np.empty((self.config.population_size,n),dtype=np.int16)
        for candidate_index in range(self.config.population_size):
            used=np.zeros(n,dtype=bool); cursor=0; positions=list(range(n))
            for index in range(n-1,0,-1):
                other=self._rng.randbelow(index+1); positions[index],positions[other]=positions[other],positions[index]
            for position in positions:
                selected,cursor=self._sample_value(position,used,cursor)
                population[candidate_index,position]=selected; used[selected]=True
        return population

    def statistics(self, population: np.ndarray, fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ranked = ranked_population(population, fitness)
        rewards = np.zeros_like(self.weights)
        punishments = np.zeros_like(self.weights)
        if self.config.rewards_enabled:
            low, high = selected_bounds(len(ranked), self.config.reward_ratio, self.config.objective, True)
            rewards = count_positions(ranked, low, high, self.config.problem_size)
        if self.config.punishments_enabled:
            low, high = selected_bounds(len(ranked), self.config.punishment_ratio, self.config.objective, False)
            punishments = count_positions(ranked, low, high, self.config.problem_size)
        return rewards, punishments

    def update(self, rewards: np.ndarray, punishments: np.ndarray) -> None:
        n = self.config.problem_size
        if rewards.shape != self.weights.shape or punishments.shape != self.weights.shape:
            raise ValueError("position statistics must match the position matrix")
        if self.config.learning_mode == "reconstruction":
            self.weights[:] = rewards * 10 + 1
            return
        for position in range(n):
            for job in range(n):
                for _ in range(int(punishments[position, job])):
                    if self.weights[position, job] > n:
                        self.weights[position, :] += 1
                        self.weights[position, job] -= n
        for position in range(n):
            for job in range(n):
                for _ in range(int(rewards[position, job])):
                    for donor in range(n):
                        if self.weights[position, donor] > n:
                            self.weights[position, job] += 1
                            self.weights[position, donor] -= 1

    def step(self, evaluator) -> tuple[np.ndarray, np.ndarray]:
        population = self.generate_population()
        fitness = np.asarray(evaluator(population))
        if fitness.shape != (self.config.population_size,):
            raise ValueError("evaluator returned an invalid fitness shape")
        rewards, punishments = self.statistics(population, fitness)
        self.update(rewards, punishments)
        return population, fitness
