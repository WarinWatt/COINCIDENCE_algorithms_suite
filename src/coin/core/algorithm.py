"""Problem-independent execution of a single-objective COIN model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from coin.problems.base import PermutationProblem, evaluate_population


class LearnablePermutationModel(Protocol):
    config: object

    def generate_population(self) -> np.ndarray: ...
    def statistics(self, population: np.ndarray, fitness: np.ndarray) -> tuple[object, object]: ...
    def update(self, rewards: object, punishments: object) -> None: ...


@dataclass(frozen=True, slots=True)
class CoinGeneration:
    generation: int
    evaluations: int
    best: float
    average: float
    worst: float


class PermutationCoinAlgorithm:
    """Run any learnable COIN model against a `PermutationProblem`.

    COIN's legacy cohort selection is scalar. Multi-objective selection is
    deliberately rejected rather than silently scalarized.
    """

    def __init__(self, model: LearnablePermutationModel, problem: PermutationProblem):
        if len(problem.objective_names) != 1:
            raise ValueError("COIN currently requires exactly one objective; scalarization is not implicit")
        problem_size = getattr(model.config, "problem_size", None)
        if problem_size != problem.dimension:
            raise ValueError("model problem_size must match problem dimension")
        self.model = model
        self.problem = problem
        self.generation = 0
        self.evaluations = 0
        self.history: list[CoinGeneration] = []

    @property
    def objective(self) -> str:
        return str(getattr(self.model.config, "objective"))

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        population = self.model.generate_population()
        objective_matrix = evaluate_population(self.problem, population)
        fitness = objective_matrix[:, 0]
        rewards, punishments = self.model.statistics(population, fitness)
        self.model.update(rewards, punishments)
        effective_population = getattr(self.model, "effective_population", None)
        effective_fitness = getattr(self.model, "effective_fitness", None)
        if effective_population is not None and effective_fitness is not None:
            population = np.asarray(effective_population)
            fitness = np.asarray(effective_fitness)
            objective_matrix = fitness.reshape(-1, 1)
        self.generation += 1
        self.evaluations += population.shape[0]
        best = float(fitness.min() if self.objective == "min" else fitness.max())
        worst = float(fitness.max() if self.objective == "min" else fitness.min())
        self.history.append(
            CoinGeneration(self.generation, self.evaluations, best, float(fitness.mean()), worst)
        )
        return population, objective_matrix

    def run(self, generations: int) -> list[CoinGeneration]:
        if generations < 0:
            raise ValueError("generations must be non-negative")
        for _ in range(generations):
            self.step()
        return self.history
