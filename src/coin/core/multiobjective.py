"""Reusable multi-objective execution for every learnable COIN model."""
from __future__ import annotations
import numpy as np
from coin.learning.pareto import nondominated_ranks, pareto_quality, pareto_selection_scores
from coin.problems.base import PermutationProblem, evaluate_population
from .algorithm import LearnablePermutationModel

class MultiObjectiveCoinAlgorithm:
    def __init__(self, model: LearnablePermutationModel, problem: PermutationProblem):
        if len(problem.objective_names) < 2:
            raise ValueError("multi-objective COIN requires at least two objectives")
        if getattr(model.config, "problem_size", None) != problem.dimension:
            raise ValueError("model problem_size must match problem dimension")
        self.model, self.problem = model, problem
        self.generation = self.evaluations = 0
        self.archive_population = np.empty((0, problem.dimension), dtype=np.int16)
        self.archive_values = np.empty((0, len(problem.objective_names)), dtype=float)
        self.history = []

    def step(self):
        population = self.model.generate_population()
        values = evaluate_population(self.problem, population)
        rewards, punishments = self.model.statistics(population, pareto_selection_scores(values))
        self.model.update(rewards, punishments)
        all_population = np.vstack((self.archive_population, population))
        all_values = np.vstack((self.archive_values, values))
        keep = nondominated_ranks(all_values) == 0
        unique = {}
        for permutation, objective_values in zip(all_population[keep], all_values[keep]):
            unique.setdefault(tuple(int(x) for x in permutation), objective_values)
        self.archive_population = np.asarray(list(unique), dtype=np.int16)
        self.archive_values = np.asarray(list(unique.values()), dtype=float)
        self.generation += 1
        self.evaluations += len(population)
        depths, spread = pareto_quality(values)
        self.history.append((self.generation, self.evaluations, self.archive_values.min(axis=0).tolist(), depths, spread))
        return population, values
