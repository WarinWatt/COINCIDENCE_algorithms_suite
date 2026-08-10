"""Reusable single- and multi-objective symmetric TSP contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class TSPInstance:
    """A closed-tour TSP instance with one matrix per minimized objective."""

    id: str
    name: str
    matrices: Mapping[str, np.ndarray]
    description: str = ""

    def __post_init__(self) -> None:
        copied: dict[str, np.ndarray] = {}
        dimension: int | None = None
        for objective, raw in self.matrices.items():
            matrix = np.asarray(raw, dtype=np.float64).copy()
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"{objective} must be a square matrix")
            if dimension is None:
                dimension = matrix.shape[0]
            if matrix.shape != (dimension, dimension):
                raise ValueError("all objective matrices must have the same shape")
            if np.any(matrix < 0) or not np.allclose(np.diag(matrix), 0):
                raise ValueError("TSP costs must be non-negative with a zero diagonal")
            matrix.setflags(write=False)
            copied[str(objective)] = matrix
        if not copied or dimension is None or dimension < 3 or dimension > 24:
            raise ValueError("a preloaded TSP instance must contain 3..24 cities")
        object.__setattr__(self, "matrices", MappingProxyType(copied))

    @property
    def dimension(self) -> int:
        return next(iter(self.matrices.values())).shape[0]

    @property
    def available_objective_names(self) -> tuple[str, ...]:
        return tuple(self.matrices)


class TSPProblem:
    """Evaluate a zero-based permutation as a closed tour (last city to first)."""

    def __init__(self, instance: TSPInstance, objectives: tuple[str, ...] | None = None):
        self.instance = instance
        self._objective_names = objectives or instance.available_objective_names
        if not self._objective_names or len(set(self._objective_names)) != len(self._objective_names):
            raise ValueError("objectives must be non-empty and unique")
        unknown = set(self._objective_names) - set(instance.matrices)
        if unknown:
            raise ValueError(f"unknown TSP objectives: {sorted(unknown)}")
        self._stack = np.stack([instance.matrices[name] for name in self._objective_names])

    @property
    def dimension(self) -> int:
        return self.instance.dimension

    @property
    def objective_names(self) -> tuple[str, ...]:
        return tuple(self._objective_names)

    def validate(self, permutation: np.ndarray) -> None:
        candidate = np.asarray(permutation)
        if candidate.shape != (self.dimension,) or not np.array_equal(
            np.sort(candidate), np.arange(self.dimension)
        ):
            raise ValueError("candidate must be a complete zero-based permutation")

    def evaluate(self, permutation: np.ndarray) -> np.ndarray:
        candidate = np.asarray(permutation, dtype=np.int64)
        self.validate(candidate)
        return self._stack[:, candidate, np.roll(candidate, -1)].sum(axis=1)

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        candidates = np.asarray(population, dtype=np.int64)
        if candidates.ndim != 2 or candidates.shape[1] != self.dimension:
            raise ValueError("population shape does not match the TSP dimension")
        if any(not np.array_equal(np.sort(row), np.arange(self.dimension)) for row in candidates):
            raise ValueError("every candidate must be a complete zero-based permutation")
        following = np.roll(candidates, -1, axis=1)
        return np.stack(
            [matrix[candidates, following].sum(axis=1) for matrix in self._stack], axis=1
        )
