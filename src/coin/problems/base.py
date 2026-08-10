"""Problem contracts shared by permutation optimization algorithms."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PermutationProblem(Protocol):
    """A problem evaluated from a complete zero-based permutation."""

    @property
    def dimension(self) -> int: ...

    @property
    def objective_names(self) -> tuple[str, ...]: ...

    def evaluate(self, permutation: np.ndarray) -> np.ndarray: ...

    def validate(self, permutation: np.ndarray) -> None: ...


def evaluate_population(
    problem: PermutationProblem, population: np.ndarray
) -> np.ndarray:
    """Evaluate a two-dimensional population into a dense objective matrix."""
    candidates = np.asarray(population)
    if candidates.ndim != 2:
        raise ValueError("population must be a two-dimensional array")
    if candidates.shape[1] != problem.dimension:
        raise ValueError("population candidate width does not match problem dimension")
    batch_evaluator = getattr(problem, "evaluate_population", None)
    if callable(batch_evaluator):
        result = np.asarray(batch_evaluator(candidates), dtype=np.float64)
        expected = (candidates.shape[0], len(problem.objective_names))
        if result.shape != expected:
            raise ValueError(f"problem returned objective matrix {result.shape}, expected {expected}")
        return result
    values = [np.asarray(problem.evaluate(candidate), dtype=np.float64) for candidate in candidates]
    if not values:
        return np.empty((0, len(problem.objective_names)), dtype=np.float64)
    result = np.vstack(values)
    expected = (candidates.shape[0], len(problem.objective_names))
    if result.shape != expected:
        raise ValueError(f"problem returned objective matrix {result.shape}, expected {expected}")
    return result
