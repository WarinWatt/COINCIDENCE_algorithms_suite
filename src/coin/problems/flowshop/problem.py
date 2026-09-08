"""Reusable permutation-problem adapter for Flow Shop scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evaluator import OBJECTIVE_NAMES, TARDINESS_OBJECTIVES, FlowShopEvaluation, evaluate_flowshop
from .metrics import validate_due_dates
from .schedule import validate_permutation, validate_processing_times
from .fast_evaluator import evaluate_flowshop_population_fast


@dataclass(frozen=True, slots=True)
class FlowShopProblem:
    processing_times: np.ndarray
    objectives: tuple[str, ...] = ("makespan",)
    due_dates: np.ndarray | None = None
    _times: np.ndarray = field(init=False, repr=False)
    _due_dates: np.ndarray | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        times = validate_processing_times(self.processing_times)
        if not self.objectives:
            raise ValueError("at least one objective must be selected")
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError("objective names must be unique")
        unknown = [name for name in self.objectives if name not in OBJECTIVE_NAMES]
        if unknown:
            raise ValueError(f"unknown Flow Shop objective(s): {', '.join(unknown)}")
        due = validate_due_dates(self.due_dates, times.shape[0])
        if TARDINESS_OBJECTIVES.intersection(self.objectives) and due is None:
            raise ValueError("due_dates are required for tardiness objectives")
        object.__setattr__(self, "_times", times)
        object.__setattr__(self, "_due_dates", due)
        object.__setattr__(self, "processing_times", times)
        object.__setattr__(self, "due_dates", due)

    @property
    def dimension(self) -> int:
        return int(self._times.shape[0])

    @property
    def number_of_machines(self) -> int:
        return int(self._times.shape[1])

    @property
    def objective_names(self) -> tuple[str, ...]:
        return tuple(self.objectives)

    @property
    def available_objective_names(self) -> tuple[str, ...]:
        if self._due_dates is None:
            return tuple(name for name in OBJECTIVE_NAMES if name not in TARDINESS_OBJECTIVES)
        return OBJECTIVE_NAMES

    def validate(self, permutation: np.ndarray) -> None:
        validate_permutation(permutation, self.dimension)

    def evaluate_details(self, permutation: np.ndarray) -> FlowShopEvaluation:
        return evaluate_flowshop(
            self._times,
            permutation,
            objective_names=self.objective_names,
            due_dates=self._due_dates,
        )

    def evaluate(self, permutation: np.ndarray) -> np.ndarray:
        return self.evaluate_details(permutation).objective_values.copy()

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        return evaluate_flowshop_population_fast(
            self._times, population, self.objective_names, self._due_dates
        )
