from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .evaluator import OBJECTIVE_NAMES, evaluate_tsptw
from .model import TSPTWInstance, validate_permutation
from .fast_evaluator import evaluate_tsptw_population_fast


@dataclass(frozen=True, slots=True)
class TSPTWProblem:
    instance: TSPTWInstance
    objectives: tuple[str, ...] = ("travel_cost",)
    hard_windows: bool = True
    penalty: float = 1_000_000.0
    _distance: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if not self.objectives or len(set(self.objectives)) != len(self.objectives):
            raise ValueError("objectives must be non-empty and unique")
        if set(self.objectives) - set(OBJECTIVE_NAMES) or self.penalty <= 0:
            raise ValueError("unknown objective or invalid penalty")
        object.__setattr__(self, "_distance", np.asarray(self.instance.distance_matrix, dtype=np.float64))

    @property
    def dimension(self): return self.instance.number_of_customers

    @property
    def objective_names(self): return tuple(self.objectives)

    @property
    def available_objective_names(self): return OBJECTIVE_NAMES

    def validate(self, permutation): validate_permutation(permutation, self.dimension)

    def evaluate_details(self, permutation): return evaluate_tsptw(self.instance, permutation, self.objectives)

    def evaluate(self, permutation):
        result = self.evaluate_details(permutation)
        values = result.objective_values.copy()
        if self.hard_windows and not result.feasible:
            violation = result.tardiness.sum() + np.count_nonzero(result.tardiness)
            values += self.penalty * violation
        return values

    def evaluate_population(self, population):
        return evaluate_tsptw_population_fast(
            self._distance, self.instance.time_windows, self.instance.service_times,
            population, self.objectives, self.hard_windows, self.penalty,
        )
