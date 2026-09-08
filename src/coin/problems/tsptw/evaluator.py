from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .model import TSPTWInstance, validate_permutation

OBJECTIVE_NAMES = (
    "travel_cost", "makespan", "total_waiting_time", "total_tardiness",
    "maximum_tardiness", "late_customers", "negative_minimum_slack",
)


@dataclass(frozen=True, slots=True)
class TSPTWEvaluation:
    objective_names: tuple[str, ...]
    objective_values: np.ndarray
    permutation: np.ndarray
    arrivals: np.ndarray
    service_starts: np.ndarray
    departures: np.ndarray
    waiting: np.ndarray
    tardiness: np.ndarray
    slack: np.ndarray
    travel_cost: float
    makespan: float
    feasible: bool


def evaluate_tsptw(instance: TSPTWInstance, permutation: np.ndarray,
                   objectives: tuple[str, ...] = ("travel_cost",)) -> TSPTWEvaluation:
    order = validate_permutation(permutation, instance.number_of_customers)
    unknown = set(objectives) - set(OBJECTIVE_NAMES)
    if not objectives or unknown:
        raise ValueError(f"unknown or empty TSPTW objectives: {sorted(unknown)}")
    distance = instance.distance_matrix
    arrivals = np.empty(len(order)); starts = np.empty(len(order)); departures = np.empty(len(order))
    waiting = np.empty(len(order)); tardiness = np.empty(len(order)); slack = np.empty(len(order))
    time = 0.0; travel = 0.0; previous = 0
    for position, customer in enumerate(order):
        node = int(customer) + 1
        leg = distance[previous, node]; travel += leg; arrivals[position] = time + leg
        opening, closing = instance.time_windows[customer]
        starts[position] = max(arrivals[position], opening)
        waiting[position] = starts[position] - arrivals[position]
        tardiness[position] = max(0.0, starts[position] - closing)
        slack[position] = closing - starts[position]
        departures[position] = starts[position] + instance.service_times[customer]
        time = departures[position]; previous = node
    travel += distance[previous, 0]
    makespan = time + distance[previous, 0]
    metrics = {
        "travel_cost": travel, "makespan": makespan,
        "total_waiting_time": waiting.sum(), "total_tardiness": tardiness.sum(),
        "maximum_tardiness": tardiness.max(initial=0),
        "late_customers": float(np.count_nonzero(tardiness)),
        "negative_minimum_slack": -float(slack.min()),
    }
    return TSPTWEvaluation(
        tuple(objectives), np.array([metrics[name] for name in objectives]), order.copy(),
        arrivals, starts, departures, waiting, tardiness, slack, travel, makespan,
        bool(np.all(tardiness == 0)),
    )
