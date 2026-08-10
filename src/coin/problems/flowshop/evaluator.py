"""Readable reference evaluator for permutation Flow Shop objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .metrics import FlowShopMetrics, calculate_metrics
from .schedule import FlowShopSchedule, build_schedule

ObjectiveName = Literal[
    "makespan",
    "total_flow_time",
    "total_tardiness",
    "maximum_tardiness",
    "total_machine_idle_time",
]

OBJECTIVE_NAMES: tuple[str, ...] = (
    "makespan",
    "total_flow_time",
    "total_tardiness",
    "maximum_tardiness",
    "total_machine_idle_time",
)
TARDINESS_OBJECTIVES = frozenset(("total_tardiness", "maximum_tardiness"))


@dataclass(frozen=True, slots=True)
class FlowShopEvaluation:
    objective_names: tuple[str, ...]
    objective_values: np.ndarray
    metrics: FlowShopMetrics
    schedule: FlowShopSchedule


def evaluate_flowshop(
    processing_times: np.ndarray,
    permutation: np.ndarray,
    *,
    objective_names: tuple[str, ...] = ("makespan",),
    due_dates: np.ndarray | None = None,
) -> FlowShopEvaluation:
    if not objective_names:
        raise ValueError("at least one objective must be selected")
    unknown = tuple(name for name in objective_names if name not in OBJECTIVE_NAMES)
    if unknown:
        raise ValueError(f"unknown Flow Shop objective(s): {', '.join(unknown)}")
    if len(set(objective_names)) != len(objective_names):
        raise ValueError("objective names must be unique")
    if TARDINESS_OBJECTIVES.intersection(objective_names) and due_dates is None:
        raise ValueError("due_dates are required for tardiness objectives")

    schedule = build_schedule(processing_times, permutation)
    metrics = calculate_metrics(schedule, processing_times, due_dates)
    values = np.asarray([getattr(metrics, name) for name in objective_names], dtype=np.float64)
    values.setflags(write=False)
    return FlowShopEvaluation(tuple(objective_names), values, metrics, schedule)
