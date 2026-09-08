"""Objective-independent reporting metrics for Flow Shop schedules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schedule import FlowShopSchedule


def validate_due_dates(due_dates: np.ndarray | None, number_of_jobs: int) -> np.ndarray | None:
    if due_dates is None:
        return None
    values = np.asarray(due_dates)
    if values.ndim != 1 or values.size != number_of_jobs:
        raise ValueError(f"due_dates must contain exactly {number_of_jobs} values")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_):
        raise TypeError("due_dates must contain numeric values")
    result = np.array(values, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("due_dates must be finite")
    if np.any(result < 0):
        raise ValueError("due_dates must be non-negative")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FlowShopMetrics:
    makespan: float
    total_flow_time: float
    average_flow_time: float
    total_tardiness: float | None
    maximum_tardiness: float | None
    average_tardiness: float | None
    total_machine_idle_time: float
    idle_time_by_machine: np.ndarray
    average_machine_utilization: float
    utilization_by_machine: np.ndarray
    total_waiting_time: float
    completion_time_by_job: np.ndarray


def calculate_metrics(
    schedule: FlowShopSchedule,
    processing_times: np.ndarray,
    due_dates: np.ndarray | None = None,
) -> FlowShopMetrics:
    times = np.asarray(processing_times, dtype=np.float64)
    completion_by_job = schedule.completion_time_by_job
    makespan = schedule.makespan
    loads = times.sum(axis=0)
    idle = makespan - loads
    utilization = loads / makespan

    waits = 0.0
    for position in range(schedule.number_of_jobs):
        for machine in range(schedule.number_of_machines):
            ready = schedule.completion_times[position, machine - 1] if machine else 0.0
            waits += schedule.start_times[position, machine] - ready

    validated_due_dates = validate_due_dates(due_dates, schedule.number_of_jobs)
    if validated_due_dates is None:
        total_tardiness = maximum_tardiness = average_tardiness = None
    else:
        tardiness = np.maximum(completion_by_job - validated_due_dates, 0.0)
        total_tardiness = float(tardiness.sum())
        maximum_tardiness = float(tardiness.max())
        average_tardiness = float(tardiness.mean())

    idle.setflags(write=False)
    utilization.setflags(write=False)
    return FlowShopMetrics(
        makespan=makespan,
        total_flow_time=float(completion_by_job.sum()),
        average_flow_time=float(completion_by_job.mean()),
        total_tardiness=total_tardiness,
        maximum_tardiness=maximum_tardiness,
        average_tardiness=average_tardiness,
        total_machine_idle_time=float(idle.sum()),
        idle_time_by_machine=idle,
        average_machine_utilization=float(utilization.mean()),
        utilization_by_machine=utilization,
        total_waiting_time=float(waits),
        completion_time_by_job=completion_by_job,
    )
