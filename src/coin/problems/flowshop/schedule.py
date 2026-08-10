"""Reference permutation Flow Shop schedule construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def validate_processing_times(processing_times: np.ndarray) -> np.ndarray:
    values = np.asarray(processing_times)
    if values.ndim != 2:
        raise ValueError("processing_times must be a two-dimensional jobs-by-machines matrix")
    if values.shape[0] < 2:
        raise ValueError("a Flow Shop instance requires at least two jobs")
    if values.shape[1] < 1:
        raise ValueError("a Flow Shop instance requires at least one machine")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_):
        raise TypeError("processing_times must contain numeric values")
    # Own the validated array: callers must not be able to mutate a problem
    # after construction, and validation must not make their source read-only.
    numeric = np.array(values, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("processing_times must be finite")
    if np.any(numeric <= 0):
        raise ValueError("processing_times must be strictly positive")
    numeric.setflags(write=False)
    return numeric


def validate_permutation(permutation: np.ndarray, number_of_jobs: int) -> np.ndarray:
    values = np.asarray(permutation)
    if values.ndim != 1:
        raise ValueError("permutation must be one-dimensional")
    if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        raise TypeError("permutation must contain integer job indices")
    candidate = np.array(values, dtype=np.int64, order="C", copy=True)
    if candidate.size != number_of_jobs:
        raise ValueError(f"permutation must contain exactly {number_of_jobs} jobs")
    if np.any(candidate < 0) or np.any(candidate >= number_of_jobs):
        raise ValueError(f"job indices must be in the range 0..{number_of_jobs - 1}")
    if np.unique(candidate).size != number_of_jobs:
        raise ValueError("permutation must contain every job exactly once")
    return candidate


@dataclass(frozen=True, slots=True)
class Operation:
    sequence_position: int
    job: int
    machine: int
    start: float
    finish: float


@dataclass(frozen=True, slots=True)
class FlowShopSchedule:
    permutation: np.ndarray
    start_times: np.ndarray
    completion_times: np.ndarray

    @property
    def makespan(self) -> float:
        return float(self.completion_times[-1, -1])

    @property
    def number_of_jobs(self) -> int:
        return int(self.permutation.size)

    @property
    def number_of_machines(self) -> int:
        return int(self.completion_times.shape[1])

    @property
    def completion_time_by_job(self) -> np.ndarray:
        result = np.empty(self.number_of_jobs, dtype=np.float64)
        result[self.permutation] = self.completion_times[:, -1]
        result.setflags(write=False)
        return result

    def operations(self) -> tuple[Operation, ...]:
        return tuple(
            Operation(
                sequence_position=position,
                job=int(job),
                machine=machine,
                start=float(self.start_times[position, machine]),
                finish=float(self.completion_times[position, machine]),
            )
            for position, job in enumerate(self.permutation)
            for machine in range(self.number_of_machines)
        )


def build_schedule(
    processing_times: np.ndarray, permutation: np.ndarray
) -> FlowShopSchedule:
    """Build the canonical earliest-start permutation Flow Shop schedule."""
    times = validate_processing_times(processing_times)
    order = validate_permutation(permutation, times.shape[0])
    starts = np.zeros((order.size, times.shape[1]), dtype=np.float64)
    completions = np.zeros_like(starts)
    for position, job in enumerate(order):
        for machine in range(times.shape[1]):
            prior_job_finish = completions[position - 1, machine] if position else 0.0
            prior_machine_finish = completions[position, machine - 1] if machine else 0.0
            starts[position, machine] = max(prior_job_finish, prior_machine_finish)
            completions[position, machine] = starts[position, machine] + times[job, machine]
    order.setflags(write=False)
    starts.setflags(write=False)
    completions.setflags(write=False)
    return FlowShopSchedule(order, starts, completions)
