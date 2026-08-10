"""Numba batch evaluator for permutation Flow Shop objective values."""
from __future__ import annotations

import numpy as np
from numba import njit, prange

OBJECTIVE_CODES = {
    "makespan": 0,
    "total_flow_time": 1,
    "total_tardiness": 2,
    "maximum_tardiness": 3,
    "total_machine_idle_time": 4,
}


@njit(cache=False, parallel=True)
def _population_is_valid(population: np.ndarray, number_of_jobs: int) -> bool:
    valid = np.ones(population.shape[0], dtype=np.uint8)
    for candidate in prange(population.shape[0]):
        seen = np.zeros(number_of_jobs, dtype=np.uint8)
        for position in range(number_of_jobs):
            job = population[candidate, position]
            if job < 0 or job >= number_of_jobs or seen[job]:
                valid[candidate] = 0
                break
            seen[job] = 1
    return bool(np.all(valid))


@njit(cache=False, parallel=True)
def _evaluate_batch(
    processing_times: np.ndarray,
    population: np.ndarray,
    due_dates: np.ndarray,
    has_due_dates: bool,
    objective_codes: np.ndarray,
) -> np.ndarray:
    size = population.shape[0]
    jobs = processing_times.shape[0]
    machines = processing_times.shape[1]
    output = np.empty((size, objective_codes.size), dtype=np.float64)
    total_processing = processing_times.sum()

    for candidate in prange(size):
        completion = np.zeros(machines, dtype=np.float64)
        total_flow_time = 0.0
        total_tardiness = 0.0
        maximum_tardiness = 0.0

        for position in range(jobs):
            job = population[candidate, position]
            for machine in range(machines):
                prior_job = completion[machine]
                prior_machine = completion[machine - 1] if machine else 0.0
                start = prior_job if prior_job >= prior_machine else prior_machine
                completion[machine] = start + processing_times[job, machine]

            final_completion = completion[machines - 1]
            total_flow_time += final_completion
            if has_due_dates:
                tardiness = final_completion - due_dates[job]
                if tardiness > 0.0:
                    total_tardiness += tardiness
                    if tardiness > maximum_tardiness:
                        maximum_tardiness = tardiness

        makespan = completion[machines - 1]
        total_idle = machines * makespan - total_processing
        for column in range(objective_codes.size):
            code = objective_codes[column]
            if code == 0:
                output[candidate, column] = makespan
            elif code == 1:
                output[candidate, column] = total_flow_time
            elif code == 2:
                output[candidate, column] = total_tardiness
            elif code == 3:
                output[candidate, column] = maximum_tardiness
            else:
                output[candidate, column] = total_idle
    return output


def evaluate_flowshop_population_fast(
    processing_times: np.ndarray,
    population: np.ndarray,
    objective_names: tuple[str, ...],
    due_dates: np.ndarray | None,
) -> np.ndarray:
    """Evaluate a validated-shape population without materializing schedules."""
    candidates = np.ascontiguousarray(population, dtype=np.int64)
    jobs = int(processing_times.shape[0])
    if candidates.ndim != 2 or candidates.shape[1] != jobs:
        raise ValueError("population candidate width does not match problem dimension")
    if not len(candidates):
        return np.empty((0, len(objective_names)), dtype=np.float64)
    if not _population_is_valid(candidates, jobs):
        raise ValueError("population candidates must contain every job exactly once")
    times = np.ascontiguousarray(processing_times, dtype=np.float64)
    due = np.empty(0, dtype=np.float64) if due_dates is None else np.ascontiguousarray(due_dates, dtype=np.float64)
    codes = np.asarray([OBJECTIVE_CODES[name] for name in objective_names], dtype=np.int64)
    return _evaluate_batch(times, candidates, due, due_dates is not None, codes)
