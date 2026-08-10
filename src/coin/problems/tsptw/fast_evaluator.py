from __future__ import annotations

import numpy as np
from numba import njit

OBJECTIVE_CODES = {
    "travel_cost": 0, "makespan": 1, "total_waiting_time": 2,
    "total_tardiness": 3, "maximum_tardiness": 4, "late_customers": 5,
    "negative_minimum_slack": 6,
}


@njit(cache=False)
def _evaluate_batch(matrix, windows, service, population, codes, hard_windows, penalty):
    count, size = population.shape
    output = np.empty((count, len(codes)), dtype=np.float64)
    for candidate in range(count):
        time = 0.0; travel = 0.0; waiting_total = 0.0; tardiness_total = 0.0
        maximum_tardiness = 0.0; late = 0.0; minimum_slack = np.inf; previous = 0
        for position in range(size):
            customer = population[candidate, position]; node = customer + 1
            leg = matrix[previous, node]; travel += leg
            arrival = time + leg; start = max(arrival, windows[customer, 0])
            waiting_total += start - arrival
            tardiness = max(0.0, start - windows[customer, 1])
            tardiness_total += tardiness
            if tardiness > maximum_tardiness: maximum_tardiness = tardiness
            if tardiness > 0: late += 1
            slack = windows[customer, 1] - start
            if slack < minimum_slack: minimum_slack = slack
            time = start + service[customer]; previous = node
        travel += matrix[previous, 0]
        makespan = time + matrix[previous, 0]
        metrics = (travel, makespan, waiting_total, tardiness_total,
                   maximum_tardiness, late, -minimum_slack)
        violation = penalty * (tardiness_total + late) if hard_windows and late else 0.0
        for objective in range(len(codes)):
            output[candidate, objective] = metrics[codes[objective]] + violation
    return output


def evaluate_tsptw_population_fast(matrix, windows, service, population, objectives,
                                    hard_windows=True, penalty=1_000_000.0):
    candidates = np.asarray(population, dtype=np.int64)
    if candidates.ndim != 2:
        raise ValueError("population must be two-dimensional")
    codes = np.asarray([OBJECTIVE_CODES[name] for name in objectives], dtype=np.int64)
    return _evaluate_batch(np.asarray(matrix, dtype=np.float64), np.asarray(windows, dtype=np.float64),
                           np.asarray(service, dtype=np.float64), candidates, codes,
                           bool(hard_windows), float(penalty))
