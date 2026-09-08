"""Shared legacy cohort and coincidence statistics."""

from __future__ import annotations

import numpy as np


def selected_bounds(size: int, ratio: int, objective: str, reward: bool) -> tuple[int, int]:
    loop = round(size * ratio / 100)
    take_top = (objective == "min") == reward
    return (0, loop) if take_top else (max(0, size - loop - 1), size)


def ranked_population(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
    order = np.argsort(fitness, kind="stable")
    return np.ascontiguousarray(population[order])


def count_edge_cycles(population: np.ndarray, low: int, high: int, size: int) -> np.ndarray:
    counts = np.zeros((size, size), dtype=np.int64)
    for candidate in population[low:high]:
        for position, prior in enumerate(candidate):
            counts[int(prior), int(candidate[(position + 1) % candidate.size])] += 1
    return counts


def count_positions(population: np.ndarray, low: int, high: int, size: int) -> np.ndarray:
    counts = np.zeros((size, size), dtype=np.int64)
    for candidate in population[low:high]:
        for position, job in enumerate(candidate):
            counts[position, int(job)] += 1
    return counts
