"""Deterministic teaching instances small enough for notebooks and classrooms."""

from __future__ import annotations

import numpy as np

from .problem import TSPInstance


def _coordinates(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, size, endpoint=False)
    radii = 35 + rng.uniform(-13, 18, size)
    points = np.column_stack((50 + radii * np.cos(angles), 50 + radii * np.sin(angles)))
    return points[rng.permutation(size)]


def _distances(points: np.ndarray) -> np.ndarray:
    delta = points[:, None, :] - points[None, :, :]
    return np.rint(np.sqrt(np.square(delta).sum(axis=2))).astype(float)


def _single(size: int, seed: int) -> TSPInstance:
    return TSPInstance(
        f"tsp-{size}", f"Teaching TSP - {size} cities",
        {"distance": _distances(_coordinates(size, seed))},
        "Deterministic Euclidean closed-tour fixture.",
    )


def _multi(size: int, seed: int) -> TSPInstance:
    return TSPInstance(
        f"motsp-{size}", f"Teaching MO-TSP - {size} cities",
        {"distance": _distances(_coordinates(size, seed)),
         "operating_cost": _distances(_coordinates(size, seed + 10_000))},
        "Bi-objective fixture: distance versus an independent operating-cost network.",
    )


_INSTANCES = tuple(_single(n, 110 + n) for n in (8, 12, 16, 20, 24)) + tuple(
    _multi(n, 410 + n) for n in (8, 12, 16, 20, 24)
)


def list_tsp_instances(*, multiobjective: bool | None = None) -> tuple[TSPInstance, ...]:
    if multiobjective is None:
        return _INSTANCES
    return tuple(item for item in _INSTANCES if (len(item.matrices) > 1) is multiobjective)


def get_tsp_instance(instance_id: str) -> TSPInstance:
    try:
        return next(item for item in _INSTANCES if item.id == instance_id)
    except StopIteration as exc:
        raise KeyError(f"unknown TSP fixture: {instance_id}") from exc
