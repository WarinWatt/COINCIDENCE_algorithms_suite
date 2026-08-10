"""Pareto ranking shared by MO-COIN problems and comparison algorithms."""
from __future__ import annotations
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

PARETO_SORTER_BACKEND = "pymoo"


def reference_nondominated_ranks(values: np.ndarray) -> np.ndarray:
    """Readable quadratic specification retained for teaching and tests."""
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or not len(points):
        raise ValueError("objective values must be a non-empty matrix")
    remaining, ranks, depth = list(range(len(points))), np.empty(len(points), dtype=np.int64), 0
    while remaining:
        front = [i for i in remaining if not any(j != i and np.all(points[j] <= points[i]) and np.any(points[j] < points[i]) for j in remaining)]
        ranks[front] = depth
        selected = set(front)
        remaining = [i for i in remaining if i not in selected]
        depth += 1
    return ranks


def nondominated_ranks(values: np.ndarray) -> np.ndarray:
    """Return minimization Pareto depth using pymoo's optimized sorter."""
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or not len(points):
        raise ValueError("objective values must be a non-empty matrix")
    ranks = np.empty(len(points), dtype=np.int64)
    for depth, front in enumerate(NonDominatedSorting().do(points)):
        ranks[np.asarray(front, dtype=np.int64)] = depth
    return ranks

def pareto_selection_scores(values: np.ndarray) -> np.ndarray:
    """Order by Pareto depth then crowding, without objective scalarization."""
    points, ranks = np.asarray(values, dtype=float), nondominated_ranks(values)
    crowding = np.zeros(len(points), dtype=float)
    for rank in np.unique(ranks):
        front = np.flatnonzero(ranks == rank)
        if len(front) <= 2:
            crowding[front] = np.inf
            continue
        for objective in range(points.shape[1]):
            ordered = front[np.argsort(points[front, objective], kind="stable")]
            crowding[ordered[[0, -1]]] = np.inf
            span = points[ordered[-1], objective] - points[ordered[0], objective]
            if span:
                crowding[ordered[1:-1]] += (points[ordered[2:], objective] - points[ordered[:-2], objective]) / span
    order = sorted(range(len(points)), key=lambda i: (int(ranks[i]), -float(crowding[i]), i))
    scores = np.empty(len(points), dtype=float)
    scores[order] = np.arange(len(points), dtype=float)
    return scores

def pareto_quality(values: np.ndarray) -> tuple[list[int], list[float]]:
    """Return population counts by Pareto depth and the rank-0 range per objective."""
    points = np.asarray(values, dtype=float)
    ranks = nondominated_ranks(points)
    counts = np.bincount(ranks).astype(int).tolist()
    front = points[ranks == 0]
    spread = (front.max(axis=0) - front.min(axis=0)).astype(float).tolist()
    return counts, spread


def comparative_pareto_indicators(solution_sets: list[np.ndarray]) -> list[dict[str, float]]:
    """Compare final sets against their pooled observed non-dominated front.

    Distances are normalized per objective.  Spread is Deb's delta for two
    objectives and its nearest-neighbour/extreme-point generalization for
    three or more objectives.  Lower convergence/spread and higher
    non-dominated ratio are better.
    """
    sets = [np.asarray(values, dtype=float) for values in solution_sets]
    if not sets or any(values.ndim != 2 or not len(values) for values in sets):
        raise ValueError("solution sets must be non-empty objective matrices")
    if len({values.shape[1] for values in sets}) != 1:
        raise ValueError("solution sets must have the same objective dimension")
    pooled = np.vstack(sets)
    pooled_ranks = nondominated_ranks(pooled)
    minimum, maximum = pooled.min(axis=0), pooled.max(axis=0)
    scale = np.where(maximum > minimum, maximum - minimum, 1.0)
    normalized_pool = (pooled - minimum) / scale
    reference = normalized_pool[pooled_ranks == 0]
    indicators, offset = [], 0
    for values in sets:
        count = len(values)
        points = normalized_pool[offset:offset + count]
        ranks = pooled_ranks[offset:offset + count]
        offset += count
        distances_to_reference = np.linalg.norm(points[:, None, :] - reference[None, :, :], axis=2)
        convergence = float(distances_to_reference.min(axis=1).mean())
        nondominated_ratio = float(np.mean(ranks == 0))
        if len(points) <= 1:
            neighbour_distances = np.empty(0)
        else:
            pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
            np.fill_diagonal(pairwise, np.inf)
            neighbour_distances = pairwise.min(axis=1)
        mean_distance = float(neighbour_distances.mean()) if len(neighbour_distances) else 0.0
        if points.shape[1] == 2:
            ordered = points[np.argsort(points[:, 0], kind="stable")]
            neighbour_distances = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
            mean_distance = float(neighbour_distances.mean()) if len(neighbour_distances) else 0.0
            reference_ordered = reference[np.argsort(reference[:, 0], kind="stable")]
            extreme_distances = [
                float(np.linalg.norm(ordered[0] - reference_ordered[0])),
                float(np.linalg.norm(ordered[-1] - reference_ordered[-1])),
            ]
        else:
            extreme_distances = []
            for objective in range(points.shape[1]):
                extreme = reference[np.argmin(reference[:, objective])]
                extreme_distances.append(float(np.linalg.norm(points - extreme, axis=1).min()))
        numerator = sum(extreme_distances) + float(np.abs(neighbour_distances - mean_distance).sum())
        denominator = sum(extreme_distances) + len(neighbour_distances) * mean_distance
        spread = 0.0 if denominator == 0 else float(numerator / denominator)
        indicators.append({
            "convergence": convergence,
            "spread": spread,
            "nondominated_ratio": nondominated_ratio,
            "reference_size": int(len(reference)),
        })
    return indicators
