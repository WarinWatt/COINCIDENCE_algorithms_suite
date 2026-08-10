"""Normalized algorithm results shared with COIN experiment runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ConvergencePoint:
    generation: int
    evaluations: int
    best: list[float]
    nondominated_count: int = 1
    pareto_depth_counts: list[int] = field(default_factory=list)
    objective_spread: list[float] = field(default_factory=list)


@dataclass(slots=True)
class AlgorithmResult:
    algorithm: str
    seed: int
    permutations: list[list[int]]
    objective_values: list[list[float]]
    runtime_seconds: float
    evaluations: int
    generations: int
    history: list[ConvergencePoint] = field(default_factory=list)
    ranks: list[int] | None = None
    crowding: list[float | None] | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    reported_objective_values: list[list[float]] = field(default_factory=list)
