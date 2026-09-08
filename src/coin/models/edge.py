"""Shared types for the two independently implemented Edge COIN variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Objective = Literal["min", "max"]
LearningMode = Literal["conservative", "reconstruction"]


@dataclass(frozen=True, slots=True)
class EdgeConfig:
    problem_size: int = 64
    population_size: int = 400
    training_rate: int = 5
    reward_ratio: int = 25
    punishment_ratio: int = 25
    objective: Objective = "max"
    learning_mode: LearningMode = "conservative"
    rewards_enabled: bool = True
    punishments_enabled: bool = True

    def __post_init__(self) -> None:
        if self.problem_size < 2:
            raise ValueError("problem_size must be at least 2")
        if self.population_size < 1:
            raise ValueError("population_size must be positive")
        if self.training_rate <= 0:
            raise ValueError("training_rate must be positive")
        if not 0 <= self.reward_ratio <= 100:
            raise ValueError("reward_ratio must be between 0 and 100")
        if not 0 <= self.punishment_ratio <= 100:
            raise ValueError("punishment_ratio must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation: int
    best: int
    average: float


def initial_edge_weights(config: EdgeConfig) -> np.ndarray:
    """Match the lasting value written by ``InitializeButtonClick``."""
    initial = round(config.problem_size * 100 / config.training_rate)
    weights = np.full(
        (config.problem_size, config.problem_size), initial, dtype=np.int64
    )
    np.fill_diagonal(weights, 0)
    return weights


def prefer_edges(
    weights: np.ndarray, preferred: np.ndarray, *, minimum: int = 1
) -> None:
    """Bias an initialized matrix toward a supplied edge mask, in place.

    Preferred edges retain their original weight. Other off-diagonal edges
    receive the smallest positive weight so they remain explorable and keep
    the legacy Edge COIN invariant intact.
    """
    if preferred.shape != weights.shape:
        raise ValueError("preferred-edge mask must match the weight matrix")
    if minimum < 1:
        raise ValueError("minimum edge weight must be positive")
    diagonal = np.eye(weights.shape[0], dtype=bool)
    weights[~np.asarray(preferred, dtype=bool) & ~diagonal] = minimum
    np.fill_diagonal(weights, 0)
    validate_weight_matrix(weights)


def validate_weight_matrix(weights: np.ndarray) -> None:
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("edge weights must be a square matrix")
    if np.any(np.diag(weights) != 0):
        raise ValueError("forbidden diagonal transitions must remain zero")
    off_diagonal = ~np.eye(weights.shape[0], dtype=bool)
    if np.any(weights[off_diagonal] < 1):
        raise ValueError("legacy edge weights must remain positive off diagonal")
