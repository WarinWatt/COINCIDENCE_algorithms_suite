"""Readable legacy reward/punishment update kernels."""

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coin.models.edge import EdgeConfig


def update_edge_weights(
    weights: np.ndarray,
    config: "EdgeConfig",
    rewards: np.ndarray,
    punishments: np.ndarray,
) -> None:
    """Apply the exact readable Edge learner to `weights` in place."""
    # Local import keeps the learning package independent during model module
    # initialization while reusing the public invariant checker at runtime.
    from coin.models.edge import validate_weight_matrix
    n = config.problem_size
    if rewards.shape != weights.shape or punishments.shape != weights.shape:
        raise ValueError("edge statistics must match the edge matrix")
    if config.learning_mode == "reconstruction":
        for prior in range(n):
            for following in range(n):
                if prior != following:
                    weights[prior, following] = 10 * rewards[prior, following] + 1
        validate_weight_matrix(weights)
        return

    for following in range(n):
        for prior in range(n):
            for _ in range(int(punishments[prior, following])):
                if weights[prior, following] > n:
                    for competitor in range(n):
                        if competitor != prior:
                            weights[prior, competitor] += 1
                    weights[prior, following] -= n - 1
    for following in range(n):
        for prior in range(n):
            for _ in range(int(rewards[prior, following])):
                for competitor in range(n):
                    if weights[prior, competitor] > n:
                        weights[prior, following] += 1
                        weights[prior, competitor] -= 1
    validate_weight_matrix(weights)
