from __future__ import annotations

import numpy as np


def descriptive(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    return {
        "best": float(data.min()), "mean": float(data.mean()), "median": float(np.median(data)),
        "standard_deviation": float(data.std()), "worst": float(data.max()),
        "interquartile_range": float(np.percentile(data, 75) - np.percentile(data, 25)),
    }
