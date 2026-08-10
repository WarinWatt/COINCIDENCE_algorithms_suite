"""Validated built-in and reproducibly generated Flow Shop instances."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import validate_due_dates
from .schedule import validate_processing_times


@dataclass(frozen=True, slots=True)
class FlowShopInstance:
    name: str
    processing_times: np.ndarray
    due_dates: np.ndarray | None = None
    _times: np.ndarray = field(init=False, repr=False)
    _due_dates: np.ndarray | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("instance name must not be empty")
        times = validate_processing_times(self.processing_times)
        due = validate_due_dates(self.due_dates, times.shape[0])
        object.__setattr__(self, "_times", times)
        object.__setattr__(self, "_due_dates", due)
        object.__setattr__(self, "processing_times", times)
        object.__setattr__(self, "due_dates", due)

    @property
    def number_of_jobs(self) -> int:
        return int(self._times.shape[0])

    @property
    def number_of_machines(self) -> int:
        return int(self._times.shape[1])

    @property
    def load_by_job(self) -> np.ndarray:
        return self._times.sum(axis=1)

    @property
    def load_by_machine(self) -> np.ndarray:
        return self._times.sum(axis=0)


SMALL_3X2 = FlowShopInstance(
    name="small-3x2",
    processing_times=np.array([[2, 3], [1, 4], [3, 2]], dtype=np.int64),
    due_dates=np.array([6, 8, 12], dtype=np.int64),
)

BUILTIN_INSTANCES: dict[str, FlowShopInstance] = {SMALL_3X2.name: SMALL_3X2}


def get_builtin_instance(name: str) -> FlowShopInstance:
    try:
        return BUILTIN_INSTANCES[name]
    except KeyError as exc:
        raise KeyError(f"unknown built-in Flow Shop instance: {name}") from exc


def generate_random_instance(
    number_of_jobs: int,
    number_of_machines: int,
    *,
    seed: int,
    processing_time_low: int = 1,
    processing_time_high: int = 99,
    name: str | None = None,
) -> FlowShopInstance:
    if number_of_jobs < 2:
        raise ValueError("number_of_jobs must be at least 2")
    if number_of_machines < 1:
        raise ValueError("number_of_machines must be at least 1")
    if processing_time_low < 1 or processing_time_high < processing_time_low:
        raise ValueError("processing-time bounds must be positive and ordered")
    rng = np.random.default_rng(seed)
    times = rng.integers(
        processing_time_low,
        processing_time_high + 1,
        size=(number_of_jobs, number_of_machines),
        dtype=np.int64,
    )
    return FlowShopInstance(
        name=name or f"random-{number_of_jobs}x{number_of_machines}-seed-{seed}",
        processing_times=times,
    )
