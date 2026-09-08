from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def validate_permutation(permutation: np.ndarray, size: int) -> np.ndarray:
    value = np.asarray(permutation, dtype=np.int64)
    if value.shape != (size,) or not np.array_equal(np.sort(value), np.arange(size)):
        raise ValueError("permutation must contain every customer exactly once")
    return value


@dataclass(frozen=True, slots=True)
class TSPTWInstance:
    name: str
    coordinates: np.ndarray
    time_windows: np.ndarray
    service_times: np.ndarray
    depot: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _coordinates: np.ndarray = field(init=False, repr=False)
    _windows: np.ndarray = field(init=False, repr=False)
    _service: np.ndarray = field(init=False, repr=False)
    _depot: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        coordinates = np.asarray(self.coordinates, dtype=np.float64)
        windows = np.asarray(self.time_windows, dtype=np.float64)
        service = np.asarray(self.service_times, dtype=np.float64)
        depot = np.asarray(self.depot, dtype=np.float64)
        if not self.name.strip() or coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
            raise ValueError("TSPTW requires a name and at least two customer coordinates")
        if windows.shape != (len(coordinates), 2) or np.any(windows[:, 0] > windows[:, 1]):
            raise ValueError("time_windows must be ordered [open, close] pairs")
        if service.shape != (len(coordinates),) or np.any(service < 0) or depot.shape != (2,):
            raise ValueError("service_times and depot have invalid shapes or values")
        if not all(np.isfinite(x).all() for x in (coordinates, windows, service, depot)):
            raise ValueError("TSPTW data must be finite")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "time_windows", windows)
        object.__setattr__(self, "service_times", service)
        object.__setattr__(self, "depot", depot)
        object.__setattr__(self, "_coordinates", coordinates)
        object.__setattr__(self, "_windows", windows)
        object.__setattr__(self, "_service", service)
        object.__setattr__(self, "_depot", depot)

    @property
    def number_of_customers(self) -> int:
        return len(self._coordinates)

    @property
    def distance_matrix(self) -> np.ndarray:
        points = np.vstack((self._depot, self._coordinates))
        delta = points[:, None, :] - points[None, :, :]
        return np.sqrt(np.sum(delta * delta, axis=2))


@dataclass(frozen=True, slots=True)
class MatrixTSPTWInstance:
    """Canonical López-Ibáñez TSPTW instance with an explicit, possibly asymmetric matrix."""
    name: str
    travel_times: np.ndarray
    node_time_windows: np.ndarray
    source: str = "López-Ibáñez TSPTW collection"
    _matrix: np.ndarray = field(init=False, repr=False)
    _windows: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        matrix = np.asarray(self.travel_times, dtype=np.float64)
        windows = np.asarray(self.node_time_windows, dtype=np.float64)
        if not self.name.strip() or matrix.ndim != 2 or matrix.shape[0] < 3 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("canonical TSPTW matrix must be square with a depot and at least two customers")
        if windows.shape != (matrix.shape[0], 2) or np.any(windows[:, 0] > windows[:, 1]):
            raise ValueError("canonical node time windows are invalid")
        if np.any(matrix < 0) or not np.isfinite(matrix).all() or not np.isfinite(windows).all():
            raise ValueError("canonical TSPTW values must be finite and non-negative")
        object.__setattr__(self, "travel_times", matrix)
        object.__setattr__(self, "node_time_windows", windows)
        object.__setattr__(self, "_matrix", matrix)
        object.__setattr__(self, "_windows", windows)

    @property
    def number_of_customers(self): return self._matrix.shape[0] - 1

    @property
    def distance_matrix(self): return self._matrix

    @property
    def time_windows(self): return self._windows[1:]

    @property
    def service_times(self): return np.zeros(self.number_of_customers)

    @property
    def depot(self): return np.array([50.0, 50.0])

    @property
    def coordinates(self):
        # Canonical files contain matrices, not coordinates. A deterministic
        # circular embedding is presentation-only and never used for evaluation.
        angle = 2 * np.pi * np.arange(self.number_of_customers) / self.number_of_customers
        return np.column_stack((50 + 42 * np.cos(angle), 50 + 42 * np.sin(angle)))


def parse_lopez_ibanez_instance(text: str, *, name: str) -> MatrixTSPTWInstance:
    tokens: list[str] = []
    for line in text.splitlines():
        content = line.split("#", 1)[0].strip()
        if content: tokens.extend(content.split())
    if not tokens: raise ValueError("empty canonical TSPTW file")
    size = int(tokens[0]); expected = 1 + size * size + size * 2
    if len(tokens) < expected: raise ValueError("truncated canonical TSPTW file")
    values = np.asarray([float(value) for value in tokens[1:expected]])
    matrix = values[:size * size].reshape(size, size)
    windows = values[size * size:].reshape(size, 2)
    return MatrixTSPTWInstance(name, matrix, windows)


def generate_tsptw_instance(size: int, seed: int, *, window_width: float = 80.0) -> TSPTWInstance:
    if size < 2 or window_width <= 0:
        raise ValueError("size and window_width must be positive")
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0, 100, (size, 2))
    guide = rng.permutation(size)
    points = np.vstack((np.array([[50.0, 50.0]]), coordinates[guide]))
    arrivals = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    openings = np.maximum(0, arrivals - rng.uniform(0, window_width * .35, size))
    closings = arrivals + rng.uniform(window_width * .35, window_width, size)
    windows = np.empty((size, 2))
    windows[guide, 0], windows[guide, 1] = openings, closings
    return TSPTWInstance(
        f"generated-{size}-seed-{seed}", coordinates, windows,
        rng.integers(0, 11, size).astype(float), np.array([50.0, 50.0]),
    )
