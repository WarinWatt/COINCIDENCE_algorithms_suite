"""Permutation-safe crossover operators not bundled with pymoo 0.6.2."""
from __future__ import annotations

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.util import default_random_state


def _cuts(n: int, rng) -> tuple[int, int]:
    left, right = sorted(rng.choice(n, 2, replace=False).tolist())
    return int(left), int(right)


def _uniform_mask(n: int, rng) -> np.ndarray:
    mask = rng.random(n) < 0.5
    if not mask.any():
        mask[int(rng.integers(n))] = True
    elif mask.all():
        mask[int(rng.integers(n))] = False
    return mask


def _fixed_mask(n: int, rng) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[rng.choice(n, max(1, n // 2), replace=False)] = True
    return mask


def _pmx(first: np.ndarray, second: np.ndarray, left: int, right: int) -> np.ndarray:
    child = np.full(len(first), -1, dtype=int)
    child[left:right + 1] = first[left:right + 1]
    position_in_second = {int(value): index for index, value in enumerate(second)}
    retained = set(int(value) for value in child[left:right + 1])
    for index in range(left, right + 1):
        value = int(second[index])
        if value in retained:
            continue
        target = index
        while left <= target <= right:
            target = position_in_second[int(first[target])]
        child[target] = value
    child[child < 0] = second[child < 0]
    return child


def _cycle_children(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(first)
    child_a = np.empty(n, dtype=int)
    child_b = np.empty(n, dtype=int)
    position_in_first = {int(value): index for index, value in enumerate(first)}
    unvisited = set(range(n))
    cycle = 0
    while unvisited:
        start = min(unvisited)
        indices = []
        index = start
        while index not in indices:
            indices.append(index)
            unvisited.discard(index)
            index = position_in_first[int(second[index])]
        source_a, source_b = ((first, second) if cycle % 2 == 0 else (second, first))
        child_a[indices] = source_a[indices]
        child_b[indices] = source_b[indices]
        cycle += 1
    return child_a, child_b


def _mx(first: np.ndarray, second: np.ndarray, cut: int) -> np.ndarray:
    prefix = first[:cut]
    retained = set(int(value) for value in prefix)
    suffix = [int(value) for value in second if int(value) not in retained]
    return np.asarray([*prefix.tolist(), *suffix], dtype=int)


def _pbx(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> np.ndarray:
    child = np.full(len(first), -1, dtype=int)
    child[mask] = first[mask]
    retained = set(int(value) for value in child[mask])
    donor = iter(int(value) for value in second if int(value) not in retained)
    for index in np.flatnonzero(~mask):
        child[index] = next(donor)
    return child


def _obx(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> np.ndarray:
    child = first.copy().astype(int)
    selected = set(int(value) for value in first[mask])
    ordered = [int(value) for value in second if int(value) in selected]
    child[np.flatnonzero(mask)] = ordered
    return child


def _lox(first: np.ndarray, second: np.ndarray, left: int, right: int) -> np.ndarray:
    """Linear-order crossover: retain a segment and order-fill without wrapping."""
    child = np.full(len(first), -1, dtype=int)
    child[left:right + 1] = first[left:right + 1]
    retained = set(int(value) for value in child[left:right + 1])
    donor = iter(int(value) for value in second if int(value) not in retained)
    for index in list(range(0, left)) + list(range(right + 1, len(first))):
        child[index] = next(donor)
    return child


class PermutationCrossover(Crossover):
    """Two-child PMX, CX, Davis MX, Syswerda OBX, or Syswerda PBX."""

    DEFINITIONS = {
        "pmx": "Goldberg-Lingle partially mapped crossover; two inclusive cuts",
        "cx": "Oliver-Smith-Holland cycle crossover; alternating cycles",
        "mx": "Davis modified crossover; one-cut prefix plus donor relative order",
        "obx": "Syswerda order-based crossover; reorder selected loci by donor order",
        "pbx": "Syswerda position-based crossover; retain selected loci and order-fill",
        "uox": "Davis uniform order-based crossover; Bernoulli loci and order-fill",
        "lox": "linear order crossover; retain segment and non-wrapping order-fill",
    }

    def __init__(self, kind: str, **kwargs):
        kind = kind.lower()
        if kind not in self.DEFINITIONS:
            raise ValueError(f"unsupported permutation crossover: {kind}")
        self.kind = kind
        super().__init__(2, 2, **kwargs)

    @default_random_state
    def _do(self, problem, X, random_state=None, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.empty((2, n_matings, n_var), dtype=int)
        for mating in range(n_matings):
            first, second = X[:, mating, :]
            if self.kind == "pmx":
                left, right = _cuts(n_var, random_state)
                children = (_pmx(first, second, left, right),
                            _pmx(second, first, left, right))
            elif self.kind == "cx":
                children = _cycle_children(first, second)
            elif self.kind == "mx":
                cut = int(random_state.integers(1, n_var))
                children = (_mx(first, second, cut), _mx(second, first, cut))
            elif self.kind == "lox":
                left, right = _cuts(n_var, random_state)
                children = (_lox(first, second, left, right),
                            _lox(second, first, left, right))
            else:
                mask = (_uniform_mask(n_var, random_state) if self.kind == "uox"
                        else _fixed_mask(n_var, random_state))
                operator = _obx if self.kind == "obx" else _pbx
                children = (operator(first, second, mask), operator(second, first, mask))
            Y[0, mating], Y[1, mating] = children
        return Y
