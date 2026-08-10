"""Deterministic random state used by readable COIN implementations."""

from __future__ import annotations

_MASK = (1 << 64) - 1
_INCREMENT = 0x9E3779B97F4A7C15


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & _MASK

    def randbelow(self, bound: int) -> int:
        if bound <= 0:
            return 0
        self.state = (self.state + _INCREMENT) & _MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK
        value ^= value >> 31
        return value % bound
