"""Chained Node-Based COIN: fill positions 0..n-1 in order."""
from __future__ import annotations
import numpy as np
from .position import PositionCoin

class CNBCoin(PositionCoin):
    variant_name = "cnb_coin"
    def generate_population(self) -> np.ndarray:
        n=self.config.problem_size; population=np.empty((self.config.population_size,n),dtype=np.int16)
        for row in range(self.config.population_size):
            used=np.zeros(n,dtype=bool); cursor=0
            for position in range(n):
                selected,cursor=self._sample_value(position,used,cursor)
                population[row,position]=selected; used[selected]=True
        return population
