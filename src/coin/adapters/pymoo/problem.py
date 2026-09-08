"""pymoo view of the reusable COIN permutation problem."""

from __future__ import annotations

import numpy as np
from pymoo.core.problem import Problem

from coin.problems.base import PermutationProblem, evaluate_population


class PymooPermutationProblem(Problem):
    def __init__(self, problem: PermutationProblem):
        self.coin_problem = problem
        super().__init__(
            n_var=problem.dimension,
            n_obj=len(problem.objective_names),
            n_ieq_constr=0,
            xl=0,
            xu=problem.dimension - 1,
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = evaluate_population(self.coin_problem, np.asarray(x))
