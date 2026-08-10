from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.crossover.erx import EdgeRecombinationCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.util.ref_dirs import get_reference_directions

from coin.problems.base import PermutationProblem, evaluate_population
from .problem import PymooPermutationProblem
from .result import AlgorithmResult, ConvergencePoint
from coin.learning.pareto import nondominated_ranks, pareto_quality


@dataclass(frozen=True, slots=True)
class PymooRunConfig:
    population_size: int
    evaluation_budget: int
    seed: int
    crossover_probability: float = 0.9
    mutation_probability: float = 0.2
    eliminate_duplicates: bool = True

    @property
    def generations(self) -> int:
        if self.evaluation_budget % self.population_size:
            raise ValueError("evaluation_budget must be divisible by population_size")
        return self.evaluation_budget // self.population_size


def _history(result) -> list[ConvergencePoint]:
    points = []
    best_so_far = None
    for generation, state in enumerate(result.history, start=1):
        values = np.asarray(state.pop.get("F"), dtype=float)
        current = values.min(axis=0)
        best_so_far = current if best_so_far is None else np.minimum(best_so_far, current)
        depths, spread = pareto_quality(values)
        points.append(ConvergencePoint(generation, int(state.evaluator.n_eval), best_so_far.tolist(), depths[0], depths, spread))
    return points


def _run(problem: PermutationProblem, config: PymooRunConfig, *, multi: bool, crossover_kind: str = "ox", multi_algorithm: str = "nsga2") -> AlgorithmResult:
    adapter = PymooPermutationProblem(problem)
    common = dict(
        pop_size=config.population_size,
        sampling=PermutationRandomSampling(),
        crossover=(OrderCrossover(prob=config.crossover_probability) if crossover_kind == "ox"
                   else EdgeRecombinationCrossover(prob=config.crossover_probability)),
        mutation=InversionMutation(prob=config.mutation_probability),
        eliminate_duplicates=config.eliminate_duplicates,
    )
    reference_directions = None
    if not multi:
        algorithm = GA(**common)
    elif multi_algorithm == "spea2":
        algorithm = SPEA2(**common)
    elif multi_algorithm == "nsga3":
        reference_directions = get_reference_directions(
            "energy", len(problem.objective_names), config.population_size, seed=config.seed
        )
        algorithm = NSGA3(ref_dirs=reference_directions, **common)
    else:
        algorithm = NSGA2(**common)
    started = perf_counter()
    result = minimize(
        adapter, algorithm, termination=("n_gen", config.generations), seed=config.seed,
        save_history=True, verbose=False,
    )
    runtime = perf_counter() - started
    x = np.atleast_2d(result.X).astype(int)
    f = np.atleast_2d(result.F).astype(float)
    actual_evaluations = int(result.algorithm.evaluator.n_eval)
    if actual_evaluations > config.evaluation_budget:
        raise RuntimeError(f"pymoo exceeded the evaluation budget: {actual_evaluations} > {config.evaluation_budget}")
    deficit = config.evaluation_budget - actual_evaluations
    if deficit:
        # Duplicate elimination may produce a short final generation. Count an
        # exact budget by evaluating deterministic permutation samples and make
        # those evaluations eligible for the returned best/Pareto set.
        rng = np.random.default_rng(config.seed + 1_000_003)
        filler_x = np.vstack([rng.permutation(problem.dimension) for _ in range(deficit)]).astype(int)
        filler_f = evaluate_population(problem, filler_x)
        combined_x, combined_f = np.vstack((x, filler_x)), np.vstack((f, filler_f))
        if multi:
            selected = nondominated_ranks(combined_f) == 0
            x, f = combined_x[selected], combined_f[selected]
        else:
            selected = int(np.argmin(combined_f[:, 0]))
            x, f = combined_x[[selected]], combined_f[[selected]]
        actual_evaluations = config.evaluation_budget
    family = {"nsga2": "NSGA-II", "spea2": "SPEA2", "nsga3": "NSGA-III"}[multi_algorithm]
    name = f"pymoo {family} {'OX' if crossover_kind == 'ox' else 'ERX'}" if multi else (
        "pymoo GA-OX" if crossover_kind == "ox" else "pymoo GA-ERX")
    ranks = np.zeros(len(x), dtype=int) if multi else None
    crowding = result.opt.get("crowding") if multi and multi_algorithm == "nsga2" else None
    metadata = {"problem_identity": id(problem), "pymoo_version": __import__("pymoo").__version__,
                "budget_fill_evaluations": deficit}
    if reference_directions is not None:
        metadata.update({"reference_direction_method": "energy", "reference_direction_count": len(reference_directions)})
    history = _history(result)
    if history and deficit:
        best = np.minimum(np.asarray(history[-1].best), f.min(axis=0)).tolist()
        depths, spread = pareto_quality(f)
        history[-1] = ConvergencePoint(history[-1].generation, config.evaluation_budget, best,
                                       depths[0], depths, spread)
    return AlgorithmResult(
        algorithm=name,
        seed=config.seed,
        permutations=x.tolist(),
        objective_values=f.tolist(),
        runtime_seconds=runtime,
        evaluations=actual_evaluations,
        generations=config.generations,
        history=history,
        ranks=None if ranks is None else np.asarray(ranks, dtype=int).tolist(),
        crowding=None if crowding is None else [None if not np.isfinite(v) else float(v) for v in np.asarray(crowding)],
        metadata=metadata,
    )


def run_pymoo_ga(problem: PermutationProblem, config: PymooRunConfig) -> AlgorithmResult:
    if len(problem.objective_names) != 1:
        raise ValueError("pymoo GA-OX requires exactly one objective")
    return _run(problem, config, multi=False)


def run_pymoo_ga_erx(problem: PermutationProblem, config: PymooRunConfig) -> AlgorithmResult:
    if len(problem.objective_names) != 1:
        raise ValueError("pymoo GA-ERX requires exactly one objective")
    return _run(problem, config, multi=False, crossover_kind="erx")


def run_pymoo_nsga2(problem: PermutationProblem, config: PymooRunConfig) -> AlgorithmResult:
    if len(problem.objective_names) < 2:
        raise ValueError("pymoo NSGA-II requires at least two objectives")
    return _run(problem, config, multi=True)


def run_pymoo_nsga2_erx(problem: PermutationProblem, config: PymooRunConfig) -> AlgorithmResult:
    if len(problem.objective_names) < 2:
        raise ValueError("pymoo NSGA-II ERX requires at least two objectives")
    return _run(problem, config, multi=True, crossover_kind="erx")


def run_pymoo_spea2(problem: PermutationProblem, config: PymooRunConfig) -> AlgorithmResult:
    if len(problem.objective_names) < 2:
        raise ValueError("pymoo SPEA2 requires at least two objectives")
    return _run(problem, config, multi=True, multi_algorithm="spea2")


def run_pymoo_spea2_erx(problem: PermutationProblem, config: PymooRunConfig) -> AlgorithmResult:
    if len(problem.objective_names) < 2:
        raise ValueError("pymoo SPEA2 ERX requires at least two objectives")
    return _run(problem, config, multi=True, crossover_kind="erx", multi_algorithm="spea2")


def run_pymoo_nsga3(problem: PermutationProblem, config: PymooRunConfig) -> AlgorithmResult:
    if len(problem.objective_names) < 3:
        raise ValueError("pymoo NSGA-III requires at least three objectives")
    return _run(problem, config, multi=True, multi_algorithm="nsga3")


def run_pymoo_nsga3_erx(problem: PermutationProblem, config: PymooRunConfig) -> AlgorithmResult:
    if len(problem.objective_names) < 3:
        raise ValueError("pymoo NSGA-III ERX requires at least three objectives")
    return _run(problem, config, multi=True, crossover_kind="erx", multi_algorithm="nsga3")
