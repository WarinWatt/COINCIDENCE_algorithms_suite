"""Fair single-objective Sudoku comparisons for four permutation algorithms."""

from __future__ import annotations

from time import perf_counter
import numpy as np

from coin.adapters.pymoo.runner import PymooRunConfig, run_pymoo_ga
from coin.core.algorithm import PermutationCoinAlgorithm
from coin.models import CNBCoin, PositionCoin, EHBSA, NHBSA, HBSAConfig
from coin.models.edge import EdgeConfig
from coin.problems.sudoku import SudokuPermutationProblem, SudokuBlockPermutationProblem
from coin.experiments.sudoku_block import run_block_algorithm

ALGORITHM_LABELS = {
    "coin": "NB-COIN · random position → value",
    "cnb_coin": "CNB-COIN · position 0→n → value",
    "ehbsa": "EHBSA · Edge Histogram",
    "nhbsa": "NHBSA · Node Histogram",
    "ga_ox": "GA-OX",
}


def _run_model(name: str, problem: SudokuPermutationProblem, population_size: int,
               generations: int, seed: int) -> dict:
    if name in ("coin", "cnb_coin"):
        model = (PositionCoin if name == "coin" else CNBCoin)(EdgeConfig(
            problem_size=problem.dimension, population_size=population_size,
            training_rate=5, reward_ratio=10, punishment_ratio=10,
            objective="min", learning_mode="conservative",
        ), seed=seed)
    elif name in ("ehbsa", "nhbsa"):
        config = HBSAConfig(
            problem_size=problem.dimension, population_size=population_size,
            selection_ratio=50, bias_ratio=0.005, sampling_mode="wt",
            template_sample_ratio=50, objective="min",
        )
        model = (EHBSA if name == "ehbsa" else NHBSA)(config, seed=seed)
    else:
        raise ValueError(name)
    algorithm = PermutationCoinAlgorithm(model, problem)
    best_value = float("inf")
    best_permutation = None
    progress = []
    started = perf_counter()
    for _ in range(generations):
        population, objectives = algorithm.step()
        index = int(np.argmin(objectives[:, 0]))
        if objectives[index, 0] < best_value:
            best_value = float(objectives[index, 0])
            best_permutation = population[index].copy()
        grid = problem.decode(best_permutation)
        metrics = problem.prime_metrics(grid)
        progress.append({"generation": algorithm.generation, "evaluations": algorithm.evaluations,
                         "penalty": int(best_value), **{k: int(v) for k, v in metrics.items() if k != "penalty"}})
    return _result(name, problem, best_permutation, best_value, progress,
                   perf_counter() - started, population_size * generations, generations, seed)


def _result(name: str, problem: SudokuPermutationProblem, permutation: np.ndarray,
            value: float, progress: list[dict], runtime: float, evaluations: int,
            generations: int, seed: int) -> dict:
    grid = problem.decode(np.asarray(permutation, dtype=int))
    metrics = problem.prime_metrics(grid)
    return {"algorithm": name, "label": ALGORITHM_LABELS[name], "seed": seed,
            "permutation": np.asarray(permutation, dtype=int).tolist(),
            "grid": grid.astype(int).ravel().tolist(), "penalty": int(value),
            "solved": metrics["invalid_groups"] == 0,
            "metrics": {key: int(value) for key, value in metrics.items()},
            "evaluations": evaluations, "generations": generations,
            "runtime_seconds": runtime, "history": progress}


def run_comparison(problem: SudokuPermutationProblem, algorithms: list[str],
                   population_size: int, generations: int, seed: int) -> dict:
    unknown = set(algorithms) - set(ALGORITHM_LABELS)
    if unknown:
        raise ValueError(f"unknown algorithms: {sorted(unknown)}")
    budget = population_size * generations
    results = []
    for name in algorithms:
        if name != "ga_ox":
            results.append(_run_model(name, problem, population_size, generations, seed))
            continue
        config = PymooRunConfig(population_size=population_size,
                                evaluation_budget=budget, seed=seed,
                                crossover_probability=0.9, mutation_probability=0.2,
                                eliminate_duplicates=False)
        started = perf_counter()
        ga = run_pymoo_ga(problem, config)
        permutation = np.asarray(ga.permutations[0], dtype=int)
        history = []
        for point in ga.history:
            penalty = int(point.best[0])
            history.append({"generation": point.generation, "evaluations": point.evaluations,
                            "penalty": penalty})
        results.append(_result(name, problem, permutation, float(ga.objective_values[0][0]),
                               history, perf_counter() - started, ga.evaluations,
                               ga.generations, seed))
    return {"evaluation_budget_per_algorithm": budget, "results": results}


def run_block_comparison(problem: SudokuBlockPermutationProblem, algorithms: list[str],
                         population_size: int, generations: int, seed: int,
                         algorithm_parameters: dict | None = None,
                         evaluation_method: str = "paper_prime",
                         stop_at_solution: bool = True) -> dict:
    unknown = set(algorithms) - set(ALGORITHM_LABELS)
    if unknown:
        raise ValueError(f"unknown algorithms: {sorted(unknown)}")
    if problem.dimension == 0:
        grid = problem.puzzle.copy()
        metrics = problem.paper_metrics(grid)
        metric_key = "previous_penalty" if evaluation_method == "legacy_hierarchical" else "paper_penalty"
        results = [{"algorithm": name, "label": f"{ALGORITHM_LABELS[name]} · skipped",
                    "seed": seed, "grid": grid.astype(int).ravel().tolist(),
                    "penalty": int(metrics[metric_key]), "paper_score": metrics["paper_score"],
                    "solved": True, "metrics": metrics, "evaluations": 0, "generations": 0,
                    "best_found_at_evaluation": 0, "best_found_at_generation": 0,
                    "solution_found_at_evaluation": 0,
                    "runtime_seconds": 0.0, "history": [], "evaluation_method": evaluation_method,
                    "stop_at_solution": stop_at_solution, "termination_reason": "obvious_fill",
                    "skipped": True, "solved_by": "obvious_fill"} for name in algorithms]
        return {"representation": "block", "evaluation_budget_per_algorithm": 0,
                "preprocessing_solved": True, "results": results}
    parameters = algorithm_parameters or {}
    results = [run_block_algorithm(name, problem, population_size, generations, seed,
                                   ALGORITHM_LABELS[name], evaluation_method=evaluation_method,
                                   stop_at_solution=stop_at_solution,
                                   **parameters.get(name, {})) for name in algorithms]
    return {"representation": "block", "evaluation_budget_per_algorithm": population_size * generations,
            "results": results}
