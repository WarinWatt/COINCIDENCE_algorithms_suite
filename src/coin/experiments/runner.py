from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from coin.adapters.pymoo.result import AlgorithmResult, ConvergencePoint
from coin.adapters.pymoo.runner import (PymooRunConfig, run_pymoo_ga, run_pymoo_ga_erx,
    run_pymoo_nsga2, run_pymoo_nsga2_erx, run_pymoo_nsga3, run_pymoo_nsga3_erx,
    run_pymoo_spea2, run_pymoo_spea2_erx)
from coin.core import MultiObjectiveCoinAlgorithm, PermutationCoinAlgorithm
from coin.models import CNBCoin, EdgeConfig, EHBSA, HBSAConfig, HybridChainCoin, HybridCoin, NHBSA, OptimizedEdgeCoin, PositionCoin, StartNodeEdgeCoin
from coin.problems.base import evaluate_population
from coin.problems.flowshop import FlowShopInstance, FlowShopProblem
from coin.problems.tsptw import MatrixTSPTWInstance, TSPTWInstance, TSPTWProblem
from coin.learning.pareto import comparative_pareto_indicators
from .configuration import ExperimentConfiguration
from .statistics import descriptive

COIN_FACTORIES = {
    "edge_coin": OptimizedEdgeCoin,
    "position_coin": PositionCoin,
    "cnb_coin": CNBCoin,
    "hybrid_coin": HybridCoin,
    "hybrid_chain": HybridChainCoin,
    "start_node_edge_coin": StartNodeEdgeCoin,
    "ehbsa": EHBSA,
    "nhbsa": NHBSA,
    "ehbsa_wo": EHBSA,
    "ehbsa_wt": EHBSA,
    "nhbsa_wo": NHBSA,
    "nhbsa_wt": NHBSA,
}
HBSA_VARIANTS = {
    "ehbsa": ("wo", EHBSA), "nhbsa": ("wo", NHBSA),
    "ehbsa_wo": ("wo", EHBSA), "ehbsa_wt": ("wt", EHBSA),
    "nhbsa_wo": ("wo", NHBSA), "nhbsa_wt": ("wt", NHBSA),
}
MO_COIN_FACTORIES = {f"mo_{name}": factory for name, factory in COIN_FACTORIES.items()}
ALGORITHM_NAMES = {
    "pymoo_ga_ox": "pymoo GA-OX", "pymoo_ga_erx": "pymoo GA-ERX",
    "pymoo_nsga2": "pymoo NSGA-II OX", "pymoo_nsga2_erx": "pymoo NSGA-II ERX",
    "pymoo_spea2": "pymoo SPEA2 OX", "pymoo_spea2_erx": "pymoo SPEA2 ERX",
    "pymoo_nsga3": "pymoo NSGA-III OX", "pymoo_nsga3_erx": "pymoo NSGA-III ERX",
    "edge_coin": "Edge-Based COIN", "position_coin": "NB-COIN · random position → value", "cnb_coin": "CNB-COIN · position 0→n → value",
    "hybrid_coin": "Hybrid Template COIN · Node template → Edge completion",
    "hybrid_chain": "Hybrid Chain COIN · Node/Edge per link", "start_node_edge_coin": "Start-Node Edge COIN",
    "ehbsa": "EHBSA · Edge Histogram", "nhbsa": "NHBSA · Node Histogram",
    "ehbsa_wo": "EHBSA-WO · Edge Histogram",
    "ehbsa_wt": "EHBSA-WT · Edge Histogram + template",
    "nhbsa_wo": "NHBSA-WO · Node Histogram",
    "nhbsa_wt": "NHBSA-WT · Node Histogram + template",
    "mo_edge_coin": "MO Edge COIN", "mo_position_coin": "MO NB-COIN", "mo_cnb_coin": "MO CNB-COIN",
    "mo_hybrid_coin": "MO Hybrid Template COIN · Node template → Edge completion",
    "mo_hybrid_chain": "MO Hybrid Chain COIN · Node/Edge per link", "mo_start_node_edge_coin": "MO Start-Node Edge COIN",
}


@dataclass(slots=True)
class ExperimentResult:
    objective_names: tuple[str, ...]
    reported_objective_names: tuple[str, ...]
    runs: list[AlgorithmResult]
    summary: dict[str, dict[str, float]]
    evaluator_identity: int

    def as_dict(self):
        return asdict(self)


class ExperimentRunner:
    def run(self, instance: FlowShopInstance | TSPTWInstance, config: ExperimentConfiguration,
            progress=None, *, hard_windows: bool = True) -> ExperimentResult:
        if isinstance(instance, FlowShopInstance):
            problem = FlowShopProblem(instance.processing_times, config.objectives, instance.due_dates)
            reported_objective_names = problem.available_objective_names
            reporting_problem = FlowShopProblem(
                instance.processing_times, reported_objective_names, instance.due_dates
            )
        elif isinstance(instance, (TSPTWInstance, MatrixTSPTWInstance)):
            problem = TSPTWProblem(instance, config.objectives, hard_windows=hard_windows)
            reported_objective_names = problem.available_objective_names
            reporting_problem = TSPTWProblem(instance, reported_objective_names, hard_windows=False)
        else:
            raise TypeError("unsupported experiment instance type")

        def attach_reported_objectives(result: AlgorithmResult) -> AlgorithmResult:
            if result.permutations:
                result.reported_objective_values = evaluate_population(
                    reporting_problem, np.asarray(result.permutations, dtype=np.int64)
                ).tolist()
            return result
        pymoo_multi = {"pymoo_nsga2", "pymoo_nsga2_erx", "pymoo_spea2", "pymoo_spea2_erx", "pymoo_nsga3", "pymoo_nsga3_erx"}
        if len(config.objectives) > 1 and any(a in COIN_FACTORIES or a.startswith("pymoo_ga_") for a in config.algorithms):
            raise ValueError("multi-objective experiments require an MO COIN or NSGA-II variant")
        if len(config.objectives) == 1 and any(a in pymoo_multi or a in MO_COIN_FACTORIES for a in config.algorithms):
            raise ValueError("multi-objective algorithm variants require multiple objectives")
        if len(config.objectives) < 3 and any(a.startswith("pymoo_nsga3") for a in config.algorithms):
            raise ValueError("NSGA-III requires at least three objectives")
        unknown = set(config.algorithms) - set(ALGORITHM_NAMES)
        if unknown:
            raise ValueError(f"unknown algorithms: {', '.join(sorted(unknown))}")
        run_specs: list[tuple[str, dict[str, object]]] = []
        for name in config.algorithms:
            grid_enabled = config.parameter_grid or name in config.parameter_grid_algorithms
            if grid_enabled and name in ("ehbsa", "nhbsa"):
                run_specs.extend((name, {
                    "selection_ratio": ratio,
                    "bias_ratio": config.hbsa_bias_ratio,
                    "sampling_mode": config.hbsa_sampling_mode,
                    "template_sample_ratio": config.hbsa_template_sample_ratio,
                }) for ratio in (50, 40, 30, 20))
            elif grid_enabled and name in ("edge_coin", "position_coin"):
                run_specs.extend(
                    (name, {"reward_ratio": ratio, "punishment_ratio": ratio, "training_rate": rate})
                    for ratio in (25, 20, 15, 10) for rate in (5, 10, 15, 20)
                )
            else:
                run_specs.append((name, {}))
        runs = []
        total = len(run_specs) * len(config.seeds)
        completed = 0
        for seed in config.seeds:
            for algorithm_name, variant in run_specs:
                if progress:
                    progress(completed, total, algorithm_name, seed)
                if algorithm_name.startswith("pymoo_"):
                    pymoo_config = PymooRunConfig(
                        config.population_size, config.evaluation_budget, seed,
                        config.crossover_probability, config.mutation_probability, config.eliminate_duplicates,
                    )
                    functions = {"pymoo_ga_ox": run_pymoo_ga, "pymoo_ga_erx": run_pymoo_ga_erx,
                                 "pymoo_nsga2": run_pymoo_nsga2, "pymoo_nsga2_erx": run_pymoo_nsga2_erx,
                                 "pymoo_spea2": run_pymoo_spea2, "pymoo_spea2_erx": run_pymoo_spea2_erx,
                                 "pymoo_nsga3": run_pymoo_nsga3, "pymoo_nsga3_erx": run_pymoo_nsga3_erx}
                    result = functions[algorithm_name](problem, pymoo_config)
                else:
                    model_config = EdgeConfig(
                        problem_size=problem.dimension, population_size=config.population_size,
                        reward_ratio=int(variant.get("reward_ratio", config.reward_ratio)),
                        punishment_ratio=int(variant.get("punishment_ratio", config.punishment_ratio)),
                        training_rate=int(variant.get("training_rate", config.training_rate)),
                        rewards_enabled=config.rewards_enabled,
                        punishments_enabled=config.punishments_enabled,
                        objective="min", learning_mode=config.learning_mode,
                    )
                    factory = MO_COIN_FACTORIES.get(algorithm_name, COIN_FACTORIES.get(algorithm_name))
                    if algorithm_name in HBSA_VARIANTS:
                        sampling_mode, _ = HBSA_VARIANTS[algorithm_name]
                        hbsa_config = HBSAConfig(
                            problem_size=problem.dimension,
                            population_size=config.population_size,
                            selection_ratio=int(variant.get("selection_ratio", config.hbsa_selection_ratio)),
                            bias_ratio=config.hbsa_bias_ratio,
                            sampling_mode=(config.hbsa_sampling_mode
                                           if algorithm_name in ("ehbsa", "nhbsa")
                                           else sampling_mode),
                            template_sample_ratio=config.hbsa_template_sample_ratio,
                            objective="min",
                        )
                        model = factory(hbsa_config, seed=seed)
                    else:
                        model = factory(model_config, seed=seed)
                    algorithm = MultiObjectiveCoinAlgorithm(model, problem) if algorithm_name in MO_COIN_FACTORIES else PermutationCoinAlgorithm(model, problem)
                    started = perf_counter()
                    if algorithm_name in MO_COIN_FACTORIES:
                        for _ in range(config.evaluation_budget // config.population_size):
                            algorithm.step()
                        runtime = perf_counter() - started
                        result = AlgorithmResult(
                            ALGORITHM_NAMES[algorithm_name], seed, algorithm.archive_population.astype(int).tolist(),
                            algorithm.archive_values.tolist(), runtime, algorithm.evaluations, algorithm.generation,
                            [ConvergencePoint(g, e, best, depths[0], depths, spread) for g, e, best, depths, spread in algorithm.history],
                            ranks=[0] * len(algorithm.archive_population), metadata={"problem_identity": id(problem)},
                        )
                        if result.evaluations != config.evaluation_budget:
                            raise RuntimeError(f"{result.algorithm} used {result.evaluations}, expected {config.evaluation_budget}")
                        if variant:
                            result.metadata["parameters"] = variant
                            result.algorithm += " · " + ", ".join(f"{key}={value}" for key, value in variant.items())
                        runs.append(attach_reported_objectives(result)); completed += 1
                        continue
                    best = float("inf")
                    best_permutation = None
                    for _ in range(config.evaluation_budget // config.population_size):
                        population, objective_matrix = algorithm.step()
                        index = int(np.argmin(objective_matrix[:, 0]))
                        if float(objective_matrix[index, 0]) < best:
                            best = float(objective_matrix[index, 0])
                            best_permutation = population[index].astype(int).tolist()
                    runtime = perf_counter() - started
                    result = AlgorithmResult(
                        ALGORITHM_NAMES[algorithm_name], seed, [best_permutation], [[best]], runtime,
                        algorithm.evaluations, algorithm.generation,
                        [ConvergencePoint(p.generation, p.evaluations, [p.best]) for p in algorithm.history],
                        metadata={"problem_identity": id(problem)},
                    )
                if result.evaluations != config.evaluation_budget:
                    raise RuntimeError(f"{result.algorithm} used {result.evaluations}, expected {config.evaluation_budget}")
                if variant:
                    result.metadata["parameters"] = variant
                    result.algorithm += " · " + ", ".join(f"{key}={value}" for key, value in variant.items())
                if algorithm_name in HBSA_VARIANTS:
                    result.metadata["sampling_mode"] = model.config.sampling_mode
                    result.metadata["template_sample_ratio"] = model.config.template_sample_ratio
                runs.append(attach_reported_objectives(result))
                completed += 1
        summary = {}
        if len(config.objectives) > 1 and runs:
            indicators = comparative_pareto_indicators([
                np.asarray(run.objective_values, dtype=float) for run in runs
            ])
            for run, values in zip(runs, indicators):
                run.metadata["pareto_indicators"] = values
                run.metadata["pareto_reference"] = "pooled_observed_nondominated_front"
        for label in dict.fromkeys(r.algorithm for r in runs):
            values = [r.objective_values[0][0] for r in runs if r.algorithm == label]
            if values:
                summary[label] = descriptive(values)
        return ExperimentResult(
            objective_names=config.objectives,
            reported_objective_names=reported_objective_names,
            runs=runs,
            summary=summary,
            evaluator_identity=id(problem),
        )
