from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pymoo = pytest.importorskip("pymoo")

from coin.adapters.pymoo import (PymooPermutationProblem, PymooRunConfig, run_pymoo_ga,
    run_pymoo_ga_erx, run_pymoo_nsga2, run_pymoo_nsga2_erx, run_pymoo_nsga3,
    run_pymoo_nsga3_erx, run_pymoo_spea2, run_pymoo_spea2_erx)
from coin.experiments import ExperimentConfiguration, ExperimentRunner
from coin.problems.flowshop import FlowShopProblem, generate_random_instance
import coin.adapters.pymoo.runner as pymoo_runner


INSTANCE = generate_random_instance(8, 3, seed=8)


def test_pymoo_problem_evaluation_is_identical_to_direct_evaluator():
    problem = FlowShopProblem(INSTANCE.processing_times, objectives=("makespan", "total_flow_time"))
    adapter = PymooPermutationProblem(problem)
    population = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]])
    out = {}
    adapter._evaluate(population, out)
    expected = np.vstack([problem.evaluate(candidate) for candidate in population])
    assert np.array_equal(out["F"], expected)
    assert adapter.coin_problem is problem


def test_short_pymoo_generation_is_filled_to_the_exact_fair_budget(monkeypatch):
    problem = FlowShopProblem(INSTANCE.processing_times, objectives=("makespan", "total_flow_time"))
    candidates = np.array([[0,1,2,3,4,5,6,7], [7,6,5,4,3,2,1,0]])
    fake = SimpleNamespace(
        X=candidates, F=problem.evaluate_population(candidates), history=[], opt={},
        algorithm=SimpleNamespace(evaluator=SimpleNamespace(n_eval=17)),
    )
    monkeypatch.setattr(pymoo_runner, "minimize", lambda *args, **kwargs: fake)

    result = run_pymoo_nsga2(problem, PymooRunConfig(10, 20, seed=9))

    assert result.evaluations == 20
    assert result.metadata["budget_fill_evaluations"] == 3
    assert result.permutations


def test_ga_ox_returns_valid_permutation_and_exact_budget():
    problem = FlowShopProblem(INSTANCE.processing_times)
    result = run_pymoo_ga(problem, PymooRunConfig(10, 30, seed=11))
    assert result.algorithm == "pymoo GA-OX"
    assert result.evaluations == 30
    assert result.generations == 3
    assert sorted(result.permutations[0]) == list(range(8))
    assert len(result.objective_values[0]) == 1
    assert result.metadata["problem_identity"] == id(problem)


def test_nsga2_returns_correct_objective_dimension_and_exact_budget():
    problem = FlowShopProblem(INSTANCE.processing_times, objectives=("makespan", "total_flow_time"))
    result = run_pymoo_nsga2(problem, PymooRunConfig(10, 30, seed=12))
    assert result.evaluations == 30
    assert all(len(values) == 2 for values in result.objective_values)
    assert all(sorted(permutation) == list(range(8)) for permutation in result.permutations)


@pytest.mark.parametrize("runner,name", [(run_pymoo_ga_erx, "pymoo GA-ERX")])
def test_erx_single_objective_baseline_is_valid_and_budget_fair(runner, name):
    problem = FlowShopProblem(INSTANCE.processing_times)
    result = runner(problem, PymooRunConfig(10, 30, seed=14))
    assert result.algorithm == name
    assert result.evaluations == 30
    assert sorted(result.permutations[0]) == list(range(8))


def test_erx_nsga2_is_valid_and_budget_fair():
    problem = FlowShopProblem(INSTANCE.processing_times, objectives=("makespan", "total_flow_time"))
    result = run_pymoo_nsga2_erx(problem, PymooRunConfig(10, 30, seed=15))
    assert result.algorithm == "pymoo NSGA-II ERX"
    assert result.evaluations == 30
    assert all(len(values) == 2 for values in result.objective_values)


@pytest.mark.parametrize("runner,name", [
    (run_pymoo_spea2, "pymoo SPEA2 OX"),
    (run_pymoo_spea2_erx, "pymoo SPEA2 ERX"),
])
def test_spea2_variants_return_a_valid_final_set_with_exact_budget(runner, name):
    problem = FlowShopProblem(INSTANCE.processing_times, objectives=("makespan", "total_flow_time"))
    result = runner(problem, PymooRunConfig(10, 30, seed=21))
    assert result.algorithm == name
    assert result.evaluations == 30
    assert all(len(values) == 2 for values in result.objective_values)
    assert all(sorted(permutation) == list(range(8)) for permutation in result.permutations)


@pytest.mark.parametrize("runner,name", [
    (run_pymoo_nsga3, "pymoo NSGA-III OX"),
    (run_pymoo_nsga3_erx, "pymoo NSGA-III ERX"),
])
def test_nsga3_variants_use_exact_reference_direction_and_evaluation_counts(runner, name):
    problem = FlowShopProblem(INSTANCE.processing_times, objectives=(
        "makespan", "total_flow_time", "total_machine_idle_time"
    ))
    result = runner(problem, PymooRunConfig(10, 30, seed=22))
    assert result.algorithm == name
    assert result.evaluations == 30
    assert result.metadata["reference_direction_method"] == "energy"
    assert result.metadata["reference_direction_count"] == 10
    assert all(len(values) == 3 for values in result.objective_values)
    assert all(sorted(permutation) == list(range(8)) for permutation in result.permutations)


def test_nsga3_rejects_two_objectives():
    problem = FlowShopProblem(INSTANCE.processing_times, objectives=("makespan", "total_flow_time"))
    with pytest.raises(ValueError, match="at least three"):
        run_pymoo_nsga3(problem, PymooRunConfig(10, 20, seed=1))


def test_ox_and_erx_diverge_once_crossover_generations_exist():
    problem=FlowShopProblem(INSTANCE.processing_times,objectives=("makespan","total_flow_time"))
    config=PymooRunConfig(10,30,seed=19)
    ox=run_pymoo_nsga2(problem,config)
    erx=run_pymoo_nsga2_erx(problem,config)
    assert ox.permutations!=erx.permutations


def test_fair_runner_uses_one_problem_identity_and_exact_budget_for_pymoo_and_coin():
    config = ExperimentConfiguration(
        algorithms=("pymoo_ga_ox", "edge_coin", "position_coin", "hybrid_coin", "start_node_edge_coin"),
        objectives=("makespan",), population_size=10, evaluation_budget=30,
        maximum_generations=3, seeds=(3, 4),
    )
    result = ExperimentRunner().run(INSTANCE, config)
    assert len(result.runs) == 10
    assert {run.evaluations for run in result.runs} == {30}
    assert {run.metadata["problem_identity"] for run in result.runs} == {result.evaluator_identity}
    assert all(run.generations == 3 for run in result.runs)
    assert len(result.summary) == 5


def test_coin_parameters_are_forwarded_and_erx_compares_under_same_budget():
    config = ExperimentConfiguration(
        algorithms=("pymoo_ga_ox", "pymoo_ga_erx", "edge_coin"), objectives=("makespan",),
        population_size=10, evaluation_budget=30, maximum_generations=3, seeds=(5,),
        training_rate=9, reward_ratio=20, punishment_ratio=10,
        rewards_enabled=True, punishments_enabled=False,
    )
    result = ExperimentRunner().run(INSTANCE, config)
    assert {run.evaluations for run in result.runs} == {30}
    assert {run.algorithm for run in result.runs} == {"pymoo GA-OX", "pymoo GA-ERX", "Edge-Based COIN"}


def test_runner_reports_every_available_objective_without_changing_evolution_budget():
    config = ExperimentConfiguration(
        algorithms=("edge_coin",), objectives=("makespan",),
        population_size=10, evaluation_budget=30, maximum_generations=3, seeds=(1,),
    )
    result = ExperimentRunner().run(INSTANCE, config)
    run = result.runs[0]

    assert result.objective_names == ("makespan",)
    assert result.reported_objective_names == (
        "makespan", "total_flow_time", "total_machine_idle_time"
    )
    assert len(run.objective_values[0]) == 1
    assert len(run.reported_objective_values[0]) == 3
    expected = FlowShopProblem(
        INSTANCE.processing_times, result.reported_objective_names
    ).evaluate(np.asarray(run.permutations[0]))
    assert np.array_equal(run.reported_objective_values[0], expected)
    assert run.evaluations == config.evaluation_budget


def test_parameter_grid_expands_coin_and_hbsa_with_parameter_metadata():
    config = ExperimentConfiguration(
        algorithms=("edge_coin", "ehbsa", "nhbsa"), objectives=("makespan",),
        population_size=10, evaluation_budget=20, maximum_generations=2, seeds=(1,),
        parameter_grid=True, hbsa_sampling_mode="wt", hbsa_template_sample_ratio=50,
    )
    result = ExperimentRunner().run(INSTANCE, config)

    assert len(result.runs) == 24  # 16 conservative COIN + 4 EHBSA + 4 NHBSA
    coin = [run for run in result.runs if run.algorithm.startswith("Edge-Based COIN")]
    histograms = [run for run in result.runs if "HBSA" in run.algorithm]
    assert len(coin) == 16
    assert {(run.metadata["parameters"]["reward_ratio"], run.metadata["parameters"]["training_rate"]) for run in coin} == {
        (ratio, rate) for ratio in (25, 20, 15, 10) for rate in (5, 10, 15, 20)
    }
    assert len(histograms) == 8
    assert {run.metadata["parameters"]["selection_ratio"] for run in histograms} == {50, 40, 30, 20}
    assert all(run.metadata["parameters"]["sampling_mode"] == "wt" for run in histograms)
    assert {run.evaluations for run in result.runs} == {20}


def test_individual_parameter_grids_expand_only_selected_algorithms():
    config = ExperimentConfiguration(
        algorithms=("edge_coin", "position_coin", "ehbsa", "nhbsa"), objectives=("makespan",),
        population_size=10, evaluation_budget=20, maximum_generations=2, seeds=(1,),
        parameter_grid_algorithms=("edge_coin", "ehbsa"),
    )
    result = ExperimentRunner().run(INSTANCE, config)

    assert len(result.runs) == 22  # 16 COIN + 1 NB-COIN + 4 EHBSA + 1 NHBSA
    assert sum(run.algorithm.startswith("Edge-Based COIN") for run in result.runs) == 16
    assert sum(run.algorithm.startswith("NB-COIN") for run in result.runs) == 1
    assert sum(run.algorithm.startswith("EHBSA") for run in result.runs) == 4
    assert sum(run.algorithm.startswith("NHBSA") for run in result.runs) == 1


def test_runner_rejects_unapproved_multiobjective_coin():
    config = ExperimentConfiguration(
        algorithms=("edge_coin", "pymoo_nsga2"), objectives=("makespan", "total_flow_time"),
        population_size=10, evaluation_budget=20, maximum_generations=2,
    )
    with pytest.raises(ValueError, match="multi-objective"):
        ExperimentRunner().run(INSTANCE, config)


def test_budget_must_be_divisible_and_within_generation_limit():
    with pytest.raises(ValueError, match="multiple"):
        ExperimentConfiguration(("edge_coin",), ("makespan",), population_size=10, evaluation_budget=25)
    with pytest.raises(ValueError, match="maximum_generations"):
        ExperimentConfiguration(("edge_coin",), ("makespan",), population_size=10, evaluation_budget=30, maximum_generations=2)
