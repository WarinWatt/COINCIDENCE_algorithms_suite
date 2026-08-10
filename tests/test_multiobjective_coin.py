import numpy as np
import pytest

from coin.experiments import ExperimentConfiguration, ExperimentRunner
from coin.learning.pareto import (PARETO_SORTER_BACKEND, comparative_pareto_indicators,
                                  nondominated_ranks, pareto_quality, pareto_selection_scores,
                                  reference_nondominated_ranks)
from coin.problems.flowshop import generate_random_instance
from coin.core import MultiObjectiveCoinAlgorithm
from coin.models import EdgeConfig, HybridChainCoin, HybridCoin, PositionCoin, StartNodeEdgeCoin
from coin.problems.flowshop import FlowShopProblem
from coin.models import EHBSA, HBSAConfig, NHBSA


def test_pareto_ranks_require_one_strict_improvement_and_keep_equal_points():
    values = np.array([[1, 4], [2, 2], [4, 1], [3, 3], [2, 2]])
    assert nondominated_ranks(values).tolist() == [0, 0, 0, 1, 0]
    scores = pareto_selection_scores(values)
    assert set(scores.astype(int)) == set(range(len(values)))


def test_pymoo_sorter_matches_reference_for_random_discrete_objectives():
    rng = np.random.default_rng(20260809)
    assert PARETO_SORTER_BACKEND == "pymoo"
    for objectives in (2, 3, 5):
        for population_size in (1, 7, 50, 200):
            values = rng.integers(0, 20, size=(population_size, objectives))
            assert np.array_equal(nondominated_ranks(values), reference_nondominated_ranks(values))


def test_pareto_quality_reports_depth_counts_and_front_spread():
    values = np.array([[1, 4], [2, 2], [4, 1], [3, 3], [5, 5]], dtype=float)
    counts, spread = pareto_quality(values)
    assert counts == [3, 1, 1]
    assert spread == [3.0, 3.0]


def test_comparative_pareto_indicators_use_the_pooled_observed_front():
    reference_set = np.asarray([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    dominated_set = np.asarray([[0.8, 0.8]])
    reference, dominated = comparative_pareto_indicators([reference_set, dominated_set])

    assert reference["convergence"] == pytest.approx(0.0)
    assert reference["spread"] == pytest.approx(0.0)
    assert reference["nondominated_ratio"] == pytest.approx(1.0)
    assert dominated["convergence"] > 0
    assert dominated["nondominated_ratio"] == pytest.approx(0.0)
    assert reference["reference_size"] == 3


@pytest.mark.parametrize("model_type", [EHBSA, NHBSA])
@pytest.mark.parametrize("sampling_mode", ["wo", "wt"])
def test_hbsa_library_builds_from_selected_elite_and_samples_permutations(model_type, sampling_mode):
    config = HBSAConfig(
        problem_size=8, population_size=10, selection_ratio=40,
        bias_ratio=0.01, sampling_mode=sampling_mode, template_sample_ratio=50,
    )
    model = model_type(config, seed=1)
    population = model.generate_population()
    fitness = np.arange(10, dtype=float)
    statistics, unused = model.statistics(population, fitness)
    model.update(statistics, unused)
    generated = model.generate_population()

    assert statistics.selected.shape == (4, 8)
    assert statistics.histogram.shape == (8, 8)
    assert all(sorted(candidate.tolist()) == list(range(8)) for candidate in generated)
    assert model.config.selection_ratio == 40
    assert model.config.bias_ratio == 0.01
    assert model.config.sampling_mode == sampling_mode


@pytest.mark.parametrize("model_type", [EHBSA, NHBSA])
def test_hbsa_with_template_uses_pairwise_improvement_replacement(model_type):
    model = model_type(HBSAConfig(
        problem_size=8, population_size=6, selection_ratio=50,
        sampling_mode="wt", template_sample_ratio=50,
    ), seed=9)
    initial = model.generate_population()
    initial_fitness = np.arange(6, dtype=float) + 10
    statistics, unused = model.statistics(initial, initial_fitness)
    model.update(statistics, unused)

    children = model.generate_population()
    parents = np.asarray(model._pending_parents).copy()
    parent_fitness = np.asarray(model._pending_parent_fitness).copy()
    assert {tuple(row) for row in parents.tolist()} == {tuple(row) for row in initial.tolist()}
    statistics, _ = model.statistics(children, parent_fitness + 100)
    assert np.array_equal(statistics.population, parents)
    assert np.array_equal(statistics.fitness, parent_fitness)

    model.update(statistics, None)
    better_children = model.generate_population()
    better_parent_fitness = np.asarray(model._pending_parent_fitness).copy()
    statistics, _ = model.statistics(better_children, better_parent_fitness - 1)
    assert np.array_equal(statistics.population, better_children)


@pytest.mark.parametrize("algorithm", [
    "mo_edge_coin", "mo_position_coin", "mo_cnb_coin", "mo_hybrid_coin", "mo_hybrid_chain", "mo_start_node_edge_coin",
])
def test_every_coin_representation_supports_multiobjective_exact_budget(algorithm):
    instance = generate_random_instance(8, 3, seed=81)
    config = ExperimentConfiguration(
        algorithms=(algorithm,), objectives=("makespan", "total_flow_time"),
        population_size=10, evaluation_budget=30, maximum_generations=3, seeds=(7,),
        reward_ratio=30, punishment_ratio=30,
    )
    result = ExperimentRunner().run(instance, config)
    run = result.runs[0]
    assert run.evaluations == 30
    assert run.generations == 3
    assert all(len(values) == 2 for values in run.objective_values)
    assert all(sorted(permutation) == list(range(8)) for permutation in run.permutations)
    assert np.all(nondominated_ranks(np.asarray(run.objective_values)) == 0)


def test_mo_coin_and_nsga2_share_evaluator_identity_and_budget():
    instance = generate_random_instance(8, 3, seed=82)
    config = ExperimentConfiguration(
        algorithms=("pymoo_nsga2", "mo_edge_coin", "mo_position_coin"),
        objectives=("makespan", "total_flow_time"), population_size=10,
        evaluation_budget=30, maximum_generations=3, seeds=(8,),
    )
    result = ExperimentRunner().run(instance, config)
    assert {run.evaluations for run in result.runs} == {30}
    assert {run.metadata["problem_identity"] for run in result.runs} == {result.evaluator_identity}


def test_position_hybrid_and_start_edge_diverge_after_learning():
    instance=generate_random_instance(8,3,seed=83)
    problem=FlowShopProblem(instance.processing_times,objectives=("makespan","total_flow_time"))
    config=EdgeConfig(problem_size=8,population_size=20,objective="min",reward_ratio=20,punishment_ratio=20,training_rate=20)
    next_populations=[]
    for model_type in (PositionCoin,HybridCoin,HybridChainCoin,StartNodeEdgeCoin):
        model=model_type(config,seed=42)
        MultiObjectiveCoinAlgorithm(model,problem).step()
        next_populations.append(model.generate_population())
    assert len({population.tobytes() for population in next_populations})==4
