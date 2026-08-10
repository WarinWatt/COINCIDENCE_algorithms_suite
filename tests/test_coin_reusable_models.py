from __future__ import annotations

import numpy as np
import pytest

from coin.core import PermutationCoinAlgorithm
from coin.models import (
    EdgeConfig,
    CNBCoin,
    HybridCoin,
    HybridChainCoin,
    OptimizedEdgeCoin,
    PositionCoin,
    ReferenceEdgeCoin,
    StartNodeEdgeCoin,
)
from coin.problems.flowshop import SMALL_3X2, FlowShopProblem


def assert_valid_population(population: np.ndarray, size: int) -> None:
    assert population.ndim == 2
    expected = np.arange(size)
    for candidate in population:
        assert np.array_equal(np.sort(candidate), expected)


def test_reference_edge_seeded_output_survives_rng_extraction():
    config = EdgeConfig(problem_size=5, population_size=3)
    actual = ReferenceEdgeCoin(config, seed=42).generate_population()
    assert actual.tolist() == [
        [3, 4, 2, 0, 1],
        [0, 3, 4, 1, 2],
        [3, 1, 0, 4, 2],
    ]


def test_position_generation_is_reproducible_and_valid():
    config = EdgeConfig(problem_size=12, population_size=30)
    first = PositionCoin(config, seed=91).generate_population()
    second = PositionCoin(config, seed=91).generate_population()
    assert np.array_equal(first, second)
    assert_valid_population(first, 12)

def test_cnb_chain_and_random_position_nb_are_distinct_valid_generators():
    config=EdgeConfig(problem_size=12,population_size=30)
    nb=PositionCoin(config,seed=91).generate_population(); cnb=CNBCoin(config,seed=91).generate_population()
    assert_valid_population(cnb,12)
    assert not np.array_equal(nb,cnb)


def test_position_statistics_use_position_job_orientation_and_legacy_cohorts():
    population = np.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])
    fitness = np.array([0, 1, 2, 3])
    model = PositionCoin(EdgeConfig(problem_size=4, population_size=4, objective="min"))
    rewards, punishments = model.statistics(population, fitness)
    assert rewards.sum() == 4
    assert punishments.sum() == 8  # inclusive Delphi lower tail
    assert rewards[0, 0] == 1
    assert rewards[1, 1] == 1
    assert punishments[0, 2] == 1
    assert punishments[0, 3] == 1


def test_position_incremental_and_reconstruction_learning():
    config = EdgeConfig(problem_size=4, population_size=4)
    model = PositionCoin(config)
    rewards = np.zeros((4, 4), dtype=np.int64)
    punishments = np.zeros_like(rewards)
    rewards[0, 1] = 1
    punishments[2, 3] = 1
    before = model.weights.sum(axis=1).copy()
    model.update(rewards, punishments)
    assert np.array_equal(model.weights.sum(axis=1), before)
    assert model.weights[0].tolist() == [79, 83, 79, 79]
    assert model.weights[2].tolist() == [81, 81, 81, 77]

    histogram = PositionCoin(EdgeConfig(problem_size=4, learning_mode="reconstruction"))
    histogram.update(rewards, np.full_like(rewards, 99))
    assert np.array_equal(histogram.weights, rewards * 10 + 1)


def test_hybrid_preserves_scattered_node_template_then_trains_both_models():
    config = EdgeConfig(problem_size=8, population_size=12)
    hybrid = HybridCoin(config, seed=7)
    standalone_position = PositionCoin(config, seed=7)
    population = hybrid.generate_population()
    templates = standalone_position.generate_population()
    assert_valid_population(population, 8)
    assert np.all(hybrid.last_template_masks[:, 0])
    assert np.array_equal(population[hybrid.last_template_masks], templates[hybrid.last_template_masks])
    retained = hybrid.last_template_masks.sum(axis=1)
    assert np.all(retained >= round(config.problem_size * .30))
    assert np.all(retained <= round(config.problem_size * .70))
    edge_before = hybrid.edge_weights.copy()
    position_before = hybrid.position_weights.copy()
    fitness = np.arange(config.population_size)
    rewards, punishments = hybrid.statistics(population, fitness)
    hybrid.update(rewards, punishments)
    assert not np.array_equal(hybrid.edge_weights, edge_before)
    assert not np.array_equal(hybrid.position_weights, position_before)
    assert hybrid.variant_name == "hybrid_coin"


def test_hybrid_chain_starts_with_node_then_mixes_node_and_edge_links():
    config = EdgeConfig(problem_size=12, population_size=30)
    first = HybridChainCoin(config, seed=17)
    second = HybridChainCoin(config, seed=17)
    population = first.generate_population()
    assert np.array_equal(population, second.generate_population())
    assert_valid_population(population, 12)
    assert np.all(first.last_source_masks[:, 0])
    assert np.any(first.last_source_masks[:, 1:])
    assert np.any(~first.last_source_masks[:, 1:])


def test_problem_interface_runs_edge_coin_on_flowshop_without_int16_objectives():
    problem = FlowShopProblem(SMALL_3X2.processing_times, objectives=("total_flow_time",))
    model = ReferenceEdgeCoin(EdgeConfig(problem_size=3, population_size=8, objective="min"), seed=4)
    algorithm = PermutationCoinAlgorithm(model, problem)
    population, objectives = algorithm.step()
    assert_valid_population(population, 3)
    assert objectives.shape == (8, 1)
    assert objectives.dtype == np.float64
    assert algorithm.evaluations == 8
    assert algorithm.history[0].best == objectives[:, 0].min()


@pytest.mark.parametrize(
    "model_factory",
    [ReferenceEdgeCoin, OptimizedEdgeCoin, PositionCoin, CNBCoin, HybridCoin, HybridChainCoin, StartNodeEdgeCoin],
)
def test_every_variant_runs_behind_permutation_problem(model_factory):
    config = EdgeConfig(problem_size=3, population_size=8, objective="min")
    algorithm = PermutationCoinAlgorithm(
        model_factory(config, seed=12), FlowShopProblem(SMALL_3X2.processing_times)
    )
    algorithm.run(2)
    assert algorithm.generation == 2
    assert algorithm.evaluations == 16


def test_reusable_runner_rejects_multiobjective_and_dimension_mismatch():
    config = EdgeConfig(problem_size=3, population_size=4, objective="min")
    multi = FlowShopProblem(SMALL_3X2.processing_times, objectives=("makespan", "total_flow_time"))
    with pytest.raises(ValueError, match="exactly one objective"):
        PermutationCoinAlgorithm(PositionCoin(config), multi)
    with pytest.raises(ValueError, match="problem_size"):
        PermutationCoinAlgorithm(
            PositionCoin(EdgeConfig(problem_size=4)), FlowShopProblem(SMALL_3X2.processing_times)
        )


def test_start_node_statistics_update_only_first_position_and_all_edge_pairs():
    population = np.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])
    fitness = np.array([0, 1, 2, 3])
    model = StartNodeEdgeCoin(EdgeConfig(problem_size=4, population_size=4, objective="min"))
    (start_rewards, edge_rewards), _ = model.statistics(population, fitness)
    assert start_rewards.tolist() == [1, 0, 0, 0]
    assert start_rewards.sum() == 1
    assert edge_rewards.sum() == 4
    assert edge_rewards[0, 1] == 1
    assert edge_rewards[3, 0] == 1  # legacy Edge cycle closure


def test_start_node_sampling_is_independent_then_edges_complete_candidate():
    model = StartNodeEdgeCoin(EdgeConfig(problem_size=4, population_size=2), seed=3)
    model.start_node_weights[:] = [0, 0, 10, 0]
    model.edge_weights[:] = 0
    model.edge_weights[2, 1] = 10
    model.edge_weights[1, 3] = 10
    model.edge_weights[3, 0] = 10
    population = model.generate_population()
    assert population.tolist() == [[2, 1, 3, 0], [2, 1, 3, 0]]
    assert_valid_population(population, 4)


def test_start_node_reward_increases_and_punishment_decreases_first_job_weight():
    config = EdgeConfig(problem_size=4, population_size=4)
    model = StartNodeEdgeCoin(config)
    zero_edges = np.zeros((4, 4), dtype=np.int64)
    before = model.start_node_weights.copy()
    model.update((np.array([0, 1, 0, 0]), zero_edges), (np.zeros(4, dtype=np.int64), zero_edges))
    assert model.start_node_weights[1] > before[1]
    after_reward = model.start_node_weights.copy()
    model.update((np.zeros(4, dtype=np.int64), zero_edges), (np.array([0, 1, 0, 0]), zero_edges))
    assert model.start_node_weights[1] < after_reward[1]


@pytest.mark.parametrize("mode", ["conservative", "reconstruction"])
def test_start_node_edge_update_matches_edge_coin_for_same_population(mode):
    config = EdgeConfig(problem_size=4, population_size=4, objective="min", learning_mode=mode)
    population = np.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]])
    fitness = np.array([0, 1, 2, 3])
    edge = ReferenceEdgeCoin(config)
    start_edge = StartNodeEdgeCoin(config)
    edge_rewards, edge_punishments = edge.statistics(population, fitness)
    rewards, punishments = start_edge.statistics(population, fitness)
    edge.update(edge_rewards, edge_punishments)
    start_edge.update(rewards, punishments)
    assert np.array_equal(start_edge.edge_weights, edge.weights)


def test_start_node_reconstruction_keeps_models_separate():
    config = EdgeConfig(problem_size=4, learning_mode="reconstruction")
    model = StartNodeEdgeCoin(config)
    start_rewards = np.array([0, 2, 1, 0])
    edge_rewards = np.arange(16).reshape(4, 4)
    zeros = (np.zeros(4, dtype=np.int64), np.zeros((4, 4), dtype=np.int64))
    model.update((start_rewards, edge_rewards), zeros)
    assert model.start_node_weights.tolist() == [1, 21, 11, 1]
    expected = edge_rewards * 10 + 1
    np.fill_diagonal(expected, 0)
    assert np.array_equal(model.edge_weights, expected)
