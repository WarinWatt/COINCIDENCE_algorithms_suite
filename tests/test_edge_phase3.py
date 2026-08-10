import numpy as np
import pytest

from coin.models.edge import EdgeConfig, initial_edge_weights, prefer_edges, validate_weight_matrix
from coin.models.edge_optimized import OptimizedEdgeCoin
from coin.models.edge_reference import ReferenceEdgeCoin
from coin.problems.knights_tour import evaluate_population


IMPLEMENTATIONS = (ReferenceEdgeCoin, OptimizedEdgeCoin)


def _models(config, seed=42):
    return ReferenceEdgeCoin(config, seed=seed), OptimizedEdgeCoin(config, seed=seed)


def _simple_fitness(population):
    # Deterministic and deliberately produces ties to verify stable sorting.
    return (population[:, 0] * 3 + population[:, 1]).astype(np.int16)


def test_initialization_and_diagonal_match_delphi_orientation():
    config = EdgeConfig(problem_size=4, population_size=3, training_rate=5)
    weights = initial_edge_weights(config)
    assert weights.tolist() == [
        [0, 80, 80, 80],
        [80, 0, 80, 80],
        [80, 80, 0, 80],
        [80, 80, 80, 0],
    ]
    validate_weight_matrix(weights)


def test_preferred_edge_initialization_keeps_only_masked_edges_high():
    config = EdgeConfig(problem_size=4, training_rate=5)
    weights = initial_edge_weights(config)
    preferred = np.array(
        [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
        dtype=np.uint8,
    )
    prefer_edges(weights, preferred)
    assert np.all(np.diag(weights) == 0)
    assert np.all(weights[preferred.astype(bool)] == 80)
    off_diagonal = ~np.eye(4, dtype=bool)
    assert np.all(weights[~preferred.astype(bool) & off_diagonal] == 1)
    validate_weight_matrix(weights)


@pytest.mark.parametrize("model_type", IMPLEMENTATIONS)
def test_sampling_produces_complete_permutations(model_type):
    config = EdgeConfig(problem_size=16, population_size=100)
    population = model_type(config, seed=123).generate_population()
    expected = np.arange(16)
    assert population.shape == (100, 16)
    for candidate in population:
        assert np.array_equal(np.sort(candidate), expected)


def test_reference_and_optimized_sampling_are_exactly_equivalent():
    config = EdgeConfig(problem_size=12, population_size=40)
    reference, optimized = _models(config, seed=8675309)
    assert np.array_equal(reference.generate_population(), optimized.generate_population())
    assert np.array_equal(reference.generate_population(), optimized.generate_population())


@pytest.mark.parametrize("objective", ["min", "max"])
def test_delphi_reward_and_punishment_boundaries_and_counts(objective):
    # Already sorted by the supplied fitness. At 25%, loop=1. The top cohort
    # has one row while the inclusive Delphi bottom cohort has two rows.
    population = np.array(
        [[0, 1, 2, 3], [0, 2, 1, 3], [1, 0, 3, 2], [3, 2, 1, 0]],
        dtype=np.int16,
    )
    fitness = np.array([0, 1, 2, 3], dtype=np.int16)
    config = EdgeConfig(
        problem_size=4,
        population_size=4,
        reward_ratio=25,
        punishment_ratio=25,
        objective=objective,
    )
    reference, optimized = _models(config)
    ref_reward, ref_punishment = reference.statistics(population, fitness)
    opt_reward, opt_punishment = optimized.statistics(population, fitness)
    assert np.array_equal(ref_reward, opt_reward)
    assert np.array_equal(ref_punishment, opt_punishment)
    assert ref_reward.sum() == (4 if objective == "min" else 8)
    assert ref_punishment.sum() == (8 if objective == "min" else 4)
    # Cycle closure is counted: first candidate contains 3 -> 0.
    selected = ref_reward if objective == "min" else ref_punishment
    assert selected[3, 0] == 1


def test_conservative_update_matches_fixed_delphi_example_and_preserves_mass():
    config = EdgeConfig(problem_size=4, population_size=4)
    reference, optimized = _models(config)
    rewards = np.zeros((4, 4), dtype=np.int64)
    punishments = np.zeros((4, 4), dtype=np.int64)
    rewards[0, 1] = 1
    punishments[2, 3] = 1
    before_mass = reference.weights.sum(axis=1).copy()
    reference.update(rewards, punishments)
    optimized.update(rewards, punishments)
    assert np.array_equal(reference.weights, optimized.weights)
    assert reference.weights.tolist() == [
        [0, 82, 79, 79],
        [80, 0, 80, 80],
        [81, 81, 0, 78],
        [80, 80, 80, 0],
    ]
    assert np.array_equal(reference.weights.sum(axis=1), before_mass)
    validate_weight_matrix(reference.weights)


def test_repeated_conservative_updates_respect_legacy_positive_floor():
    config = EdgeConfig(problem_size=4, population_size=4)
    reference, optimized = _models(config)
    rewards = np.zeros((4, 4), dtype=np.int64)
    punishments = np.zeros((4, 4), dtype=np.int64)
    punishments[0, 1] = 100
    reference.update(rewards, punishments)
    optimized.update(rewards, punishments)
    assert np.array_equal(reference.weights, optimized.weights)
    # Each punishment has net -2 at n=4 and stops once the pre-update value
    # is no longer greater than n, so 80 reaches 4 and remains there.
    assert reference.weights[0, 1] == 4
    assert np.all(reference.weights[~np.eye(4, dtype=bool)] >= 1)
    assert np.all(np.diag(reference.weights) == 0)


def test_reconstruction_update_matches_ten_r_plus_one():
    config = EdgeConfig(problem_size=4, population_size=4, learning_mode="reconstruction")
    reference, optimized = _models(config)
    rewards = np.arange(16, dtype=np.int64).reshape(4, 4)
    punishments = np.full((4, 4), 99, dtype=np.int64)
    reference.update(rewards, punishments)
    optimized.update(rewards, punishments)
    expected = rewards * 10 + 1
    np.fill_diagonal(expected, 0)
    assert np.array_equal(reference.weights, expected)
    assert np.array_equal(optimized.weights, expected)


@pytest.mark.parametrize("mode", ["conservative", "reconstruction"])
def test_one_generation_is_equivalent(mode):
    config = EdgeConfig(
        problem_size=8,
        population_size=24,
        reward_ratio=25,
        punishment_ratio=25,
        learning_mode=mode,
    )
    reference, optimized = _models(config, seed=2025)
    ref_population, ref_fitness = reference.step(_simple_fitness)
    opt_population, opt_fitness = optimized.step(_simple_fitness)
    assert np.array_equal(ref_population, opt_population)
    assert np.array_equal(ref_fitness, opt_fitness)
    assert np.array_equal(reference.weights, optimized.weights)
    assert reference.history == optimized.history


def test_multiple_generations_and_history_are_equivalent():
    config = EdgeConfig(problem_size=10, population_size=30)
    reference, optimized = _models(config, seed=9)
    reference.run(5, _simple_fitness)
    optimized.run(5, _simple_fitness)
    assert reference.history == optimized.history
    assert np.array_equal(reference.weights, optimized.weights)
    assert [record.generation for record in reference.history] == [1, 2, 3, 4, 5]


def test_phase3_integrates_with_knight_evaluator_without_web_layer():
    config = EdgeConfig(problem_size=64, population_size=20, objective="max")
    reference, optimized = _models(config, seed=63)
    ref_population, ref_fitness = reference.step(evaluate_population)
    opt_population, opt_fitness = optimized.step(evaluate_population)
    assert np.array_equal(ref_population, opt_population)
    assert np.array_equal(ref_fitness, opt_fitness)
    assert np.array_equal(reference.weights, optimized.weights)


def test_run_stops_after_target_is_reached():
    config = EdgeConfig(problem_size=4, population_size=5, objective="max")
    for model_type in IMPLEMENTATIONS:
        model = model_type(config, seed=1)
        history = model.run(10, lambda population: np.full(5, 63), target_score=63)
        assert len(history) == 1
