import numpy as np
import pytest

from coin.core import MultiObjectiveCoinAlgorithm, PermutationCoinAlgorithm
from coin.models import EdgeConfig, OptimizedEdgeCoin
from coin.problems.tsp import TSPInstance, TSPProblem, get_tsp_instance, list_tsp_instances


def test_catalog_preloads_single_and_multiobjective_sizes_through_24():
    single = list_tsp_instances(multiobjective=False)
    multi = list_tsp_instances(multiobjective=True)
    assert [item.dimension for item in single] == [8, 12, 16, 20, 24]
    assert [item.dimension for item in multi] == [8, 12, 16, 20, 24]
    assert all(item.available_objective_names == ("distance",) for item in single)
    assert all(item.available_objective_names == ("distance", "operating_cost") for item in multi)


def test_tsp_closes_the_tour_and_batch_matches_reference():
    matrix = np.array([[0, 2, 9, 4], [2, 0, 3, 8], [9, 3, 0, 1], [4, 8, 1, 0]])
    problem = TSPProblem(TSPInstance("four", "Four", {"distance": matrix}))
    population = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    assert problem.evaluate(population[0]).tolist() == [10]
    assert np.array_equal(problem.evaluate_population(population), np.array([[10], [10]]))


def test_invalid_permutations_and_unknown_objectives_are_rejected():
    instance = get_tsp_instance("motsp-8")
    with pytest.raises(ValueError, match="unknown"):
        TSPProblem(instance, ("emissions",))
    with pytest.raises(ValueError, match="complete"):
        TSPProblem(instance).evaluate(np.zeros(8, dtype=int))


def test_single_and_multiobjective_coin_run_on_preloaded_tsp():
    single = TSPProblem(get_tsp_instance("tsp-8"))
    single_run = PermutationCoinAlgorithm(
        OptimizedEdgeCoin(EdgeConfig(problem_size=8, population_size=10, objective="min"), seed=1), single
    )
    population, objectives = single_run.step()
    assert population.shape == (10, 8) and objectives.shape == (10, 1)

    multi = TSPProblem(get_tsp_instance("motsp-8"))
    multi_run = MultiObjectiveCoinAlgorithm(
        OptimizedEdgeCoin(EdgeConfig(problem_size=8, population_size=10, objective="min"), seed=1), multi
    )
    population, objectives = multi_run.step()
    assert population.shape == (10, 8) and objectives.shape == (10, 2)
    assert len(multi_run.archive_population) >= 1
