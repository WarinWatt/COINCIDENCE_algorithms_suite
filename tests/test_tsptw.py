import numpy as np
import pytest
from pathlib import Path

from coin.problems.tsptw import (MatrixTSPTWInstance, TSPTWInstance, TSPTWProblem,
    evaluate_tsptw, generate_tsptw_instance, parse_lopez_ibanez_instance)
from coin.experiments import ExperimentConfiguration, ExperimentRunner


INSTANCE = TSPTWInstance(
    "line", np.array([[1, 0], [2, 0], [3, 0]], float),
    np.array([[0, 5], [4, 8], [0, 5]], float), np.array([1, 1, 1], float),
    np.array([0, 0], float),
)


def test_reference_schedule_reports_waiting_tardiness_and_return_to_depot():
    result = evaluate_tsptw(INSTANCE, np.array([0, 1, 2]), (
        "travel_cost", "makespan", "total_waiting_time", "total_tardiness", "late_customers"
    ))
    assert result.travel_cost == 6
    assert result.makespan == 10
    assert result.waiting.tolist() == [0, 1, 0]
    assert result.tardiness.tolist() == [0, 0, 1]
    assert result.objective_values.tolist() == [6, 10, 1, 1, 1]
    assert not result.feasible


def test_hard_problem_penalizes_but_soft_problem_reports_raw_objectives():
    permutation = np.array([0, 1, 2])
    soft = TSPTWProblem(INSTANCE, ("travel_cost", "total_tardiness"), hard_windows=False)
    hard = TSPTWProblem(INSTANCE, ("travel_cost", "total_tardiness"), hard_windows=True, penalty=100)
    assert soft.evaluate(permutation).tolist() == [6, 1]
    assert hard.evaluate(permutation).tolist() == [206, 201]


def test_generated_instance_has_a_valid_guided_feasible_tour_somewhere():
    instance = generate_tsptw_instance(20, 7)
    assert instance.coordinates.shape == (20, 2)
    assert instance.time_windows.shape == (20, 2)
    assert np.all(instance.time_windows[:, 0] <= instance.time_windows[:, 1])


def test_instance_and_permutation_validation():
    with pytest.raises(ValueError, match="permutation"):
        evaluate_tsptw(INSTANCE, np.array([0, 0, 2]))
    with pytest.raises(ValueError, match="time_windows"):
        TSPTWInstance("bad", np.zeros((2, 2)), np.array([[2, 1], [0, 1]]), np.zeros(2))


def test_tsptw_uses_the_reusable_experiment_runner():
    instance = generate_tsptw_instance(8, 3, window_width=120)
    config = ExperimentConfiguration(
        algorithms=("edge_coin",), objectives=("travel_cost",), population_size=6,
        evaluation_budget=12, maximum_generations=2, seeds=(1,),
    )
    result = ExperimentRunner().run(instance, config, hard_windows=False)
    assert result.objective_names == ("travel_cost",)
    assert result.reported_objective_names == TSPTWProblem(instance).available_objective_names
    assert result.runs[0].evaluations == 12


def test_tsptw_hbsa_wo_and_wt_variants_report_their_sampling_mode():
    instance = generate_tsptw_instance(8, 3, window_width=120)
    config = ExperimentConfiguration(
        algorithms=("ehbsa_wo", "ehbsa_wt", "nhbsa_wo", "nhbsa_wt"),
        objectives=("travel_cost",), population_size=6,
        evaluation_budget=12, maximum_generations=2, seeds=(1,),
    )
    runs = ExperimentRunner().run(instance, config, hard_windows=False).runs
    assert [run.metadata["sampling_mode"] for run in runs] == ["wo", "wt", "wo", "wt"]


def test_canonical_parser_matches_the_publishers_solution_checker_example():
    path = Path(__file__).parents[1] / "src/coin/problems/tsptw/benchmarks/SolomonPotvinBengio/rc_201.1.txt"
    instance = parse_lopez_ibanez_instance(path.read_text(), name=path.name)
    permutation = np.asarray([13,14,18,4,9,5,7,8,6,16,19,17,1,11,3,12,10,2,15]) - 1
    result = evaluate_tsptw(instance, permutation, ("travel_cost", "makespan"))
    assert isinstance(instance, MatrixTSPTWInstance)
    assert result.feasible
    assert result.travel_cost == pytest.approx(545.33, abs=.01)
    assert result.makespan == pytest.approx(592.06, abs=.01)


@pytest.mark.parametrize("hard_windows", [False, True])
def test_numba_population_evaluator_matches_reference_for_every_objective(hard_windows):
    instance = generate_tsptw_instance(12, 19, window_width=70)
    objectives = TSPTWProblem(instance).available_objective_names
    problem = TSPTWProblem(instance, objectives, hard_windows=hard_windows)
    rng = np.random.default_rng(4)
    population = np.vstack([rng.permutation(problem.dimension) for _ in range(25)])
    expected = np.vstack([problem.evaluate(candidate) for candidate in population])
    actual = problem.evaluate_population(population)
    assert np.allclose(actual, expected)


def test_numba_population_evaluator_matches_canonical_asymmetric_matrix():
    path = Path(__file__).parents[1] / "src/coin/problems/tsptw/benchmarks/SolomonPotvinBengio/rc_201.1.txt"
    instance = parse_lopez_ibanez_instance(path.read_text(), name=path.name)
    problem = TSPTWProblem(instance, ("travel_cost", "makespan", "total_tardiness"), hard_windows=False)
    rng = np.random.default_rng(8)
    population = np.vstack([rng.permutation(problem.dimension) for _ in range(20)])
    assert np.allclose(problem.evaluate_population(population),
                       np.vstack([problem.evaluate(candidate) for candidate in population]))
