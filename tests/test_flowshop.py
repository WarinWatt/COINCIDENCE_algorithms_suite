from __future__ import annotations

import numpy as np
import pytest

from coin.problems.base import PermutationProblem, evaluate_population
from coin.problems.flowshop import (
    SMALL_3X2,
    FlowShopInstance,
    FlowShopProblem,
    build_schedule,
    evaluate_flowshop,
    generate_random_instance,
)


ORDER = np.array([0, 1, 2], dtype=np.int64)


def test_known_small_instance_completion_and_start_matrices():
    schedule = build_schedule(SMALL_3X2.processing_times, ORDER)
    assert schedule.start_times.tolist() == [[0, 2], [2, 5], [3, 9]]
    assert schedule.completion_times.tolist() == [[2, 5], [3, 9], [6, 11]]
    assert schedule.completion_time_by_job.tolist() == [5, 9, 11]
    assert schedule.makespan == 11


def test_operation_records_use_the_same_schedule_calculation():
    operations = build_schedule(SMALL_3X2.processing_times, ORDER).operations()
    assert len(operations) == 6
    assert (operations[3].job, operations[3].machine, operations[3].start, operations[3].finish) == (1, 1, 5, 9)


def test_all_required_objectives_and_reporting_metrics():
    result = evaluate_flowshop(
        SMALL_3X2.processing_times,
        ORDER,
        objective_names=(
            "makespan", "total_flow_time", "total_tardiness",
            "maximum_tardiness", "total_machine_idle_time",
        ),
        due_dates=SMALL_3X2.due_dates,
    )
    assert result.objective_values.tolist() == [11, 25, 1, 1, 7]
    metrics = result.metrics
    assert metrics.average_flow_time == pytest.approx(25 / 3)
    assert metrics.average_tardiness == pytest.approx(1 / 3)
    assert metrics.idle_time_by_machine.tolist() == [5, 2]
    assert metrics.utilization_by_machine.tolist() == pytest.approx([6 / 11, 9 / 11])
    assert metrics.average_machine_utilization == pytest.approx(15 / 22)
    assert metrics.total_waiting_time == 10
    assert metrics.completion_time_by_job.tolist() == [5, 9, 11]


def test_completion_times_are_reported_by_job_not_sequence_position():
    order = np.array([2, 0, 1])
    result = evaluate_flowshop(SMALL_3X2.processing_times, order)
    assert result.metrics.completion_time_by_job.tolist() == [8, 12, 5]
    assert result.objective_values.shape == (1,)


def test_tardiness_is_optional_for_reporting_but_required_as_objective():
    result = evaluate_flowshop(SMALL_3X2.processing_times, ORDER)
    assert result.metrics.total_tardiness is None
    assert result.metrics.maximum_tardiness is None
    with pytest.raises(ValueError, match="due_dates are required"):
        evaluate_flowshop(
            SMALL_3X2.processing_times, ORDER, objective_names=("total_tardiness",)
        )


@pytest.mark.parametrize(
    "permutation,exception,message",
    [
        ([0, 1], ValueError, "exactly 3"),
        ([0, 1, 1], ValueError, "every job exactly once"),
        ([0, 1, 3], ValueError, "range 0..2"),
        ([0.0, 1.0, 2.0], TypeError, "integer job indices"),
        ([[0, 1, 2]], ValueError, "one-dimensional"),
    ],
)
def test_invalid_permutations(permutation, exception, message):
    with pytest.raises(exception, match=message):
        build_schedule(SMALL_3X2.processing_times, np.asarray(permutation))


@pytest.mark.parametrize(
    "matrix,exception,message",
    [
        ([1, 2, 3], ValueError, "two-dimensional"),
        ([[1, 2]], ValueError, "at least two jobs"),
        ([[], []], ValueError, "at least one machine"),
        ([[1, 0], [2, 3]], ValueError, "strictly positive"),
        ([[1, np.inf], [2, 3]], ValueError, "finite"),
        ([[True], [False]], TypeError, "numeric"),
    ],
)
def test_invalid_processing_matrices(matrix, exception, message):
    with pytest.raises(exception, match=message):
        FlowShopProblem(np.asarray(matrix))


@pytest.mark.parametrize("due_dates", ([1, 2], [1, 2, 3, 4], [1, np.nan, 3], [1, -1, 3]))
def test_invalid_due_dates(due_dates):
    with pytest.raises((TypeError, ValueError)):
        FlowShopProblem(SMALL_3X2.processing_times, due_dates=np.asarray(due_dates))


def test_problem_protocol_and_objective_order():
    problem = FlowShopProblem(
        SMALL_3X2.processing_times,
        objectives=("total_flow_time", "makespan"),
    )
    assert isinstance(problem, PermutationProblem)
    assert problem.dimension == 3
    assert problem.number_of_machines == 2
    assert problem.objective_names == ("total_flow_time", "makespan")
    assert problem.evaluate(ORDER).tolist() == [25, 11]
    assert "total_tardiness" not in problem.available_objective_names


def test_population_adapter_returns_population_by_objective_matrix():
    problem = FlowShopProblem(
        SMALL_3X2.processing_times,
        objectives=("makespan", "total_flow_time"),
    )
    result = evaluate_population(problem, np.array([[0, 1, 2], [2, 0, 1]]))
    assert result.shape == (2, 2)
    assert result.tolist() == [[11, 25], [12, 25]]


def test_fast_batch_matches_reference_for_every_objective_and_fractional_times():
    rng = np.random.default_rng(20260807)
    times = rng.uniform(0.25, 20.0, size=(12, 5))
    due_dates = rng.uniform(20.0, 180.0, size=12)
    objectives = (
        "makespan", "total_flow_time", "total_tardiness",
        "maximum_tardiness", "total_machine_idle_time",
    )
    problem = FlowShopProblem(times, objectives=objectives, due_dates=due_dates)
    population = np.asarray([rng.permutation(12) for _ in range(31)])
    fast = evaluate_population(problem, population)
    reference = np.vstack([problem.evaluate(permutation) for permutation in population])
    assert np.allclose(fast, reference, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("population", [
    [[0, 1, 1]], [[0, 1, 3]], [[0, -1, 2]],
])
def test_fast_batch_rejects_invalid_permutations(population):
    problem = FlowShopProblem(SMALL_3X2.processing_times)
    with pytest.raises(ValueError, match="every job exactly once"):
        evaluate_population(problem, np.asarray(population))


def test_empty_population_has_stable_objective_shape():
    problem = FlowShopProblem(SMALL_3X2.processing_times)
    result = evaluate_population(problem, np.empty((0, 3), dtype=np.int64))
    assert result.shape == (0, 1)


def test_objective_validation_rejects_empty_unknown_and_duplicate_names():
    with pytest.raises(ValueError, match="at least one"):
        FlowShopProblem(SMALL_3X2.processing_times, objectives=())
    with pytest.raises(ValueError, match="unknown"):
        FlowShopProblem(SMALL_3X2.processing_times, objectives=("energy",))
    with pytest.raises(ValueError, match="unique"):
        FlowShopProblem(SMALL_3X2.processing_times, objectives=("makespan", "makespan"))


def test_random_fixture_is_reproducible_and_bounded():
    first = generate_random_instance(20, 5, seed=2026, processing_time_low=3, processing_time_high=7)
    second = generate_random_instance(20, 5, seed=2026, processing_time_low=3, processing_time_high=7)
    assert np.array_equal(first.processing_times, second.processing_times)
    assert first.processing_times.shape == (20, 5)
    assert first.processing_times.min() >= 3
    assert first.processing_times.max() <= 7


def test_instance_loads_and_immutable_source_copy():
    source = np.array([[1, 2], [3, 4]])
    instance = FlowShopInstance("copy-check", source)
    source[0, 0] = 99
    assert instance.processing_times.tolist() == [[1, 2], [3, 4]]
    assert instance.load_by_job.tolist() == [3, 7]
    assert instance.load_by_machine.tolist() == [4, 6]
    with pytest.raises(ValueError):
        instance.processing_times[0, 0] = 8


def test_problem_owns_immutable_copies_without_freezing_caller_arrays():
    times = np.array([[1.0, 2.0], [3.0, 4.0]])
    due = np.array([5.0, 9.0])
    problem = FlowShopProblem(times, due_dates=due)
    times[0, 0] = 99
    due[0] = 99
    assert problem.processing_times.tolist() == [[1, 2], [3, 4]]
    assert problem.due_dates.tolist() == [5, 9]
    with pytest.raises(ValueError):
        problem.processing_times[0, 0] = 8


def test_fractional_processing_times_are_supported_without_truncation():
    times = np.array([[0.5, 1.25], [1.0, 0.5]])
    result = evaluate_flowshop(times, np.array([0, 1]))
    assert result.metrics.makespan == pytest.approx(2.25)
    assert result.metrics.total_flow_time == pytest.approx(4.0)
