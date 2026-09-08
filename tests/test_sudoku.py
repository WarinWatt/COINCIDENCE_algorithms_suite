import numpy as np

from coin.experiments.sudoku import run_comparison, run_block_comparison
from coin.problems.sudoku import FIXTURES, PRIMES, TARGET_PRODUCT, SudokuPermutationProblem, SudokuBlockPermutationProblem


def solution_permutation(problem: SudokuPermutationProblem, solution: str) -> np.ndarray:
    available: dict[int, list[int]] = {}
    for token, digit in enumerate(problem.token_values):
        available.setdefault(int(digit), []).append(token)
    result = []
    for position in problem.blank_positions:
        digit = int(solution[int(position)])
        result.append(available[digit].pop())
    return np.asarray(result, dtype=np.int16)


def test_known_completion_is_zero_and_every_prime_product_matches():
    fixture = FIXTURES[0]
    problem = SudokuPermutationProblem(fixture.puzzle)
    permutation = solution_permutation(problem, fixture.solution)
    grid = problem.decode(permutation)
    metrics = problem.prime_metrics(grid)
    assert metrics == {"penalty": 0, "invalid_groups": 0, "duplicate_excess": 0,
                       "squared_excess": 0, "worst_group": 0, "solved_groups": 27}
    assert np.all(problem.group_products(grid) == TARGET_PRODUCT)


def test_permutation_preserves_global_digit_multiplicity_and_clues():
    fixture = FIXTURES[0]
    problem = SudokuPermutationProblem(fixture.puzzle)
    candidate = np.arange(problem.dimension - 1, -1, -1)
    grid = problem.decode(candidate).ravel()
    assert np.array_equal(np.bincount(grid, minlength=10)[1:], np.full(9, 9))
    clues = problem.puzzle != 0
    assert np.array_equal(grid[clues], problem.puzzle[clues])


def test_every_imported_fixture_is_a_valid_global_permutation_problem():
    assert len(FIXTURES) == 24
    for fixture in FIXTURES:
        problem = SudokuPermutationProblem(fixture.puzzle)
        assert problem.dimension == fixture.puzzle.count("0")
        grid = problem.decode(np.arange(problem.dimension))
        assert np.array_equal(np.bincount(grid.ravel(), minlength=10)[1:], np.full(9, 9))


def test_all_four_algorithms_use_identical_exact_budget():
    problem = SudokuPermutationProblem(FIXTURES[0].puzzle)
    result = run_comparison(problem, ["coin", "ehbsa", "nhbsa", "ga_ox"], 10, 2, 1)
    assert result["evaluation_budget_per_algorithm"] == 20
    assert [item["algorithm"] for item in result["results"]] == ["coin", "ehbsa", "nhbsa", "ga_ox"]
    assert {item["evaluations"] for item in result["results"]} == {20}
    assert all(len(item["grid"]) == 81 for item in result["results"])


def test_paper_block_encoding_guarantees_all_nine_boxes_and_score_162_at_solution():
    fixture = FIXTURES[0]
    problem = SudokuBlockPermutationProblem(fixture.puzzle)
    solution = np.fromiter((int(x) for x in fixture.solution), dtype=np.int16)
    chromosome = np.concatenate([solution[positions] for positions, _ in problem.segments])
    grid = problem.decode(chromosome)
    assert all(np.prod(PRIMES[np.asarray(grid[r:r+3, c:c+3]).ravel() - 1]) == TARGET_PRODUCT
               for r in (0, 3, 6) for c in (0, 3, 6))
    assert problem.paper_metrics(grid)["paper_score"] == 162
    assert problem.evaluate(chromosome)[0] == 0


def test_new_multiobjective_evaluators_are_zero_at_solution_and_have_expected_dimensions():
    fixture = FIXTURES[0]
    problem = SudokuBlockPermutationProblem(fixture.puzzle, obvious_fill=False)
    solution = np.fromiter((int(x) for x in fixture.solution), dtype=np.int16)
    chromosome = np.concatenate([solution[positions] for positions, _ in problem.segments])
    population = np.asarray([chromosome])
    expected = {"mo_pareto": 2, "mo_bands_stacks": 6,
                "mo_block_responsibility": 9, "mo_digit_conflicts": 9,
                "mo_worst_total": 2, "mo_invalid_severity": 2}
    for method, width in expected.items():
        objectives = problem.evaluate_population_objectives(population, method)
        assert objectives.shape == (1, width)
        assert np.all(objectives == 0)


def test_every_new_mo_evaluator_runs_with_archive_and_exact_budget():
    puzzle = next(f.puzzle for f in FIXTURES if f.id == "missing-42")
    methods = ["mo_bands_stacks", "mo_block_responsibility", "mo_digit_conflicts",
               "mo_worst_total", "mo_invalid_severity"]
    for method in methods:
        result = run_block_comparison(SudokuBlockPermutationProblem(puzzle, obvious_fill=False),
                                      ["coin"], 10, 2, 3, evaluation_method=method)["results"][0]
        assert result["mo_coin"] is True
        assert result["evaluations"] == 20
        assert len(result["objective_names"]) == len(result["pareto_archive"][0]["objectives"])
        assert result["pareto_metrics"]["archive_size"] >= 1


def test_block_algorithms_preserve_boxes_and_exact_budget():
    problem = SudokuBlockPermutationProblem(next(f.puzzle for f in FIXTURES if f.id == "missing-42"))
    result = run_block_comparison(problem, ["coin", "ehbsa", "nhbsa", "ga_ox"], 10, 2, 1)
    assert {item["evaluations"] for item in result["results"]} == {20}
    for item in result["results"]:
        grid = np.asarray(item["grid"]).reshape(9, 9)
        assert all(len(np.unique(grid[r:r+3, c:c+3])) == 9 for r in (0, 3, 6) for c in (0, 3, 6))


def test_obvious_fill_solution_skips_stochastic_budget():
    problem = SudokuBlockPermutationProblem(next(f.puzzle for f in FIXTURES if f.id == "paper-01"))
    assert problem.dimension == 0
    result = run_block_comparison(problem, ["coin", "nhbsa"], 100, 500, 1)
    assert result["preprocessing_solved"] is True
    assert result["evaluation_budget_per_algorithm"] == 0
    assert all(item["solved"] and item["skipped"] and item["evaluations"] == 0
               for item in result["results"])


def test_disabling_obvious_fill_runs_the_requested_population_budget():
    puzzle = next(f.puzzle for f in FIXTURES if f.id == "paper-01")
    problem = SudokuBlockPermutationProblem(puzzle, obvious_fill=False)
    assert problem.dimension == puzzle.count("0")
    result = run_block_comparison(problem, ["coin"], 10, 2, 1)
    assert result["evaluation_budget_per_algorithm"] == 20
    assert result["results"][0]["evaluations"] == 20
    assert not result["results"][0].get("skipped", False)


def test_best_and_solution_times_are_reported_as_function_evaluations():
    puzzle = next(f.puzzle for f in FIXTURES if f.id == "paper-01")
    result = run_block_comparison(
        SudokuBlockPermutationProblem(puzzle, obvious_fill=False),
        ["coin"], 10, 3, 1,
    )["results"][0]
    assert 1 <= result["best_found_at_evaluation"] <= result["evaluations"]
    assert result["best_found_at_generation"] == (
        (result["best_found_at_evaluation"] - 1) // 10 + 1
    )
    if result["solved"]:
        assert 1 <= result["solution_found_at_evaluation"] <= result["evaluations"]


def test_optimal_solution_stops_before_the_requested_budget_by_default():
    solved_grid = next(f.solution for f in FIXTURES if f.id == "missing-12")
    puzzle = "0" + solved_grid[1:]
    problem = SudokuBlockPermutationProblem(puzzle, obvious_fill=False)
    stopped = run_block_comparison(problem, ["coin"], 20, 100, 4)["results"][0]
    assert stopped["solved"]
    assert stopped["termination_reason"] == "optimal_solution"
    assert stopped["evaluations"] == 20
    assert stopped["solution_found_at_evaluation"] == 1
    continued = run_block_comparison(problem, ["coin"], 20, 3, 4,
                                     stop_at_solution=False)["results"][0]
    assert continued["evaluations"] == 60
    assert continued["termination_reason"] == "evaluation_budget"


def test_numba_population_evaluator_equals_readable_reference():
    puzzle = next(f.puzzle for f in FIXTURES if f.id == "paper-106")
    problem = SudokuBlockPermutationProblem(puzzle, obvious_fill=False)
    rng = np.random.default_rng(2026)
    population = np.asarray([problem.random_candidate(rng) for _ in range(64)], dtype=np.int16)
    assert np.array_equal(problem.evaluate_population_metrics(population),
                          problem.evaluate_population_reference(population))


def test_ehbsa_and_nhbsa_support_template_and_no_template_sampling():
    puzzle = next(f.puzzle for f in FIXTURES if f.id == "paper-106")
    problem = SudokuBlockPermutationProblem(puzzle, obvious_fill=False)
    for name in ("ehbsa", "nhbsa"):
        for mode in ("wt", "wo"):
            result = run_block_comparison(
                problem, [name], 10, 3, 1,
                {name: {"sampling_mode": mode, "template_sample_ratio": 50}},
            )["results"][0]
            assert result["evaluations"] == 30
            assert result["sampling_mode"] == mode
            grid = np.asarray(result["grid"]).reshape(9, 9)
            assert all(len(np.unique(grid[r:r+3, c:c+3])) == 9
                       for r in (0, 3, 6) for c in (0, 3, 6))


def test_mo_pareto_guidance_reports_row_column_objectives_and_depths():
    puzzle = next(f.puzzle for f in FIXTURES if f.id == "paper-106")
    problem = SudokuBlockPermutationProblem(puzzle, obvious_fill=False)
    result = run_block_comparison(problem, ["coin"], 12, 3, 1,
                                  evaluation_method="mo_pareto")["results"][0]
    assert result["evaluations"] == 36
    assert result["evaluation_method"] == "mo_pareto"
    assert result["metrics"]["row_duplicate_excess"] >= 0
    assert result["metrics"]["column_duplicate_excess"] >= 0
    assert all(point["nondominated_count"] >= 1 and point["pareto_depth_count"] >= 1
               for point in result["history"])


def test_mo_coin_preserves_a_unique_external_nondominated_archive():
    puzzle = next(f.puzzle for f in FIXTURES if f.id == "missing-42")
    problem = SudokuBlockPermutationProblem(puzzle, obvious_fill=False)
    result = run_block_comparison(problem, ["coin"], 20, 4, 7,
                                  evaluation_method="mo_pareto")["results"][0]
    archive = result["pareto_archive"]
    objectives = np.asarray([item["objectives"] for item in archive])
    assert result["mo_coin"] is True
    assert result["objective_names"] == ["row_duplicate_excess", "column_duplicate_excess"]
    assert len(archive) == result["pareto_metrics"]["archive_size"] >= 1
    assert len({tuple(item["grid"]) for item in archive}) == len(archive)
    for i in range(len(objectives)):
        assert not any(np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i])
                       for j in range(len(objectives)) if j != i)
    assert all(point["archive_size"] >= 1 for point in result["history"])


def test_mo_coin_is_deterministic_for_a_fixed_seed():
    puzzle = next(f.puzzle for f in FIXTURES if f.id == "missing-42")
    first = run_block_comparison(SudokuBlockPermutationProblem(puzzle, obvious_fill=False),
                                 ["coin"], 16, 3, 11, evaluation_method="mo_pareto")["results"][0]
    second = run_block_comparison(SudokuBlockPermutationProblem(puzzle, obvious_fill=False),
                                  ["coin"], 16, 3, 11, evaluation_method="mo_pareto")["results"][0]
    assert first["pareto_archive"] == second["pareto_archive"]
    assert first["history"] == second["history"]

