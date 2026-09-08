import numpy as np
import pytest

from coin.problems.knights_tour import (
    LEGAL_MOVE,
    evaluate_knight_tour,
    evaluate_population,
    is_legal_knight_move,
)
from coin.termination import knight_tour_complete


# Complete tours generated once and retained as deterministic parity fixtures.
OPEN_TOUR = np.array([
    41, 56, 50, 40, 57, 51, 61, 55, 38, 23, 6, 12, 2, 8, 25, 10,
    0, 17, 32, 49, 59, 53, 63, 46, 31, 14, 4, 21, 15, 5, 11, 1,
    16, 26, 9, 24, 34, 19, 36, 42, 48, 58, 52, 62, 47, 30, 20, 3,
    13, 7, 22, 39, 29, 44, 54, 60, 45, 35, 18, 28, 43, 37, 27, 33,
], dtype=np.int16)

CLOSED_TOUR = np.array([
    46, 63, 53, 47, 62, 52, 58, 48, 33, 16, 1, 11, 5, 15, 30, 13,
    7, 22, 39, 54, 60, 50, 56, 41, 24, 9, 3, 18, 8, 2, 12, 6,
    23, 29, 14, 31, 37, 43, 26, 20, 35, 45, 28, 38, 55, 61, 51, 57,
    40, 25, 10, 0, 17, 32, 49, 59, 42, 27, 44, 34, 19, 4, 21, 36,
], dtype=np.int16)


def test_every_legal_knight_displacement_is_accepted():
    expected = {(1, 2), (2, 1)}
    observed = set()
    for prior in range(64):
        prior_row, prior_col = divmod(prior, 8)
        for following in range(64):
            next_row, next_col = divmod(following, 8)
            displacement = (abs(next_row - prior_row), abs(next_col - prior_col))
            assert is_legal_knight_move(prior, following) is (displacement in expected)
            if displacement in expected:
                observed.add((abs(next_row - prior_row), abs(next_col - prior_col)))
    assert observed == expected
    assert int(LEGAL_MOVE.sum()) == 336


@pytest.mark.parametrize("prior,following", [(0, 0), (0, 1), (0, 8), (0, 9), (0, 18), (-1, 6), (0, 64)])
def test_illegal_displacements_are_rejected(prior, following):
    assert not is_legal_knight_move(prior, following)


def test_open_complete_tour_has_63_path_moves_and_is_not_closed():
    result = evaluate_knight_tour(OPEN_TOUR, validate=True)
    assert result.score == 63
    assert result.valid_path_moves == 63
    assert result.visits_all_squares
    assert not result.is_closed


def test_closed_complete_tour_scores_64():
    result = evaluate_knight_tour(CLOSED_TOUR, validate=True)
    assert result.score == 64
    assert result.valid_path_moves == 63
    assert result.visits_all_squares
    assert result.is_closed


@pytest.mark.parametrize("score,expected", [(62, False), (63, True), (64, True)])
def test_termination_threshold(score, expected):
    assert knight_tour_complete(score) is expected


@pytest.mark.parametrize(
    "tour",
    [OPEN_TOUR[:-1], np.concatenate((OPEN_TOUR[:-1], OPEN_TOUR[:1])), np.arange(1, 65)],
)
def test_invalid_permutations_are_detected_in_validation_mode(tour):
    with pytest.raises(ValueError):
        evaluate_knight_tour(tour, validate=True)


def test_population_evaluator_matches_individual_results():
    population = np.stack((OPEN_TOUR, CLOSED_TOUR))
    assert evaluate_population(population, validate=True).tolist() == [63, 64]


def test_legal_closing_edge_alone_does_not_make_an_invalid_path_closed():
    candidate = np.arange(64, dtype=np.int16)
    # 63 -> 0 is not legal, so choose a permutation whose endpoints do form a
    # knight move while its interior remains mostly invalid.
    candidate[-1], candidate[10] = candidate[10], candidate[-1]
    result = evaluate_knight_tour(candidate, validate=True)
    assert result.visits_all_squares
    assert result.valid_path_moves < 63
    assert not result.is_closed


def test_closing_edge_cannot_turn_62_path_moves_into_a_false_open_tour():
    candidate = None
    for first in range(64):
        for second in range(first + 1, 64):
            trial = CLOSED_TOUR.copy()
            trial[first], trial[second] = trial[second], trial[first]
            path_moves = sum(
                int(LEGAL_MOVE[trial[index], trial[index + 1]])
                for index in range(63)
            )
            if path_moves == 62 and LEGAL_MOVE[trial[-1], trial[0]]:
                candidate = trial
                break
        if candidate is not None:
            break

    assert candidate is not None
    result = evaluate_knight_tour(candidate, validate=True)
    assert result.valid_path_moves == 62
    assert result.score == 62
    assert not result.is_closed
