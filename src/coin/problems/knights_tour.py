"""Knight's Tour evaluation matching ``KnightTourEvaluateButtonClick``.

The Python core uses zero-based square identifiers. The Delphi evaluator uses
one-based identifiers, but its coordinate calculation produces the same board
adjacency after subtracting one from every identifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numba import njit

BOARD_SIZE = 8
SQUARE_COUNT = BOARD_SIZE * BOARD_SIZE


def _build_legal_move_table() -> np.ndarray:
    table = np.zeros((SQUARE_COUNT, SQUARE_COUNT), dtype=np.uint8)
    for prior in range(SQUARE_COUNT):
        prior_row, prior_col = divmod(prior, BOARD_SIZE)
        for following in range(SQUARE_COUNT):
            next_row, next_col = divmod(following, BOARD_SIZE)
            row_delta = abs(next_row - prior_row)
            col_delta = abs(next_col - prior_col)
            table[prior, following] = (
                (row_delta == 1 and col_delta == 2)
                or (row_delta == 2 and col_delta == 1)
            )
    table.setflags(write=False)
    return table


LEGAL_MOVE = _build_legal_move_table()


@dataclass(frozen=True, slots=True)
class KnightTourEvaluation:
    score: int
    valid_path_moves: int
    visits_all_squares: bool
    is_closed: bool


def _as_tour_array(tour: Sequence[int] | np.ndarray) -> np.ndarray:
    values = np.asarray(tour)
    if values.ndim != 1:
        raise ValueError("a Knight's Tour must be a one-dimensional sequence")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("square identifiers must be integers")
    return np.ascontiguousarray(values, dtype=np.int16)


def _is_complete_permutation(tour: np.ndarray) -> bool:
    if tour.size != SQUARE_COUNT:
        return False
    seen = np.zeros(SQUARE_COUNT, dtype=np.uint8)
    for square in tour:
        value = int(square)
        if value < 0 or value >= SQUARE_COUNT or seen[value]:
            return False
        seen[value] = 1
    return True


def _validate_complete_permutation(tour: np.ndarray) -> None:
    if tour.size != SQUARE_COUNT:
        raise ValueError("validation mode requires exactly 64 square identifiers")
    if not _is_complete_permutation(tour):
        raise ValueError("tour must be a permutation of square identifiers 0..63")


@njit(cache=False)
def _evaluate_one(tour: np.ndarray, legal_move: np.ndarray) -> tuple[int, int]:
    valid_path_moves = 0
    for index in range(tour.size - 1):
        valid_path_moves += legal_move[tour[index], tour[index + 1]]

    # A closing edge is a bonus only after every one of the 63 path edges is
    # legal.  Otherwise a broken path with 62 legal edges plus a legal
    # endpoint-to-start edge would be a false score 63.
    closing_move = 0
    if tour.size == SQUARE_COUNT and valid_path_moves == SQUARE_COUNT - 1:
        closing_move = legal_move[tour[-1], tour[0]]
    return valid_path_moves + closing_move, valid_path_moves


@njit(cache=False)
def _evaluate_population_kernel(
    population: np.ndarray, legal_move: np.ndarray
) -> np.ndarray:
    fitness = np.empty(population.shape[0], dtype=np.int16)
    for candidate_index in range(population.shape[0]):
        score = 0
        for position in range(population.shape[1] - 1):
            score += legal_move[
                population[candidate_index, position],
                population[candidate_index, position + 1],
            ]
        if (
            population.shape[1] == SQUARE_COUNT
            and score == SQUARE_COUNT - 1
        ):
            score += legal_move[
                population[candidate_index, -1], population[candidate_index, 0]
            ]
        fitness[candidate_index] = score
    return fitness


def is_legal_knight_move(prior: int, following: int) -> bool:
    """Return whether two zero-based square identifiers form a knight move."""
    if not 0 <= prior < SQUARE_COUNT or not 0 <= following < SQUARE_COUNT:
        return False
    return bool(LEGAL_MOVE[prior, following])


def evaluate_knight_tour(
    tour: Sequence[int] | np.ndarray, *, validate: bool = False
) -> KnightTourEvaluation:
    """Evaluate path moves and the Delphi-compatible closing transition.

    Validation is optional because the Delphi evaluator itself does not verify
    uniqueness. Rich completeness flags are always conservative: a tour is
    complete only when it is a full permutation of all 64 squares.
    """
    values = _as_tour_array(tour)
    if validate:
        _validate_complete_permutation(values)
    elif values.size and (values.min() < 0 or values.max() >= SQUARE_COUNT):
        raise ValueError("square identifiers must be in the range 0..63")

    score, valid_path_moves = _evaluate_one(values, LEGAL_MOVE)
    visits_all_squares = _is_complete_permutation(values)
    is_closed = bool(
        visits_all_squares
        and valid_path_moves == SQUARE_COUNT - 1
        and values.size
        and LEGAL_MOVE[values[-1], values[0]]
    )
    return KnightTourEvaluation(
        score=int(score),
        valid_path_moves=int(valid_path_moves),
        visits_all_squares=visits_all_squares,
        is_closed=is_closed,
    )


def evaluate_population(
    population: np.ndarray, *, validate: bool = False
) -> np.ndarray:
    """Evaluate a contiguous population using the compiled lookup-table loop."""
    values = np.asarray(population)
    if values.ndim != 2:
        raise ValueError("population must be a two-dimensional array")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("square identifiers must be integers")
    values = np.ascontiguousarray(values, dtype=np.int16)
    if values.size and (values.min() < 0 or values.max() >= SQUARE_COUNT):
        raise ValueError("square identifiers must be in the range 0..63")
    if validate:
        for candidate in values:
            _validate_complete_permutation(candidate)
    return _evaluate_population_kernel(values, LEGAL_MOVE)
