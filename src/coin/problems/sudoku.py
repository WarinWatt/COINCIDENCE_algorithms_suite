"""Sudoku as a global missing-token permutation optimization problem."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit

PRIMES = np.asarray([2, 3, 5, 7, 11, 13, 17, 19, 23], dtype=np.int64)
TARGET_PRODUCT = int(np.prod(PRIMES))


@njit(cache=False)
def evaluate_block_population_numba(population: np.ndarray, base_grid: np.ndarray,
                                    chromosome_positions: np.ndarray) -> np.ndarray:
    """Evaluate all block chromosomes without Python candidate/group loops.

    Columns: paper penalty, invalid row/column groups, squared excess,
    worst group, first-web-version hierarchical penalty, row duplicate
    distance, and column duplicate distance.
    """
    count = population.shape[0]
    results = np.empty((count, 7), dtype=np.int64)
    for candidate_index in range(count):
        grid = base_grid.copy()
        for position_index in range(chromosome_positions.shape[0]):
            grid[chromosome_positions[position_index]] = population[candidate_index, position_index]
        duplicate_excess = 0
        invalid_groups = 0
        squared_excess = 0
        worst_group = 0
        row_duplicates = 0
        column_duplicates = 0
        for group_index in range(18):
            counts = np.zeros(10, dtype=np.int16)
            if group_index < 9:
                row = group_index
                for column in range(9):
                    counts[grid[row * 9 + column]] += 1
            else:
                column = group_index - 9
                for row in range(9):
                    counts[grid[row * 9 + column]] += 1
            group_duplicates = 0
            group_squared = 0
            for digit in range(1, 10):
                excess = counts[digit] - 1
                if excess > 0:
                    group_duplicates += excess
                    group_squared += excess * excess
            if group_duplicates > 0:
                invalid_groups += 1
            duplicate_excess += group_duplicates
            if group_index < 9:
                row_duplicates += group_duplicates
            else:
                column_duplicates += group_duplicates
            squared_excess += group_squared
            if group_duplicates > worst_group:
                worst_group = group_duplicates
        legacy = (duplicate_excess * 100_000 + invalid_groups * 1_000
                  + squared_excess * 10 + worst_group)
        results[candidate_index, 0] = duplicate_excess
        results[candidate_index, 1] = invalid_groups
        results[candidate_index, 2] = squared_excess
        results[candidate_index, 3] = worst_group
        results[candidate_index, 4] = legacy
        results[candidate_index, 5] = row_duplicates
        results[candidate_index, 6] = column_duplicates
    return results


@njit(cache=False)
def evaluate_block_mo_numba(population: np.ndarray, base_grid: np.ndarray,
                            chromosome_positions: np.ndarray, method_code: int) -> np.ndarray:
    """Evaluate decomposed Sudoku objectives (all minimized).

    1: three horizontal bands + three vertical stacks
    2: cross-block conflict responsibility for each of nine blocks
    3: row/column duplicate responsibility for each digit
    4: worst group + total duplicate excess
    5: invalid group count + squared duplicate severity
    """
    width = 6 if method_code == 1 else (9 if method_code in (2, 3) else 2)
    results = np.zeros((population.shape[0], width), dtype=np.int64)
    for candidate_index in range(population.shape[0]):
        grid = base_grid.copy()
        for position_index in range(chromosome_positions.shape[0]):
            grid[chromosome_positions[position_index]] = population[candidate_index, position_index]
        if method_code == 2:
            for first in range(81):
                r1, c1 = first // 9, first % 9
                b1 = (r1 // 3) * 3 + c1 // 3
                for second in range(first + 1, 81):
                    r2, c2 = second // 9, second % 9
                    b2 = (r2 // 3) * 3 + c2 // 3
                    if b1 != b2 and grid[first] == grid[second] and (r1 == r2 or c1 == c2):
                        results[candidate_index, b1] += 1
                        results[candidate_index, b2] += 1
            continue
        total, invalid, squared, worst = 0, 0, 0, 0
        for group_index in range(18):
            counts = np.zeros(10, dtype=np.int16)
            if group_index < 9:
                for column in range(9):
                    counts[grid[group_index * 9 + column]] += 1
            else:
                column = group_index - 9
                for row in range(9):
                    counts[grid[row * 9 + column]] += 1
            group_excess = 0
            for digit in range(1, 10):
                excess = counts[digit] - 1
                if excess > 0:
                    group_excess += excess
                    squared += excess * excess
                    if method_code == 3:
                        results[candidate_index, digit - 1] += excess
            if group_excess > 0:
                invalid += 1
            total += group_excess
            if group_excess > worst:
                worst = group_excess
            if method_code == 1:
                if group_index < 9:
                    results[candidate_index, group_index // 3] += group_excess
                else:
                    results[candidate_index, 3 + (group_index - 9) // 3] += group_excess
        if method_code == 4:
            results[candidate_index, 0], results[candidate_index, 1] = worst, total
        elif method_code == 5:
            results[candidate_index, 0], results[candidate_index, 1] = invalid, squared
    return results


@dataclass(frozen=True, slots=True)
class SudokuFixture:
    id: str
    name: str
    puzzle: str
    solution: str
    note: str
    source: str = "Controlled fixture"
    rating: float | None = None


_SOLUTION = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"


def _masked_solution(indices: tuple[int, ...]) -> str:
    return "".join("0" if index in indices else value for index, value in enumerate(_SOLUTION))


FIXTURES: tuple[SudokuFixture, ...] = (
    SudokuFixture("missing-12", "Controlled · 12 missing", _masked_solution((0, 4, 8, 10, 18, 22, 30, 38, 46, 54, 62, 70)), _SOLUTION,
                  "A deterministic clue-mask sample from a verified solution; an introductory convergence fixture."),
    SudokuFixture(
        "missing-18", "Permutation warm-up · 18 missing",
        _masked_solution((0, 4, 8, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 62, 70, 78)),
        _SOLUTION,
        "A compact 18-token instance for observing learning and exact-budget comparisons.",
    ),
    SudokuFixture("missing-36", "Controlled · 36 missing", _masked_solution(tuple(range(0, 72, 2))), _SOLUTION,
                  "Every second cell is hidden from a verified completion, creating a reproducible 36-variable fixture."),
    SudokuFixture("missing-42", "Controlled · 42 missing", _masked_solution(tuple(range(42))), _SOLUTION,
                  "The first 42 cells are hidden; intentionally structured to expose the effect of clue placement."),
    SudokuFixture("paper-01", "Paper No. 1 · Easy · 38 givens", "009000100217000368000207000064103580070000030150428079000589000485000293006302800", "",
                  "Benchmark No. 1 reproduced from Fig. 1 of the original NB-COIN paper.", "Waiyapara et al. Sudoku NB-COIN paper"),
    SudokuFixture("paper-11", "Paper No. 11 · Easy · 34 givens", "290701000530060100100300040000590004015004689000180003002600090360040700940805000", "",
                  "Benchmark No. 11 reproduced from Fig. 8 of the original paper.", "Waiyapara et al. Sudoku NB-COIN paper"),
    SudokuFixture("paper-27", "Paper No. 27 · Medium · 30 givens", "010506020300000006009104500090005003070302050030009060000240190900000002050601080", "",
                  "Benchmark No. 27 reproduced from Fig. 8 of the original paper.", "Waiyapara et al. Sudoku NB-COIN paper"),
    SudokuFixture("paper-29", "Paper No. 29 · Medium · 29 givens", "040200000008064700060050000806002349009000000304008172030070000000801560002030000", "",
                  "Benchmark No. 29 reproduced from Fig. 8 of the original paper.", "Waiyapara et al. Sudoku NB-COIN paper"),
    SudokuFixture("paper-77", "Paper No. 77 · Difficult · 28 givens", "500000009904030006300907005000200000090010008038000940400000002003509600002401300", "",
                  "Benchmark No. 77 reproduced from Fig. 8 of the original paper.", "Waiyapara et al. Sudoku NB-COIN paper"),
    SudokuFixture("paper-106", "Paper No. 106 · Difficult · 24 givens", "000407000001000700400000003020309040040010090000500800500000008084060530300000002", "",
                  "Benchmark No. 106 reproduced from Fig. 8 of the original paper.", "Waiyapara et al. Sudoku NB-COIN paper"),
    SudokuFixture(
        "missing-24", "Intermediate · 24 missing",
        _masked_solution((0, 3, 6, 8, 10, 13, 16, 18, 21, 24, 26, 28, 31, 34, 36, 40, 44, 46, 49, 52, 54, 60, 68, 76)),
        _SOLUTION,
        "A larger permutation with the same known completion; useful after the warm-up.",
    ),
    SudokuFixture(
        "missing-30", "Challenge · 30 missing",
        _masked_solution((0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 62, 70)),
        _SOLUTION,
        "A deliberately larger search space; stochastic solvers may require more evaluations.",
    ),
    SudokuFixture("exchange-easy-0000183b305c", "Exchange Easy · 1.2 · #0000183b305c",
        "050703060007000800000816000000030000005000100730040086906000204840572093000409000", "",
        "Public-domain Sudoku Exchange puzzle; QQWing unique solution, Sukaku Explainer rating 1.2.", "Sudoku Exchange Puzzle Bank", 1.2),
    SudokuFixture("exchange-easy-0001d5d6314e", "Exchange Easy · 1.2 · #0001d5d6314e",
        "302401809001000300000000000040708010780502036000090000200609003900000008800070005", "",
        "Public-domain Sudoku Exchange puzzle; QQWing unique solution, Sukaku Explainer rating 1.2.", "Sudoku Exchange Puzzle Bank", 1.2),
    SudokuFixture("exchange-easy-000212406270", "Exchange Easy · 1.2 · #000212406270",
        "000823001003000400070000052300960010000102000010038006830000040002000900600789000", "",
        "Public-domain Sudoku Exchange puzzle; QQWing unique solution, Sukaku Explainer rating 1.2.", "Sudoku Exchange Puzzle Bank", 1.2),
    SudokuFixture("exchange-medium-0000847b216e", "Exchange Medium · 2.3 · #0000847b216e",
        "020900000048000031000063020009407003003080200400105600030570000250000180000006050", "",
        "Public-domain Sudoku Exchange puzzle; Sukaku Explainer rating 2.3.", "Sudoku Exchange Puzzle Bank", 2.3),
    SudokuFixture("exchange-medium-0000c45fb232", "Exchange Medium · 2.0 · #0000c45fb232",
        "100800570000009210090040000300900050007000300020006008000020040071400000064007003", "",
        "Public-domain Sudoku Exchange puzzle; Sukaku Explainer rating 2.0.", "Sudoku Exchange Puzzle Bank", 2.0),
    SudokuFixture("exchange-medium-0000fc7c6b96", "Exchange Medium · 2.3 · #0000fc7c6b96",
        "002000800005020100460000029130060052009080400000302000006070200700000008020519070", "",
        "Public-domain Sudoku Exchange puzzle; Sukaku Explainer rating 2.3.", "Sudoku Exchange Puzzle Bank", 2.3),
    SudokuFixture("exchange-hard-00005f662e09", "Exchange Hard · 3.4 · #00005f662e09",
        "080200400570000100002300000820090005000715000700020041000006700003000018007009050", "",
        "Public-domain Sudoku Exchange puzzle; Sukaku Explainer rating 3.4.", "Sudoku Exchange Puzzle Bank", 3.4),
    SudokuFixture("exchange-hard-00009e4900fa", "Exchange Hard · 2.6 · #00009e4900fa",
        "600050007030000000080409200015300000008000300000007590009501030000000080200070004", "",
        "Public-domain Sudoku Exchange puzzle; Sukaku Explainer rating 2.6.", "Sudoku Exchange Puzzle Bank", 2.6),
    SudokuFixture("exchange-hard-0000b2c3fc62", "Exchange Hard · 4.3 · #0000b2c3fc62",
        "210950004090060037000700000000000308920000015805000000000002000680010040100047096", "",
        "Public-domain Sudoku Exchange puzzle; Sukaku Explainer rating 4.3.", "Sudoku Exchange Puzzle Bank", 4.3),
    SudokuFixture("exchange-diabolical-00015097c6c3", "Exchange Diabolical · 7.2 · #00015097c6c3",
        "083020090000800100029300008000098700070000060006740000300006980002005000010030540", "",
        "Public-domain Sudoku Exchange puzzle; Sukaku Explainer rating 7.2.", "Sudoku Exchange Puzzle Bank", 7.2),
    SudokuFixture("exchange-diabolical-0001d2888928", "Exchange Diabolical · 7.1 · #0001d2888928",
        "200050006010000090600801003007090600000703000900080002100000005060902010003060200", "",
        "Public-domain Sudoku Exchange puzzle; Sukaku Explainer rating 7.1.", "Sudoku Exchange Puzzle Bank", 7.1),
    SudokuFixture("exchange-diabolical-000274921f39", "Exchange Diabolical · 7.1 · #000274921f39",
        "590000007040010083008034900001402000069000820000109300004670200980040030700000016", "",
        "Public-domain Sudoku Exchange puzzle; Sukaku Explainer rating 7.1.", "Sudoku Exchange Puzzle Bank", 7.1),
)


class SudokuPermutationProblem:
    """Decode a permutation of unique missing tokens into their digit values.

    Tokens are unique to satisfy the reusable permutation interface. Multiple
    tokens may decode to the same digit; this exposes genotype symmetry while
    preserving the required global digit multiplicities by construction.
    """

    objective_names = ("hierarchical_prime_penalty",)

    def __init__(self, puzzle: str):
        normalized = "".join(ch for ch in puzzle if ch in "0123456789.").replace(".", "0")
        if len(normalized) != 81:
            raise ValueError("Sudoku puzzle must contain exactly 81 cells")
        self.puzzle = np.fromiter((int(ch) for ch in normalized), dtype=np.int16)
        self.blank_positions = np.flatnonzero(self.puzzle == 0).astype(np.int16)
        counts = np.bincount(self.puzzle, minlength=10)
        values: list[int] = []
        for digit in range(1, 10):
            missing = 9 - int(counts[digit])
            if missing < 0:
                raise ValueError(f"digit {digit} occurs more than nine times")
            values.extend([digit] * missing)
        if len(values) != len(self.blank_positions):
            raise ValueError("clues do not have valid global Sudoku digit multiplicities")
        self.token_values = np.asarray(values, dtype=np.int16)

    @property
    def dimension(self) -> int:
        return len(self.blank_positions)

    def validate(self, permutation: np.ndarray) -> None:
        candidate = np.asarray(permutation)
        if candidate.shape != (self.dimension,):
            raise ValueError("permutation has the wrong length")
        if not np.array_equal(np.sort(candidate), np.arange(self.dimension)):
            raise ValueError("candidate must be a zero-based permutation")

    def decode(self, permutation: np.ndarray) -> np.ndarray:
        self.validate(permutation)
        grid = self.puzzle.copy()
        grid[self.blank_positions] = self.token_values[np.asarray(permutation, dtype=int)]
        return grid.reshape(9, 9)

    @staticmethod
    def group_products(grid: np.ndarray) -> np.ndarray:
        groups = [grid[row, :] for row in range(9)]
        groups.extend(grid[:, column] for column in range(9))
        groups.extend(grid[box_row:box_row + 3, box_col:box_col + 3].ravel()
                      for box_row in (0, 3, 6) for box_col in (0, 3, 6))
        return np.asarray([np.prod(PRIMES[np.asarray(group, dtype=int) - 1]) for group in groups], dtype=np.int64)

    @staticmethod
    def prime_metrics(grid: np.ndarray) -> dict[str, int]:
        groups = [grid[row, :] for row in range(9)]
        groups.extend(grid[:, column] for column in range(9))
        groups.extend(grid[box_row:box_row + 3, box_col:box_col + 3].ravel()
                      for box_row in (0, 3, 6) for box_col in (0, 3, 6))
        products = SudokuPermutationProblem.group_products(grid)
        invalid = int(np.count_nonzero(products != TARGET_PRODUCT))
        duplicate_excess = 0
        squared_excess = 0
        worst_group = 0
        row_duplicate_excess = 0
        column_duplicate_excess = 0
        for group_index, group in enumerate(groups):
            counts = np.bincount(np.asarray(group, dtype=int), minlength=10)[1:]
            excess = np.maximum(counts - 1, 0)
            group_duplicates = int(excess.sum())
            duplicate_excess += group_duplicates
            squared_excess += int(np.square(excess).sum())
            worst_group = max(worst_group, group_duplicates)
        penalty = duplicate_excess * 100_000 + invalid * 1_000 + squared_excess * 10 + worst_group
        return {"penalty": penalty, "invalid_groups": invalid,
                "duplicate_excess": duplicate_excess, "squared_excess": squared_excess,
                "worst_group": worst_group, "solved_groups": 27 - invalid}

    def evaluate(self, permutation: np.ndarray) -> np.ndarray:
        return np.asarray([self.prime_metrics(self.decode(permutation))["penalty"]], dtype=np.float64)

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        return np.asarray([self.evaluate(candidate) for candidate in population], dtype=np.float64)


class SudokuBlockPermutationProblem:
    """Paper-faithful Sudoku encoding: one constrained permutation per 3x3 block.

    A chromosome is the concatenation of the still-missing digits in each block.
    Consequently every decoded candidate has nine valid 3x3 blocks by
    construction; optimization only has to remove duplicates from rows/columns.
    """

    objective_names = ("paper_penalty",)

    def __init__(self, puzzle: str, *, obvious_fill: bool = True):
        normalized = "".join(ch for ch in puzzle if ch in "0123456789.").replace(".", "0")
        if len(normalized) != 81:
            raise ValueError("Sudoku puzzle must contain exactly 81 cells")
        self.original_puzzle = np.fromiter((int(ch) for ch in normalized), dtype=np.int16).reshape(9, 9)
        self.obvious_fill_enabled = bool(obvious_fill)
        self.puzzle = (self._fill_obvious_singles(self.original_puzzle)
                       if self.obvious_fill_enabled else self.original_puzzle.copy())
        self.inferred_count = int(np.count_nonzero(self.puzzle) - np.count_nonzero(self.original_puzzle))
        self.segments: list[tuple[np.ndarray, np.ndarray]] = []
        for br in (0, 3, 6):
            for bc in (0, 3, 6):
                positions, fixed = [], []
                for dr in range(3):
                    for dc in range(3):
                        value = int(self.puzzle[br + dr, bc + dc])
                        if value:
                            if value in fixed:
                                raise ValueError("a 3x3 block contains duplicate clues")
                            fixed.append(value)
                        else:
                            positions.append((br + dr) * 9 + bc + dc)
                digits = [d for d in range(1, 10) if d not in fixed]
                if len(digits) != len(positions):
                    raise ValueError("invalid clues in a 3x3 block")
                self.segments.append((np.asarray(positions, dtype=np.int16), np.asarray(digits, dtype=np.int16)))
        self.offsets = np.cumsum([0] + [len(p) for p, _ in self.segments]).astype(np.int16)
        self.chromosome_positions = np.concatenate([positions for positions, _ in self.segments]).astype(np.int16)
        self._flat_puzzle = self.puzzle.ravel().astype(np.int16)

    @staticmethod
    def _fill_obvious_singles(puzzle: np.ndarray) -> np.ndarray:
        """Repeat naked and hidden singles before probabilistic search."""
        grid = np.asarray(puzzle, dtype=np.int16).copy()
        while True:
            candidates: dict[tuple[int, int], set[int]] = {}
            for row, col in zip(*np.where(grid == 0)):
                used = set(grid[row, :]) | set(grid[:, col]) | set(grid[row//3*3:row//3*3+3, col//3*3:col//3*3+3].ravel())
                candidates[(int(row), int(col))] = set(range(1, 10)) - used
            fills = {cell: next(iter(values)) for cell, values in candidates.items() if len(values) == 1}
            units = [[(r, c) for c in range(9)] for r in range(9)] + [[(r, c) for r in range(9)] for c in range(9)]
            units += [[(r, c) for r in range(br, br+3) for c in range(bc, bc+3)] for br in (0,3,6) for bc in (0,3,6)]
            for unit in units:
                for digit in range(1, 10):
                    cells = [cell for cell in unit if cell in candidates and digit in candidates[cell]]
                    if len(cells) == 1:
                        fills.setdefault(cells[0], digit)
            if not fills:
                return grid
            for (row, col), digit in fills.items():
                grid[row, col] = digit

    @property
    def dimension(self) -> int:
        return int(self.offsets[-1])

    def random_candidate(self, rng: np.random.Generator) -> np.ndarray:
        return np.concatenate([rng.permutation(digits) for _, digits in self.segments]).astype(np.int16)

    def decode(self, chromosome: np.ndarray) -> np.ndarray:
        chromosome = np.asarray(chromosome, dtype=np.int16)
        if chromosome.shape != (self.dimension,):
            raise ValueError("chromosome has the wrong length")
        grid = self.puzzle.ravel().copy()
        for block, (positions, digits) in enumerate(self.segments):
            values = chromosome[self.offsets[block]:self.offsets[block + 1]]
            if not np.array_equal(np.sort(values), digits):
                raise ValueError("each block segment must permute its missing digits")
            grid[positions] = values
        return grid.reshape(9, 9)

    @staticmethod
    def paper_metrics(grid: np.ndarray) -> dict[str, int]:
        groups = [grid[r, :] for r in range(9)] + [grid[:, c] for c in range(9)]
        unique_score = sum(len(np.unique(group)) for group in groups)
        products = np.asarray([np.prod(PRIMES[np.asarray(group, dtype=int) - 1]) for group in groups], dtype=np.int64)
        invalid = int(np.count_nonzero(products != TARGET_PRODUCT))
        penalty = 162 - int(unique_score)
        squared_excess = 0
        worst_group = 0
        row_duplicate_excess = 0
        column_duplicate_excess = 0
        for group_index, group in enumerate(groups):
            counts = np.bincount(np.asarray(group, dtype=int), minlength=10)[1:]
            excess = np.maximum(counts - 1, 0)
            squared_excess += int(np.square(excess).sum())
            group_excess = int(excess.sum())
            worst_group = max(worst_group, group_excess)
            if group_index < 9:
                row_duplicate_excess += group_excess
            else:
                column_duplicate_excess += group_excess
        previous_penalty = penalty * 100_000 + invalid * 1_000 + squared_excess * 10 + worst_group
        return {"penalty": penalty, "paper_penalty": penalty, "paper_score": int(unique_score),
                "invalid_groups": invalid, "solved_groups": 27 - invalid,
                "duplicate_excess": penalty, "squared_excess": squared_excess,
                "worst_group": worst_group, "previous_penalty": previous_penalty,
                "row_duplicate_excess": row_duplicate_excess,
                "column_duplicate_excess": column_duplicate_excess}

    def evaluate(self, chromosome: np.ndarray) -> np.ndarray:
        return np.asarray([self.paper_metrics(self.decode(chromosome))["penalty"]], dtype=np.float64)

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        metrics = self.evaluate_population_metrics(population)
        return metrics[:, 0:1].astype(np.float64)

    def evaluate_population_metrics(self, population: np.ndarray) -> np.ndarray:
        candidates = np.ascontiguousarray(population, dtype=np.int16)
        if candidates.ndim != 2 or candidates.shape[1] != self.dimension:
            raise ValueError("population has the wrong chromosome width")
        return evaluate_block_population_numba(candidates, self._flat_puzzle,
                                               self.chromosome_positions)

    def evaluate_population_objectives(self, population: np.ndarray, method: str) -> np.ndarray:
        if method == "mo_pareto":
            return self.evaluate_population_metrics(population)[:, 5:7]
        codes = {"mo_bands_stacks": 1, "mo_block_responsibility": 2,
                 "mo_digit_conflicts": 3, "mo_worst_total": 4,
                 "mo_invalid_severity": 5}
        try:
            code = codes[method]
        except KeyError as exc:
            raise ValueError(f"unknown multi-objective evaluator: {method}") from exc
        candidates = np.ascontiguousarray(population, dtype=np.int16)
        return evaluate_block_mo_numba(candidates, self._flat_puzzle,
                                       self.chromosome_positions, code)

    def evaluate_population_reference(self, population: np.ndarray) -> np.ndarray:
        rows = []
        for candidate in population:
            metrics = self.paper_metrics(self.decode(candidate))
            rows.append((metrics["paper_penalty"], metrics["invalid_groups"],
                         metrics["squared_excess"], metrics["worst_group"],
                         metrics["previous_penalty"], metrics["row_duplicate_excess"],
                         metrics["column_duplicate_excess"]))
        return np.asarray(rows, dtype=np.int64)


def fixture_by_id(fixture_id: str) -> SudokuFixture:
    for fixture in FIXTURES:
        if fixture.id == fixture_id:
            return fixture
    raise KeyError(fixture_id)
