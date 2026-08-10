"""Block-preserving Sudoku solvers matching Waiyapara et al.'s representation."""

from __future__ import annotations

from time import perf_counter
import numpy as np

from coin.problems.sudoku import SudokuBlockPermutationProblem
from coin.learning.pareto import nondominated_ranks, pareto_quality, pareto_selection_scores


MO_OBJECTIVE_NAMES = {
    "mo_pareto": ["row_duplicate_excess", "column_duplicate_excess"],
    "mo_bands_stacks": ["horizontal_band_rows_1_3", "horizontal_band_rows_4_6",
                         "horizontal_band_rows_7_9", "vertical_stack_columns_1_3",
                         "vertical_stack_columns_4_6", "vertical_stack_columns_7_9"],
    "mo_block_responsibility": [f"block_{index}_cross_conflicts" for index in range(1, 10)],
    "mo_digit_conflicts": [f"digit_{digit}_conflicts" for digit in range(1, 10)],
    "mo_worst_total": ["worst_group", "total_duplicate_excess"],
    "mo_invalid_severity": ["invalid_group_count", "squared_duplicate_severity"],
}


def _objective_archive(populations, metrics, objective_sets, maximum_size: int):
    """Return a unique, diverse external non-dominated archive.

    Sudoku has a common optimum rather than a permanent trade-off, but during
    search the archive preserves candidates that exchange row quality for
    column quality.  Duplicate genotypes are removed and crowding order is
    used if the archive grows beyond the teaching-lab bound.
    """
    population = np.vstack(populations)
    all_metrics = np.vstack(metrics)
    objectives = np.vstack(objective_sets).astype(float)
    keep = nondominated_ranks(objectives) == 0
    population, all_metrics, objectives = population[keep], all_metrics[keep], objectives[keep]
    unique_indices, seen = [], set()
    for index, chromosome in enumerate(population):
        key = tuple(int(value) for value in chromosome)
        if key not in seen:
            seen.add(key); unique_indices.append(index)
    population, all_metrics, objectives = population[unique_indices], all_metrics[unique_indices], objectives[unique_indices]
    if len(population) > maximum_size:
        order = np.argsort(pareto_selection_scores(objectives), kind="stable")[:maximum_size]
        population, all_metrics, objectives = population[order], all_metrics[order], objectives[order]
    return population, all_metrics, objectives


def _mo_environmental_selection(populations, metrics, objective_sets, size: int):
    population = np.vstack(populations)
    all_metrics = np.vstack(metrics)
    objectives = np.vstack(objective_sets)
    order = np.argsort(pareto_selection_scores(objectives), kind="stable")[:size]
    return population[order], all_metrics[order], objectives[order]


def _preserved_positions(n, ratio, rng):
    count = min(n, max(0, round(n * ratio / 100)))
    return set(int(x) for x in rng.choice(n, count, replace=False)) if count else set()


def _sample_position(problem, weights, rng, template=None, preserve_ratio=0, random_positions=False):
    parts = []
    for b, (_, digits) in enumerate(problem.segments):
        n = len(digits)
        if not n:
            continue
        preserved = _preserved_positions(n, preserve_ratio, rng) if template is not None else set()
        template_values = template[problem.offsets[b]:problem.offsets[b + 1]] if template is not None else None
        reserved = {pos: int(np.flatnonzero(digits == template_values[pos])[0]) for pos in preserved}
        unused = [index for index in range(n) if index not in reserved.values()]; out = np.empty(n,dtype=np.int16)
        positions=list(range(n))
        if random_positions and template is None: rng.shuffle(positions)
        for pos in positions:
            if pos in preserved:
                out[pos]=digits[reserved[pos]]; continue
            probs = np.asarray([weights[b][pos, j] for j in unused], dtype=float)
            probs /= probs.sum()
            chosen = int(rng.choice(len(unused), p=probs))
            out[pos]=digits[unused.pop(chosen)]
        parts.extend(out.tolist())
    return np.asarray(parts, dtype=np.int16)


def _sample_edge(problem, weights, rng, template=None, preserve_ratio=0):
    parts = []
    for b, (_, digits) in enumerate(problem.segments):
        n = len(digits)
        if not n:
            continue
        preserved = _preserved_positions(n, preserve_ratio, rng) if template is not None else set()
        template_values = template[problem.offsets[b]:problem.offsets[b + 1]] if template is not None else None
        reserved = {pos: int(np.flatnonzero(digits == template_values[pos])[0]) for pos in preserved}
        unused = [index for index in range(n) if index not in reserved.values()]; order = []
        for pos in range(n):
            if pos in preserved:
                selected = reserved[pos]
            elif not order:
                selected = unused.pop(int(rng.integers(len(unused))))
            else:
                probs = np.asarray([weights[b][order[-1], j] for j in unused], dtype=float)
                probs /= probs.sum(); pick = int(rng.choice(len(unused), p=probs)); selected = unused.pop(pick)
            order.append(selected)
        parts.extend(digits[order])
    return np.asarray(parts, dtype=np.int16)


def _learn(problem, population, fitness, weights, kind, reward, punish, strength):
    order = np.argsort(fitness); nr = max(1, round(len(order) * reward / 100)); npun = max(1, round(len(order) * punish / 100)) if punish else 0
    selections = [(order[:nr], strength / nr)]
    if npun:
        selections.append((order[-npun:], -strength / npun))
    for selected, delta in selections:
        for ci in selected:
            candidate = population[ci]
            for b, (_, digits) in enumerate(problem.segments):
                lo, hi = problem.offsets[b:b+2]; values = candidate[lo:hi]
                index = {int(d): i for i, d in enumerate(digits)}
                if kind == "edge":
                    for a, z in zip(values[:-1], values[1:]): weights[b][index[int(a)], index[int(z)]] += delta
                else:
                    for pos, value in enumerate(values): weights[b][pos, index[int(value)]] += delta
    for matrix in weights:
        np.clip(matrix, .01, 100., out=matrix)


def _learn_block_responsibility(problem, population, objectives, weights, kind,
                                reward, punish, strength):
    """Assign positive/negative credit only to the responsible block model."""
    for b, (_, digits) in enumerate(problem.segments):
        order = np.argsort(objectives[:, b], kind="stable")
        nr = max(1, round(len(order) * reward / 100))
        npun = max(1, round(len(order) * punish / 100)) if punish else 0
        selections = [(order[:nr], strength / nr)]
        if npun:
            selections.append((order[-npun:], -strength / npun))
        index = {int(digit): position for position, digit in enumerate(digits)}
        lo, hi = problem.offsets[b:b + 2]
        for selected, delta in selections:
            for candidate_index in selected:
                values = population[candidate_index, lo:hi]
                if kind == "edge":
                    for first, second in zip(values[:-1], values[1:]):
                        weights[b][index[int(first)], index[int(second)]] += delta
                else:
                    for position, value in enumerate(values):
                        weights[b][position, index[int(value)]] += delta
        np.clip(weights[b], .01, 100., out=weights[b])


def _ox(a, b, rng):
    n = len(a)
    if n < 2: return a.copy()
    x, y = sorted(rng.choice(n, 2, replace=False)); child = np.full(n, -1, dtype=np.int16); child[x:y+1] = a[x:y+1]
    fill = [v for v in b if v not in child]; child[child < 0] = fill
    return child


def run_block_algorithm(name: str, problem: SudokuBlockPermutationProblem, population_size: int,
                        generations: int, seed: int, label: str, *, reward_ratio: int = 15,
                        punishment_ratio: int = 15, learning_step: float = .15,
                        crossover_probability: float = .3, mutation_probability: float = .3,
                        evaluation_method: str = "paper_prime", sampling_mode: str = "wt",
                        template_sample_ratio: int = 50,
                        stop_at_solution: bool = True) -> dict:
    rng = np.random.default_rng(seed); started = perf_counter(); best = None; best_value = 10**9; history = []
    best_found_at_evaluation = 0; solution_found_at_evaluation = None
    weights = [np.ones((len(d), len(d)), dtype=float) for _, d in problem.segments]
    previous = None; archive_population = None; archive_metrics = None; archive_objectives = None
    for generation in range(1, generations + 1):
        parents = None
        if name == "ga_ox" and previous is not None:
            ranked = np.argsort(previous[1]); elite = previous[0][ranked[:max(2, population_size // 2)]]; candidates = []
            while len(candidates) < population_size:
                p1, p2 = elite[rng.integers(len(elite), size=2)]; child = []
                for b in range(9):
                    lo, hi = problem.offsets[b:b+2]; segment = _ox(p1[lo:hi], p2[lo:hi], rng)
                    if rng.random() > crossover_probability:
                        segment = p1[lo:hi].copy()
                    if len(segment) > 1 and rng.random() < mutation_probability:
                        i, j = rng.choice(len(segment), 2, replace=False); segment[i], segment[j] = segment[j], segment[i]
                    child.extend(segment)
                candidates.append(child)
            population = np.asarray(candidates, dtype=np.int16)
        else:
            sampler = _sample_edge if name == "ehbsa" else _sample_position
            use_template = name in ("ehbsa", "nhbsa") and sampling_mode == "wt" and previous is not None
            parents = previous if use_template else None
            population = np.asarray([sampler(problem, weights, rng,
                                    parents[0][i % len(parents[0])] if parents is not None else None,
                                    template_sample_ratio if use_template else 0,
                                    **({"random_positions": name == "coin"} if sampler is _sample_position else {}))
                                    for i in range(population_size)])
        population_metrics = problem.evaluate_population_metrics(population)
        is_mo = evaluation_method in MO_OBJECTIVE_NAMES
        metric_column = 4 if evaluation_method == "legacy_hierarchical" else 0
        metric_key = "previous_penalty" if evaluation_method == "legacy_hierarchical" else "paper_penalty"
        if is_mo:
            objective_values = problem.evaluate_population_objectives(population, evaluation_method)
            ranks = nondominated_ranks(objective_values)
            fitness = pareto_selection_scores(objective_values)
        else:
            objective_values = None
            ranks = np.zeros(len(population), dtype=np.int32)
            fitness = population_metrics[:, metric_column].astype(float)
        raw_selection_values = population_metrics[:, 0].astype(float) if is_mo else fitness
        raw_index = int(np.argmin(raw_selection_values))
        raw_value = int(raw_selection_values[raw_index])
        if raw_value < best_value:
            best_value = raw_value
            best = population[raw_index].copy()
            best_found_at_evaluation = (generation - 1) * population_size + raw_index + 1
        if solution_found_at_evaluation is None:
            solved_indices = np.flatnonzero(population_metrics[:, 0] == 0)
            if len(solved_indices):
                solution_found_at_evaluation = ((generation - 1) * population_size
                                                + int(solved_indices[0]) + 1)
        if parents is not None:
            if is_mo:
                population, population_metrics, objective_values = _mo_environmental_selection(
                    [parents[0], population], [parents[2], population_metrics],
                    [parents[3], objective_values], population_size)
                ranks = nondominated_ranks(objective_values)
                fitness = pareto_selection_scores(objective_values)
            else:
                keep_parent = parents[1] <= fitness
                population[keep_parent] = parents[0][keep_parent]
                fitness[keep_parent] = parents[1][keep_parent]
                population_metrics = problem.evaluate_population_metrics(population)
                fitness = population_metrics[:, metric_column].astype(float)
        previous = (population.copy(), fitness.copy(), population_metrics.copy(),
                    None if objective_values is None else objective_values.copy())
        if is_mo:
            archive_population, archive_metrics, archive_objectives = _objective_archive(
                [archive_population, population] if archive_population is not None else [population],
                [archive_metrics, population_metrics] if archive_metrics is not None else [population_metrics],
                [archive_objectives, objective_values] if archive_objectives is not None else [objective_values],
                maximum_size=max(32, population_size * 4))
        metrics = problem.paper_metrics(problem.decode(best)); point = {"generation": generation,
                  "evaluations": generation * population_size, **metrics,
                  "pareto_depth_count": int(ranks.max()) + 1,
                  "nondominated_count": int(np.count_nonzero(ranks == 0))};
        if is_mo:
            depth_counts, front_spread = pareto_quality(objective_values)
            point.update({"pareto_depth_counts": depth_counts,
                          "front_spread": front_spread,
                          "archive_size": int(len(archive_population)),
                          "archive_objective_count": int(len(np.unique(archive_objectives, axis=0)))})
        point["penalty"] = int(metrics["paper_penalty"] if is_mo else metrics[metric_key]); history.append(point)
        # Zero paper penalty is an exact Sudoku certificate. Population evaluation is
        # vectorized, so the current batch has already been evaluated; stop before
        # learning or generating another batch.
        if stop_at_solution and solution_found_at_evaluation is not None:
            break
        if name != "ga_ox":
            kind = "edge" if name == "ehbsa" else "position"
            # Paper NB-COIN: reward/punishment 25%, k=.4. NHBSA is positive-only.
            if evaluation_method == "mo_block_responsibility":
                _learn_block_responsibility(problem, population, objective_values, weights, kind,
                                            reward_ratio, punishment_ratio if name in ("coin","cnb_coin") else 0,
                                            learning_step)
            else:
                _learn(problem, population, fitness, weights, kind, reward_ratio,
                       punishment_ratio if name in ("coin","cnb_coin") else 0, learning_step)
    grid = problem.decode(best); metrics = problem.paper_metrics(grid)
    result = {"algorithm": name, "label": label, "seed": seed, "grid": grid.ravel().tolist(),
            "penalty": int(best_value), "paper_score": metrics["paper_score"], "solved": metrics["paper_penalty"] == 0,
            "metrics": metrics, "evaluations": len(history) * population_size,
            "generations": len(history),
            "runtime_seconds": perf_counter() - started, "history": history,
            "best_found_at_evaluation": int(best_found_at_evaluation),
            "best_found_at_generation": int((best_found_at_evaluation - 1) // population_size + 1),
            "solution_found_at_evaluation": (None if solution_found_at_evaluation is None
                                               else int(solution_found_at_evaluation)),
            "evaluation_method": evaluation_method, "sampling_mode": sampling_mode,
            "template_sample_ratio": template_sample_ratio,
            "stop_at_solution": stop_at_solution,
            "termination_reason": ("optimal_solution" if solution_found_at_evaluation is not None
                                   and stop_at_solution else "evaluation_budget")}
    if evaluation_method in MO_OBJECTIVE_NAMES:
        result.update({
            "mo_coin": True,
            "objective_names": MO_OBJECTIVE_NAMES[evaluation_method],
            "pareto_archive": [{
                "objectives": [int(value) for value in objective],
                "paper_penalty": int(candidate_metrics[0]),
                "grid": problem.decode(chromosome).ravel().astype(int).tolist(),
            } for chromosome, candidate_metrics, objective in zip(
                archive_population, archive_metrics, archive_objectives)],
            "pareto_metrics": {
                "archive_size": int(len(archive_population)),
                "unique_objective_vectors": int(len(np.unique(archive_objectives, axis=0))),
                "spread": (archive_objectives.max(axis=0) - archive_objectives.min(axis=0)).astype(float).tolist(),
                "nondominated_ratio": float(np.mean(nondominated_ranks(objective_values) == 0)),
            },
        })
    return result
