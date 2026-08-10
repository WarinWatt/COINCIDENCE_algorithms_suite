# COINCIDENCE Algorithms Suite

A reusable Python research and teaching library for COINcidence-based
permutation optimization. The repository publishes the algorithmic core only:
no web application, API, database, Docker image, queue, or experiment-result
store.

COIN learns recurring relationships among good permutations and uses the
learned probability model to construct the next population. The same problem
contract can therefore be used to compare COIN representations, histogram
sampling algorithms, and permutation baselines under identical evaluators,
seeds, and objective-evaluation budgets.

## Why this repository exists

The suite modernizes research implementations into readable, testable Python
components suitable for:

- reproducible experiments in permutation and multi-objective optimization;
- teaching representation, selection, probabilistic learning, and Pareto depth;
- extending COIN to a new problem by implementing one small problem contract;
- comparing a reference implementation with accelerated NumPy/Numba evaluators;
- preserving legacy behavior while making new variants explicit and testable.

## Algorithms

| Family | Variant | What is learned / how a candidate is generated |
|---|---|---|
| COIN | Edge COIN | directed adjacency weight `W[previous, next]` |
| COIN | NB-COIN | choose an unfilled position randomly, then sample its value |
| COIN | CNB-COIN | visit positions `0..n-1`, sampling each value |
| COIN | Hybrid Template | retain position 0 and a scattered 30–70% Node template; Edge fills holes |
| COIN | Hybrid Chain | start with Node at position 0; choose Node or Edge at each subsequent link |
| COIN | Start-Node Edge | learn a starting-node distribution and directed edges |
| EDA | EHBSA-WO / EHBSA-WT | edge histogram sampling without/with a punched template |
| EDA | NHBSA-WO / NHBSA-WT | node-position histogram sampling without/with a punched template |
| pymoo adapter | GA, NSGA-II, NSGA-III, SPEA2 | OX and ERX permutation comparisons where applicable |

Every COIN representation can run with scalar cohort selection or reusable
multi-objective selection based on nondominated depth and diversity. MO runs
maintain a unique external nondominated archive.

## Included problem contracts

- permutation Flow Shop Scheduling, including Taillard instances;
- TSP with Time Windows;
- TSP and bi-objective MO-TSP controlled fixtures of 8, 12, 16, 20, and 24 cities;
- Sudoku block-permutation encoding and decomposed objectives;
- Knight's Tour evaluation;
- RNA secondary-structure helix selection, with optional ViennaRNA scoring.

The small TSP/MO-TSP preloads are deterministic teaching fixtures. They are
intended for examples and regression tests and are **not** presented as
published best-known benchmark instances.

## Installation

```bash
git clone https://github.com/WarinWatt/COINCIDENCE_algorithms_suite.git
cd COINCIDENCE_algorithms_suite
python -m pip install -e .
```

Development, tests, RNA scoring, and notebooks are optional:

```bash
python -m pip install -e ".[test]"
python -m pip install -e ".[rna]"
python -m pip install -e ".[notebook]"
pytest
```

## Minimal single-objective example

```python
from coin.core import PermutationCoinAlgorithm
from coin.models import CNBCoin, EdgeConfig
from coin.problems.tsp import TSPProblem, get_tsp_instance

problem = TSPProblem(get_tsp_instance("tsp-16"), ("distance",))
config = EdgeConfig(
    problem_size=problem.dimension,
    population_size=100,
    reward_ratio=10,
    punishment_ratio=10,
    training_rate=5,
    objective="min",
)

algorithm = PermutationCoinAlgorithm(CNBCoin(config, seed=1), problem)
algorithm.run(100)
print(algorithm.history[-1])
```

## Minimal multi-objective example

```python
from coin.core import MultiObjectiveCoinAlgorithm
from coin.models import EdgeConfig, OptimizedEdgeCoin
from coin.problems.tsp import TSPProblem, get_tsp_instance

problem = TSPProblem(
    get_tsp_instance("motsp-16"),
    ("distance", "operating_cost"),
)
config = EdgeConfig(
    problem_size=problem.dimension,
    population_size=100,
    reward_ratio=10,
    punishment_ratio=10,
    training_rate=5,
    objective="min",
)

algorithm = MultiObjectiveCoinAlgorithm(OptimizedEdgeCoin(config, seed=1), problem)
for _ in range(100):
    algorithm.step()

print(algorithm.archive_values)       # nondominated objective vectors
print(algorithm.archive_population)   # corresponding tours
```

The executable [`examples/tsp_coin_tutorial.ipynb`](examples/tsp_coin_tutorial.ipynb)
adds learning-progress and Pareto-front plots.

## Implementing a new problem

A reusable problem supplies four things:

```python
class MyPermutationProblem:
    dimension: int
    objective_names: tuple[str, ...]
    def validate(self, permutation): ...
    def evaluate(self, permutation): ...
    # Optional but recommended for speed:
    def evaluate_population(self, population): ...
```

The chromosome must be a complete zero-based permutation. Return a NumPy
vector with one value per objective. Objectives are assumed to be minimized in
the present MO selection and adapters; transform maximization values explicitly
when defining a new contract.

## Fair experiment checklist

For defensible comparisons:

1. reuse the identical problem/evaluator instance for every algorithm;
2. compare equal objective-evaluation budgets, normally `population × generations`;
3. publish every seed and parameter setting, not only the best run;
4. separate search objectives from reporting-only metrics;
5. report best/mean/dispersion for one objective;
6. report convergence, spread/diversity, archive size, and nondominated ratio
   for multiple objectives; add hypervolume only with a declared reference point;
7. retain the final permutations or nondominated set so results can be re-evaluated.

## Repository layout

```text
src/coin/core/          reusable scalar and MO execution
src/coin/models/        COIN, hybrid, EHBSA, and NHBSA models
src/coin/learning/      statistics, reward/punishment, Pareto selection
src/coin/problems/      reusable evaluators and controlled fixtures
src/coin/adapters/      optional comparison-framework adapters
src/coin/experiments/   fair-budget orchestration and statistics
examples/               executable research/teaching notebooks
tests/                  equivalence, invariants, budgets, and regression tests
```

## Citation

If this software contributes to a publication, cite the software release and
the relevant original algorithm/problem paper. GitHub can export the software
citation directly from [`CITATION.cff`](CITATION.cff). A ready-to-copy APA and
BibTeX entry is provided in [`CITATION.md`](CITATION.md).

The Flow Shop work motivating part of this modernization is:

> Srimongkolkul, O., & Chongstitvatana, P. (2013). Application of Node Based
> Coincidence algorithm for flow shop scheduling problems. *2013 10th
> International Joint Conference on Computer Science and Software Engineering
> (JCSSE)*, 49–52.

## Scope and license

Code is licensed under the [Apache License 2.0](LICENSE). External benchmark
files and optional tools retain their own terms; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The library is research
software: validate results independently before operational use.
