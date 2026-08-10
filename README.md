# COINCIDENCE Algorithms Suite

A reusable Python library for COINcidence-based permutation optimization. This
repository publishes the algorithmic core only: no web application, API,
database, Docker image, or experiment-result store.

## Algorithms

| Family | Variant | Sampling representation |
|---|---|---|
| COIN | Edge COIN | `W[previous, next]` |
| COIN | NB-COIN | randomly choose an unfilled position, then sample its value |
| COIN | CNB-COIN | visit positions `0 → n-1`, then sample each value |
| COIN | Hybrid Template | retain position 0 and a scattered 30–70% Node template; Edge fills holes |
| COIN | Hybrid Chain | Node at position 0; choose Node or Edge independently at every next link |
| COIN | Start-Node Edge | learn the starting node and directed edges |
| EDA | EHBSA-WO / EHBSA-WT | edge histogram, without/with punched template |
| EDA | NHBSA-WO / NHBSA-WT | node-position histogram, without/with punched template |

Every COIN representation can run with scalar selection or Pareto-depth and
diversity selection through the reusable permutation-problem interface.

## Included problem contracts

- Flow Shop Scheduling, including Taillard instances
- TSP with Time Windows
- Sudoku block-permutation encoding
- Knight's Tour evaluation
- RNA secondary-structure permutation search (optional ViennaRNA scoring)
- TSP and bi-objective MO-TSP, with deterministic preloaded fixtures of 8, 12, 16, 20, and 24 cities

These modules are research examples and reusable evaluators, not applications.

See `examples/tsp_coin_tutorial.ipynb` for an executable tutorial covering a
closed tour, single-objective learning progress, a two-objective Pareto
archive, and a Pareto-front plot.

## Installation

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[test,rna]"
pytest
```

## Minimal example

```python
from coin.core import PermutationCoinAlgorithm
from coin.models import EdgeConfig, CNBCoin
from coin.problems.flowshop import FlowShopProblem

processing_times = [[3, 2], [2, 4], [4, 1]]
problem = FlowShopProblem(processing_times, objectives=("makespan",))
model = CNBCoin(EdgeConfig(
    problem_size=problem.dimension,
    population_size=100,
    reward_ratio=10,
    punishment_ratio=10,
    training_rate=5,
    objective="min",
), seed=1)

algorithm = PermutationCoinAlgorithm(model, problem)
algorithm.run(200)
print(algorithm.history[-1])
```

## NB-COIN and CNB-COIN

Both models learn the same matrix `W[position, value]`. They differ in the
generation policy:

- **NB-COIN** randomizes which unfilled position is sampled next.
- **CNB-COIN** follows the positional chain from 0 through `n-1`.

Keeping both variants explicit makes their convergence and inductive bias
experimentally testable.

## Multi-objective use

`MultiObjectiveCoinAlgorithm` evaluates one shared population evaluator,
assigns nondominated depth, preserves diversity, and maintains an external
archive. The experiment runner exposes MO forms of Edge, NB, CNB, Hybrid
Template, Hybrid Chain, and Start-Node Edge COIN.

## Citation

Use the repository's `CITATION.cff`. If a particular problem adapter is used,
also cite the corresponding benchmark and original algorithm paper.

## License

Code is licensed under the [Apache License 2.0](LICENSE). Documentation is
provided under CC BY 4.0. External benchmark files retain their original terms;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
