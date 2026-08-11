# ROSE permutation models

The suite exposes four reusable scalar-objective permutation models:

- `ROSE` / **ROSE-MeanRef** — original averaged-reference baseline.
- `TemplateROSE` / **Template-ROSE-MeanRef** — original model with HBSA punched-template reconstruction.
- `ROSESingleRef` / **ROSE-SingleRef** — version 2; one real reference and its empirical signed-distance distribution.
- `TemplateROSESingleRef` / **Template-ROSE-SingleRef** — version 2 for removed template items.

The original class names remain backward-compatible MeanRef aliases.

The model learns `N[j,k] = P(pos(j)=k)` and
`D[i,j,delta] = P(pos(j)-pos(i)=delta)`. SingleRef selects a reference from a
fixed, random or full window. Uniform selection is the default; experimental
confidence selection uses capped inverse entropy. It samples only free positions
from a log-space mixture of exact and relative evidence, so collision shifting
and permutation repair are unnecessary.

`diagnostics()` reports estimator/reference entropy, reference frequency,
fallback rates, sampled-distance mean/SD, diversity, and template counts.
`pair_statistics(i,j)` reports count, mean, min, max, SD, entropy, mode, and the
number of observed signed distances. The dense relative tensor uses `O(n^3)`
memory; sparse storage is a future option for large permutations.
