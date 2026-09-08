# ROSE permutation models

The suite exposes four reusable scalar-objective permutation models:

- `ROSE` / **ROSE-MultiRef Mean** — original baseline using mean predictions from several references.
- `TemplateROSE` / **Template-ROSE-MultiRef Mean** — original model with HBSA punched-template reconstruction.
- `ROSESingleRef` / **ROSE-SingleRef Range** — selects one real reference and samples its learned signed-distance range.
- `TemplateROSESingleRef` / **Template-ROSE-SingleRef Range** — the range sampler for removed template items.
- `ROSESingleRefHistogram` / **ROSE-SingleRef Histogram** — retains the full empirical signed-distance distribution.
- `TemplateROSESingleRefHistogram` / **Template-ROSE-SingleRef Histogram** — punched-template empirical-histogram baseline.

The original class and algorithm IDs remain backward compatible. Only their
display names now state the sampling behavior explicitly.

## Compact estimators

ROSE stores exact node probabilities `N[j,p] = P(pos(j)=p)` and five compact
pair matrices: sample count, mean, minimum, maximum, and standard deviation of
`pos(j)-pos(i)`. Every matrix is `n × n`; no pairwise distance histogram is
stored. Total estimator memory is therefore `O(n²)`.

**ROSE-MultiRef Mean** predicts a target position from every reference in its
active window and averages those predictions. **ROSE-SingleRef Range** selects
one actual reference uniformly by default (or experimentally by inverse SD),
then draws an integer distance from a truncated normal parameterized by the
learned mean and SD inside the observed `[minimum, maximum]` range.

Both variants mix relative evidence with exact-position evidence using
`node_weight`, apply temperature-controlled masked softmax over free positions,
and never require collision shifting or permutation repair. A zero-SD pair uses
its rounded mean. An unusable pair falls back to the exact-position estimator.

`diagnostics()` reports estimator bytes, mean pairwise SD, reference frequency,
fallback rates, sampled and realized distance mean/SD, offspring diversity, and
template counts. `pair_statistics(i,j)` returns count, mean, minimum, maximum,
and standard deviation directly from the compact matrices.

## Empirical-histogram research baseline

The Histogram variants preserve the alternative implementation developed while
clarifying SingleRef. They store `H[i,j,d]`, select one real reference, and
score free positions from the complete empirical signed-distance distribution.
This can retain separated modes that mean/SD range sampling smooths over, but
requires `O(n³)` memory. Keeping it under an explicit name makes that trade-off
measurable without changing either compact ROSE definition.
