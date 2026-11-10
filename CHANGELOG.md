# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
While the version is below 1.0, minor versions (0.x) may contain breaking changes.

## [0.8.0] - 2026-11-11

### Breaking changes

Removed and renamed functions are gone without a deprecation period:

| Old | Use instead | Why |
|---|---|---|
| `association.gaussain_rank_correlation` | `association.gaussian_rank_correlation` | misspelled name |
| `skewness.medeen_skew` | `skewness.meeden_skew` | misspelled name (Groeneveld & Meeden, 1984) |
| `skewness.hossain_adnan_skew` | `skewness.meeden_skew` | numerically identical: both are (mean - median) / mean(\|x - median\|) |
| `dispersion.cole_index_of_dispersion` | `dispersion.coefficient_of_variation` | redundant: Cole's index is exactly (1 + CV²) / n |
| `skewness.left_quantile_weight(x, q)` | `kurtosis.quantile_weight(x, q)` with q in (0, 0.5) | merged: one tail-weight function for both tails; moved to `kurtosis` because it measures tail weight, not skewness |
| `skewness.right_quantile_weight(x, q)` | `kurtosis.quantile_weight(x, q)` with q in (0.5, 1) (default 0.75) | merged, see above |
| `association.symmetric_normalized_chatterjee_xi` | `max(normalized_chatterjee_xi(x, y), normalized_chatterjee_xi(y, x))` | redundant: without ties it is exactly `symmetric_chatterjee_xi` · (n + 1) / (n − 2) |
| `kurtosis.moors_kurt` | `scipy.stats.kurtosis(x, fisher=False)` | numerically identical: var(z²) + 1 is the moment kurtosis |
| `variation.gibbs_m2` | `variation.gibbs_m1` | redundant: M2 is exactly M1 · k / (k − 1), with k the number of categories |
| `variation.negative_extropy` | `variation.extropy` | misnamed: it returns extropy as defined by Lad et al. (2015), which is already non-negative; values are unchanged |

Functions whose results change:

- `dispersion.quartile_coefficient_of_dispersion` now returns (Q3 - Q1) / (Q3 + Q1), the
  coefficient of quartile variation of Bonett (2006). Before, it returned twice that
  (IQR / midhinge). This makes it consistent with `dispersion.coefficient_of_range`,
  (max - min) / (max + min).
- `association.concordance_rate` is now the four-quadrant concordance rate of paired
  *changes* (Critchley, Lee & Ho, 2010). The quadrants are centered at 0. A square
  exclusion zone is set by the new `exclusion=` keyword (default `0.0`: only pairs where
  neither variable changed are dropped). The result is agreeing pairs / kept pairs, in [0, 1].
  Before, it counted quadrants around the means with an uncited ± SEM zone, divided by all
  n, in [-1, 1].
- `association.rank_divergence`: rank 1 is now the largest value, as in Dodds et al. (2023).
  Before, ranks were reversed. New keyword `normalize=` (default `False`) divides by the
  paper's normalization constant, so the result is in [0, 1].
- All functions return raw values, without consistency (normalizing) constants:
  - `dispersion.robust_coefficient_of_variation` no longer multiplies by 1.4826,
  - `dispersion.shapiro_coefficient_of_variation` no longer multiplies by 0.75,
  - `dispersion.standard_quantile_absolute_deviation` no longer applies the finite-sample
    constant K_n.

  Each docstring names the constant to multiply by, if needed.
- `skewness.auc_skew_gamma` uses the paper's estimator: a midpoint grid in (0, 0.5) that
  excludes the minimum and maximum, the mean of the generalized Bowley coefficients, and
  Hyndman-Fan type 8 quantiles.
- `central_tendency.grenanders_m`: the defaults of `p` and `k` now grow with the sample size
  as recommended by Hall (1982): p = n^(2/7) and k = ceil(n^(3/5)). The old fixed defaults
  (p = 1.001, k = 2) estimated the density-weighted mean ∫x f^(p+1) / ∫f^(p+1) rather than
  the mode, which is biased for skewed data even in large samples. Explicitly passed `p`
  and `k` behave as before; tiny samples (n <= k) return NaN.
- `skewness.pearson_mode_skew` estimates the mode with `central_tendency.half_sample_mode`.
  Before, it used `scipy.stats.mode`, which returns the smallest value when all values are
  distinct, so for continuous data the result was (mean - min) / SD.
- `association.rank_minrelation_coefficient` normalizes by m² as in the paper, instead of m² + 1.
- `association.normalized_chatterjee_xi` is clipped at -1 (Dalitz et al., 2024, eq. 8).
  NaNs now drop the whole pair from both the numerator and the normalizer.

### Fixed

- `association.zhang_i` returned a NumPy scalar instead of a Python `float`.
- `variation.avdev`, `variation.b_index` and `variation.range_vr` returned 1 (maximum
  variation) for data with a single category; they now return 0, like the other measures.
- Association functions no longer return NaN with an "input is constant" warning for
  small-scale data (e.g. values around 1e-9) or data with a large level and a small
  spread (e.g. prices). The constant check used a tolerance; it is now exact.
- Association functions convert their inputs to float64. Before, large integer inputs could
  overflow silently (e.g. `morisita_horn_similarity` returned 0.058 instead of 1.0 for
  int64 values around 4e9), and non-numeric input failed with a cryptic numpy error; it now
  raises `ValueError`.
- Input warnings of association functions (constant input, inf, missing values, length
  mismatch) now point at the caller's line instead of inside the library.
- Association functions validate their inputs in one pass: lengths and infinite values are
  checked on the raw input, incomplete pairs are dropped, and the remaining pairs are checked
  for size and constant variables. Before, input that became constant only after dropping the
  pairs with missing values could return a number (e.g. -0.217 for `symmetric_chatterjee_xi`)
  or inf instead of NaN.
- `variation.renyi_entropy(alpha=np.inf)` returns the min-entropy instead of raising.
- `central_tendency.grenanders_m` accepts numpy integers for `k`.
- `association.winsorized_correlation` and `association.rank_divergence` validate their
  parameters before checking the arrays, so invalid parameters always raise `ValueError`.
- Edge cases return NaN instead of warnings or division by zero:
  `variation.mcintosh_d` (one observation), `dispersion.gini_mean_difference` (empty
  input), `dispersion.standard_quantile_absolute_deviation` (all-NaN input).

### Changed

- Typing: all public inputs are annotated as `numpy.typing.ArrayLike`, so lists, tuples
  and arrays of any numeric dtype pass type checkers. Numeric inputs are converted to
  float64 internally, which also prevents integer overflow. Categorical inputs are
  flattened, so a scalar counts as one observation.
- `skewness.forhad_shorna_rank_skew` documents that it uses average ranks for ties (the
  paper uses competition ranking), which keeps f(-x) = -f(x).
- Docstrings: corrected references and typos, and documented the exact relationships
  between related functions (e.g. Rényi entropy with alpha = 2 = -log2(1 - Gibbs M1)).
- Docstrings: every function now states what it computes, with the formula in plain text
  (in backticks) and a description in words next to it. See Also entries use the numpydoc
  `name : description` format and point at real functions.
- `variation.renyi_entropy`: the error for a negative `alpha` now says "non-negative"
  (`alpha = 0` is valid).
- `association.zhang_i` documents that ties in x are broken by input order (inherited from
  Chatterjee's xi), so with ties the result depends on the order of the observations.
- `central_tendency.standard_trimmed_harrell_davis_quantile` documents that its highest
  density interval is found numerically (Brent's method), as in the reference implementation.

### Performance

- Importing any submodule except `association` takes about 25 ms instead of 340 ms, because
  scipy is imported only inside the functions that need it.
- `association.normalized_chatterjee_xi` computes its normalizer in closed form.

### Added

- New measures:
  - `central_tendency.shorth`: mean of the shortest half (Andrews et al., 1972).
  - `dispersion.length_of_shortest_half` (Rousseeuw & Leroy, 1988).
  - `association.coefficient_of_monotone_association`: CMA, monotone association on the AUC
    scale (Walz, Eberl & Gneiting, 2025).
  - `association.l1_concordance_correlation`: rho_1 agreement coefficient (Vallejos, Osorio &
    Ferrer, 2025).
  - `association.gini_correlation` (Schechtman & Yitzhaki, 1987).
  - `variation.mndif`: MNDif, the mean difference analogue (Wilcox, 1973).
  - `variation.brillouin_index`: Brillouin's diversity index, in nats (Brillouin, 1956).
- Every function has an `Examples` section; the examples run as doctests in the test suite.
- `obscure_stats.__version__`.
- The wheel ships `py.typed` and the license file (PEP 639 metadata), plus project URLs and keywords.
- Releases are built and published by a tag-triggered workflow with PyPI trusted publishing
  and attestations. GitHub releases carry Sigstore signatures of the distributions.

## Earlier versions

See the [GitHub releases](https://github.com/glevv/obscure_stats/releases) for 0.7.1 and earlier.

[0.8.0]: https://github.com/glevv/obscure_stats/releases/tag/0.8.0
