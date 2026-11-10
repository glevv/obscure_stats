# obscure_stats

| Category | Badges |
| --- | --- |
| CI/CD | [![CI build and test status][github-actions-badge]](https://github.com/glevv/obscure_stats/actions/workflows/package.yml) [![Release workflow status][github-publish-badge]](https://github.com/glevv/obscure_stats/actions/workflows/publish.yml) [![Test coverage on Coveralls][coverage-badge]](https://coveralls.io/github/glevv/obscure_stats) |
| Security | [![CodeQL analysis status][code-ql-badge]](https://github.com/glevv/obscure_stats/actions/workflows/codeql.yml) [![Dependabot enabled][dependabot-badge]](https://github.com/dependabot/dependabot-core) [![OpenSSF Scorecard score][open-ssf-badge]](https://securityscorecards.dev/viewer/?uri=github.com/glevv/obscure_stats) |
| Package | [![Supported Python versions][pypi-ver-badge]](https://pypi.org/project/obscure_stats/) [![Latest version on PyPI][pypi-pub-badge]](https://pypi.org/project/obscure_stats/) [![Total downloads from PyPI][downloads-badge]](https://pepy.tech/project/obscure_stats) |
| Meta | [![Managed with uv][uv-badge]](https://github.com/astral-sh/uv) [![Linted and formatted with Ruff][ruff-badge]](https://github.com/astral-sh/ruff) [![Type checked with pyrefly][pyrefly-badge]](https://github.com/facebook/pyrefly) [![License: MIT][license-badge]](LICENSE.txt) [![Zenodo DOI 10.5281/zenodo.10206933][doi-badge]](https://doi.org/10.5281/zenodo.10206933) |

[github-actions-badge]: https://github.com/glevv/obscure_stats/actions/workflows/package.yml/badge.svg
[github-publish-badge]: https://github.com/glevv/obscure_stats/actions/workflows/publish.yml/badge.svg
[coverage-badge]: https://coveralls.io/repos/github/glevv/obscure_stats/badge.svg
[code-ql-badge]: https://github.com/glevv/obscure_stats/actions/workflows/codeql.yml/badge.svg
[dependabot-badge]: https://img.shields.io/badge/Dependabot-active-brightgreen.svg
[open-ssf-badge]: https://api.securityscorecards.dev/projects/github.com/glevv/obscure_stats/badge
[pypi-ver-badge]: https://img.shields.io/pypi/pyversions/obscure_stats?logo=Python
[pypi-pub-badge]: https://img.shields.io/pypi/v/obscure_stats?logo=PyPI
[downloads-badge]: https://static.pepy.tech/badge/obscure_stats
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[pyrefly-badge]: https://img.shields.io/endpoint?url=https://pyrefly.org/badge.json
[license-badge]: https://img.shields.io/badge/license-MIT-9400d3.svg
[doi-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.10206933.svg

## Highlights

`obscure_stats` is a small Python package that includes a lot of useful but lesser-known statistical functions and
builds on top of `numpy` and `scipy`.

- Every function cites the paper it implements.
- Functions return raw statistics, without hidden normalizing constants.
- Inputs can be lists, tuples, NumPy arrays or pandas Series; the package is fully typed.

## Installation

```bash
pip install obscure_stats
```

or, with [uv](https://github.com/astral-sh/uv):

```bash
uv add obscure_stats
```

## Usage example

```python
from obscure_stats.central_tendency import standard_trimmed_harrell_davis_quantile
from obscure_stats.dispersion import standard_quantile_absolute_deviation

data = [1.83, 1.01, 100.12, 1.20, 0.99, 0.87, 1.13, 100.01, 0.75, 1.03]
central_tendency = standard_trimmed_harrell_davis_quantile(data)
dispersion = standard_quantile_absolute_deviation(data)
print(f"Robust measure of central tendency = {central_tendency:.2f}±{dispersion:.2f}")
# Robust measure of central tendency = 1.09±0.39
```

## Available functions

### Central tendency (`obscure_stats.central_tendency`)

- Contraharmonic Mean - `contraharmonic_mean`
- Gastwirth's Location - `gastwirth_location`
- Grenander's Mode - `grenanders_m`
- Half-Sample Mode - `half_sample_mode`
- Hodges-Lehmann-Sen Location - `hodges_lehmann_sen_location`
- Midhinge - `midhinge`
- Midmean - `midmean`
- Midrange - `midrange`
- Shorth - `shorth`
- Standard Trimmed Harrell-Davis Quantile - `standard_trimmed_harrell_davis_quantile`
- Tau Measure of Location - `tau_location`
- Trimean - `trimean`

### Dispersion (`obscure_stats.dispersion`)

- Coefficient of Range - `coefficient_of_range`
- Coefficient of Variation - `coefficient_of_variation`
- Fisher's Index of Dispersion - `fisher_index_of_dispersion`
- Gini Mean Difference - `gini_mean_difference`
- Inter-Expectile Range - `inter_expectile_range`
- Length of the Shortest Half - `length_of_shortest_half`
- Linear Coefficient of Variation - `coefficient_of_lvariation`
- Morisita Index of Dispersion - `morisita_index_of_dispersion`
- Quartile Coefficient of Dispersion - `quartile_coefficient_of_dispersion`
- Robust Coefficient of Variation - `robust_coefficient_of_variation`
- Shamos Estimator - `shamos_estimator`
- Shapiro Coefficient of Variation - `shapiro_coefficient_of_variation`
- Standard Quantile Absolute Deviation - `standard_quantile_absolute_deviation`
- Studentized Range - `studentized_range`

### Skewness (`obscure_stats.skewness`)

- Area Under the Skewness Curve - `auc_skew_gamma`
- Bickel Mode Skewness Coefficient - `bickel_mode_skew`
- Bowley Skewness Coefficient - `bowley_skew`
- Cumulative Skewness Coefficient - `cumulative_skew`
- Forhad-Shorna Rank Skewness Coefficient - `forhad_shorna_rank_skew`
- Groeneveld Range Skewness Coefficient - `groeneveld_range_skew`
- Kelly Skewness Coefficient - `kelly_skew`
- Meeden Skewness Coefficient - `meeden_skew`
- Pearson Median Skewness Coefficient - `pearson_median_skew`
- Pearson Mode Skewness Coefficient - `pearson_mode_skew`

### Kurtosis (`obscure_stats.kurtosis`)

- Crow-Siddiqui Kurtosis Coefficient - `crow_siddiqui_kurt`
- Hogg Kurtosis Coefficient - `hogg_kurt`
- Moors Octile Kurtosis Coefficient - `moors_octile_kurt`
- Quantile Weight - `quantile_weight`
- Reza-Ma Kurtosis Coefficient - `reza_ma_kurt`
- Schmid-Trede Measure of Peakedness - `schmid_trede_peakedness`
- Staudte Kurtosis Coefficient - `staudte_kurt`

### Association (`obscure_stats.association`)

- Blomqvist's Beta - `blomqvist_beta`
- Coefficient of Monotone Association - `coefficient_of_monotone_association`
- Concordance Correlation Coefficient - `concordance_correlation`
- Concordance Rate - `concordance_rate`
- Fechner Correlation Coefficient - `fechner_correlation`
- Gaussian Rank Correlation Coefficient - `gaussian_rank_correlation`
- Gini Correlation Coefficient - `gini_correlation`
- L1 Concordance Correlation Coefficient - `l1_concordance_correlation`
- Morisita-Horn Similarity - `morisita_horn_similarity`
- Normalized Chatterjee Xi Correlation Coefficient - `normalized_chatterjee_xi`
- Quantile Correlation Coefficient - `quantile_correlation`
- Rank Minrelation Coefficient - `rank_minrelation_coefficient`
- Rank-Turbulence Divergence - `rank_divergence`
- Symmetric Chatterjee Xi Correlation Coefficient - `symmetric_chatterjee_xi`
- Tanimoto Similarity - `tanimoto_similarity`
- Tukey's Correlation Coefficient - `tukey_correlation`
- Winsorized Correlation Coefficient - `winsorized_correlation`
- Zhang I Correlation Coefficient - `zhang_i`

### Qualitative variation (`obscure_stats.variation`)

- AVDev - `avdev`
- B Index - `b_index`
- Brillouin Index - `brillouin_index`
- Extropy - `extropy`
- Gibbs M1 - `gibbs_m1`
- McIntosh's D - `mcintosh_d`
- MNDif - `mndif`
- ModVR - `mod_vr`
- RanVR - `range_vr`
- Rényi Entropy - `renyi_entropy`

## Citation

If you use `obscure_stats` in your work, please cite it using the metadata in [CITATION.cff](CITATION.cff)
(GitHub shows it under "Cite this repository") or the DOI
[10.5281/zenodo.10206933](https://doi.org/10.5281/zenodo.10206933).

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Contributing

See the [contribution guidelines](CONTRIBUTING.md) for how to set up the project, run the checks and propose changes.

## Security Policy

See the [security policy](SECURITY.md) for how to report a vulnerability.

## Accessibility

See the [accessibility statement](ACCESSIBILITY.md) for our accessibility commitments and how to report a barrier.

## License

This project is licensed under the [MIT License](LICENSE.txt).
