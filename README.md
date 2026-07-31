# obscure_stats

| | |
| --- | --- |
| CI/CD | [![CI][github-actions-badge]](https://github.com/glevv/obscure_stats/actions/workflows/package.yml) [![CD][github-publish-badge]](https://github.com/glevv/obscure_stats/actions/workflows/publish.yml) [![Coverage Status][coveradge-badge]](https://coveralls.io/github/glevv/obscure_stats) |
| Security | [![CodeQL][code-ql-badge]](https://github.com/glevv/obscure_stats/actions/workflows/codeql.yml) [![Dependabot][dependabot-badge]](https://github.com/dependabot/dependabot-core) [![OpenSSF Scorecard][open-ssf-badge]](https://securityscorecards.dev/viewer/?uri=github.com/glevv/obscure_stats) |
| Package | [![PyPI - Python Version][pypi-ver-badge]](https://pypi.org/project/obscure_stats/) [![PyPI][pypi-pub-badge]](https://pypi.org/project/obscure_stats/) [![Downloads][downloads-badge]](https://pepy.tech/project/obscure_stats) |
| Meta | [![uv][uv-badge]](https://github.com/astral-sh/uv) [![Ruff][ruff-badge]](https://github.com/astral-sh/ruff) [![pyrefly][pyrefly-badge]](https://github.com/facebook/pyrefly) [![License - MIT][license-badge]](https://spdx.org/licenses/) [![DOI][doi-badge]](https://doi.org/10.5281/zenodo.10206933)

[github-actions-badge]: https://github.com/glevv/obscure_stats/actions/workflows/package.yml/badge.svg
[github-publish-badge]: https://github.com/glevv/obscure_stats/actions/workflows/publish.yml/badge.svg
[coveradge-badge]: https://coveralls.io/repos/github/glevv/obscure_stats/badge.svg
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

## Highlights:

`obscure_stats` is a small Python package that includes a lot of useful but lesser-known statistical functions and builds on top of `numpy` and `scipy`.

## Current API list

- Collection of measures of central tendency - `obscure_stats.central_tendency`:
    * Contraharmonic Mean - `contraharmonic_mean`;
    * Gastwirth's Location - `gastwirth_location`;
    * Grenander's Mode - `grenanders_m`;
    * Half-Sample Mode - `half_sample_mode`;
    * Hodges-Lehmann-Sen Location - `hodges_lehmann_sen_location`;
    * Midhinge - `midhinge`;
    * Midmean - `midmean`;
    * Midrange - `midrange`;
    * Standard Trimmed Harrell-Davis Quantile - `standard_trimmed_harrell_davis_quantile`;
    * Tau Measure of Location - `tau_location`;
    * Trimean - `trimean`.
- Collection of measures of dispersion - `obscure_stats.dispersion`:
    * Coefficient of Range - `coefficient_of_range`;
    * Coefficient of Variation - `coefficient_of_variation`;
    * Cole's Index of Dispersion - `cole_index_of_dispersion`;
    * Fisher's Index of Dispersion - `fisher_index_of_dispersion`;
    * Gini Mean Difference - `gini_mean_difference`;
    * Linear Coefficient of Variation - `coefficient_of_lvariation`;
    * Inter-expectile Range - `inter_expectile_range`;
    * Morisita Index of Dispersion - `morisita_index_of_dispersion`;
    * Quartile Coefficient of Dispersion - `quartile_coefficient_of_dispersion`;
    * Robust Coefficient of Variation - `robust_coefficient_of_variation`;
    * Shamos Estimator - `shamos_estimator`;
    * Standard Quantile Absolute Deviation - `standard_quantile_absolute_deviation`;
    * Studentized Range - `studentized_range`.
- Collection of measures of skewness - `obscure_stats.skewness`:
    * Area Under the Skewness Curve - `auc_skew_gamma`;
    * Bickel Mode Skewness Coefficient - `bickel_mode_skew`;
    * Bowley Skewness Coefficient - `bowley_skew`;
    * Cumulative Skewness Coefficient - `cumulative_skew`;
    * Forhad-Shorna Rank Skewness Coefficient - `forhad_shorna_rank_skew`;
    * Groeneveld Range Skewness Coefficient - `groeneveld_range_skew`;
    * Hossain-Adnan Skewness Coefficient - `hossain_adnan_skew`;
    * Kelly Skewness Coefficient - `kelly_skew`;
    * Left Quantile Weight - `left_quantile_weight`;
    * Medeen Skewness Coefficient - `medeen_skew`;
    * Pearson Median Skewness Coefficient - `pearson_median_skew`;
    * Pearson Mode Skewness Coefficient - `pearson_mode_skew`;
    * Right Quantile Weight - `right_quantile_weight`.
- Collection of measures of kurtosis - `obscure_stats.kurtosis`:
    * Crow-Siddiqui Kurtosis Coefficient - `crow_siddiqui_kurt`;
    * Hogg Kurtosis Coefficient - `hogg_kurt`;
    * Moors Kurtosis Coefficient - `moors_kurt`;
    * Moors Octile Kurtosis Coefficient - `moors_octile_kurt`;
    * Reza-Ma Kurtosis Coefficient - `reza_ma_kurt`;
    * Schmid-Trede measure of Peakedness - `schmid_trede_peakedness`;
    * Staudte Kurtosis Coefficient - `staudte_kurt`.
- Collection of measures of association - `obscure_stats.association`:
    * Blomqvist's Beta - `blomqvist_beta`;
    * Concordance Correlation Coefficient - `concordance_correlation`;
    * Concordance Rate - `concordance_rate`;
    * Fechner Correlation Coefficient - `fechner_correlation`;
    * Gaussian Rank Correlation Coefficient - `gaussain_rank_correlation`;
    * Morisita-Horn Similarity - `morisita_horn_similarity`;
    * Normalized Chatterjee Xi Correlation Coefficient - `normalized_chatterjee_xi`;
    * Quantile Correlation Coefficient - `quantile_correlation`;
    * Rank Minrelation Coefficient - `rank_minrelation_coefficient`;
    * Rank-Turbulence Divergence - `rank_divergence`;
    * Symmetric Chatterjee Xi Correlation Coefficient - `symmetric_chatterjee_xi`;
    * Tanimoto Similarity - `tanimoto_similarity`;
    * Tukey's Correlation Coefficient - `tukey_correlation`;
    * Winsorized Correlation Coefficient - `winsorized_correlation`;
    * Zhang I Correlation Coefficient - `zhang_i`.
- Collection of measures of qualitative variation - `obscure_stats.variation`:
    * AVDev - `avdev`;
    * B Index - `b_index`;
    * Gibbs M1 - `gibbs_m1`;
    * Gibbs M2 - `gibbs_m2`;
    * McIntosh's D - `mcintosh_d`;
    * ModVR - `mod_vr`;
    * Negative Extropy - `negative_extropy`;
    * RanVR - `range_vr`;
    * Rényi entropy - `renyi_entropy`.

## Installation

```bash
>>> pip install obscure_stats
```

## Usage Example

```python
>>> from obscure_stats.central_tendency import standard_trimmed_harrell_davis_quantile
>>> from obscure_stats.dispersion import standard_quantile_absolute_deviation

>>> data = [1.83, 1.01, 100.12, 1.20, 0.99, 0.87, 1.13, 100.01, 0.75, 1.03]
>>> central_tendency = standard_trimmed_harrell_davis_quantile(data)
>>> dispersion = standard_quantile_absolute_deviation(data)
>>> print(f"Robust measure of central tendency = {central_tendency:.2f}±{dispersion:.2f}")
```

```
Out[1]:
Robust measure of central tendency = 1.09±0.42
```

## Code of Conduct

Code of Conduct for this project can be found [here](CODE_OF_CONDUCT.md).

## Contributing

Contribution guidelines for this project can be found [here](CONTRIBUTING.md).

## Security Policy

Security Policy for this project can be found [here](SECURITY.md).

## License

The content of this repository is licensed under a [MIT license](https://github.com/glevv/obscure_stats/blob/main/LICENSE.txt).

This repository bundles several libraries that are compatibly licensed. A full list can be found [here](https://github.com/glevv/obscure_stats/blob/main/LICENSES_bundled.txt).
