# obscure_stats

| | |
| --- | --- |
| CI/CD | [![CI](https://github.com/glevv/obscure_stats/actions/workflows/package.yml/badge.svg)](https://github.com/glevv/obscure_stats/actions/workflows/package.yml) [![CD](https://github.com/glevv/obscure_stats/actions/workflows/publish.yml/badge.svg)](https://github.com/glevv/obscure_stats/actions/workflows/publish.yml) [![Coverage](https://codecov.io/github/glevv/obscure_stats/coverage.svg?branch=main)](https://codecov.io/gh/glevv/obscure_stats) |
| Security | [![CodeQL](https://github.com/glevv/obscure_stats/actions/workflows/codeql.yml/badge.svg)](https://github.com/glevv/obscure_stats/actions/workflows/codeql.yml) [![Dependabot](https://img.shields.io/badge/Dependabot-active-brightgreen.svg)](https://github.com/dependabot/dependabot-core) [![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/glevv/obscure_stats/badge)](https://securityscorecards.dev/viewer/?uri=github.com/glevv/obscure_stats) |
| Package | [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/obscure_stats?logo=Python)](https://pypi.org/project/obscure_stats/) [![PyPI](https://img.shields.io/pypi/v/obscure_stats?logo=PyPI)](https://pypi.org/project/obscure_stats/) [![Downloads](https://static.pepy.tech/badge/obscure_stats)](https://pepy.tech/project/obscure_stats) |
| Meta | [![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/) [![License - MIT](https://img.shields.io/badge/license-MIT-9400d3.svg)](https://spdx.org/licenses/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10206933.svg)](https://doi.org/10.5281/zenodo.10206933)

## Highlights:

`obscure_stats` is a small Python package that includes a lot of useful but lesser-known statistical functions and builds on top of `numpy` and `scipy`.

## Current API list

- Collection of measures of central tendency - `obscure_stats/central_tendency`:
    * Contraharmonic Mean;
    * Gastwirth's Location;
    * Grenander's Mode;
    * Half-Sample Mode;
    * Hodges-Lehmann-Sen Location;
    * Midhinge;
    * Midmean;
    * Midrange;
    * Standard Trimmed Harrell-Davis Quantile;
    * Tau Measure of Location;
    * Trimean.
- Collection of measures of dispersion - `obscure_stats/dispersion`:
    * Coefficient of Range;
    * Coefficient of Variation;
    * Cole's Index of Dispersion;
    * Fisher's Index of Dispersion;
    * Gini Mean Difference;
    * Linear Coefficient of Variation;
    * Inter-expectile Range;
    * Morisita Index of Dispersion;
    * Quartile Coefficient of Dispersion;
    * Robust Coefficient of Variation;
    * Shamos Estimator;
    * Standard Quantile Absolute Deviation;
    * Studentized Range.
- Collection of measures of skewness - `obscure_stats/skewness`:
    * Area Under the Skewness Curve;
    * Bickel Mode Skewness Coefficient;
    * Bowley Skewness Coefficient;
    * Forhad-Shorna Rank Skewness Coefficient;
    * Groeneveld Range Skewness Coefficient;
    * Hossain-Adnan Skewness Coefficient;
    * Kelly Skewness Coefficient;
    * L-Skewness Coefficient;
    * Left Quantile Weight;
    * Medeen Skewness Coefficient;
    * Pearson Median Skewness Coefficient;
    * Pearson Mode Skewness Coefficient;
    * Right Quantile Weight;
- Collection of measures of kurtosis - `obscure_stats/kurtosis`:
    * Crow-Siddiqui Kurtosis;
    * L-Kurtosis;
    * Hogg Kurtosis;
    * Moors Kurtosis;
    * Moors Octile Kurtosis;
    * Reza-Ma Kurtosis;
    * Schmid-Trede measure of Peakedness;
    * Staudte Kurtosis.
- Collection of measures of association - `obscure_stats/association`:
    * Blomqvist's Beta;
    * Chatterjee Xi Correlation Coefficient;
    * Concordance Correlation Coefficient;
    * Concordance Rate;
    * Fechner Correlation Coefficient;
    * Gaussian Rank Correlation Coefficient;
    * Normalized Chatterjee Xi Correlation Coefficient;
    * Quantile Correlation Coefficient;
    * Rank Minrelation Coefficient;
    * Symmetric Chatterjee Xi Correlation Coefficient;
    * Tanimoto Similarity;
    * Tukey's Correlation Coefficient;
    * Winsorized Correlation Coefficient;
    * Zhang I Correlation Coefficient.
- Collection of measures of qualitative variation - `obscure_stats/variation`:
    * AVDev;
    * B Index;
    * Gibbs M1;
    * Gibbs M2;
    * McIntosh's D;
    * ModVR;
    * Negative Extropy;
    * RanVR;
    * Rényi entropy.

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
