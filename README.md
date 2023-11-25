# obscure_stats

| | |
| --- | --- |
| CI/CD |[![CI - Test](https://github.com/glevv/obscure_stats/actions/workflows/package.yml/badge.svg)](https://github.com/glevv/obscure_stats/actions/workflows/package.yml) [![Coverage](https://codecov.io/github/glevv/obscure_stats/coverage.svg?branch=main)](https://codecov.io/gh/glevv/obscure_stats)
| Package | [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/obscure_stats?logo=Python)](https://pypi.org/project/obscure_stats/) [![PyPI](https://img.shields.io/pypi/v/obscure_stats?logo=PyPI)](https://pypi.org/project/obscure_stats) |
| Meta | [![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/) [![License - MIT](https://img.shields.io/badge/license-MIT-9400d3.svg)](https://spdx.org/licenses/) [![DOI](https://zenodo.org/badge/163630824.svg)](https://zenodo.org/badge/latestdoi/163630824)

## Highlights:

`obscure_stats` is a Python package that includes a lot of useful but less known statistical functions and builds on top of `numpy` and `scipy`.

## Current API list

- Collection of measures of central tendency - `obscure_stats/central_tendency`:
    * Contraharmonic Mean;
    * Half-Sample Mode;
    * Hodges-Lehmann-Sen Location;
    * Midhinge;
    * Midmean;
    * Midrange;
    * Standard Trimmed Harrell-Davis Quantile;
    * Trimean.
- Collection of measures of dispersion - `obscure_stats/dispersion`:
    * Coefficient of Variation;
    * Dispersion Ratio;
    * Linear Coefficient of Variation;
    * Lloyds Index;
    * Morisita Index;
    * Quartile Coefficient of Dispersion;
    * Robust Coefficient of Variation;
    * Standard Quantile Absolute Deviation;
    * Studentized Range.
- Collection of measures of skewness - `obscure_stats/skewness`:
    * Area Under the Skewness Curve (weighted and unweighted);
    * Bickel Mode Skewness Coefficient;
    * Bowley Skewness Coefficient;
    * Forhad-Shorna Rank Skewness Coefficient;
    * Groeneveld Skewness Coefficient;
    * Hossain-Adnan Skewness Coefficient;
    * Kelly Skewness Coefficient;
    * Medeen Skewness Coefficient;
    * Pearson Median Skewness Coefficient;
    * Pearson Mode Skewness Coefficient (original and halfmode modification).
- Collection of measures of kurtosis - `obscure_stats/kurtosis`:
    * Crow-Siddiqui Kurtosis;
    * Hogg Kurtosis;
    * Moors Kurtosis;
    * Moors Octile Kurtosis;
    * Reza-Ma Kurtosis.
- Collection of measures of association - `obscure_stats/association`:
    * Chatterjee Xi correlation Coefficient (original and symmetric versions);
    * Concordance Correlation Coefficient;
    * Concordance Rate;
    * Tanimoto Similarity;
    * Zhang I Correlation Coefficient.
- Collection of measures of qualitative variation - `obscure_stats/variation`:
    * AVDev;
    * B Index;
    * Extropy;
    * Gibbs M1;
    * Gibbs M2;
    * ModVR;
    * RanVR.

## Installation

`pip install obscure_stats`

## License

The content of this repository is licensed under a [MIT license](https://github.com/glevv/obscure_stats/blob/main/LICENSE).
