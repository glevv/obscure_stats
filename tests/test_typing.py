"""Static typing guard: public functions must accept plain Python sequences.

pyrefly (strict) checks this file in CI; the tests also run the calls.
"""

from __future__ import annotations

import math

from obscure_stats.association import blomqvist_beta
from obscure_stats.central_tendency import midhinge
from obscure_stats.dispersion import standard_quantile_absolute_deviation
from obscure_stats.kurtosis import moors_octile_kurt
from obscure_stats.skewness import bowley_skew
from obscure_stats.variation import gibbs_m1


def test_numeric_functions_accept_sequences() -> None:
    """Test that lists and tuples type-check and run for numeric inputs."""
    values: list[float] = [1.0, 2.0, 4.0, 8.0, 16.0, 3.0, 5.0, 7.0]
    results: list[float] = [
        midhinge(values),
        standard_quantile_absolute_deviation(tuple(values)),
        moors_octile_kurt(values),
        bowley_skew([1, 2, 4, 8, 16, 3, 5, 7]),
        blomqvist_beta(values, values[::-1]),
    ]
    if not all(math.isfinite(r) for r in results):
        msg = f"Expected finite results, got {results}"
        raise ValueError(msg)


def test_categorical_functions_accept_sequences() -> None:
    """Test that a list of strings type-checks and runs for categorical inputs."""
    if not 0.0 < gibbs_m1(["a", "b", "a", "c"]) < 1.0:
        msg = "Expected Gibbs M1 in (0, 1) for mixed categories."
        raise ValueError(msg)
