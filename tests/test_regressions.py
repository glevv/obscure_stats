"""Regression tests pinning fixed bugs against independent reference formulas."""

from __future__ import annotations

import math
import typing

import numpy as np
import numpy.typing as npt
import pytest

from obscure_stats.association import concordance_correlation
from obscure_stats.central_tendency import (
    half_sample_mode,
    hodges_lehmann_sen_location,
    tau_location,
)
from obscure_stats.dispersion import (
    fisher_index_of_dispersion,
    gini_mean_difference,
    morisita_index_of_dispersion,
    shamos_estimator,
    standard_quantile_absolute_deviation,
)
from obscure_stats.skewness import (
    auc_skew_gamma,
    cumulative_skew,
    left_quantile_weight,
    right_quantile_weight,
)
from obscure_stats.variation import avdev, b_index, negative_extropy


def _check(condition: bool, msg: str = "Result does not match reference.") -> None:  # noqa: FBT001
    if not condition:
        raise ValueError(msg)


def _hsm_reference(x: npt.NDArray[np.float64]) -> float:
    """Straightforward port of R modeest::hsm (last-tie convention)."""
    y = np.sort(x)
    while len(y) >= 4:  # noqa: PLR2004
        k = math.ceil(len(y) / 2) - 1
        widths = [y[i + k] - y[i] for i in range(len(y) - k)]
        j = max(i for i, w in enumerate(widths) if w == min(widths))
        if widths[j] == 0:
            return float(y[j])
        y = y[j : j + k + 1]
    if len(y) == 3:  # noqa: PLR2004
        z = 2 * y[1] - y[0] - y[2]
        if z < 0:
            return float(y[:2].mean())
        if z > 0:
            return float(y[1:].mean())
        return float(y[1])
    return float(y.mean())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hsm_matches_reference(seed: int) -> None:
    """HSM must consider every window and keep k  1 points per step."""
    x = np.random.default_rng(seed).gamma(2.0, size=257)
    _check(half_sample_mode(x) == pytest.approx(_hsm_reference(x)))


def test_hsm_three_points() -> None:
    """Three-point case averages the two closest points."""
    _check(half_sample_mode(np.array([0.0, 1.0, 10.0])) == pytest.approx(0.5))
    _check(half_sample_mode(np.array([0.0, 9.0, 10.0])) == pytest.approx(9.5))


def test_hls_walsh_averages() -> None:
    """HL is the median of Walsh averages over i <= j, not the full matrix."""
    x = np.array([1.0, 2.0, 10.0])
    i, j = np.triu_indices(len(x))
    _check(
        hodges_lehmann_sen_location(x) == pytest.approx(np.median((x[i] + x[j]) / 2))
    )


def test_shamos_excludes_diagonal() -> None:
    """Shamos is the median over i < j; zero self-differences must be excluded."""
    x = np.random.default_rng(3).normal(size=40)
    i, j = np.triu_indices(len(x), k=1)
    _check(shamos_estimator(x) == pytest.approx(np.median(np.abs(x[i] - x[j]))))


def test_gini_mean_difference_reference() -> None:
    """O(N log N) GMD equals the pairwise definition."""
    x = np.random.default_rng(4).normal(size=60)
    n = len(x)
    ref = np.abs(x[:, None] - x[None, :]).sum() / (n * (n - 1))
    _check(gini_mean_difference(x) == pytest.approx(ref))


def test_fisher_index_formula() -> None:
    """Fisher's index is sum((x - mean)^2) / mean."""
    x = np.random.default_rng(5).poisson(5, size=30).astype(float)
    ref = np.sum((x - x.mean()) ** 2) / x.mean()
    _check(fisher_index_of_dispersion(x) == pytest.approx(ref))


@pytest.mark.parametrize(
    "func",
    [
        fisher_index_of_dispersion,
        morisita_index_of_dispersion,
        gini_mean_difference,
        standard_quantile_absolute_deviation,
        cumulative_skew,
    ],
)
def test_nans_do_not_change_result(func: typing.Callable[..., float]) -> None:
    """Sample size must count only non-NaN observations."""
    x = np.random.default_rng(6).poisson(5, size=30).astype(float) + 1
    _check(func(np.r_[x, [np.nan] * 10]) == pytest.approx(func(x)))


@pytest.mark.parametrize("shift", [0.0, 0.5, 3.0])
def test_concordance_correlation_reference(shift: float) -> None:
    """Lin's CCC = 2 cov / (var_x  var_y  (mean_x - mean_y)^2)."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=200)
    y = x + rng.normal(scale=0.5, size=200) + shift
    ref = (
        2
        * np.cov(x, y, ddof=0)[0, 1]
        / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    )
    _check(concordance_correlation(x, y) == pytest.approx(ref))


def test_uniform_categories_max_variation() -> None:
    """B index and AVDEV equal 1 for perfectly even categories."""
    u = np.array(list("aabbccdd"))
    _check(b_index(u) == pytest.approx(1.0))
    _check(avdev(u) == pytest.approx(1.0))


def test_negative_extropy_constant() -> None:
    """A single category must not produce NaN."""
    _check(negative_extropy(np.array(list("aaaa"))) == 0.0)


def test_quantile_weights_symmetric() -> None:
    """For symmetric data left and right tail weights agree in sign and size."""
    x = np.random.default_rng(8).normal(size=100_000)
    lqw, rqw = left_quantile_weight(x, 0.25), right_quantile_weight(x, 0.75)
    _check(rqw > 0)
    _check(lqw == pytest.approx(rqw, abs=0.02))


def test_auc_skew() -> None:
    """Symmetric data is ~0, right-skewed data clearly positive, and bounded."""
    rng = np.random.default_rng(9)
    _check(abs(auc_skew_gamma(rng.normal(size=20_000), dp=0.01)) < 0.05)  # noqa: PLR2004
    res = auc_skew_gamma(rng.exponential(size=20_000))
    _check(0.2 < res <= 1.0)  # noqa: PLR2004


def test_tau_location_zero_mad() -> None:
    """MAD == 0 must not produce NaN."""
    _check(tau_location(np.array([1.0, 1.0, 1.0, 1.0, 5.0])) == 1.0)
