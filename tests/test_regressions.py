"""Regression tests pinning fixed bugs against independent reference formulas."""

from __future__ import annotations

import math
import typing

import numpy as np
import numpy.typing as npt
import pytest

from obscure_stats.association import (
    concordance_correlation,
)
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
from obscure_stats.kurtosis import quantile_weight
from obscure_stats.skewness import (
    auc_skew_gamma,
    cumulative_skew,
)
from obscure_stats.variation import (
    avdev,
    b_index,
    extropy,
)


def _check(condition: bool, msg: str = "Result does not match reference.") -> None:  # noqa: FBT001
    """Raise ValueError with msg if the condition is false."""
    if not condition:
        raise ValueError(msg)


def _hsm_reference(x: npt.NDArray[np.number]) -> float:
    """Calculate HSM. Port of R modeest::hsm (last-tie convention)."""
    y = np.sort(x)
    while len(y) >= 4:
        k = math.ceil(len(y) / 2) - 1
        widths = [y[i + k] - y[i] for i in range(len(y) - k)]
        j = max(i for i, w in enumerate(widths) if w == min(widths))
        if widths[j] == 0:
            return float(y[j])
        y = y[j : j + k + 1]
    if len(y) == 3:
        z = 2 * y[1] - y[0] - y[2]
        if z < 0:
            return float(y[:2].mean())
        if z > 0:
            return float(y[1:].mean())
        return float(y[1])
    return float(y.mean())


@pytest.mark.parametrize("data", ["gamma_data", "exponential_data", "hsm_test_data"])
def test_hsm_matches_reference(data: str, request: pytest.FixtureRequest) -> None:
    """Check if HSM consider every window and keep k  1 points per step."""
    x = np.asarray(request.getfixturevalue(data), dtype=np.float64)
    _check(half_sample_mode(x) == pytest.approx(_hsm_reference(x)))


def test_hsm_three_points(three_point_data: npt.NDArray[np.number]) -> None:
    """Check three points with the closest pair on the left (others: test_hsm)."""
    _check(half_sample_mode(three_point_data) == pytest.approx(1.5))


def test_hls_walsh_averages(
    three_point_data: npt.NDArray[np.number], all_nan_array: npt.NDArray[np.number]
) -> None:
    """Test whether Walsh sums are computed over upper triangle, not the full matrix."""
    i, j = np.triu_indices(len(three_point_data))
    walsh = (three_point_data[i] + three_point_data[j]) / 2
    _check(
        hodges_lehmann_sen_location(three_point_data) == pytest.approx(np.median(walsh))
    )
    _check(np.isnan(hodges_lehmann_sen_location(all_nan_array)))


def test_shamos_excludes_diagonal(normal_data: npt.NDArray[np.number]) -> None:
    """Test whether zero self-differences are excluded."""
    i, j = np.triu_indices(len(normal_data), k=1)
    _check(
        shamos_estimator(normal_data)
        == pytest.approx(np.median(np.abs(normal_data[i] - normal_data[j])))
    )


def test_gini_mean_difference_reference(normal_data: npt.NDArray[np.number]) -> None:
    """Test whether O(N log N) GMD equals the pairwise definition."""
    n = len(normal_data)
    ref = np.abs(normal_data[:, None] - normal_data[None, :]).sum() / (n * (n - 1))
    _check(gini_mean_difference(normal_data) == pytest.approx(ref))


def test_fisher_index_formula(poisson_data: npt.NDArray[np.number]) -> None:
    """Test whether Fisher's index is sum((x - mean)^2) / mean."""
    ref = np.sum((poisson_data - poisson_data.mean()) ** 2) / poisson_data.mean()
    _check(fisher_index_of_dispersion(poisson_data) == pytest.approx(ref))


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
def test_nans_do_not_change_result(
    func: typing.Callable[..., float], poisson_data: npt.NDArray[np.number]
) -> None:
    """Check if sample size must count only non-NaN observations."""
    x = poisson_data + 1
    _check(func(np.r_[x, [np.nan] * 10]) == pytest.approx(func(x)))


@pytest.mark.parametrize("shift", [0.0, 0.5, 3.0])
def test_concordance_correlation_reference(
    shift: float,
    normal_data: npt.NDArray[np.number],
    exponential_data: npt.NDArray[np.number],
) -> None:
    """Test whether Lin's CCC = 2 cov / (var_x  var_y  (mean_x - mean_y)^2)."""
    y = normal_data + 0.5 * exponential_data + shift
    ref = (
        2
        * np.cov(normal_data, y, ddof=0)[0, 1]
        / (normal_data.var() + y.var() + (normal_data.mean() - y.mean()) ** 2)
    )
    _check(concordance_correlation(normal_data, y) == pytest.approx(ref))


def test_uniform_categories_max_variation(
    even_categories: npt.NDArray[np.str_],
) -> None:
    """Test whether B index and AVDEV equal 1 for perfectly even categories."""
    _check(b_index(even_categories) == pytest.approx(1.0))
    _check(avdev(even_categories) == pytest.approx(1.0))


def test_extropy_constant(c_array_obj: npt.NDArray[np.str_]) -> None:
    """Check that a single category must not produce NaN."""
    _check(extropy(c_array_obj[:1]) == 0.0)


def test_quantile_weights_symmetric(normal_data: npt.NDArray[np.number]) -> None:
    """Test if symmetric data left and right tail weights agree in sign and size."""
    lqw, rqw = quantile_weight(normal_data, 0.25), quantile_weight(normal_data, 0.75)
    _check(rqw > 0)
    _check(lqw == pytest.approx(rqw, abs=0.02))


def test_auc_skew(
    normal_data: npt.NDArray[np.number], exponential_data: npt.NDArray[np.number]
) -> None:
    """Check that symmetric data is ~0, right-skewed data positive and bounded."""
    _check(abs(auc_skew_gamma(normal_data, dp=0.01)) < 0.05)
    res = auc_skew_gamma(exponential_data)
    _check(0.2 < res <= 1.0)


def test_tau_location_zero_mad(const_array_float: npt.NDArray[np.number]) -> None:
    """Test whether MAD == 0 must not produce NaN."""
    x = const_array_float.copy()
    x[0] = 5.0  # most values are equal, so the MAD is 0
    _check(tau_location(x) == 1.0)


def test_cum_skew_all_nans(all_nan_array: npt.NDArray[np.number]) -> None:
    """Test whether cumulative skew would return nan for nan array."""
    _check(np.isnan(cumulative_skew(all_nan_array)))


def test_hsm_all_nans(all_nan_array: npt.NDArray[np.number]) -> None:
    """Test whether HSM would return nan for nan array."""
    _check(np.isnan(half_sample_mode(all_nan_array)))
