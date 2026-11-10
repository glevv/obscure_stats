"""Collection of tests of skewness module."""

import math
import typing

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from obscure_stats.central_tendency import (
    half_sample_mode,
)
from obscure_stats.skewness import (
    auc_skew_gamma,
    bickel_mode_skew,
    bowley_skew,
    cumulative_skew,
    forhad_shorna_rank_skew,
    groeneveld_range_skew,
    kelly_skew,
    meeden_skew,
    pearson_median_skew,
    pearson_mode_skew,
)

all_functions = [
    auc_skew_gamma,
    bickel_mode_skew,
    bowley_skew,
    cumulative_skew,
    forhad_shorna_rank_skew,
    groeneveld_range_skew,
    kelly_skew,
    meeden_skew,
    pearson_median_skew,
    pearson_mode_skew,
]


@pytest.mark.parametrize("func", all_functions)
@pytest.mark.parametrize(
    "data", ["x_list_float", "x_list_int", "x_array_int", "x_array_float"]
)
def test_mock_aggregation_functions(
    func: typing.Callable[..., float], data: str, request: pytest.FixtureRequest
) -> None:
    """Test for different data types."""
    data = request.getfixturevalue(data)
    func(data)


@pytest.mark.parametrize("func", all_functions)
def test_skew_sensibility(
    func: typing.Callable[..., float],
    uniform_data: npt.NDArray[np.number],
    exponential_data: npt.NDArray[np.number],
) -> None:
    """Test for result correctness."""
    no_skew_res = func(uniform_data)
    left_skew_res = func(exponential_data)
    if no_skew_res > left_skew_res:
        msg = (
            f"Skewness in the first case should be lower, "
            f"got {no_skew_res} > {left_skew_res}"
        )
        raise ValueError(msg)


def test_rank_skew(rank_skewness_test_data: npt.NDArray[np.number]) -> None:
    """Test for correctness of Rank skewness coefficient."""
    res = forhad_shorna_rank_skew(rank_skewness_test_data)
    if res != pytest.approx(0.93809, rel=1e-2):
        msg = f"Results from the test and paper do not match, got {res} instead of 0.93809."
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_statistic_with_nans(
    func: typing.Callable[..., float], x_array_nan: npt.NDArray[np.number]
) -> None:
    """Test for different data types."""
    if math.isnan(func(x_array_nan)):
        msg = "Statistic should not return nans."
        raise ValueError(msg)


@pytest.mark.parametrize("dp", [0, 0.3])
def test_dp_in_auc_skew(x_array_float: npt.NDArray[np.number], dp: float) -> None:
    """Test for correctness of dp parameter value."""
    with pytest.raises(ValueError, match="Parameter dp should be in range"):
        auc_skew_gamma(x_array_float, dp=dp)


@pytest.mark.parametrize(
    "func",
    [
        auc_skew_gamma,
        bickel_mode_skew,
        bowley_skew,
        cumulative_skew,
        forhad_shorna_rank_skew,
        groeneveld_range_skew,
        kelly_skew,
        meeden_skew,
        pearson_median_skew,
        pearson_mode_skew,
    ],
)
def test_change_sign(
    func: typing.Callable[..., float], x_array_float: npt.NDArray[np.number]
) -> None:
    """Test change of sign of statistic if array changed sign."""
    res1 = -func(x_array_float)
    res2 = func(-x_array_float)
    if res1 != pytest.approx(res2):
        msg = f"Statistic should change sign, got {res1} and {res2}."
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_invariance_mult(
    func: typing.Callable[..., float], x_array_float: npt.NDArray[np.number]
) -> None:
    """Test coefficients for invariance to multiplication."""
    res1 = func(x_array_float)
    res2 = func(x_array_float * 10)
    if res1 != pytest.approx(res2):
        msg = f"Statistic should be invariant to multiplication, got {res1} and {res2}."
        raise ValueError(msg)


@given(
    arrays(
        dtype=np.float64,
        shape=array_shapes(max_dims=1),
        elements=st.floats(allow_nan=True, allow_infinity=True),
    )
)
@pytest.mark.parametrize("func", all_functions)
def test_fuzz_skewnesses(
    func: typing.Callable[..., float], data: npt.NDArray[np.number]
) -> None:
    """Test all functions with fuzz."""
    func(data)


def test_auc_skew_matches_paper_estimator(
    rank_skewness_test_data: npt.NDArray[np.number],
) -> None:
    """Test against eq. 8 of Arachchige & Prendergast (mean over a midpoint grid)."""
    j = 100
    ps = 0.5 * (np.arange(1, j + 1) - 0.5) / j
    lo = np.quantile(rank_skewness_test_data, ps, method="median_unbiased")
    hi = np.quantile(rank_skewness_test_data, 1 - ps, method="median_unbiased")
    med = np.quantile(rank_skewness_test_data, 0.5, method="median_unbiased")
    ref = np.mean((hi + lo - 2 * med) / (hi - lo))
    if auc_skew_gamma(rank_skewness_test_data, dp=0.005) != pytest.approx(ref):
        msg = "Result does not match reference."
        raise ValueError(msg)


def test_pearson_mode_skew_uses_half_sample_mode(
    rank_skewness_test_data: npt.NDArray[np.number],
) -> None:
    """Test that continuous data uses the half sample mode, not the minimum."""
    ref = (
        rank_skewness_test_data.mean() - half_sample_mode(rank_skewness_test_data)
    ) / rank_skewness_test_data.std()
    if pearson_mode_skew(rank_skewness_test_data) != pytest.approx(ref):
        msg = "Result does not match reference."
        raise ValueError(msg)
    if pearson_mode_skew(rank_skewness_test_data) == pytest.approx(
        (rank_skewness_test_data.mean() - rank_skewness_test_data.min())
        / rank_skewness_test_data.std()
    ):
        msg = "The mode must not be the minimum."
        raise ValueError(msg)


def test_pearson_mode_skew_discrete_mode(hsm_test_data: npt.NDArray[np.number]) -> None:
    """Test that a dominant repeated value (2) is used as the mode."""
    if pearson_mode_skew(hsm_test_data) != pytest.approx(
        (hsm_test_data.mean() - 2.0) / hsm_test_data.std()
    ):
        msg = "Result does not match reference."
        raise ValueError(msg)
