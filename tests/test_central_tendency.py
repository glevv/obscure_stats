"""Collection of tests of central tendency module."""

from __future__ import annotations

import math
import typing

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

# pyrefly: ignore [untyped-import]
from scipy import stats

from obscure_stats.central_tendency import (
    contraharmonic_mean,
    gastwirth_location,
    grenanders_m,
    half_sample_mode,
    hodges_lehmann_sen_location,
    midhinge,
    midmean,
    midrange,
    standard_trimmed_harrell_davis_quantile,
    tau_location,
    trimean,
)
from obscure_stats.central_tendency.central_tendency import _beta_hdi

all_functions = [
    contraharmonic_mean,
    gastwirth_location,
    grenanders_m,
    half_sample_mode,
    hodges_lehmann_sen_location,
    midhinge,
    midmean,
    midrange,
    standard_trimmed_harrell_davis_quantile,
    tau_location,
    trimean,
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


def test_thdm(thdme_test_data: npt.NDArray[np.number]) -> None:
    """Test for correctness of Trimmed Harrel-Davis median."""
    result = standard_trimmed_harrell_davis_quantile(thdme_test_data)
    if result != pytest.approx(0.6268, rel=1e-4):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)
    if result > standard_trimmed_harrell_davis_quantile(thdme_test_data, q=0.99):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)
    if result < standard_trimmed_harrell_davis_quantile(thdme_test_data, q=0.01):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)


def test_edge_cases(x_array_float: npt.NDArray[np.number]) -> None:
    """Test for edge cases."""
    result = standard_trimmed_harrell_davis_quantile(x_array_float[:1])
    if result != pytest.approx(x_array_float[0], rel=1e-4):
        msg = "Result does not match expected output."
        raise ValueError(msg)
    result = standard_trimmed_harrell_davis_quantile(x_array_float[:0])
    if not math.isnan(result):
        msg = "Result does not match expected output."
        raise ValueError(msg)


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_q_in_sthdq(x_array_float: npt.NDArray[np.number], q: float) -> None:
    """Test for correctness of q parameter value."""
    with pytest.raises(ValueError, match="Parameter q should be in range"):
        standard_trimmed_harrell_davis_quantile(x_array_float, q=q)


def test_hls(
    hls_test_data: npt.NDArray[np.number], hls_test_data_big: list[int]
) -> None:
    """Test for correctness of Hodges-Lehmann-Sen."""
    result = hodges_lehmann_sen_location(hls_test_data)
    if result != pytest.approx(3.5):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)
    # pyrefly: ignore [bad-argument-type]
    result = hodges_lehmann_sen_location(hls_test_data_big)
    if result != pytest.approx(5.75):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)


def test_hsm(hsm_test_data: npt.NDArray[np.number]) -> None:
    """Test for correctness of Half Sample Mode."""
    result = half_sample_mode(hsm_test_data)
    if result != pytest.approx(2.0):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)
    result = half_sample_mode(hsm_test_data[:3])
    if result != pytest.approx(2):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)
    result = half_sample_mode(hsm_test_data[1:4])
    if result != pytest.approx(2):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)
    result = half_sample_mode(hsm_test_data[2:5])
    if result != pytest.approx(2):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_statistic_with_nans(
    func: typing.Callable[..., float], x_array_nan: npt.NDArray[np.number]
) -> None:
    """Test for different data types."""
    if math.isnan(func(x_array_nan)):
        msg = "Statistic should not return nans."
        raise ValueError(msg)


@pytest.mark.parametrize("c", [0.0, -1.0])
def test_tau_location(x_array_float: npt.NDArray[np.number], c: float) -> None:
    """Test that function will raise error if c parameter is incorrect."""
    with pytest.raises(ValueError, match="Parameter c should be strictly"):
        tau_location(x_array_float, c=c)


def test_grenaders_m(x_array_float: npt.NDArray[np.number]) -> None:
    """Test that function will raise error if p or k parameters are incorrect."""
    with pytest.raises(ValueError, match="Parameter p should be a float"):
        grenanders_m(x_array_float, p=1)
    with pytest.raises(ValueError, match="Parameter k should be an integer"):
        grenanders_m(x_array_float, k=0)
    with pytest.raises(ValueError, match="Parameter k should be an integer"):
        # pyrefly: ignore [bad-argument-type]
        grenanders_m(x_array_float, k=1.5)
    if not math.isnan(grenanders_m(x_array_float, k=len(x_array_float))):
        msg = "Statistic should return nans."
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
@pytest.mark.parametrize("seed", [1, 42, 99])
def test_sensibility(func: typing.Callable[..., float], seed: int) -> None:
    """Test central tendencies convergence for normal distribution."""
    rng = np.random.default_rng(seed)
    res = func(rng.normal(loc=10.0, size=100))
    if res != pytest.approx(10.0, abs=0.75):
        msg = f"Central tendency for N(10, 1) should be close to 10.0, got {res}."
        raise ValueError(msg)


def _best_mass(a: float, b: float, width: float, grid: int = 20_001) -> float:
    """Find largest probability mass of any interval [l, l + width] by brute force."""
    left = np.linspace(0.0, max(0.0, 1.0 - width), grid)
    dist = stats.beta(a, b)
    return float(np.max(dist.cdf(np.minimum(left + width, 1.0)) - dist.cdf(left)))


BRANCH_CASES = [
    pytest.param(3.0, 4.0, 1.0, id="width>=1"),
    pytest.param(0.5, 0.5, 0.2, id="u-shaped-symmetric"),
    pytest.param(0.3, 0.8, 0.2, id="u-shaped-left-heavy"),
    pytest.param(0.8, 0.3, 0.2, id="u-shaped-right-heavy"),
    pytest.param(1.0, 1.0, 0.3, id="uniform"),
    pytest.param(0.7, 5.0, 0.1, id="decreasing"),
    pytest.param(5.0, 0.7, 0.1, id="increasing"),
    pytest.param(5.5, 5.5, 0.3, id="middle-symmetric"),
    pytest.param(2.0, 30.0, 0.2, id="middle-mode-near-0"),
    pytest.param(30.0, 2.0, 0.2, id="middle-mode-near-1"),
    pytest.param(1001.0 * 0.9, 1001.0 * 0.1, 1000**-0.5, id="middle-peaked"),
]


@pytest.mark.parametrize(("a", "b", "width"), BRANCH_CASES)
def test_beta_hdi_is_highest_density(a: float, b: float, width: float) -> None:
    """Test if returned interval matches the width, lies in [0, 1], and has max mass."""
    lo, hi = _beta_hdi(a, b, width)
    if not 0.0 <= lo <= hi <= 1.0:
        msg = f"Interval ({lo}, {hi}) is outside [0, 1]."
        raise ValueError(msg)
    if hi - lo != pytest.approx(min(width, 1.0)):
        msg = f"Interval width {hi - lo} != {width}."
        raise ValueError(msg)
    mass = stats.beta(a, b).cdf(hi) - stats.beta(a, b).cdf(lo)
    if mass < _best_mass(a, b, width) - 1e-6:
        msg = f"Interval mass {mass} is below the brute-force maximum."
        raise ValueError(msg)


def test_beta_hdi_middle_equal_density_endpoints() -> None:
    """Test unimodal case for HDI endpoints."""
    a, b, width = 4.0, 9.0, 0.25
    lo, hi = _beta_hdi(a, b, width)
    pdf = stats.beta(a, b).pdf  # pyrefly: ignore [missing-attribute]
    if pdf(lo) != pytest.approx(pdf(hi), rel=1e-6):
        msg = "HDI endpoints should have equal density."
        raise ValueError(msg)


@given(
    arrays(
        dtype=np.float64,
        shape=array_shapes(max_dims=1),
        elements=st.floats(allow_nan=True, allow_infinity=True),
    )
)
@pytest.mark.parametrize("func", all_functions)
def test_fuzz_central_tendencies(
    func: typing.Callable[..., float], data: npt.NDArray[np.number]
) -> None:
    """Test all functions with fuzz."""
    func(data)
