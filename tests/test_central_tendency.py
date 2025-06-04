"""Collection of tests of central tendecy module."""

from __future__ import annotations

import math
import typing

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

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
    func: typing.Callable, data: str, request: pytest.FixtureRequest
) -> None:
    """Test for different data types."""
    data = request.getfixturevalue(data)
    func(data)


def test_thdm(thdme_test_data: npt.NDArray) -> None:
    """Test the case for correctness of Trimmed Harrel Davis median."""
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


def test_edge_cases(x_array_float: npt.NDArray) -> None:
    """Simple tets case for edge cases."""
    result = standard_trimmed_harrell_davis_quantile(x_array_float[:1])
    if result != pytest.approx(x_array_float[0], rel=1e-4):
        msg = "Result does not match expected output."
        raise ValueError(msg)
    result = standard_trimmed_harrell_davis_quantile(x_array_float[:0])
    if not math.isnan(result):
        msg = "Result does not match expected output."
        raise ValueError(msg)


@pytest.mark.parametrize("q", [0.0, 1.0])
def test_q_in_sthdq(x_array_float: npt.NDArray, q: float) -> None:
    """Simple tets case for correctnes of q."""
    with pytest.raises(ValueError, match="Parameter q should be in range"):
        standard_trimmed_harrell_davis_quantile(x_array_float, q=q)


def test_hls(hls_test_data: npt.NDArray, hls_test_data_big: list[int]) -> None:
    """Simple tets case for correctness of Hodges-Lehmann-Sen."""
    result = hodges_lehmann_sen_location(hls_test_data)
    if result != pytest.approx(3.5):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)
    result = hodges_lehmann_sen_location(hls_test_data_big)  # type: ignore[arg-type]
    if result != pytest.approx(5.75):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)


def test_hsm(hsm_test_data: npt.NDArray) -> None:
    """Simple tets case for correctness of Half Sample Mode."""
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
def test_statistic_with_nans(func: typing.Callable, x_array_nan: npt.NDArray) -> None:
    """Test for different data types."""
    if math.isnan(func(x_array_nan)):
        msg = "Statistic should not return nans."
        raise ValueError(msg)


@pytest.mark.parametrize("c", [0.0, -1.0])
def test_tau_location(x_array_float: npt.NDArray, c: float) -> None:
    """Test that function will raise error if c parameter is incorrect."""
    with pytest.raises(ValueError, match="Parameter c should be strictly"):
        tau_location(x_array_float, c=c)


def test_grenaders_m(x_array_float: npt.NDArray) -> None:
    """Test that function will raise error if p or k parameters are incorrect."""
    with pytest.raises(ValueError, match="Parameter p should be a float"):
        grenanders_m(x_array_float, p=1)
    with pytest.raises(ValueError, match="Parameter k should be an integer"):
        grenanders_m(x_array_float, k=0)
    with pytest.raises(ValueError, match="Parameter k should be an integer"):
        grenanders_m(x_array_float, k=1.5)  # type: ignore[arg-type]
    if not math.isnan(grenanders_m(x_array_float, k=len(x_array_float))):
        msg = "Statistic should return nans."
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
@pytest.mark.parametrize("seed", [1, 42, 99])
def test_sensibility(func: typing.Callable, seed: int) -> None:
    """Test central tendencies convergence for normal distribution."""
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=100)
    res = func(normal)
    if res == pytest.approx(0.0):
        msg = f"Cental tendency for normal distribution should be 0.0, got {res}."
        raise ValueError(msg)


@given(
    arrays(
        dtype=np.float64,
        shape=array_shapes(max_dims=1),
        elements=st.floats(allow_nan=True, allow_infinity=True),
    )
)
@pytest.mark.parametrize("func", all_functions)
def test_fuzz_central_tendencies(func: typing.Callable, data: npt.NDArray) -> None:
    """Test all functions with fuzz."""
    func(data)
