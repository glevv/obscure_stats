"""Collection of tests of kurtosis module."""

import math
import typing

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from obscure_stats.kurtosis import (
    crow_siddiqui_kurt,
    hogg_kurt,
    moors_octile_kurt,
    quantile_weight,
    reza_ma_kurt,
    schmid_trede_peakedness,
    staudte_kurt,
)

all_functions = [
    quantile_weight,
    crow_siddiqui_kurt,
    hogg_kurt,
    moors_octile_kurt,
    reza_ma_kurt,
    schmid_trede_peakedness,
    staudte_kurt,
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
def test_kurt_sensibility(
    func: typing.Callable[..., float],
    uniform_data: npt.NDArray[np.number],
    laplace_data: npt.NDArray[np.number],
) -> None:
    """Test for result correctness."""
    platy_res = func(uniform_data)
    lepto_res = func(laplace_data)
    if platy_res > lepto_res:
        msg = (
            f"Kurtosis in the first case should be lower, got {platy_res} > {lepto_res}"
        )
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_statistic_with_nans(
    func: typing.Callable[..., float], x_array_nan: npt.NDArray[np.number]
) -> None:
    """Test for different data types."""
    if math.isnan(func(x_array_nan)):
        msg = "Statistic should not return nans."
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_invariance_add(
    func: typing.Callable[..., float], x_array_float: npt.NDArray[np.number]
) -> None:
    """Test coefficients for invariance to addition."""
    res1 = func(x_array_float)
    res2 = func(x_array_float + 10)
    if res1 != pytest.approx(res2):
        msg = f"Statistic should be invariant to addition, got {res1} and {res2}."
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


@pytest.mark.parametrize(
    "func",
    # quantile_weight measures one tail, so reflecting the data swaps the tails
    [f for f in all_functions if f is not quantile_weight],
)
def test_change_sign(
    func: typing.Callable[..., float], x_array_float: npt.NDArray[np.number]
) -> None:
    """Test change of sign of statistic if array changed sign."""
    res1 = func(x_array_float)
    res2 = func(-x_array_float)
    if res1 != pytest.approx(res2):
        msg = f"Statistic should change sign, got {res1} and {res2}."
        raise ValueError(msg)


@given(
    arrays(
        dtype=np.float64,
        shape=array_shapes(max_dims=1),
        elements=st.floats(allow_nan=True, allow_infinity=True),
    )
)
@pytest.mark.parametrize("func", all_functions)
def test_fuzz_kurtosises(
    func: typing.Callable[..., float], data: npt.NDArray[np.number]
) -> None:
    """Test all functions with fuzz."""
    func(data)


def test_quantile_weight_mirror(x_array_float: npt.NDArray[np.number]) -> None:
    """Test that the left tail of x is the right tail of the reflected data."""
    left = quantile_weight(x_array_float, q=0.25)
    if left != pytest.approx(quantile_weight(-x_array_float, q=0.75)):
        msg = "Left quantile weight of x should equal the right one of -x."
        raise ValueError(msg)


def test_quantile_weight_left_tail_sensibility(
    uniform_data: npt.NDArray[np.number], exponential_data: npt.NDArray[np.number]
) -> None:
    """Test that a heavy left tail (reflected exponential) gives a higher LQW."""
    no_tail_res = quantile_weight(uniform_data, q=0.25)
    left_tail_res = quantile_weight(-exponential_data, q=0.25)
    if no_tail_res > left_tail_res:
        msg = (
            f"Left tail weight in the first case should be lower, "
            f"got {no_tail_res} > {left_tail_res}"
        )
        raise ValueError(msg)


@pytest.mark.parametrize("q", [0.0, 0.5, 1.0])
def test_q_in_quantile_weight(x_array_float: npt.NDArray[np.number], q: float) -> None:
    """Test for correctness of q parameter value."""
    with pytest.raises(ValueError, match="Parameter q should be in range"):
        quantile_weight(x_array_float, q=q)
