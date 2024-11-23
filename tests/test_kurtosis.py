"""Collection of tests of kurtosis module."""

import math
import typing

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from obscure_stats.kurtosis import (
    crow_siddiqui_kurt,
    hogg_kurt,
    l_kurt,
    left_quantile_weight,
    moors_kurt,
    moors_octile_kurt,
    reza_ma_kurt,
    right_quantile_weight,
    schmid_trede_peakedness,
    staudte_kurt,
)

all_functions = [
    crow_siddiqui_kurt,
    hogg_kurt,
    l_kurt,
    left_quantile_weight,
    moors_kurt,
    moors_octile_kurt,
    reza_ma_kurt,
    right_quantile_weight,
    schmid_trede_peakedness,
    staudte_kurt,
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


@pytest.mark.parametrize("func", all_functions)
@pytest.mark.parametrize("seed", [1, 42, 99])
def test_kurt_sensibility(func: typing.Callable, seed: int) -> None:
    """Testing for result correctness."""
    rng = np.random.default_rng(seed)
    platy = rng.uniform(size=99)
    lepto = rng.laplace(size=99)
    platy_res = func(platy)
    lepto_res = func(lepto)
    if func.__name__ == "right_quantile_weight":
        # ugly but more harmonized this way
        platy_res = -platy_res
        lepto_res = -lepto_res
    if platy_res > lepto_res:
        msg = (
            f"Kurtosis in the first case should be lower, got {platy_res} > {lepto_res}"
        )
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_statistic_with_nans(func: typing.Callable, x_array_nan: np.ndarray) -> None:
    """Test for different data types."""
    if math.isnan(func(x_array_nan)):
        msg = "Statistic should not return nans."
        raise ValueError(msg)


@pytest.mark.parametrize("func", [right_quantile_weight, left_quantile_weight])
@pytest.mark.parametrize("q", [0.0, 1.0])
def test_q_in_qw(x_array_float: np.ndarray, func: typing.Callable, q: float) -> None:
    """Simple tets case for correctnes of q."""
    with pytest.raises(ValueError, match="Parameter q should be in range"):
        func(x_array_float, q=q)


@given(
    arrays(
        dtype=np.float64,
        shape=array_shapes(max_dims=1),
        elements=st.floats(allow_nan=True, allow_infinity=True),
    )
)
@pytest.mark.parametrize("func", all_functions)
def test_fuzz_kurtosises(func: typing.Callable, data: np.ndarray) -> None:
    """Test all functions with fuzz."""
    func(data)
