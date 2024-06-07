"""Collection of tests of kurtosis module."""

import typing

import numpy as np
import pytest
from hypothesis import given  # type: ignore[import-not-found]
from hypothesis import strategies as st  # type: ignore[import-not-found]
from hypothesis.extra.numpy import (  # type: ignore[import-not-found]
    array_shapes,
    arrays,
)
from obscure_stats.kurtosis import (
    crow_siddiqui_kurt,
    hogg_kurt,
    l_kurt,
    moors_kurt,
    moors_octile_kurt,
    reza_ma_kurt,
)

all_functions = [
    crow_siddiqui_kurt,
    hogg_kurt,
    l_kurt,
    moors_kurt,
    moors_octile_kurt,
    reza_ma_kurt,
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
    platy = np.round(rng.uniform(size=100), 2)
    lepto = np.round(rng.exponential(size=100), 2)
    platy_res = func(platy)
    lepto_res = func(lepto)
    if platy_res > lepto_res:
        msg = (
            f"Kurtosis in the first case should be lower, got {platy_res} > {lepto_res}"
        )
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_statistic_with_nans(func: typing.Callable, x_array_nan: np.ndarray) -> None:
    """Test for different data types."""
    if np.isnan(func(x_array_nan)):
        msg = "Statistic should not return nans."
        raise ValueError(msg)


@given(
    arrays(
        dtype=np.float64,
        shape=array_shapes(),
        elements=st.floats(allow_nan=True, allow_infinity=True),
    )
)
@pytest.mark.parametrize("func", all_functions)
def test_fuzz_all(func: typing.Callable, data: np.ndarray) -> None:
    """Test all functions with fuzz."""
    func(data)
