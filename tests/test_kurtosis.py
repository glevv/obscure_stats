"""Collection of tests of kurtosis."""

import typing

import numpy as np
import pytest

from obscure_stats.kurtosis import (
    crow_siddiqui_kurt,
    hogg_kurt,
    moors_kurt,
    moors_octile_kurt,
    reza_ma_kurt,
)


@pytest.mark.parametrize(
    "func",
    (
        moors_kurt,
        moors_octile_kurt,
        hogg_kurt,
        crow_siddiqui_kurt,
        reza_ma_kurt,
    ),
)
@pytest.mark.parametrize(
    "data", ("x_list_float", "x_list_int", "x_array_int", "x_array_float")
)
def test_mock_aggregation_functions(
    request, func: typing.Callable, data: typing.Iterable
) -> None:
    """Test for different data types."""
    data = request.getfixturevalue(data)
    func(data)


@pytest.mark.parametrize(
    "func",
    (
        moors_kurt,
        moors_octile_kurt,
        hogg_kurt,
        crow_siddiqui_kurt,
        reza_ma_kurt,
    ),
)
@pytest.mark.parametrize("seed", (1, 42, 99))
def test_kurt_sensibility(func: typing.Callable, seed: int):
    """Testing for result correctness."""
    rng = np.random.default_rng(seed)
    platy = np.round(rng.uniform(size=100), 2)
    lepto = np.round(rng.exponential(size=100), 2)
    assert func(platy) <= func(lepto)
