"""Collection of tests of dispersion."""

import typing

import numpy as np
import pytest

from obscure_stats.dispersion import (
    coefficient_of_lvariation,
    coefficient_of_variation,
    dispersion_ratio,
    efficiency,
    hoover_index,
    kirkwood_coefficient_of_variation,
    lloyds_index,
    morisita_index,
    quartile_coef_of_dispersion,
    sqad,
    studentized_range,
)


@pytest.mark.parametrize(
    "func",
    (
        coefficient_of_lvariation,
        coefficient_of_variation,
        kirkwood_coefficient_of_variation,
        dispersion_ratio,
        efficiency,
        hoover_index,
        lloyds_index,
        morisita_index,
        quartile_coef_of_dispersion,
        sqad,
        studentized_range,
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
        coefficient_of_lvariation,
        coefficient_of_variation,
        kirkwood_coefficient_of_variation,
        dispersion_ratio,
        efficiency,
        hoover_index,
        lloyds_index,
        morisita_index,
        quartile_coef_of_dispersion,
        sqad,
    ),
)
@pytest.mark.parametrize("seed", (1, 42, 99))
def test_dispersion_sensibility(func: typing.Callable, seed: int):
    """Testing for result correctness."""
    rng = np.random.default_rng(seed)
    low_disp = np.round(rng.exponential(scale=1, size=100) + 1, 2)
    high_disp = np.round(rng.exponential(scale=10, size=100) + 1, 2)
    assert func(low_disp) <= func(high_disp)
