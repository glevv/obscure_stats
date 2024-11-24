"""Collection of tests of dispersion module."""

import math
import typing

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from obscure_stats.dispersion import (
    coefficient_of_lvariation,
    coefficient_of_range,
    coefficient_of_variation,
    cole_index_of_dispersion,
    dispersion_ratio,
    fisher_index_of_dispersion,
    gini_mean_difference,
    inter_expectile_range,
    morisita_index_of_dispersion,
    quartile_coefficient_of_dispersion,
    robust_coefficient_of_variation,
    shamos_estimator,
    standard_quantile_absolute_deviation,
    studentized_range,
)

all_functions = [
    coefficient_of_lvariation,
    coefficient_of_range,
    coefficient_of_variation,
    cole_index_of_dispersion,
    dispersion_ratio,
    fisher_index_of_dispersion,
    gini_mean_difference,
    inter_expectile_range,
    morisita_index_of_dispersion,
    quartile_coefficient_of_dispersion,
    robust_coefficient_of_variation,
    shamos_estimator,
    standard_quantile_absolute_deviation,
    studentized_range,
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


@pytest.mark.parametrize(
    "func",
    [
        coefficient_of_lvariation,
        coefficient_of_variation,
        cole_index_of_dispersion,
        robust_coefficient_of_variation,
        dispersion_ratio,
        fisher_index_of_dispersion,
        gini_mean_difference,
        inter_expectile_range,
        morisita_index_of_dispersion,
        quartile_coefficient_of_dispersion,
        standard_quantile_absolute_deviation,
        shamos_estimator,
        coefficient_of_range,
    ],
)
@pytest.mark.parametrize("seed", [1, 42, 99])
def test_dispersion_sensibility(func: typing.Callable, seed: int) -> None:
    """Testing for result correctness."""
    rng = np.random.default_rng(seed)
    low_disp = np.round(rng.exponential(scale=1, size=99) + 1, 2)
    high_disp = np.round(rng.exponential(scale=10, size=99) + 1, 2)
    low_disp_res = func(low_disp)
    high_disp_res = func(high_disp)
    if low_disp_res > high_disp_res:
        msg = (
            f"Dispersion in the first case should be lower, "
            f"got {low_disp_res} > {high_disp_res}"
        )
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_statistic_with_nans(func: typing.Callable, x_array_nan: np.ndarray) -> None:
    """Test for different data types."""
    if math.isnan(func(x_array_nan)):
        msg = "Statistic should not return nans."
        raise ValueError(msg)


@given(
    arrays(
        dtype=np.float64,
        shape=array_shapes(max_dims=1),
        elements=st.floats(allow_nan=True, allow_infinity=True),
    )
)
@pytest.mark.parametrize("func", all_functions)
def test_fuzz_dispersions(func: typing.Callable, data: np.ndarray) -> None:
    """Test all functions with fuzz."""
    func(data)
