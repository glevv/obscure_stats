"""Collection of tests of dispersion."""

import typing

import numpy as np
import pytest
from obscure_stats.dispersion import (
    coefficient_of_lvariation,
    coefficient_of_range,
    coefficient_of_variation,
    cole_index_of_dispersion,
    dispersion_ratio,
    fisher_index_of_dispersion,
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
    morisita_index_of_dispersion,
    quartile_coefficient_of_dispersion,
    robust_coefficient_of_variation,
    shamos_estimator,
    standard_quantile_absolute_deviation,
    studentized_range,
]


@pytest.mark.parametrize(
    "func",
    all_functions,
)
@pytest.mark.parametrize(
    "data",
    ["x_list_float", "x_list_int", "x_array_int", "x_array_float"],
)
def test_mock_aggregation_functions(
    func: typing.Callable,
    data: str,
    request: pytest.FixtureRequest,
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
    low_disp = np.round(rng.exponential(scale=1, size=100) + 1, 2)
    high_disp = np.round(rng.exponential(scale=10, size=100) + 1, 2)
    low_disp_res = func(low_disp)
    high_disp_res = func(high_disp)
    if low_disp_res > high_disp_res:
        msg = (
            f"Dispersion in the first case should be lower, "
            f"got {low_disp_res} > {high_disp_res}"
        )
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        coefficient_of_lvariation,
        coefficient_of_variation,
        robust_coefficient_of_variation,
        quartile_coefficient_of_dispersion,
        coefficient_of_range,
    ],
)
def test_cv_corner_cases(func: typing.Callable) -> None:
    """Testing for very small central tendency in CV calculation."""
    x = [0.0, 0.0, 0.0, 0.0, 1e-9, 0.0, 0.0]
    with pytest.warns(match="Statistic is undefined"):
        if func(x) is not np.inf:
            msg = "Dispersion should be inf."
            raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    all_functions,
)
def test_statistic_with_nans(
    func: typing.Callable,
    x_array_nan: np.ndarray,
) -> None:
    """Test for different data types."""
    if np.isnan(func(x_array_nan)):
        msg = "Statistic should not return nans."
        raise ValueError(msg)
