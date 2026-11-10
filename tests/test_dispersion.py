"""Collection of tests of dispersion module."""

import math
import typing
import warnings

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from obscure_stats.dispersion import (
    coefficient_of_lvariation,
    coefficient_of_range,
    coefficient_of_variation,
    fisher_index_of_dispersion,
    gini_mean_difference,
    inter_expectile_range,
    length_of_shortest_half,
    morisita_index_of_dispersion,
    quartile_coefficient_of_dispersion,
    robust_coefficient_of_variation,
    shamos_estimator,
    shapiro_coefficient_of_variation,
    standard_quantile_absolute_deviation,
    studentized_range,
)

all_functions = [
    length_of_shortest_half,
    coefficient_of_lvariation,
    coefficient_of_range,
    coefficient_of_variation,
    fisher_index_of_dispersion,
    gini_mean_difference,
    inter_expectile_range,
    morisita_index_of_dispersion,
    quartile_coefficient_of_dispersion,
    robust_coefficient_of_variation,
    shapiro_coefficient_of_variation,
    shamos_estimator,
    standard_quantile_absolute_deviation,
    studentized_range,
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


def test_zero_array() -> None:
    """Test length_of_shortest_half on zero array."""
    if not math.isnan(length_of_shortest_half([])):
        msg = "Statistic should return nan."
        raise ValueError(msg)


def test_length_of_shortest_half(
    shorth_data: npt.NDArray[np.number],
) -> None:
    """Test a hand-computed example with two equally short halves."""
    res = length_of_shortest_half(shorth_data)
    if res != pytest.approx(2.0):
        msg = "Result does not match reference."
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        coefficient_of_lvariation,
        coefficient_of_variation,
        robust_coefficient_of_variation,
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
def test_dispersion_sensibility(
    func: typing.Callable[..., float],
    exponential_data: npt.NDArray[np.number],
    exponential_wide_data: npt.NDArray[np.number],
) -> None:
    """Test for result correctness."""
    low_disp_res = func(exponential_data)
    high_disp_res = func(exponential_wide_data)
    if low_disp_res > high_disp_res:
        msg = (
            f"Dispersion in the first case should be lower, "
            f"got {low_disp_res} > {high_disp_res}"
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


@pytest.mark.parametrize(
    "func",
    [
        gini_mean_difference,
        inter_expectile_range,
        shamos_estimator,
        standard_quantile_absolute_deviation,
        studentized_range,
    ],
)
def test_invariance_add(
    func: typing.Callable[..., float], x_array_float: npt.NDArray[np.number]
) -> None:
    """Test coefficients for invariance to addition."""
    res1 = func(x_array_float)
    res2 = func(x_array_float + 10)
    if res1 != pytest.approx(res2):
        msg = f"Statistic should be invariant to addition, got {res1} and {res2}."
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        quartile_coefficient_of_dispersion,
        robust_coefficient_of_variation,
        coefficient_of_lvariation,
        coefficient_of_range,
        coefficient_of_variation,
    ],
)
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
def test_fuzz_dispersions(
    func: typing.Callable[..., float], data: npt.NDArray[np.number]
) -> None:
    """Test all functions with fuzz."""
    func(data)


def test_sqad_has_no_consistency_constant(
    x_array_float: npt.NDArray[np.number],
) -> None:
    """Test that SQAD returns the raw 68.27% quantile of |x - median|."""
    dev = np.abs(x_array_float - np.median(x_array_float))
    ref = np.quantile(dev, 0.6826894921370850)
    if standard_quantile_absolute_deviation(x_array_float) != pytest.approx(ref):
        msg = "Result does not match reference."
        raise ValueError(msg)


def test_robust_cvs_have_no_consistency_constant(
    x_array_float: npt.NDArray[np.number],
) -> None:
    """Test that robust CVs return raw MAD / median and IQR / median."""
    med = np.median(x_array_float)
    q1, q3 = np.quantile(x_array_float, [0.25, 0.75])
    mad = np.median(np.abs(x_array_float - med))
    if robust_coefficient_of_variation(x_array_float) != pytest.approx(mad / med):
        msg = "Result does not match reference."
        raise ValueError(msg)
    if shapiro_coefficient_of_variation(x_array_float) != pytest.approx(
        (q3 - q1) / med
    ):
        msg = "Result does not match reference."
        raise ValueError(msg)


def test_degenerate_inputs_return_nan(
    x_array_float: npt.NDArray[np.number], x_array_nan: npt.NDArray[np.number]
) -> None:
    """Test that empty or all-NaN inputs return NaN without warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if not math.isnan(standard_quantile_absolute_deviation(x_array_nan[:1])):
            msg = "Expected NaN for all-NaN input."
            raise ValueError(msg)
        if not math.isnan(gini_mean_difference(x_array_float[:0])):
            msg = "Expected NaN for empty input."
            raise ValueError(msg)
    if gini_mean_difference(x_array_float[:1]) != 0.0:
        msg = "Expected 0 for one observation."
        raise ValueError(msg)
