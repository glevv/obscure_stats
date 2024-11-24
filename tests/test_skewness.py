"""Collection of tests of skewness module."""

import math
import typing

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from obscure_stats.skewness import (
    auc_skew_gamma,
    bickel_mode_skew,
    bowley_skew,
    cumulative_skew,
    forhad_shorna_rank_skew,
    groeneveld_skew,
    hossain_adnan_skew,
    kelly_skew,
    l_skew,
    left_quantile_weight,
    medeen_skew,
    pearson_median_skew,
    pearson_mode_skew,
    right_quantile_weight,
    wauc_skew_gamma,
)

all_functions = [
    auc_skew_gamma,
    bickel_mode_skew,
    bowley_skew,
    cumulative_skew,
    forhad_shorna_rank_skew,
    groeneveld_skew,
    hossain_adnan_skew,
    kelly_skew,
    l_skew,
    left_quantile_weight,
    medeen_skew,
    pearson_median_skew,
    pearson_mode_skew,
    right_quantile_weight,
    wauc_skew_gamma,
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
def test_skew_sensibility(func: typing.Callable, seed: int) -> None:
    """Testing for result correctness."""
    rng = np.random.default_rng(seed)
    # round for the mode estimators to work properly
    no_skew = np.round(rng.uniform(size=100), 2)
    left_skew = np.round(rng.exponential(size=100) + 1, 2)
    no_skew_res = func(no_skew)
    left_skew_res = func(left_skew)
    if func.__name__ == "right_quantile_weight":
        # ugly but more harmonized this way
        no_skew_res = -no_skew_res
        left_skew_res = -left_skew_res
    if no_skew_res > left_skew_res:
        msg = (
            f"Skewness in the first case should be lower, "
            f"got {no_skew_res} > {left_skew_res}"
        )
        raise ValueError(msg)


def test_rank_skew(rank_skewness_test_data: np.ndarray) -> None:
    """Simple tets case for correctness of Rank skewness coefficient."""
    if forhad_shorna_rank_skew(rank_skewness_test_data) != pytest.approx(
        0.93809, rel=1e-4
    ):
        msg = "Results from the test and paper do not match."
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
def test_fuzz_skewnesses(func: typing.Callable, data: np.ndarray) -> None:
    """Test all functions with fuzz."""
    func(data)
