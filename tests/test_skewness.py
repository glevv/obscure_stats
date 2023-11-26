"""Collection of tests of skewness."""

import typing

import numpy as np
import pytest
from obscure_stats.skewness import (
    auc_skew_gamma,
    bickel_mode_skew,
    bowley_skew,
    forhad_shorna_rank_skew,
    groeneveld_skew,
    hossain_adnan_skew,
    kelly_skew,
    medeen_skew,
    pearson_median_skew,
    pearson_mode_skew,
    wauc_skew_gamma,
)


@pytest.mark.parametrize(
    "func",
    [
        auc_skew_gamma,
        wauc_skew_gamma,
        bowley_skew,
        forhad_shorna_rank_skew,
        groeneveld_skew,
        hossain_adnan_skew,
        kelly_skew,
        medeen_skew,
        pearson_median_skew,
        pearson_mode_skew,
        bickel_mode_skew,
    ],
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
        auc_skew_gamma,
        wauc_skew_gamma,
        bowley_skew,
        forhad_shorna_rank_skew,
        groeneveld_skew,
        hossain_adnan_skew,
        kelly_skew,
        medeen_skew,
        pearson_median_skew,
        pearson_mode_skew,
        bickel_mode_skew,
    ],
)
@pytest.mark.parametrize("seed", [1, 42, 99])
def test_skew_sensibility(func: typing.Callable, seed: int) -> None:
    """Testing for result correctness."""
    rng = np.random.default_rng(seed)
    no_skew = np.round(rng.normal(size=100), 2)
    left_skew = np.round(rng.exponential(size=100) + 1, 2)
    if func(no_skew) > func(left_skew):
        msg = "Skewness in the first case should be lower."
        raise ValueError(msg)


def test_rank_skew(rank_skewness_test_data: np.ndarray) -> None:
    """Simple tets case for correctness of Rank skewness coefficient."""
    if forhad_shorna_rank_skew(rank_skewness_test_data) != pytest.approx(
        0.93809, rel=1e-4
    ):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        auc_skew_gamma,
        wauc_skew_gamma,
        bowley_skew,
        forhad_shorna_rank_skew,
        groeneveld_skew,
        hossain_adnan_skew,
        kelly_skew,
        medeen_skew,
        pearson_median_skew,
        pearson_mode_skew,
        bickel_mode_skew,
    ],
)
def test_statistic_with_nans(
    func: typing.Callable,
    x_array_nan: np.ndarray,
) -> None:
    """Test for different data types."""
    if np.isnan(func(x_array_nan)):
        msg = "Statistic should not return nans."
        raise ValueError(msg)
