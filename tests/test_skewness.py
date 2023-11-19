"""Collection of tests of skewness."""

import typing

import numpy as np
import pytest
from obscure_stats.skewness import (
    auc_skew_gamma,
    bowley_skew,
    forhad_shorna_rank_skew,
    groeneveld_skew,
    hossain_adnan_skew,
    kelly_skew,
    medeen_skew,
    pearson_halfmode_skew,
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
        pearson_halfmode_skew,
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
        pearson_halfmode_skew,
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


def test_rank_skew() -> None:
    """Simple tets case for correctness of Rank skewness coefficient."""
    x = np.asarray(
        (
            73.3,
            80.5,
            50.4,
            64.8,
            74.0,
            72.8,
            72.0,
            59.7,
            90.9,
            76.9,
            71.4,
            45.6,
            77.5,
            60.6,
            67.5,
            54.6,
            71.0,
            66.0,
            71.0,
            74.0,
            72.7,
            73.6,
            97.5,
            89.6,
            70.5,
            78.1,
            84.6,
            92.5,
            76.9,
            76.9,
            59.0,
            82.4,
            56.8,
            83.0,
            76.5,
            72.6,
            65.9,
            70.0,
            130.0,
            76.9,
            88.2,
            63.4,
            123.7,
            65.6,
            80.2,
            84.7,
            82.6,
            76.5,
            80.6,
            72.3,
            99.6,
            80.7,
            73.3,
            77.4,
            68.1,
            74.6,
            70.5,
            58.8,
            93.7,
            61.3,
            76.9,
            78.2,
            85.4,
            72.2,
            100.0,
            55.7,
            79.3,
            109.0,
            84.4,
            76.4,
            86.4,
            67.7,
            74.0,
            92.3,
            76.9,
            64.5,
            88.7,
            72.4,
            65.7,
            73.6,
            79.6,
            64.1,
            76.9,
            68.6,
            73.2,
            66.3,
            70.0,
            91.9,
            55.5,
            100.0,
            79.6,
            72.7,
            78.1,
            68.3,
            65.9,
            74.0,
            67.3,
            66.3,
            96.0,
            73.8,
            70.0,
            50.5,
            73.0,
            55.0,
            80.0,
            84.0,
            50.9,
        ),
    )
    if forhad_shorna_rank_skew(x) != pytest.approx(0.93809, rel=1e-4):
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
        pearson_halfmode_skew,
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
