"""Collection of tests of variation."""

import typing

import numpy as np
import pytest
from obscure_stats.variation import (
    avdev,
    b_index,
    extropy,
    gibbs_m1,
    gibbs_m2,
    mod_vr,
    range_vr,
)


@pytest.mark.parametrize(
    "func",
    [
        avdev,
        b_index,
        extropy,
        gibbs_m1,
        gibbs_m2,
        mod_vr,
        range_vr,
    ],
)
@pytest.mark.parametrize("data", ["c_list_obj", "c_array_obj"])
def test_mock_variation_functions(
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
        avdev,
        b_index,
        gibbs_m1,
        gibbs_m2,
        mod_vr,
        range_vr,
        extropy,
    ],
)
@pytest.mark.parametrize("seed", [1, 42, 99])
def test_var_sensibility_higher_better(func: typing.Callable, seed: int) -> None:
    """Testing for result correctness."""
    rng = np.random.default_rng(seed)
    low_var = rng.choice(["a", "b", "c", "d"], p=[0.25, 0.25, 0.25, 0.25], size=100)
    high_var = rng.choice(["a", "b", "c", "d"], p=[0.75, 0.15, 0.05, 0.05], size=100)
    low_res = func(low_var)
    high_res = func(high_var)
    if low_res < high_res:
        msg = f"Statistic value should be higher, got {low_res} < {high_res}"
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        avdev,
        b_index,
        extropy,
        gibbs_m1,
        gibbs_m2,
        mod_vr,
        range_vr,
    ],
)
def test_statistic_with_nans(
    func: typing.Callable,
    c_array_nan: np.ndarray,
) -> None:
    """Test for different data types."""
    if np.isnan(func(c_array_nan)):
        msg = "Statistic should not return nans."
        raise ValueError(msg)
