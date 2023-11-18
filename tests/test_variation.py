"""Collection of tests of variation."""

import typing

import numpy as np
import pytest
from obscure_stats.variation import (
    ada_index,
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
        ada_index,
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
        ada_index,
        b_index,
        gibbs_m1,
        gibbs_m2,
        mod_vr,
        range_vr,
    ],
)
@pytest.mark.parametrize("seed", [1, 42, 99])
def test_var_sensibility_higher_better(func: typing.Callable, seed: int) -> None:
    """Testing for result correctness."""
    rng = np.random.default_rng(seed)
    low_var = rng.choice(["a", "b", "c", "d"], p=[0.25, 0.25, 0.25, 0.25], size=100)
    high_var = rng.choice(["a", "b", "c", "d"], p=[0.75, 0.15, 0.05, 0.05], size=100)
    if func(low_var) < func(high_var):
        msg = "Variation value in the first case should be higher."
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        extropy,
    ],
)
@pytest.mark.parametrize("seed", [1, 42, 99])
def test_var_sensibility_lower_better(func: typing.Callable, seed: int) -> None:
    """Testing for result correctness."""
    rng = np.random.default_rng(seed)
    low_var = rng.choice(["a", "b", "c", "d"], p=[0.25, 0.25, 0.25, 0.25], size=100)
    high_var = rng.choice(["a", "b", "c", "d"], p=[0.75, 0.15, 0.05, 0.05], size=100)
    if func(low_var) > func(high_var):
        msg = "Variation value in the first case should be lower."
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        ada_index,
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
        msg = "Statistics should support nans."
        raise ValueError(msg)
