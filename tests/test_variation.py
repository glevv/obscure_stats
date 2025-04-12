"""Collection of tests of variation module."""

import math
import typing

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from obscure_stats.variation import (
    avdev,
    b_index,
    gibbs_m1,
    gibbs_m2,
    mcintosh_d,
    mod_vr,
    negative_extropy,
    range_vr,
    renyi_entropy,
)

all_functions = [
    avdev,
    b_index,
    gibbs_m1,
    gibbs_m2,
    mcintosh_d,
    mod_vr,
    negative_extropy,
    range_vr,
    renyi_entropy,
]


@pytest.mark.parametrize("func", all_functions)
@pytest.mark.parametrize("data", ["c_list_obj", "c_array_obj"])
def test_mock_variation_functions(
    func: typing.Callable, data: str, request: pytest.FixtureRequest
) -> None:
    """Test for different data types."""
    data = request.getfixturevalue(data)
    func(data)


@pytest.mark.parametrize("func", all_functions)
@pytest.mark.parametrize("seed", [1, 42, 99])
def test_var_sensibility_higher_better(func: typing.Callable, seed: int) -> None:
    """Test for result correctness."""
    rng = np.random.default_rng(seed)
    low_var = rng.choice(["a", "b", "c", "d"], p=[0.25, 0.25, 0.25, 0.25], size=100)
    high_var = rng.choice(["a", "b", "c", "d"], p=[0.75, 0.15, 0.05, 0.05], size=100)
    low_var_res = func(low_var)
    high_var_res = func(high_var)
    if low_var_res < high_var_res:
        msg = f"Statistic value should be higher, got {low_var_res} < {high_var_res}"
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_statistic_with_nans(func: typing.Callable, c_array_nan: np.ndarray) -> None:
    """Test for different data types."""
    if math.isnan(func(c_array_nan)):
        msg = "Statistic should not return nans."
        raise ValueError(msg)


def test_renyi_entropy_edgecases(c_array_obj: np.ndarray) -> None:
    """Test for different edgecases of Renyi entropy."""
    with pytest.raises(ValueError, match="alpha should be positive"):
        renyi_entropy(c_array_obj, alpha=-1)
    renyi_0 = renyi_entropy(c_array_obj, alpha=0)
    if renyi_0 != pytest.approx(2.321928):
        msg = f"Results from the test and paper do not match, got {renyi_0}"
        raise ValueError(msg)
    renyi_1 = renyi_entropy(c_array_obj, alpha=1)
    if renyi_1 != pytest.approx(2.040373):
        msg = f"Results from the test and paper do not match, got {renyi_1}"
        raise ValueError(msg)
    renyi_2 = renyi_entropy(np.arange(10), alpha=1, normalize=True)
    if renyi_2 != 1.0:
        msg = f"Results from the test and paper do not match, got {renyi_2}"
        raise ValueError(msg)


@given(
    arrays(
        dtype=np.object_,
        shape=array_shapes(max_dims=1),
        elements=st.characters(),
    )
)
@pytest.mark.parametrize("func", all_functions)
def test_fuzz_variations(func: typing.Callable, data: np.ndarray) -> None:
    """Test all functions with fuzz."""
    func(data)
