"""Collection of tests of variation module."""

import math
import typing
import warnings
from functools import partial

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from obscure_stats.variation import (
    avdev,
    b_index,
    brillouin_index,
    extropy,
    gibbs_m1,
    mcintosh_d,
    mndif,
    mod_vr,
    range_vr,
    renyi_entropy,
)

all_functions = [
    avdev,
    b_index,
    brillouin_index,
    extropy,
    gibbs_m1,
    mcintosh_d,
    mndif,
    mod_vr,
    range_vr,
    renyi_entropy,
    partial(renyi_entropy, normalize=True),
]


@pytest.mark.parametrize("func", all_functions)
@pytest.mark.parametrize("data", ["c_list_obj", "c_array_obj"])
def test_mock_variation_functions(
    func: typing.Callable[..., float], data: str, request: pytest.FixtureRequest
) -> None:
    """Test for different data types."""
    data = request.getfixturevalue(data)
    func(data)


@pytest.mark.parametrize("func", all_functions)
def test_var_sensibility_higher_better(
    func: typing.Callable[..., float],
    even_categories: npt.NDArray[np.str_],
    uneven_categories: npt.NDArray[np.str_],
) -> None:
    """Test for result correctness."""
    low_var_res = func(even_categories)
    high_var_res = func(uneven_categories)
    if low_var_res < high_var_res:
        msg = f"Statistic value should be higher, got {low_var_res} < {high_var_res}"
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_statistic_with_nans(
    func: typing.Callable[..., float], c_array_nan: npt.NDArray[np.str_]
) -> None:
    """Test for different data types."""
    if math.isnan(func(c_array_nan)):
        msg = "Statistic should not return nans."
        raise ValueError(msg)


def test_renyi_entropy_edgecases(
    c_array_obj: npt.NDArray[np.str_], even_categories: npt.NDArray[np.str_]
) -> None:
    """Test for different edge cases of Renyi entropy."""
    with pytest.raises(ValueError, match="alpha should be non-negative"):
        renyi_entropy(c_array_obj, alpha=-1)
    renyi_0 = renyi_entropy(c_array_obj, alpha=0)
    if renyi_0 != pytest.approx(2.321928):
        msg = f"Results from the test and paper do not match, got {renyi_0}"
        raise ValueError(msg)
    renyi_1 = renyi_entropy(c_array_obj, alpha=1)
    if renyi_1 != pytest.approx(2.040373):
        msg = f"Results from the test and paper do not match, got {renyi_1}"
        raise ValueError(msg)
    renyi_2 = renyi_entropy(even_categories, normalize=True)
    if renyi_2 != pytest.approx(1.0):
        msg = f"For uniformly distributed array normalized entropy should be 1.0, got {renyi_2}"
        raise ValueError(msg)
    renyi_3 = renyi_entropy(c_array_obj[0], normalize=True)
    if renyi_3 != pytest.approx(0.0):
        msg = f"For constant array entropy should be 0.0, got {renyi_3}"
        raise ValueError(msg)


@given(
    arrays(
        dtype=np.object_,
        shape=array_shapes(max_dims=1),
        elements=st.characters(),
    )
)
@pytest.mark.parametrize("func", all_functions)
def test_fuzz_variations(
    func: typing.Callable[..., float], data: npt.NDArray[np.str_]
) -> None:
    """Test all functions with fuzz."""
    func(data)


def test_renyi_entropy_min_entropy(c_array_obj: npt.NDArray[np.str_]) -> None:
    """Test that infinite alpha returns min-entropy instead of crashing."""
    ref = -math.log2(5 / 11)  # the most frequent category ("a") has 5 of 11
    if renyi_entropy(c_array_obj, alpha=math.inf) != pytest.approx(ref):
        msg = "Result does not match reference."
        raise ValueError(msg)
    if renyi_entropy(c_array_obj, alpha=200) != pytest.approx(ref, rel=1e-2):
        msg = "Large alpha should approach the min-entropy."
        raise ValueError(msg)


def test_documented_variation_relationships(c_array_obj: npt.NDArray[np.str_]) -> None:
    """Test the Notes formulas (Gibbs M1, Renyi, McIntosh D; AVDev, Strong's D_w)."""
    n = len(c_array_obj)
    m1 = gibbs_m1(c_array_obj)
    if renyi_entropy(c_array_obj, alpha=2) != pytest.approx(-math.log2(1 - m1)):
        msg = "Result does not match reference."
        raise ValueError(msg)
    ref = (1 - math.sqrt(1 - m1)) * n / (n - math.sqrt(n))
    if mcintosh_d(c_array_obj) != pytest.approx(ref):
        msg = "Result does not match reference."
        raise ValueError(msg)
    counts = np.sort(np.unique(c_array_obj, return_counts=True)[1])[::-1]
    k = len(counts)
    strong = np.max(np.cumsum(counts) / n - np.arange(1, k + 1) / k)
    if (1 - avdev(c_array_obj)) * (k - 1) / k != pytest.approx(strong):
        msg = "Result does not match reference."
        raise ValueError(msg)


def test_mcintosh_d_single_observation(c_array_obj: npt.NDArray[np.str_]) -> None:
    """Test that one observation returns NaN without warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if not math.isnan(mcintosh_d(c_array_obj[:1])):
            msg = "Expected NaN for one observation."
            raise ValueError(msg)


def test_mndif_reference(c_array_obj: npt.NDArray[np.str_]) -> None:
    """Test MNDif against a hand calculation (frequencies 5, 2, 2, 1, 1)."""
    ref = 1 - 18 / (11 * 4)  # sum of |f_i - f_j| over pairs is 18
    if mndif(c_array_obj) != pytest.approx(ref):
        msg = "Result does not match reference."
        raise ValueError(msg)


def test_brillouin_index_reference(c_array_obj: npt.NDArray[np.str_]) -> None:
    """Test Brillouin's index against a hand calculation (frequencies 5, 2, 2, 1, 1)."""
    ref = math.log(math.factorial(11) / (math.factorial(5) * 2 * 2)) / 11
    if brillouin_index(c_array_obj) != pytest.approx(ref):
        msg = "Result does not match reference."
        raise ValueError(msg)


def test_brillouin_index_below_shannon(even_categories: npt.NDArray[np.str_]) -> None:
    """Test that Brillouin's index is below the Shannon entropy in nats."""
    shannon_nats = renyi_entropy(even_categories, alpha=1) * math.log(2)
    if not brillouin_index(even_categories) < shannon_nats:
        msg = "Brillouin's index should be below the Shannon entropy."
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_single_category(
    func: typing.Callable[..., float], const_array_float: npt.NDArray[np.number]
) -> None:
    """Test that a single category (no variation) gives 0."""
    if func(const_array_float) != pytest.approx(0.0):
        msg = "A single category should give 0."
        raise ValueError(msg)


def test_brillouin_index_empty(empty_array: npt.NDArray[np.number]) -> None:
    """Test that an empty input returns NaN."""
    if not math.isnan(brillouin_index(empty_array)):
        msg = "Expected NaN for an empty input."
        raise ValueError(msg)
