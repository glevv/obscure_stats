"""Collection of tests of association."""

import math
import typing
from functools import partial

import numpy as np
import pytest

from obscure_stats.association import (
    chatterjeexi,
    concordance_corr,
    quadrant_count_ratio,
    tanimoto_similarity,
    zhangi,
)


@pytest.mark.parametrize(
    "func",
    (
        zhangi,
        chatterjeexi,
        concordance_corr,
        quadrant_count_ratio,
        partial(quadrant_count_ratio, exclusion_zone=True),
        tanimoto_similarity,
    ),
)
@pytest.mark.parametrize(
    "x_array", ("x_list_float", "x_list_int", "x_array_int", "x_array_float")
)
@pytest.mark.parametrize(
    "y_array", ("y_list_float", "y_list_int", "y_array_int", "y_array_float")
)
def test_mock_association_functions(
    request, func: typing.Callable, x_array: typing.Iterable, y_array: typing.Iterable
) -> None:
    """Test for different data types."""
    x_array = request.getfixturevalue(x_array)
    y_array = request.getfixturevalue(y_array)
    func(x_array, y_array)


@pytest.mark.parametrize(
    "func",
    (
        concordance_corr,
        quadrant_count_ratio,
        partial(quadrant_count_ratio, exclusion_zone=True),
        tanimoto_similarity,
    ),
)
def test_signed_corr_sensibility(func: typing.Callable):
    """Testing for result correctness."""
    x = np.arange(11)
    y = -x
    assert func(x, y) < 0


@pytest.mark.parametrize(
    "func",
    (
        zhangi,
        chatterjeexi,
    ),
)
def test_unsigned_corr_sensibility(func: typing.Callable):
    """Testing for result correctness."""
    x = np.arange(11)
    y = -x
    w = np.r_[2, np.ones(shape=(10,))]
    assert func(x, y) > func(x, w)


@pytest.mark.parametrize(
    "func",
    (
        concordance_corr,
        zhangi,
        chatterjeexi,
        quadrant_count_ratio,
        partial(quadrant_count_ratio, exclusion_zone=True),
    ),
)
def test_const(func: typing.Callable):
    """Testing for constant input."""
    x = [1, 1, 1, 1, 1, 1]
    y = [-1, -2, -3, -4, -5, -6]
    with pytest.warns(match="An input array is constant"):
        assert func(x, y) is np.nan


@pytest.mark.parametrize(
    "func",
    (
        concordance_corr,
        quadrant_count_ratio,
        partial(quadrant_count_ratio, exclusion_zone=True),
        tanimoto_similarity,
    ),
)
def test_invariance(func: typing.Callable):
    """Testing for invariance."""
    x = [1, 2, 3, 4, 5, 5]
    y = [-1, -2, -2, -4, -5, -6]
    assert func(x, y) == func(y, x)
