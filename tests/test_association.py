"""Collection of tests of association."""

import typing

import numpy as np
import pytest
from obscure_stats.association import (
    chatterjeexi,
    concordance_corrcoef,
    concordance_rate,
    tanimoto_similarity,
    zhangi,
)


@pytest.mark.parametrize(
    "func",
    [
        zhangi,
        chatterjeexi,
        concordance_corrcoef,
        concordance_rate,
        tanimoto_similarity,
    ],
)
@pytest.mark.parametrize(
    "x_array",
    ["x_list_float", "x_list_int", "x_array_int", "x_array_float"],
)
@pytest.mark.parametrize(
    "y_array",
    ["y_list_float", "y_list_int", "y_array_int", "y_array_float"],
)
def test_mock_association_functions(
    func: typing.Callable,
    x_array: str,
    y_array: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test for different data types."""
    x_array = request.getfixturevalue(x_array)
    y_array = request.getfixturevalue(y_array)
    func(x_array, y_array)


@pytest.mark.parametrize(
    "func",
    [
        concordance_corrcoef,
        concordance_rate,
        tanimoto_similarity,
    ],
)
def test_signed_corr_sensibility(func: typing.Callable) -> None:
    """Testing for result correctness."""
    x = np.arange(11)
    y = -x
    if func(x, y) > 0:
        msg = "Corr coeff should be negative."
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        zhangi,
        chatterjeexi,
    ],
)
def test_unsigned_corr_sensibility(func: typing.Callable) -> None:
    """Testing for result correctness."""
    x = np.arange(11)
    y = -x
    w = np.r_[2, np.ones(shape=(10,))]
    if func(x, y) < func(x, w):
        msg = "Corr coeff higher in the first case."
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        concordance_corrcoef,
        zhangi,
        chatterjeexi,
        concordance_rate,
    ],
)
def test_const(func: typing.Callable) -> None:
    """Testing for constant input."""
    x = [1, 1, 1, 1, 1, 1]
    y = [-1, -2, -3, -4, -5, -6]
    with pytest.warns(match="is constant"):
        if func(x, y) is not np.nan:
            msg = "Corr coef should be 0 with constant input."
            raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        concordance_corrcoef,
        concordance_rate,
        tanimoto_similarity,
    ],
)
def test_invariance(func: typing.Callable) -> None:
    """Testing for invariance."""
    x = [1, 2, 3, 4, 5, 5]
    y = [-1, -2, -2, -4, -5, -6]
    if func(x, y) != func(y, x):
        msg = "Corr coef should symmetrical."
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        zhangi,
        chatterjeexi,
        concordance_corrcoef,
        concordance_rate,
        tanimoto_similarity,
    ],
)
def test_notfinite_association(
    func: typing.Callable,
    x_array_nan: np.ndarray,
    x_array_int: np.ndarray,
    y_array_inf: np.ndarray,
    y_array_int: np.ndarray,
) -> None:
    """Test for correct nan behaviour."""
    if np.isnan(func(x_array_nan, y_array_int)):
        msg = "Corr coef should support nans."
        raise ValueError(msg)
    with pytest.warns(match="contains inf"):
        if not np.isnan(func(x_array_int, y_array_inf)):
            msg = "Corr coef should support infs."
            raise ValueError(msg)
