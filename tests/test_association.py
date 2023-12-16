"""Collection of tests of association."""

import typing

import numpy as np
import pytest
from obscure_stats.association import (
    chatterjeexi,
    concordance_corrcoef,
    concordance_rate,
    symmetric_chatterjeexi,
    tanimoto_similarity,
    zhangi,
)

all_functions = [
    chatterjeexi,
    concordance_corrcoef,
    concordance_rate,
    symmetric_chatterjeexi,
    tanimoto_similarity,
    zhangi,
]


@pytest.mark.parametrize(
    "func",
    all_functions,
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
def test_signed_corr_sensibility(
    func: typing.Callable, y_array_float: np.ndarray
) -> None:
    """Testing for result correctness."""
    if func(y_array_float, -y_array_float) > 0:
        msg = "Corr coeff should be negative."
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        zhangi,
        chatterjeexi,
        symmetric_chatterjeexi,
    ],
)
def test_unsigned_corr_sensibility(
    func: typing.Callable, y_array_float: np.ndarray
) -> None:
    """Testing for result correctness."""
    w = np.ones(shape=len(y_array_float))
    w[0] = 2
    if func(y_array_float, -y_array_float) < func(y_array_float, w):
        msg = "Corr coeff higher in the first case."
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        concordance_corrcoef,
        zhangi,
        chatterjeexi,
        concordance_rate,
        symmetric_chatterjeexi,
    ],
)
def test_const(func: typing.Callable, y_array_float: np.ndarray) -> None:
    """Testing for constant input."""
    x = np.ones(shape=(len(y_array_float),))
    with pytest.warns(match="is constant"):
        if func(x, y_array_float) is not np.nan:
            msg = "Corr coef should be 0 with constant input."
            raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        concordance_corrcoef,
        concordance_rate,
        tanimoto_similarity,
        symmetric_chatterjeexi,
    ],
)
def test_invariance(
    func: typing.Callable, x_array_float: np.ndarray, y_array_float: np.ndarray
) -> None:
    """Testing for invariance."""
    if pytest.approx(func(x_array_float, y_array_float)) != pytest.approx(
        func(y_array_float, x_array_float)
    ):
        msg = "Corr coef should symmetrical."
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    all_functions,
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
    with pytest.warns(match="too many missing values"):
        func(x_array_nan[:2], x_array_int[:2])
    with pytest.warns(match="contains inf"):
        if not np.isnan(func(x_array_int, y_array_inf)):
            msg = "Corr coef should support infs."
            raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    all_functions,
)
def test_unequal_arrays(
    func: typing.Callable,
    x_array_int: np.ndarray,
    y_array_int: np.ndarray,
) -> None:
    """Test for unequal arrays."""
    with pytest.warns(match="Lenghts of the inputs do not match"):
        func(x_array_int[:4], y_array_int[:3])
