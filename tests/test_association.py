"""Collection of tests of association module."""

import math
import typing

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from obscure_stats.association import (
    blomqvist_beta,
    chatterjee_xi,
    concordance_correlation,
    concordance_rate,
    fechner_correlation,
    gaussain_rank_correlation,
    morisita_horn_similarity,
    normalized_chatterjee_xi,
    quantile_correlation,
    rank_divergence,
    rank_minrelation_coefficient,
    symmetric_chatterjee_xi,
    tanimoto_similarity,
    tukey_correlation,
    winsorized_correlation,
    zhang_i,
)

all_functions = [
    blomqvist_beta,
    chatterjee_xi,
    concordance_correlation,
    concordance_rate,
    fechner_correlation,
    gaussain_rank_correlation,
    normalized_chatterjee_xi,
    morisita_horn_similarity,
    quantile_correlation,
    rank_minrelation_coefficient,
    rank_divergence,
    symmetric_chatterjee_xi,
    tanimoto_similarity,
    tukey_correlation,
    winsorized_correlation,
    zhang_i,
]


@pytest.mark.parametrize("func", all_functions)
@pytest.mark.parametrize(
    "x_array", ["x_list_float", "x_list_int", "x_array_int", "x_array_float"]
)
@pytest.mark.parametrize(
    "y_array", ["y_list_float", "y_list_int", "y_array_int", "y_array_float"]
)
def test_mock_association_functions(
    func: typing.Callable, x_array: str, y_array: str, request: pytest.FixtureRequest
) -> None:
    """Test for different data types."""
    x_array = request.getfixturevalue(x_array)
    y_array = request.getfixturevalue(y_array)
    func(x_array, y_array)


@pytest.mark.parametrize(
    "func",
    [
        blomqvist_beta,
        concordance_correlation,
        concordance_rate,
        fechner_correlation,
        gaussain_rank_correlation,
        quantile_correlation,
        rank_minrelation_coefficient,
        tanimoto_similarity,
        tukey_correlation,
        winsorized_correlation,
    ],
)
def test_signed_corr_sensibility(
    func: typing.Callable, y_array_float: npt.NDArray
) -> None:
    """Test for result correctness."""
    res = func(-2 * y_array_float, y_array_float)
    if res > 0:
        msg = f"Corr coeff should be negative, got {res}"
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        chatterjee_xi,
        morisita_horn_similarity,
        rank_divergence,
        symmetric_chatterjee_xi,
        zhang_i,
    ],
)
def test_unsigned_corr_sensibility(
    func: typing.Callable, y_array_float: npt.NDArray
) -> None:
    """Test for result correctness."""
    w = np.ones(shape=len(y_array_float))
    w[0] = 2
    res_ideal = func(y_array_float, -y_array_float)
    res_normal = func(y_array_float, w)
    if res_ideal < res_normal:
        msg = f"Corr coeff higher in the first case, got {res_ideal} < {res_normal}"
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        blomqvist_beta,
        chatterjee_xi,
        concordance_correlation,
        concordance_rate,
        fechner_correlation,
        gaussain_rank_correlation,
        normalized_chatterjee_xi,
        quantile_correlation,
        rank_minrelation_coefficient,
        rank_divergence,
        tukey_correlation,
        symmetric_chatterjee_xi,
        winsorized_correlation,
        zhang_i,
    ],
)
def test_const(func: typing.Callable, y_array_float: npt.NDArray) -> None:
    """Test for constant input."""
    x = np.ones(shape=(len(y_array_float),))
    with pytest.warns(match="is constant"):
        res = func(x, y_array_float)
        if not math.isnan(res):
            msg = f"Corr coef should be 0 with constant input, got {res}"
            raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_const_after_prep(func: typing.Callable, x_array_float: npt.NDArray) -> None:
    """Test the second prep edge case input."""
    corr_test_data = np.ones(shape=len(x_array_float))
    corr_test_data[0] = np.nan
    res = func(x_array_float, corr_test_data)
    if not math.isnan(res):
        msg = f"Corr coef should be 0 with constant input, got {res}"
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        blomqvist_beta,
        concordance_correlation,
        concordance_rate,
        fechner_correlation,
        gaussain_rank_correlation,
        morisita_horn_similarity,
        rank_divergence,
        tanimoto_similarity,
        symmetric_chatterjee_xi,
        winsorized_correlation,
    ],
)
def test_invariance(
    func: typing.Callable, x_array_float: npt.NDArray, y_array_float: npt.NDArray
) -> None:
    """Test for invariance."""
    xy = func(x_array_float, y_array_float)
    yx = func(y_array_float, x_array_float)
    if pytest.approx(xy) != pytest.approx(yx):
        msg = f"Corr coef should symmetrical, got {xy}, {yx}"
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_notfinite_association(
    func: typing.Callable,
    x_array_nan: npt.NDArray,
    x_array_int: npt.NDArray,
    y_array_inf: npt.NDArray,
    y_array_int: npt.NDArray,
) -> None:
    """Test for correct nan behaviour."""
    if math.isnan(func(x_array_nan, y_array_int)):
        msg = "Corr coef should support nans."
        raise ValueError(msg)
    with pytest.warns(match="too many missing values"):
        func(x_array_nan[:2], x_array_int[:2])
    with pytest.warns(match="contains inf"):
        if not math.isnan(func(x_array_int, y_array_inf)):
            msg = "Corr coef should support infs."
            raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_unequal_arrays(
    func: typing.Callable, x_array_int: npt.NDArray, y_array_int: npt.NDArray
) -> None:
    """Test for unequal arrays."""
    with pytest.warns(match="Lenghts of the inputs do not match"):
        func(x_array_int[:4], y_array_int[:3])


@pytest.mark.parametrize("a", [-1, 0])
def test_a_in_rank_div(
    x_array_float: npt.NDArray, y_array_float: npt.NDArray, a: float
) -> None:
    """Simple tets case for correctnes of a."""
    with pytest.raises(ValueError, match="Parameter a should be > 0"):
        rank_divergence(x_array_float, y_array_float, a=a)


@pytest.mark.parametrize("func", all_functions)
def test_corr_boundaries(func: typing.Callable, y_array_float: npt.NDArray) -> None:
    """Test for result correctness."""
    res = func(y_array_float, -y_array_float)
    if abs(res) > 1:
        msg = f"Corr coeff should not be higher than 1, got {res}"
        raise ValueError(msg)


def test_concordance(x_array_float: npt.NDArray, y_array_float: npt.NDArray) -> None:
    """Test coefficients for invariance to addition."""
    res1 = concordance_correlation(x_array_float, y_array_float)
    res2 = concordance_correlation(x_array_float, y_array_float + 1.0)
    if res1 == pytest.approx(res2):
        msg = f"Concordance should be different, got {res1} and {res2}."
        raise ValueError(msg)


@given(
    arrays(
        dtype=np.float64,
        shape=array_shapes(max_dims=1),
        elements=st.floats(allow_nan=True, allow_infinity=True),
    ),
    arrays(
        dtype=np.float64,
        shape=array_shapes(max_dims=1),
        elements=st.floats(allow_nan=True, allow_infinity=True),
    ),
)
@pytest.mark.parametrize("func", all_functions)
def test_fuzz_associations(
    func: typing.Callable, x: npt.NDArray, y: npt.NDArray
) -> None:
    """Test all functions with fuzz."""
    func(x, y)
