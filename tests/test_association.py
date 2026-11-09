"""Collection of tests of association module."""

import math
import typing
import warnings

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from obscure_stats.association import (
    blomqvist_beta,
    coefficient_of_monotone_association,
    concordance_correlation,
    concordance_rate,
    fechner_correlation,
    gaussian_rank_correlation,
    gini_correlation,
    l1_concordance_correlation,
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
    l1_concordance_correlation,
    gini_correlation,
    coefficient_of_monotone_association,
    blomqvist_beta,
    concordance_correlation,
    concordance_rate,
    fechner_correlation,
    gaussian_rank_correlation,
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
    func: typing.Callable[..., float],
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
        blomqvist_beta,
        concordance_correlation,
        fechner_correlation,
        gaussian_rank_correlation,
        quantile_correlation,
        rank_minrelation_coefficient,
        tanimoto_similarity,
        tukey_correlation,
        winsorized_correlation,
    ],
)
def test_signed_corr_sensibility(
    func: typing.Callable[..., float], y_array_float: npt.NDArray[np.number]
) -> None:
    """Test for result correctness."""
    res = func(-2 * y_array_float, y_array_float)
    if res > 0:
        msg = f"Corr coeff should be negative, got {res}"
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        morisita_horn_similarity,
        rank_divergence,
        symmetric_chatterjee_xi,
        zhang_i,
    ],
)
def test_unsigned_corr_sensibility(
    func: typing.Callable[..., float], y_array_float: npt.NDArray[np.number]
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
        concordance_correlation,
        concordance_rate,
        fechner_correlation,
        gaussian_rank_correlation,
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
def test_const(
    func: typing.Callable[..., float], y_array_float: npt.NDArray[np.number]
) -> None:
    """Test for constant input."""
    x = np.ones(shape=(len(y_array_float),))
    with pytest.warns(match="is constant"):
        res = func(x, y_array_float)
    if not math.isnan(res):
        msg = f"Corr coef should be 0 with constant input, got {res}"
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_const_after_prep(
    func: typing.Callable[..., float], x_array_float: npt.NDArray[np.number]
) -> None:
    """Test the second prep edge case input."""
    corr_test_data = np.ones(shape=len(x_array_float))
    corr_test_data[0] = np.nan
    res = func(x_array_float, corr_test_data)
    if not math.isnan(res):
        msg = f"Corr coef should be nan with constant input, got {res}"
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        blomqvist_beta,
        concordance_correlation,
        concordance_rate,
        fechner_correlation,
        gaussian_rank_correlation,
        morisita_horn_similarity,
        rank_divergence,
        tanimoto_similarity,
        symmetric_chatterjee_xi,
        winsorized_correlation,
    ],
)
def test_invariance(
    func: typing.Callable[..., float],
    x_array_float: npt.NDArray[np.number],
    y_array_float: npt.NDArray[np.number],
) -> None:
    """Test for invariance."""
    xy = func(x_array_float, y_array_float)
    yx = func(y_array_float, x_array_float)
    if pytest.approx(xy) != pytest.approx(yx):
        msg = f"Corr coef should symmetrical, got {xy}, {yx}"
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_notfinite_association(
    func: typing.Callable[..., float],
    x_array_nan: npt.NDArray[np.number],
    x_array_int: npt.NDArray[np.number],
    y_array_inf: npt.NDArray[np.number],
    y_array_int: npt.NDArray[np.number],
) -> None:
    """Test for correct NaN behavior."""
    if math.isnan(func(x_array_nan, y_array_int)):
        msg = "Corr coef should support nans."
        raise ValueError(msg)
    with pytest.warns(match="too many missing values"):
        func(x_array_nan[:2], x_array_int[:2])
    with pytest.warns(match="contains inf"):
        res = func(x_array_int, y_array_inf)
    if not math.isnan(res):
        msg = "Corr coef should support infs."
        raise ValueError(msg)


@pytest.mark.parametrize("func", all_functions)
def test_unequal_arrays(
    func: typing.Callable[..., float],
    x_array_int: npt.NDArray[np.number],
    y_array_int: npt.NDArray[np.number],
) -> None:
    """Test for unequal arrays."""
    with pytest.warns(match="Lengths of the inputs do not match"):
        func(x_array_int[:4], y_array_int[:3])


@pytest.mark.parametrize("func", all_functions)
def test_unequal_arrays_after_prep(
    func: typing.Callable[..., float],
    x_array_int: npt.NDArray[np.number],
    x_array_nan: npt.NDArray[np.number],
) -> None:
    """Test for unequal arrays after preparation."""
    res = func(x_array_nan[:2], x_array_int[:2])
    if not math.isnan(res):
        msg = f"Corr coef should be nan with incorrect input, got {res}"
        raise ValueError(msg)


@pytest.mark.parametrize("a", [-1, 0])
def test_a_in_rank_div(
    x_array_float: npt.NDArray[np.number],
    y_array_float: npt.NDArray[np.number],
    a: float,
) -> None:
    """Test for correctness of a parameter value."""
    with pytest.raises(ValueError, match="Parameter a should be > 0"):
        rank_divergence(x_array_float, y_array_float, a=a)


@pytest.mark.parametrize("k", [-1.0, 1.0])
def test_k_in_winsorized_corr(
    x_array_float: npt.NDArray[np.number],
    y_array_float: npt.NDArray[np.number],
    k: float,
) -> None:
    """Test for correctness of q parameter value."""
    with pytest.raises(ValueError, match="Parameter k should be in range"):
        winsorized_correlation(x_array_float, y_array_float, k=k)


@pytest.mark.parametrize("func", all_functions)
def test_corr_boundaries(
    func: typing.Callable[..., float], y_array_float: npt.NDArray[np.number]
) -> None:
    """Test for result correctness."""
    res = func(y_array_float, -y_array_float)
    if abs(res) > 1:
        msg = f"Corr coeff should not be higher than 1, got {res}"
        raise ValueError(msg)


def test_concordance(
    x_array_float: npt.NDArray[np.number], y_array_float: npt.NDArray[np.number]
) -> None:
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
    func: typing.Callable[..., float],
    x: npt.NDArray[np.number],
    y: npt.NDArray[np.number],
) -> None:
    """Test all functions with fuzz."""
    func(x, y)


@pytest.mark.parametrize("func", all_functions)
@pytest.mark.parametrize(("scale", "shift"), [(1e-9, 0.0), (1.0, 1e6), (1e-4, 100.0)])
def test_small_spread_is_not_constant(
    func: typing.Callable[..., float],
    scale: float,
    shift: float,
    y_array_float: npt.NDArray[np.number],
) -> None:
    """Test that small-scale or high-level data is not flagged as constant."""
    x = np.arange(len(y_array_float), dtype=float) * scale + shift
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = func(x, y_array_float)
    if any("is constant" in str(w.message) for w in rec) or math.isnan(res):
        msg = f"{func.__name__} flagged non-constant input as constant."
        raise ValueError(msg)


def test_gini_correlation(
    x_tied_list: list[int],
    y_tied_list: list[int],
) -> None:
    """Test the U-statistic ratio of Schechtman & Yitzhaki with ties."""
    n = len(x_tied_list)
    num = sum(
        (x_tied_list[i] - x_tied_list[j]) * np.sign(y_tied_list[i] - y_tied_list[j])
        for i in range(n)
        for j in range(i + 1, n)
    )
    den = sum(
        abs(x_tied_list[i] - x_tied_list[j]) for i in range(n) for j in range(i + 1, n)
    )
    res = gini_correlation(x_tied_list, y_tied_list)
    if res != pytest.approx(num / den):
        msg = "Result does not match reference."
        raise ValueError(msg)


def test_l1_concordance_correlation_reference(
    l1_corr_data_big: npt.NDArray[np.number],
    l1_corr_data_small: npt.NDArray[np.number],
) -> None:
    """Test the Bland-Altman example against R L1pack::l1ccc (U-statistic)."""
    res = l1_concordance_correlation(l1_corr_data_big, l1_corr_data_small)
    if res != pytest.approx(0.76416850, abs=1e-8):
        msg = "Result does not match reference."
        raise ValueError(msg)
    res = l1_concordance_correlation(l1_corr_data_big, l1_corr_data_big)
    if res != 1.0:
        msg = "Result does not match reference."
        raise ValueError(msg)
