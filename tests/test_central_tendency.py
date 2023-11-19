"""Collection of tests of central tendecies."""

from __future__ import annotations

import typing

import numpy as np
import pytest
from obscure_stats.central_tendency import (
    contraharmonic_mean,
    half_sample_mode,
    hodges_lehmann_sen_location,
    midhinge,
    midmean,
    midrange,
    standard_trimmed_harrell_davis_quantile,
    trimean,
)


@pytest.mark.parametrize(
    "func",
    [
        contraharmonic_mean,
        midhinge,
        midmean,
        midrange,
        trimean,
        hodges_lehmann_sen_location,
        standard_trimmed_harrell_davis_quantile,
        half_sample_mode,
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


def test_thdm() -> None:
    """Simple tets case for correctness of Trimmed Harrel Davis median."""
    x = np.asarray(
        (-0.565, -0.106, -0.095, 0.363, 0.404, 0.633, 1.371, 1.512, 2.018, 100_000),
    )
    result = standard_trimmed_harrell_davis_quantile(x)
    if result != pytest.approx(0.6268, rel=1e-4):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)


def test_edge_cases() -> None:
    """Simple tets case for edge cases."""
    x = np.asarray([1])
    result = standard_trimmed_harrell_davis_quantile(x)
    if result != pytest.approx(1.0, rel=1e-4):
        msg = "Result does not match expected output."
        raise ValueError(msg)


def test_q_in_sthdq(x_array_float: np.ndarray) -> None:
    """Simple tets case for correctnes of q."""
    with pytest.raises(ValueError, match="Parameter q should be in range"):
        standard_trimmed_harrell_davis_quantile(x_array_float, q=1)
    with pytest.raises(ValueError, match="Parameter q should be in range"):
        standard_trimmed_harrell_davis_quantile(x_array_float, q=0)


def test_hls() -> None:
    """Simple tets case for correctness of Hodges-Lehmann-Sen."""
    x = np.asarray((1, 5, 2, 2, 7, 4, 1, 6))
    result = hodges_lehmann_sen_location(x)
    if result != pytest.approx(3.5):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)


def test_hsm() -> None:
    """Simple tets case for correctness of Half Sample Mode."""
    x = np.asarray((1, 2, 2, 2, 7, 4, 1, 6))
    result = half_sample_mode(x)
    if result != pytest.approx(2.0):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)
    result = half_sample_mode(x[:3])
    if result != pytest.approx(2):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)
    result = half_sample_mode(x[1:4])
    if result != pytest.approx(2):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)
    result = half_sample_mode(x[2:5])
    if result != pytest.approx(2):
        msg = "Results from the test and paper do not match."
        raise ValueError(msg)


@pytest.mark.parametrize(
    "func",
    [
        contraharmonic_mean,
        midhinge,
        midmean,
        midrange,
        trimean,
        hodges_lehmann_sen_location,
        standard_trimmed_harrell_davis_quantile,
        half_sample_mode,
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
