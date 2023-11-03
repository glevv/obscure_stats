"""Collection of tests of central tendecies."""

import typing

import pytest

from obscure_stats.central_tendency import (
    contraharmonic_mean,
    hodges_lehmann_sen_location,
    midhinge,
    midmean,
    midrange,
    trimean,
    trimmed_harrell_davis_median,
)


@pytest.mark.parametrize(
    "func",
    (
        contraharmonic_mean,
        midhinge,
        midmean,
        midrange,
        trimean,
        hodges_lehmann_sen_location,
        trimmed_harrell_davis_median,
    ),
)
@pytest.mark.parametrize(
    "data", ("x_list_float", "x_list_int", "x_array_int", "x_array_float")
)
def test_mock_aggregation_functions(
    request, func: typing.Callable, data: typing.Iterable
) -> None:
    """Test for different data types."""
    data = request.getfixturevalue(data)
    func(data)


def test_thdm():
    """Simple tets case for correctness of Trimmed Harrel Davis median."""
    x = (-0.565, -0.106, -0.095, 0.363, 0.404, 0.633, 1.371, 1.512, 2.018, 100_000)
    result = trimmed_harrell_davis_median(x)
    assert result == pytest.approx(0.6268, rel=1e-4)


def test_hls():
    """Simple tets case for correctness of Hodges-Lehmann-Sen."""
    x = (1, 5, 2, 2, 7, 4, 1, 6)
    result = hodges_lehmann_sen_location(x)
    assert result == pytest.approx(3.5)
    x = (10**100, 10**100, 2, 2, 7, 4, 1, 6)
    result = hodges_lehmann_sen_location(x)
    assert result == pytest.approx(5.75)
