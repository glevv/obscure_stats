"""Collection of pytest fixtures."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture(scope="session")
def x_list_float() -> list[float]:
    """Return a list of floats."""
    return [-1.0, 2.0, 0.0, -2.0, 3.0, 11.0, 0.0, 1.0]


@pytest.fixture(scope="session")
def x_list_int() -> list[int]:
    """Return a list of integers."""
    return [-1, 2, 0, -2, 3, 11, 0, 1]


@pytest.fixture(scope="session")
def x_array_int(x_list_int: list[int]) -> npt.NDArray[np.number]:
    """Return an array of integers."""
    return np.asarray(x_list_int, dtype="int")


@pytest.fixture(scope="session")
def x_array_float(x_list_float: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """Return an array of floats."""
    return np.asarray(x_list_float, dtype="float")


@pytest.fixture(scope="session")
def x_array_nan(x_array_float: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """Return an array of floats with a NaN."""
    temp = x_array_float.copy()
    temp[0] = np.nan
    return temp


@pytest.fixture(scope="session")
def y_list_float() -> list[float]:
    """Return a list of floats."""
    return [-1.0, 1.0, 9.0, 7.0, 0.0, 1.0, 0.0, 1.0]


@pytest.fixture(scope="session")
def y_list_int() -> list[int]:
    """Return a list of integers."""
    return [-1, 1, 9, 7, 0, 1, 0, 1]


@pytest.fixture(scope="session")
def y_array_int(y_list_int: list[int]) -> npt.NDArray[np.number]:
    """Return an array of integers."""
    return np.asarray(y_list_int, dtype="int")


@pytest.fixture(scope="session")
def y_array_float(y_list_float: list[float]) -> npt.NDArray[np.number]:
    """Return an array of floats."""
    return np.asarray(y_list_float, dtype="float")


@pytest.fixture(scope="session")
def y_array_inf(y_array_float: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """Return an array of floats with an inf."""
    temp = y_array_float.copy()
    temp[1] = np.inf
    return temp


@pytest.fixture(scope="session")
def c_list_obj() -> list[str]:
    """Return a list of strings."""
    return ["a", "b", "r", "a", "c", "a", "d", "a", "b", "r", "a"]


@pytest.fixture(scope="session")
def c_array_obj(c_list_obj: list[str]) -> npt.NDArray[np.number]:
    """Return an array of strings."""
    return np.asarray(c_list_obj)


@pytest.fixture(scope="session")
def c_array_nan(c_array_obj: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
    """Return an array of strings with a NaN."""
    temp = c_array_obj.copy()
    temp[1] = np.nan
    return temp


@pytest.fixture(scope="session")
def rank_skewness_test_data() -> npt.NDArray[np.number]:
    """Return test data from the paper for Rank Skew."""
    return np.asarray(
        (
            73.3,
            80.5,
            50.4,
            64.8,
            74.0,
            72.8,
            72.0,
            59.7,
            90.9,
            76.9,
            71.4,
            45.6,
            77.5,
            60.6,
            67.5,
            54.6,
            71.0,
            66.0,
            71.0,
            74.0,
            72.7,
            73.6,
            97.5,
            89.6,
            70.5,
            78.1,
            84.6,
            92.5,
            76.9,
            76.9,
            59.0,
            82.4,
            56.8,
            83.0,
            76.5,
            72.6,
            65.9,
            70.0,
            130.0,
            76.9,
            88.2,
            63.4,
            123.7,
            65.6,
            80.2,
            84.7,
            82.6,
            76.5,
            80.6,
            72.3,
            99.6,
            80.7,
            73.3,
            77.4,
            68.1,
            74.6,
            70.5,
            58.8,
            93.7,
            61.3,
            76.9,
            78.2,
            85.4,
            72.2,
            100.0,
            55.7,
            79.3,
            109.0,
            84.4,
            76.4,
            86.4,
            67.7,
            74.0,
            92.3,
            76.9,
            64.5,
            88.7,
            72.4,
            65.7,
            73.6,
            79.6,
            64.1,
            76.9,
            68.6,
            73.2,
            66.3,
            70.0,
            91.9,
            55.5,
            100.0,
            79.6,
            72.7,
            78.1,
            68.3,
            65.9,
            74.0,
            67.3,
            66.3,
            96.0,
            73.8,
            70.0,
            50.5,
            73.0,
            55.0,
            80.0,
            84.0,
            50.9,
        )
    )


@pytest.fixture(scope="session")
def thdme_test_data() -> npt.NDArray[np.number]:
    """Return test data from the paper for Trimmed Harrell-Davis median."""
    return np.asarray(
        (-0.565, -0.106, -0.095, 0.363, 0.404, 0.633, 1.371, 1.512, 2.018, 100_000)
    )


@pytest.fixture(scope="session")
def hls_test_data() -> npt.NDArray[np.number]:
    """Return test data from the paper for Hodges-Lehmann-Sen estimator."""
    return np.asarray((1, 5, 2, 2, 7, 4, 1, 6))


@pytest.fixture(scope="session")
def hsm_test_data() -> npt.NDArray[np.number]:
    """Return test data for Half Sample Mode."""
    return np.asarray((1, 2, 2, 2, 7, 4, 1, 6))


@pytest.fixture(scope="session")
def hls_test_data_big() -> list[int]:
    """Return test data from the paper for Hodges-Lehmann-Sen estimator."""
    return [10**100, 10**100, 2, 2, 7, 4, 1, 6]


@pytest.fixture(scope="session")
def x_tied_list() -> list[int]:
    """Return test data for the rank correlations."""
    return [0, 0, 1, 1, 2, 3, 4, 6, 7, 8, 8]


@pytest.fixture(scope="session")
def y_tied_list() -> list[int]:
    """Return test data for the rank correlations."""
    return [2, 2, 2, 1, 4, 4, 4, 0, 9, 1, 3]


@pytest.fixture(scope="session")
def l1_corr_data_big() -> npt.NDArray[np.number]:
    """Return test data for L1 Concordance correlation."""
    return np.array(
        [
            494,
            395,
            516,
            434,
            476,
            557,
            413,
            442,
            650,
            433,
            417,
            656,
            267,
            478,
            178,
            423,
            427,
        ],
        dtype=float,
    )


@pytest.fixture(scope="session")
def l1_corr_data_small() -> npt.NDArray[np.number]:
    """Return test data L1 Concordance correlation."""
    return np.array(
        [
            512,
            430,
            520,
            428,
            500,
            600,
            364,
            380,
            658,
            445,
            432,
            626,
            260,
            477,
            259,
            350,
            451,
        ],
        dtype=float,
    )


@pytest.fixture(scope="session")
def shorth_data() -> npt.NDArray[np.number]:
    """Return test data for shorth and length of shortest half."""
    return np.array([1.0, 2.0, 3.0, 4.0, 10.0])
