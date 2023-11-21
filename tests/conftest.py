"""Collection of fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def x_list_float() -> list[float]:
    """List of floats."""
    return [2.0, 0.0, 2.0, 3.0, 11.0, 0.0, 1.0]


@pytest.fixture(scope="session")
def x_list_int() -> list[int]:
    """List of ints."""
    return [2, 0, 2, 3, 11, 0, 1]


@pytest.fixture(scope="session")
def x_array_int(x_list_int: list[int]) -> np.ndarray:
    """Array of ints."""
    return np.asarray(x_list_int, dtype="int")


@pytest.fixture(scope="session")
def x_array_float(x_list_float: np.ndarray) -> np.ndarray:
    """Array of float."""
    return np.asarray(x_list_float, dtype="float")


@pytest.fixture(scope="session")
def x_array_nan(x_array_float: np.ndarray) -> np.ndarray:
    """Array of float with nan."""
    temp = x_array_float.copy()
    temp[0] = np.nan
    return temp


@pytest.fixture(scope="session")
def y_list_float() -> list[float]:
    """List of floats."""
    return [1.0, 9.0, 7.0, 0.0, 1.0, 0.0, 1.0]


@pytest.fixture(scope="session")
def y_list_int() -> list[int]:
    """List of ints."""
    return [1, 9, 7, 0, 1, 0, 1]


@pytest.fixture(scope="session")
def y_array_int(y_list_int: list[int]) -> np.ndarray:
    """Array of ints."""
    return np.asarray(y_list_int, dtype="int")


@pytest.fixture(scope="session")
def y_array_float(y_list_float: list[float]) -> np.ndarray:
    """Array of float."""
    return np.asarray(y_list_float, dtype="float")


@pytest.fixture(scope="session")
def y_array_inf(y_array_float: np.ndarray) -> np.ndarray:
    """Array of float with nan."""
    temp = y_array_float.copy()
    temp[1] = np.inf
    return temp


@pytest.fixture(scope="session")
def c_list_obj() -> list[str]:
    """List of objects."""
    return ["a", "b", "r", "a", "c", "a", "d", "a", "b", "r", "a"]


@pytest.fixture(scope="session")
def c_array_obj(c_list_obj: list[str]) -> np.ndarray:
    """Array of objects."""
    return np.asarray(c_list_obj, dtype="object")


@pytest.fixture(scope="session")
def c_array_nan(c_array_obj: np.ndarray) -> np.ndarray:
    """Array of objects."""
    temp = c_array_obj.copy()
    temp[1] = None
    return temp


@pytest.fixture(scope="session")
def rank_skewness_test_data() -> np.ndarray:
    """Test data from the paper for Rank Skew."""
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
        ),
    )


@pytest.fixture(scope="session")
def thdme_test_data() -> np.ndarray:
    """Test data from the paper for Trimmed Harrles-Davies median."""
    return np.asarray(
        (-0.565, -0.106, -0.095, 0.363, 0.404, 0.633, 1.371, 1.512, 2.018, 100_000),
    )


@pytest.fixture(scope="session")
def hls_test_data() -> np.ndarray:
    """Test data from the paper for Hodges-Lehmann-Sen estimator."""
    return np.asarray((1, 5, 2, 2, 7, 4, 1, 6))


@pytest.fixture(scope="session")
def hsm_test_data() -> np.ndarray:
    """Test data for Half Sample Mode."""
    return np.asarray((1, 2, 2, 2, 7, 4, 1, 6))


@pytest.fixture(scope="session")
def hls_test_data_big() -> list[int]:
    """Test data from the paper for Hodges-Lehmann-Sen estimator."""
    return [10**100, 10**100, 2, 2, 7, 4, 1, 6]
