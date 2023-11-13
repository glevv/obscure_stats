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
def c_list_obj() -> list[str]:
    """List of objects."""
    return ["a", "b", "r", "a", "c", "a", "d", "a", "b", "r", "a"]


@pytest.fixture(scope="session")
def c_array_obj(c_list_obj: list[str]) -> np.ndarray:
    """Array of objects."""
    return np.asarray(c_list_obj, dtype="object")
