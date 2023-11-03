"""Collection of fixtures."""

import typing

import numpy as np
import pytest


@pytest.fixture(scope="session")
def x_list_float() -> typing.Generator:
    """List of floats."""
    yield [2.0, 0.0, 2.0, 3.0, 11.0, 0.0, 1.0]


@pytest.fixture(scope="session")
def x_list_int() -> typing.Generator:
    """List of ints."""
    yield [2, 0, 2, 3, 11, 0, 1]


@pytest.fixture(scope="session")
def x_array_int(x_list_int: typing.Generator) -> typing.Generator:
    """Array of ints."""
    yield np.asarray(x_list_int, dtype="int")


@pytest.fixture(scope="session")
def x_array_float(x_list_float: typing.Generator) -> typing.Generator:
    """Array of float."""
    yield np.asarray(x_list_float, dtype="float")


@pytest.fixture(scope="session")
def y_list_float() -> typing.Generator:
    """List of floats."""
    yield [1.0, 9.0, 7.0, 0.0, 1.0, 0.0, 1.0]


@pytest.fixture(scope="session")
def y_list_int() -> typing.Generator:
    """List of ints."""
    yield [1, 9, 7, 0, 1, 0, 1]


@pytest.fixture(scope="session")
def y_array_int(y_list_int: typing.Generator) -> typing.Generator:
    """Array of ints."""
    yield np.asarray(y_list_int, dtype="int")


@pytest.fixture(scope="session")
def y_array_float(y_list_float: typing.Generator) -> typing.Generator:
    """Array of float."""
    yield np.asarray(y_list_float, dtype="float")


@pytest.fixture(scope="session")
def c_list_obj() -> typing.Generator:
    """List of objects."""
    yield ["a", "b", "r", "a", "c", "a", "d", "a", "b", "r", "a"]


@pytest.fixture(scope="session")
def c_array_obj(c_list_obj: typing.Generator) -> typing.Generator:
    """Array of objects."""
    yield np.asarray(c_list_obj, dtype="object")
