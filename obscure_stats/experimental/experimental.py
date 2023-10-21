"""
Module for experimental stats
"""


import numpy as np
from scipy import stats  # type: ignore


def coefficient_of_gvariation(x: np.ndarray) -> float:
    """
    Function for calculating geometric coefficient of gvariation

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    cgv : float
        The value of coefficient of gvariation.
    """
    return stats.gstd(x) / stats.gmean(x)


def coefficient_of_range(x: np.ndarray) -> float:
    """
    Function for calculating coefficient of range (range divided by midrange)

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    cr : float
        The value of coefficient of range.
    """
    maximum = np.nanmax(x)
    minimum = np.nanmin(x)
    return (maximum - minimum) / (maximum + minimum)


def range_skew(x: np.ndarray) -> float:
    """
    Function for calculating range skew

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    rsk : float
        The value of range skew.
    """
    mean = np.nanmean(x)
    return (np.nanmax(x) - mean) / (mean - np.nanmin(x))


def theil_t_index(x: np.ndarray) -> float:
    """
    Function for calculating Theil T index

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    tti : float
        The value of Theil T Index.
    """
    mean = np.nanmean(x)
    return np.nanmean((x / mean) * np.log(x / mean))


def theil_l_index(x: np.ndarray) -> float:
    """
    Function for calculating Theil L index

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    tli : float
        The value of Theil L Index.
    """
    return np.nanmean(np.log(x / np.nanmean(x)))


def hrel(x: np.ndarray) -> float:
    """
    Function for calculating Relative Entropy

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    hrel : float
        The value of relative entropy.
    """
    freq = np.unique(x, return_counts=True)[1] / len(x)
    return -np.sum(freq * np.log2(freq)) / np.log2(len(freq))


def gini(x: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of a numpy array.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    gin : float
        The value of Gini index.

    References
    ----------
    https://github.com/oliviaguest/gini/blob/master/gini.py
    """
    n = len(x)
    x -= np.nanmin(x)
    x += np.finfo(np.float32).eps
    x = np.sort(x)
    index = np.indices(x.shape)
    return (np.nansum((2 * index - n - 1) * x)) / (n * np.nansum(x))


def rq_index(x: np.ndarray) -> float:
    """
    Function for calculating RQ Index.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    rqi : float
        The value of oscilation RQ index.
    """
    freq = np.unique(x, return_counts=True)[1] / len(x)
    return 1 - np.sum(np.square((0.5 - freq) / 0.5) * freq)


def dm_index(x: np.ndarray) -> float:
    """
    Function for calculating Deviation from the Mode Index.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    dmi : float
        The value of oscilation DM index.
    """
    n = len(x)
    freq = np.unique(x, return_counts=True)[1] / n
    k = len(freq)
    mode_prob = freq.max()
    return 1 - (np.sum(mode_prob - freq[freq < mode_prob]) / (n * max(k - 1, 1)))


def oscillation_coefficient(x: np.ndarray) -> float:
    """
    Function for calculating Oscilation Coefficient

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    oc : float
        The value of oscilation coefficient.
    """
    return (np.nanmax(x) - np.nanmin(x)) / np.nanmean(x)


def hodges_lehmann_sen_location(x: np.ndarray) -> float:
    """
    Function for calculating Hodges-Lehmann-Sen robust location measure (pseudomedian).

    NOTE: this metric uses cartesian product, so the time and memory complexity
    are N^2. It is best to not use it on large arrays.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    hls : float
        The value of Hodges-Lehmann-Sen estimator.

    Reference
    -------
    Hodges, J. L.; Lehmann, E. L. (1963).
    Estimation of location based on ranks.
    Annals of Mathematical Statistics. 34 (2): 598-611.
    """
    product = np.meshgrid(x, x, sparse=True)
    return np.median(product[0] + product[1]) * 0.5


def coefficient_of_unalikeability(x: np.ndarray) -> float:
    """
    Function for calculating coefficient of unalikeability.
    It ranges from 0 to 1. The higher the value, the more unalike the data are.

    NOTE: this metric uses cartesian product, so the time and memory complexity
    are N^2. It is best to not use it on large arrays.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    cu : float
        The value of coefficient of unalikeability.

    Reference
    ---------
    Kader, Gary & Perry, Mike. (2007).
    Variability for Categorical Variables.
    Journal of Statistics Education. 15.
    """
    product = np.meshgrid(x, x, sparse=True)
    N = len(x)
    return (product[0] == product[1]).sum() / (N**2 - N)
