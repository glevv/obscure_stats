"""
Module for measures of categorical variations
"""


import numpy as np
from scipy import stats  # type: ignore


def mod_vr(x: np.ndarray) -> float:
    """
    Function for calculating Mode Variation Ratio.
    This ratio could be interpreted as the probability of
    category not being the most frequent.

    Low values of ModVR correspond to small amount of variation
    and high values to larger amounts of variation.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose mode variation ratio is desired.

    Returns
    -------
    modvr : float
        The value of mode variation ratio.

    References
    ----------
    Wilcox, Allen R. (June 1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.
    """
    cnts = np.unique(x, return_counts=True)[1]
    n = len(x)
    return 1 - np.max(cnts) / n


def range_vr(x: np.ndarray) -> float:
    """
    Function for calculating Range Variation Ratio.
    Ratio of frequencies of the least and the most common categories.
    This ratio is similar to range or peak-to-peak for real values.

    Low values of RanVR correspond to small amount of variation
    and high values to larger amounts of variation.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose range variation ratio is desired.

    Returns
    -------
    ranvr : float
        The value of range variation ratio.

    References
    ----------
    Wilcox, Allen R. (June 1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.
    """
    cnts = np.unique(x, return_counts=True)[1]
    return np.min(cnts) / np.max(cnts)


def gibbs_m1(x: np.ndarray) -> float:
    """
    Function for calculating Gibbs M1 Index.
    M1 can be interpreted as one minus the likelihood that a random pair
    of samples will belong to the same category (standardized likelihood of
    a random pair falling in the same category).

    Low values of G1 correspond to small amount of variation
    and high values to larger amounts of variation.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Gibbs M1 index is desired.

    Returns
    -------
    m1 : float
        The value of Gibbs M1 index.

    References
    ----------
    Gibbs, Jack P., Poston Jr, Dudley L. (March 1975).
    The Division of Labor: Conceptualization and Related Measures".
    Social Forces, 53 (3): 468-476.

    See also
    -------
    Gini's index of mutability;
    Gini Concentration Index;
    Simpson's measure of diversity;
    Bachi's index of linguistic homogeneity;
    Mueller and Schuessler's index of qualitative variation;
    Gibbs and Martin's index of industry diversification;
    Lieberson's index;
    Blau's index in sociology, psychology and management studies;
    Special case of Tsallis entropy (α = 2).
    """
    freq = np.unique(x, return_counts=True)[1] / len(x)
    return 1 - np.sum(np.square(freq))


def gibbs_m2(x: np.ndarray) -> float:
    """
    Function for calculating Gibbs M2 Index.
    M2 can be interpreted as the ratio of the variance of
    the multinomial distribution to the variance of a binomial distribution.

    Low values of G2 correspond to small amount of variation
    and high values to larger amounts of variation.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Gibbs M2 index is desired.

    Returns
    -------
    m2 : float
        The value of Gibbs M2 index.

    References
    ----------
    Gibbs, Jack P., Poston Jr, Dudley L. (March 1975).
    The Division of Labor: Conceptualization and Related Measures".
    Social Forces, 53 (3): 468-476.
    """
    freq = np.unique(x, return_counts=True)[1] / len(x)
    k = len(freq)
    return (k / (k - 1)) * (1 - np.sum(np.square(freq)))


def b_index(x: np.ndarray) -> float:
    """
    Function for calculating B Index.
    Normalized to 0-1 range geometric mean of probabilities of all categories.

    Low values of BIn correspond to small amount of variation
    and high values to larger amounts of variation.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose B index is desired.

    Returns
    -------
    bi : float
        The value of B index.

    References
    ----------
    Wilcox, Allen R. (June 1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.
    """
    n = len(x)
    freq = np.unique(x, return_counts=True)[1] / n
    return 1 - np.sqrt(1 - np.square(stats.gmean(freq * len(freq) / n)))


def ada_index(x: np.ndarray) -> float:
    """
    Function for calculating Average Deviation Analogue.
    Normalized to 0-1 range categorical analog of the mean deviation.

    Low values of AdaIn correspond to small amount of variation
    and high values to larger amounts of variation.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose ADA index is desired.

    Returns
    -------
    adi : float
        The value of ADA index.

    References
    ----------
    Wilcox, Allen R. (June 1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.
    """
    n = len(x)
    freq = np.unique(x, return_counts=True)[1] / n
    k = len(freq)
    mean = n / k
    return 1 - (np.sum(np.abs(freq - mean)) / (2 * mean * max(k - 1, 1)))


def extropy(x: np.ndarray) -> float:
    """
    Function for calculating Information Extropy (bits).
    Measure complementary to entropy.

    Low values of extropy correspond to high amount of variation
    and high values to smaller amounts of variation.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose extropy is desired.

    Returns
    -------
    ext : float
        The value of extropy.

    References
    ----------
    Lad, F., Sanfilippo, G., & Agro, G. (2015).
    Extropy: Complementary dual of entropy.
    Statistical Science, 30(1), 40-58.
    """
    freq = np.unique(x, return_counts=True)[1] / len(x)
    p = 1 - freq + 1e-7
    return np.sum(p * np.log2(p))


def coefficient_of_unalikeability(x: np.ndarray) -> float:
    """
    Function for calculating coefficient of unalikeability.
    It ranges from 0 to 1. The higher the value, the more unalike the data are.

    NOTE: this statisstic uses cartesian product, so the time and memory complexity
    are N^2. It is best to not use it on large arrays.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    cu : float
        The value of coefficient of unalikeability.

    References
    ----------
    Kader, Gary & Perry, Mike. (2007).
    Variability for Categorical Variables.
    Journal of Statistics Education. 15.
    """
    product = np.meshgrid(x, x, sparse=True)
    n = len(x)
    return ((product[0] == product[1]).sum() - n) / (n**2 - n)
