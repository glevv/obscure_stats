"""Module for measures of categorical variations."""


from collections import Counter

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


def mod_vr(x: np.ndarray) -> float:
    """Calculate Mode Variation Ratio.

    This ratio could be interpreted as the probability of
    category not being the most frequent.

    Low values of Mode VR correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    modvr : float
        The value of mode variation ratio.

    References
    ----------
    Wilcox, A. R. (1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.
    """
    cnts = np.asarray(list(Counter(x).values()))
    n = len(x)
    return 1 - np.max(cnts) / n


def range_vr(x: np.ndarray) -> float:
    """Calculate Range Variation Ratio.

    Ratio of frequencies of the least and the most common categories.
    This ratio is similar to range or peak-to-peak for real values.

    Low values of Range VR correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    ranvr : float
        The value of range variation ratio.

    References
    ----------
    Wilcox, A. R. (1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.
    """
    cnts = np.asarray(list(Counter(x).values()))
    return np.min(cnts) / np.max(cnts)


def gibbs_m1(x: np.ndarray) -> float:
    """Calculate Gibbs M1 Index.

    M1 can be interpreted as one minus the likelihood that a random pair
    of samples will belong to the same category (standardized likelihood of
    a random pair falling in the same category).

    Low values of Gibbs M1 correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    m1 : float
        The value of Gibbs M1 index.

    References
    ----------
    Gibbs, J. P.; Poston Jr, D. L. (1975).
    The Division of Labor: Conceptualization and Related Measures.
    Social Forces, 53 (3): 468-476.

    See Also
    --------
    Gini's index of mutability;
    Gini Concentration Index;
    Simpson's measure of diversity;
    Bachi's index of linguistic homogeneity;
    Mueller and Schuessler's index of qualitative variation;
    Gibbs and Martin's index of industry diversification;
    Lieberson's index;
    Blau's index in sociology, psychology and management studies;
    Special case of Tsallis entropy (alpha = 2).
    """
    freq = np.asarray(list(Counter(x).values())) / len(x)
    return 1 - np.sum(np.square(freq))


def gibbs_m2(x: np.ndarray) -> float:
    """Calculate Gibbs M2 Index.

    M2 can be interpreted as the ratio of the variance of
    the multinomial distribution to the variance of a binomial distribution.

    Low values of Gibbs M2 correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    m2 : float
        The value of Gibbs M2 index.

    References
    ----------
    Gibbs, J. P.; Poston Jr, D. L. (1975).
    The Division of Labor: Conceptualization and Related Measures.
    Social Forces, 53 (3): 468-476.
    """
    freq = np.asarray(list(Counter(x).values())) / len(x)
    k = len(freq)
    return (k / (k - 1)) * (1 - np.sum(np.square(freq)))


def b_index(x: np.ndarray) -> float:
    """Calculate B Index.

    Normalized to 0-1 range geometric mean of probabilities of all categories.

    Low values of B Index correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    bi : float
        The value of B index.

    References
    ----------
    Wilcox, A. R. (1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.
    """
    n = len(x)
    freq = np.asarray(list(Counter(x).values())) / n
    return 1 - np.sqrt(1 - np.square(stats.gmean(freq * len(freq) / n)))


def avdev(x: np.ndarray) -> float:
    """Calculate Average Deviation Analogue.

    Normalized to 0-1 range categorical analogue of the mean deviation.

    Low values of AVDev correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    avd : float
        The value of AVDev.

    References
    ----------
    Wilcox, A. R. (1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.
    """
    n = len(x)
    freq = np.asarray(list(Counter(x).values())) / n
    k = len(freq)
    mean = n / k
    return 1 - (np.sum(np.abs(freq - mean)) / (2 * mean * max(k - 1, 1)))


def negative_extropy(x: np.ndarray) -> float:
    """Calculate Negative Information Extropy (bits).

    This measure is complementary to entropy.
    This implementation inverses the sign of extropy to align it with
    all other measures.

    Low values of negative extropy correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    ext : float
        The value of negative extropy.

    References
    ----------
    Lad, F.; Sanfilippo, G.; Agro, G. (2015).
    Extropy: Complementary dual of entropy.
    Statistical Science, 30(1), 40-58.
    """
    freq = np.asarray(list(Counter(x).values())) / len(x)
    p = 1.0 - freq + 1e-7
    return -np.sum(p * np.log2(p))
