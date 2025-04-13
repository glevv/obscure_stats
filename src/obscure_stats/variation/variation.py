"""Module for measures of categorical variations."""

import math

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
    cnts = np.unique(x, return_counts=True, equal_nan=True)[1]
    return float(1 - np.max(cnts) / len(x))


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
    cnts = np.unique(x, return_counts=True, equal_nan=True)[1]
    return float(np.min(cnts) / np.max(cnts))


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
    freq = np.unique(x, return_counts=True, equal_nan=True)[1] / len(x)
    return float(1 - np.sum(freq**2))


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
    freq = np.unique(x, return_counts=True, equal_nan=True)[1] / len(x)
    k = len(freq)
    return float((k / (k - 1)) * (1 - np.sum(freq**2))) if k > 1 else 0.0


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
    freq = np.unique(x, return_counts=True, equal_nan=True)[1] / n
    return float(1 - (1 - (stats.gmean(freq * len(freq) / n)) ** 2) ** 0.5)


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
    freq = np.unique(x, return_counts=True, equal_nan=True)[1] / n
    k = len(freq)
    mean = n / k
    return float(1 - (np.sum(np.abs(freq - mean)) / (2 * mean * max(k - 1, 1))))


def renyi_entropy(x: np.ndarray, alpha: float = 2, *, normalize: bool = False) -> float:
    """Calculate Renyi entropy (bits).

    Rényi entropy is a quantity that generalizes various notions of entropy,
    including Hartley entropy, Shannon entropy, collision entropy, and min-entropy.

    Low values of Rényi entropy correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Input array.
    alpha : float, default = 2
        Order of the Rényi entropy.
    normalize : bool, default = False
        Whether to normalize the result.

    Returns
    -------
    ren : float
        The value of Rényi entropy.

    References
    ----------
    Rényi, A. (1961).
    On measures of information and entropy.
    Proceedings of the fourth Berkeley Symposium on Mathematics,
    Statistics and Probability 1960. pp. 547-561.
    """
    if alpha < 0:
        msg = "Parameter alpha should be positive!"
        raise ValueError(msg)
    freq = np.unique(x, return_counts=True, equal_nan=True)[1] / len(x)
    n = len(freq)
    if n <= 1:
        # return 0 if an array is constant
        return 0.0
    normalizer = math.log2(n) if normalize else 1.0
    if alpha == 1:
        # return Shannon entropy to avoid division by 0
        return float(-np.sum(freq * np.log2(freq)) / normalizer)
    return float(1 / (1 - alpha) * math.log2(np.sum(freq**alpha)) / normalizer)


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
    freq = np.unique(x, return_counts=True, equal_nan=True)[1] / len(x)
    p_inv = 1.0 - freq
    return float(-np.sum(p_inv * np.log2(p_inv)))


def mcintosh_d(x: np.ndarray) -> float:
    """Calculate McIntosh's D.

    Ranges from 0 to 1, where 0 corresponds to no diversity,
    and 1 to maximum diversity.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    mid : float
        The value of McIntosh's D.

    References
    ----------
    McIntosh, R. P. (1967).
    An index of diversity and the relation of certain concepts to diversity.
    Ecology, 48(3), 392-404.
    """
    n = len(x)
    counts = np.unique(x, return_counts=True, equal_nan=True)[1]
    return float((n - np.sum(counts**2) ** 0.5) / (n - n**0.5))
