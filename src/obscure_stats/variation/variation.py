# Copyright (c) 2023 Hleb Levitski
# Licensed under the MIT License. See LICENSE in the project root for details.

"""Module for measures of categorical variations.

scipy is imported lazily inside functions: it takes ~0.3 s to import.
"""

import math

import numpy as np
import numpy.typing as npt


def mod_vr(x: npt.ArrayLike) -> float:
    """Calculate Mode Variation Ratio.

    This ratio could be interpreted as the probability of
    category not being the most frequent.

    Low values of Mode VR correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Categories (any hashable values, e.g. strings or integer codes);
        each distinct value is one category.

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
    x = np.ravel(x)
    cnts = np.unique(x, return_counts=True, equal_nan=True)[1]
    return float(1 - np.max(cnts) / len(x))


def range_vr(x: npt.ArrayLike) -> float:
    """Calculate Range Variation Ratio.

    Ratio of frequencies of the least and the most common categories.
    This ratio is similar to range or peak-to-peak for real values.

    Low values of Range VR correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Categories (any hashable values, e.g. strings or integer codes);
        each distinct value is one category.

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
    x = np.ravel(x)
    cnts = np.unique(x, return_counts=True, equal_nan=True)[1]
    return float(np.min(cnts) / np.max(cnts))


def gibbs_m1(x: npt.ArrayLike) -> float:
    """Calculate Gibbs M1 Index.

    M1 can be interpreted as one minus the likelihood that a random pair
    of samples will belong to the same category (standardized likelihood of
    a random pair falling in the same category).

    Low values of Gibbs M1 correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Categories (any hashable values, e.g. strings or integer codes);
        each distinct value is one category.

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

    Notes
    -----
    M1 = 1 - sum(p_i^2), i.e. one minus the Simpson index. `renyi_entropy`
    with alpha = 2 equals -log2(1 - M1) and `mcintosh_d` equals
    (1 - sqrt(1 - M1)) * n / (n - sqrt(n)), so for a fixed n all three order
    samples identically.
    """
    x = np.ravel(x)
    freq = np.unique(x, return_counts=True, equal_nan=True)[1] / len(x)
    return float(1 - np.sum(freq**2))


def gibbs_m2(x: npt.ArrayLike) -> float:
    """Calculate Gibbs M2 Index.

    M2 can be interpreted as the ratio of the variance of
    the multinomial distribution to the variance of a binomial distribution.

    Low values of Gibbs M2 correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Categories (any hashable values, e.g. strings or integer codes);
        each distinct value is one category.

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
    x = np.ravel(x)
    freq = np.unique(x, return_counts=True, equal_nan=True)[1] / len(x)
    k = len(freq)
    return float((k / (k - 1)) * (1 - np.sum(freq**2))) if k > 1 else 0.0


def b_index(x: npt.ArrayLike) -> float:
    """Calculate B Index.

    Normalized to 0-1 range geometric mean of probabilities of all categories.

    Low values of B Index correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Categories (any hashable values, e.g. strings or integer codes);
        each distinct value is one category.

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
    x = np.ravel(x)
    # pyrefly: ignore [untyped-import]
    from scipy import stats  # noqa: PLC0415  (lazy, see module docstring)

    n = len(x)
    freq = np.unique(x, return_counts=True, equal_nan=True)[1] / n
    return float(1 - (1 - (stats.gmean(freq * len(freq))) ** 2) ** 0.5)


def avdev(x: npt.ArrayLike) -> float:
    """Calculate Average Deviation Analogue.

    Normalized to 0-1 range categorical analogue of the mean deviation.

    Low values of AVDev correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Categories (any hashable values, e.g. strings or integer codes);
        each distinct value is one category.

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
    x = np.ravel(x)
    n = len(x)
    counts = np.unique(x, return_counts=True, equal_nan=True)[1]
    k = len(counts)
    mean = n / k
    return float(1 - (np.sum(np.abs(counts - mean)) / (2 * mean * max(k - 1, 1))))


def renyi_entropy(
    x: npt.ArrayLike, alpha: float = 2, *, normalize: bool = False
) -> float:
    """Calculate Renyi entropy (bits).

    Rényi entropy is a quantity that generalizes various notions of entropy,
    including Hartley entropy, Shannon entropy, collision entropy, and min-entropy.

    Low values of Rényi entropy correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Categories (any hashable values, e.g. strings or integer codes);
        each distinct value is one category.
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

    Notes
    -----
    With alpha = 2 (collision entropy) the value equals -log2(1 - M1), a
    monotone transform of Gibbs M1 (`gibbs_m1`); alpha = 1 gives Shannon
    entropy.
    """
    x = np.ravel(x)
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
    if math.isinf(alpha):
        # min-entropy
        return float(-math.log2(np.max(freq)) / normalizer)
    return float(math.log2(np.sum(freq**alpha)) / (1 - alpha) / normalizer)


def negative_extropy(x: npt.ArrayLike) -> float:
    """Calculate Negative Information Extropy (bits).

    This measure is complementary to entropy.
    Extropy J = -sum((1 - p) log2(1 - p)) is non-negative and grows with
    variation, like the other measures in this module.

    Low values of negative extropy correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Categories (any hashable values, e.g. strings or integer codes);
        each distinct value is one category.

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
    x = np.ravel(x)
    freq = np.unique(x, return_counts=True, equal_nan=True)[1] / len(x)
    p_inv = 1.0 - freq
    terms = p_inv * np.log2(np.where(p_inv > 0, p_inv, 1.0))
    return float(-np.sum(terms)) + 0.0  # avoid -0.0


def mcintosh_d(x: npt.ArrayLike) -> float:
    """Calculate McIntosh's D.

    Ranges from 0 to 1, where 0 corresponds to no diversity,
    and 1 to maximum diversity.

    Parameters
    ----------
    x : array_like
        Categories (any hashable values, e.g. strings or integer codes);
        each distinct value is one category.

    Returns
    -------
    mid : float
        The value of McIntosh's D.

    References
    ----------
    McIntosh, R. P. (1967).
    An index of diversity and the relation of certain concepts to diversity.
    Ecology, 48(3), 392-404.

    Notes
    -----
    D = (1 - sqrt(sum(p_i^2))) * n / (n - sqrt(n)), a monotone transform of
    Gibbs M1 (`gibbs_m1`) for a fixed n.
    """
    x = np.ravel(x)
    n = len(x)
    if n <= 1:
        return np.nan
    counts = np.unique(x, return_counts=True, equal_nan=True)[1]
    return float((n - np.sum(counts**2) ** 0.5) / (n - n**0.5))
