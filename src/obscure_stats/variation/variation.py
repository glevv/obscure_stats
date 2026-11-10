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
    float
        The value of mode variation ratio.

    References
    ----------
    Wilcox, A. R. (1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.

    Examples
    --------
    >>> from obscure_stats.variation import mod_vr
    >>> x = ["a", "a", "b", "b", "b", "c"]
    >>> round(mod_vr(x), 4)
    0.5
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
    float
        The value of range variation ratio.

    References
    ----------
    Wilcox, A. R. (1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.

    Examples
    --------
    >>> from obscure_stats.variation import range_vr
    >>> x = ["a", "a", "b", "b", "b", "c"]
    >>> round(range_vr(x), 4)
    0.3333
    """
    x = np.ravel(x)
    cnts = np.unique(x, return_counts=True, equal_nan=True)[1]
    if len(cnts) <= 1:
        # a single category has no variation
        return 0.0
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
    float
        The value of Gibbs M1 index.

    Notes
    -----
    `M1 = 1 - sum(p_i^2)`, i.e. one minus the Simpson index: one minus the sum of the
    squared category proportions, which is the probability that two randomly chosen
    observations fall into different categories. `renyi_entropy` with alpha = 2 equals
    `-log2(1 - M1)`, minus the base-2 logarithm of one minus M1, and `mcintosh_d` equals
    `(1 - sqrt(1 - M1)) * n / (n - sqrt(n))`, one minus the square root of one minus M1,
    rescaled by `n / (n - sqrt(n))`. So for a fixed n all three order samples
    identically.

    References
    ----------
    Gibbs, J. P.; Poston Jr, D. L. (1975).
    The Division of Labor: Conceptualization and Related Measures.
    Social Forces, 53 (3): 468-476.

    Examples
    --------
    >>> from obscure_stats.variation import gibbs_m1
    >>> x = ["a", "a", "b", "b", "b", "c"]
    >>> round(gibbs_m1(x), 4)
    0.6111
    """
    x = np.ravel(x)
    freq = np.unique(x, return_counts=True, equal_nan=True)[1] / len(x)
    return float(1 - np.sum(freq**2))


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
    float
        The value of B index.

    References
    ----------
    Wilcox, A. R. (1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.

    Examples
    --------
    >>> from obscure_stats.variation import b_index
    >>> x = ["a", "a", "b", "b", "b", "c"]
    >>> round(b_index(x), 4)
    0.5822
    """
    x = np.ravel(x)
    # pyrefly: ignore [untyped-import]
    from scipy import stats  # noqa: PLC0415  (lazy, see module docstring)

    n = len(x)
    freq = np.unique(x, return_counts=True, equal_nan=True)[1] / n
    if len(freq) <= 1:
        # a single category has no variation
        return 0.0
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
    float
        The value of AVDev.

    Notes
    -----
    `AVDev = 1 - k / (k - 1) * sum(|p_i - 1/k|) / 2`, where p_i are the category
    proportions: one minus half the summed distance between the proportions and equal
    proportions, rescaled to [0, 1]. Strong's dominance index D_w (Strong, 2002), the
    largest gap between the Lorenz curve of the ranked categories and the line of equal
    proportions, equals `(1 - AVDev) * (k - 1) / k`.

    References
    ----------
    Wilcox, A. R. (1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.

    Examples
    --------
    >>> from obscure_stats.variation import avdev
    >>> x = ["a", "a", "b", "b", "b", "c"]
    >>> round(avdev(x), 4)
    0.75
    """
    x = np.ravel(x)
    n = len(x)
    counts = np.unique(x, return_counts=True, equal_nan=True)[1]
    k = len(counts)
    if k <= 1:
        # a single category has no variation
        return 0.0
    mean = n / k
    return float(1 - (np.sum(np.abs(counts - mean)) / (2 * mean * (k - 1))))


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
    float
        The value of Rényi entropy.

    Notes
    -----
    With alpha = 2 (collision entropy) the value equals `-log2(1 - M1)`, minus the
    base-2 logarithm of the sum of the squared category proportions, a monotone
    transform of Gibbs M1 (`gibbs_m1`); alpha = 1 gives Shannon entropy.

    References
    ----------
    Rényi, A. (1961).
    On measures of information and entropy.
    Proceedings of the fourth Berkeley Symposium on Mathematics,
    Statistics and Probability 1960. pp. 547-561.

    Examples
    --------
    >>> from obscure_stats.variation import renyi_entropy
    >>> x = ["a", "a", "b", "b", "b", "c"]
    >>> round(renyi_entropy(x), 4)
    1.3626
    """
    x = np.ravel(x)
    if alpha < 0:
        msg = "Parameter alpha should be non-negative."
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


def extropy(x: npt.ArrayLike) -> float:
    """Calculate information extropy (bits).

    This measure is complementary to entropy. Extropy `J = -sum((1 - p) log2(1 - p))`,
    where p is the proportion of each category: for every category, take one minus its
    proportion, multiply by its base-2 logarithm, sum, and change the sign. It is
    non-negative and grows with variation, like the other measures in this module.

    Low values of extropy correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Categories (any hashable values, e.g. strings or integer codes);
        each distinct value is one category.

    Returns
    -------
    float
        The value of extropy.

    References
    ----------
    Lad, F.; Sanfilippo, G.; Agro, G. (2015).
    Extropy: Complementary dual of entropy.
    Statistical Science, 30(1), 40-58.

    Examples
    --------
    >>> from obscure_stats.variation import extropy
    >>> x = ["a", "a", "b", "b", "b", "c"]
    >>> round(extropy(x), 4)
    1.1092
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
    float
        The value of McIntosh's D.

    Notes
    -----
    `D = (1 - sqrt(sum(p_i^2))) * n / (n - sqrt(n))`: one minus the square root of the
    sum of the squared category proportions, rescaled by `n / (n - sqrt(n))` so that it
    ranges from 0 to 1. It is a monotone transform of Gibbs M1 (`gibbs_m1`) for a fixed
    n.

    References
    ----------
    McIntosh, R. P. (1967).
    An index of diversity and the relation of certain concepts to diversity.
    Ecology, 48(3), 392-404.

    Examples
    --------
    >>> from obscure_stats.variation import mcintosh_d
    >>> x = ["a", "a", "b", "b", "b", "c"]
    >>> round(mcintosh_d(x), 4)
    0.6361
    """
    x = np.ravel(x)
    n = len(x)
    if n <= 1:
        return np.nan
    counts = np.unique(x, return_counts=True, equal_nan=True)[1]
    return float((n - np.sum(counts**2) ** 0.5) / (n - n**0.5))


def mndif(x: npt.ArrayLike) -> float:
    """Calculate MNDif, the mean difference analogue.

    `MNDif = 1 - sum over pairs i < j of |f_i - f_j| / (n (k - 1))`, where f_i are the
    category frequencies, n is the number of observations and k the number of
    categories. In words: one minus the sum of the absolute differences between the
    frequencies of every pair of categories, divided by n times k - 1. It is the
    categorical analogue of the mean difference (the Gini coefficient applied to
    category frequencies) and ranges from 0 (all observations in one category) to 1 (all
    categories equally frequent).

    Low values of MNDif correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Categories (any hashable values, e.g. strings or integer codes);
        each distinct value is one category.

    Returns
    -------
    float
        The value of MNDif.

    Notes
    -----
    Only categories that occur in x are counted. Unlike `avdev`, which uses the
    deviations of the frequencies from their mean, MNDif uses the differences
    between the frequencies themselves. The pairwise sum is computed in
    O(k log k) from the sorted frequencies.

    References
    ----------
    Wilcox, A. R. (1973).
    Indices of Qualitative Variation and Political Measurement.
    The Western Political Quarterly. 26 (2): 325-343.

    Examples
    --------
    >>> from obscure_stats.variation import mndif
    >>> x = ["a", "a", "b", "b", "b", "c"]
    >>> round(mndif(x), 4)
    0.6667
    """
    x = np.ravel(x)
    n = len(x)
    counts = np.sort(np.unique(x, return_counts=True, equal_nan=True)[1])
    k = len(counts)
    if k <= 1:
        return 0.0
    pair_sum = np.sum((2 * np.arange(1, k + 1) - k - 1) * counts)
    return float(1 - pair_sum / (n * (k - 1)))


def brillouin_index(x: npt.ArrayLike) -> float:
    """Calculate Brillouin's diversity index (nats).

    `H_B = (ln(n!) - sum ln(f_i!)) / n`, where f_i are the category frequencies and n is
    the number of observations. In words: the natural logarithm of the number of
    distinct orderings of the observations (n factorial divided by the product of the
    factorials of the category frequencies), divided by n.

    It is the information per observation of a fully known collection, rather
    than an estimate for a population sampled at random. It is always below the
    Shannon entropy of the same data (in nats) and approaches it as n grows.

    Low values of Brillouin's index correspond to lower variation and
    high values to higher variation.

    Parameters
    ----------
    x : array_like
        Categories (any hashable values, e.g. strings or integer codes);
        each distinct value is one category.

    Returns
    -------
    float
        The value of Brillouin's index, in nats.

    Notes
    -----
    The natural logarithm is used, as in the original definition. Divide by ln(2) (~
    0.6931) to express the value in bits, the unit of `renyi_entropy`. The factorials
    are evaluated through the log-gamma function, `ln(m!) = ln Gamma(m + 1)`, so large
    counts do not overflow.

    References
    ----------
    Brillouin, L. (1956).
    Science and Information Theory.
    Academic Press, New York.

    Margalef, R. (1958).
    Information theory in ecology.
    General Systems, 3, 36-71.

    Examples
    --------
    >>> from obscure_stats.variation import brillouin_index
    >>> x = ["a", "a", "b", "b", "b", "c"]
    >>> round(brillouin_index(x), 4)
    0.6824
    """
    # pyrefly: ignore [untyped-import]
    from scipy import special  # noqa: PLC0415  (lazy, see module docstring)

    x = np.ravel(x)
    n = len(x)
    if n == 0:
        return np.nan
    counts = np.unique(x, return_counts=True, equal_nan=True)[1]
    log_orderings = special.gammaln(n + 1) - np.sum(special.gammaln(counts + 1))
    return float(log_orderings / n) + 0.0  # avoid -0.0
