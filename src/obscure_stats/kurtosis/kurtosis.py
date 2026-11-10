# Copyright (c) 2023 Hleb Levitski
# Licensed under the MIT License. See LICENSE in the project root for details.

"""Module for measures of kurtosis."""

import numpy as np
import numpy.typing as npt


def moors_octile_kurt(x: npt.ArrayLike) -> float:
    """Calculate Moors measure of kurtosis based on octiles (uncentered, unscaled).

    It is `((O7 - O5) + (O3 - O1)) / (O6 - O2)`, where `Oi = Q(i/8)` is the i-th octile.
    In words: the distance from the fifth to the seventh octile plus the distance from
    the first to the third, divided by the interquartile range.

    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of Moors' octile kurtosis.

    References
    ----------
    Moors, J. J. A. (1988).
    A quantile alternative for kurtosis.
    Journal of the Royal Statistical Society. Series D, 37(1):25-32.

    Examples
    --------
    >>> from obscure_stats.kurtosis import moors_octile_kurt
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(moors_octile_kurt(x), 4)
    1.24
    """
    x = np.asarray(x, dtype=np.float64)
    o1, o2, o3, o5, o6, o7 = np.nanquantile(x, [0.125, 0.25, 0.375, 0.625, 0.75, 0.875])
    return float(((o7 - o5) + (o3 - o1)) / (o6 - o2))


def hogg_kurt(x: npt.ArrayLike) -> float:
    """Calculate Hogg's kurtosis coefficient.

    It is `(U(0.05) - L(0.05)) / (U(0.5) - L(0.5))`, where U(a) is the mean of the
    values at or above the upper a-quantile and L(a) the mean of the values at or below
    the lower a-quantile. In words: the mean of the top 5% minus the mean of the bottom
    5%, divided by the mean of the upper half minus the mean of the lower half.

    It is based on means of values between different percentiles (uncentered, unscaled).
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of Hogg's kurtosis coefficient.

    References
    ----------
    Hogg, R. V. (1972).
    More light on the kurtosis and related statistics.
    Journal of the American Statistical Association, 67(338):422-424.

    Examples
    --------
    >>> from obscure_stats.kurtosis import hogg_kurt
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(hogg_kurt(x), 4)
    2.2
    """
    x = np.asarray(x, dtype=np.float64)
    p05, p50, p95 = np.nanquantile(x, [0.05, 0.5, 0.95])
    masked_p95 = np.where(x >= p95, x, np.nan)
    masked_p05 = np.where(x <= p05, x, np.nan)
    masked_p50g = np.where(x >= p50, x, np.nan)
    masked_p50l = np.where(x <= p50, x, np.nan)
    return float(
        (np.nanmean(masked_p95) - np.nanmean(masked_p05))
        / (np.nanmean(masked_p50g) - np.nanmean(masked_p50l))
    )


def crow_siddiqui_kurt(x: npt.ArrayLike) -> float:
    """Calculate Crow & Siddiqui kurtosis coefficient.

    It is `(Q(0.975) - Q(0.025)) / (Q(0.75) - Q(0.25))`, where Q is the sample quantile:
    the width of the central 95% of the data divided by the interquartile range.

    It is based on quartiles and percentiles (uncentered, unscaled) and
    tries to compare two different measures of dispersion of the same
    sample.
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of Crow & Siddiqui kurtosis coefficient.

    References
    ----------
    Crow, E. L.; Siddiqui, M. (1967).
    Robust estimation of location.
    Journal of the American Statistical Association, 62(318):353-389.

    Examples
    --------
    >>> from obscure_stats.kurtosis import crow_siddiqui_kurt
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(crow_siddiqui_kurt(x), 4)
    3.184
    """
    x = np.asarray(x, dtype=np.float64)
    p025, p25, p75, p975 = np.nanquantile(x, [0.025, 0.25, 0.75, 0.975])
    return float((p975 - p025) / (p75 - p25))


def reza_ma_kurt(x: npt.ArrayLike) -> float:
    """Calculate Reza & Ma kurtosis coefficient.

    It is `((H15 - H9) + (H7 - H1)) / (H15 - H1)`, where `Hi = Q(i/16)` is the i-th
    hexadecile. In words: the distance from the 9th to the 15th hexadecile plus the
    distance from the 1st to the 7th, divided by the distance from the 1st to the 15th.

    It is based on hexadeciles (uncentered, unscaled) and is very
    similar to Moors' octile kurtosis.
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of Reza & Ma kurtosis coefficient.

    References
    ----------
    Reza, M. S.; Ma, J. (2017).
    Quantile Kurtosis in ICA and Integrated Feature Extraction for Classification
    ICIC 2017, LNCS 10361, pp. 681-692. Springer.

    Examples
    --------
    >>> from obscure_stats.kurtosis import reza_ma_kurt
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(reza_ma_kurt(x), 4)
    0.8433
    """
    x = np.asarray(x, dtype=np.float64)
    h1, h7, h9, h15 = np.nanquantile(x, [0.0625, 0.4375, 0.5625, 0.9375])
    return float(((h15 - h9) + (h7 - h1)) / (h15 - h1))


def staudte_kurt(x: npt.ArrayLike) -> float:
    """Calculate Staudte kurtosis coefficient.

    It is `(Q(0.9) - Q(0.1)) / (Q(2/3) - Q(1/3))`, where Q is the sample quantile: the
    distance between the 10th and the 90th percentiles divided by the distance between
    the two terciles.

    It is based on inter-percentile ranges (uncentered, unscaled) and
    tries to compare two different measures of dispersion of the same
    sample.
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of Staudte kurtosis coefficient.

    References
    ----------
    Staudte, R. G. (2017).
    Inference for quantile measures of kurtosis, peakedness, and tail weight.
    Communications in Statistics-Theory and Methods, 46(7), 3148-3163.

    Examples
    --------
    >>> from obscure_stats.kurtosis import staudte_kurt
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(staudte_kurt(x), 4)
    3.1385
    """
    x = np.asarray(x, dtype=np.float64)
    p10, p33, p66, p90 = np.nanquantile(x, [0.1, 1 / 3, 2 / 3, 0.9])
    return float((p90 - p10) / (p66 - p33))


def schmid_trede_peakedness(x: npt.ArrayLike) -> float:
    """Calculate Schmid and Trede measure of peakedness P.

    It is `(Q(0.875) - Q(0.125)) / (Q(0.75) - Q(0.25))`, where Q is the sample quantile:
    the distance between the first and the seventh octiles divided by the interquartile
    range.

    It is based on inter-percentile ranges (uncentered, unscaled) and
    tries to compare two different measures of dispersion of the same
    sample.
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of measure of peakedness P.

    References
    ----------
    Schmid, F.; Trede, M. (2003).
    Simple tests for peakedness, fat tails and leptokurtosis based on quantiles.
    Computational Statistics and Data Analysis, 43, 1-12.

    Examples
    --------
    >>> from obscure_stats.kurtosis import schmid_trede_peakedness
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(schmid_trede_peakedness(x), 4)
    1.84
    """
    x = np.asarray(x, dtype=np.float64)
    p125, p25, p75, p875 = np.nanquantile(x, [0.125, 0.25, 0.75, 0.875])
    return float((p875 - p125) / (p75 - p25))


def quantile_weight(x: npt.ArrayLike, q: float = 0.75) -> float:
    """Calculate quantile weight (QW), a robust measure of tail weight.

    It measures how heavy one tail of the distribution is: q in (0.5, 1)
    gives the right tail (right quantile weight, RQW) and q in (0, 0.5) the
    left tail (left quantile weight, LQW). It is the quantile skewness of the
    outer half on that side, so it does not look at the peak of the
    distribution.

    For symmetric data both tails are equal and rank distributions like kurtosis. For
    skewed data the two tails differ: the difference `quantile_weight(x, q) -
    quantile_weight(x, 1 - q)` measures the asymmetry of the tails, which is a different
    property from the asymmetry of the middle of the distribution measured by
    `bowley_skew`.

    Parameters
    ----------
    x : array_like
        Numeric values.
    q : float, default = 0.75
        Tail and quantile level: (0.5, 1) for the right tail, (0, 0.5) for the
        left tail. Brys et al. study 0.75 / 0.25 and 0.875 / 0.125; the outer
        levels have more power, the inner ones a higher breakdown point.

    Returns
    -------
    float
        The value of the quantile weight.

    See Also
    --------
    obscure_stats.skewness.bowley_skew : Quartile skewness of the middle half.

    Notes
    -----
    `RQW(q) = (Q((1+q)/2) + Q(1-q/2) - 2 Q(0.75)) / (Q((1+q)/2) - Q(1-q/2))`, where Q is
    the sample quantile: in words, the quantile skewness of the upper half of the data.
    The left tail is computed as the right tail of the reflected data, so
    `quantile_weight(x, q)` equals `quantile_weight(-x, 1 - q)`.

    References
    ----------
    Brys, G.; Hubert, M.; Struyf, A. (2006).
    Robust measures of tail weight.
    Computational Statistics and Data Analysis 50(3), 733-759.

    Examples
    --------
    >>> from obscure_stats.kurtosis import quantile_weight
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(quantile_weight(x), 4)
    0.4468
    """
    x = np.asarray(x, dtype=np.float64)
    if not 0.0 < q < 1.0 or q == 0.5:
        msg = "Parameter q should be in range (0, 0.5) or (0.5, 1)."
        raise ValueError(msg)
    if q < 0.5:
        x, q = -x, 1.0 - q  # the left tail of x is the right tail of -x
    lower, middle, upper = np.nanquantile(x, [1 - q / 2, 0.75, (1 + q) / 2])
    return float((lower + upper - 2 * middle) / (upper - lower))
