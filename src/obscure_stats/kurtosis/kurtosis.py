"""Module for measures of kurtosis."""

import numpy as np
import numpy.typing as npt
from scipy import special, stats  # type: ignore[import-untyped]


def l_kurt(x: npt.NDArray) -> float:
    """Calculate standardized linear kurtosis.

    This measure is a 4th linear moment, which is an
    alternative to conventional moments.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    lkr : float
        The value of L-Kurtosis.

    References
    ----------
    Hosking, J. R. M. (1990).
    L-moments: analysis and estimation of distributions
    using linear combinations of order statistics.
    Journal of the Royal Statistical Society, Series B. 52 (1): 105-124.
    """
    _x = np.sort(x)
    n = len(_x)
    common = 1 / special.comb(n - 1, (0, 1, 2, 4)) / n
    betas = [
        common[i] * np.nansum(special.comb(np.arange(i, n), i) * _x[i:])
        for i in range(4)
    ]
    l4 = 20 * betas[3] - 30 * betas[2] + 12 * betas[1] - betas[0]
    l2 = 2 * betas[1] - betas[0]
    return float(l4 / l2)


def moors_kurt(x: npt.NDArray) -> float:
    """Calculate Moor's vision of kurtosis, based on Z score.

    The kurtosis can now be seen as a measure of the dispersion of
    squared Z around its expectation.
    Alternatively it can be seen to be a measure of the dispersion
    of Z around +1 and -1.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    mk : float
        The value of Moor's kurtosis.

    References
    ----------
    Moors, J. J. A. (1986).
    The meaning of kurtosis: Darlington reexamined.
    The American Statistician, 40 (4): 283-284,
    """
    return float(np.nanvar(stats.zscore(x, nan_policy="omit") ** 2) + 1)


def moors_octile_kurt(x: npt.NDArray) -> float:
    """Calculate Moors measure of kurtosis based on octiles (uncentered, unscaled).

    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    mok : float
        The value of Moor's octile kurtosis.

    References
    ----------
    Moors, J. J. A. (1988).
    A quantile alternative for kurtosis.
    Journal of the Royal Statistical Society. Series D, 37(1):25-32.
    """
    o1, o2, o3, o5, o6, o7 = np.nanquantile(x, [0.125, 0.25, 0.375, 0.625, 0.75, 0.875])
    return float(((o7 - o5) + (o3 - o1)) / (o6 - o2))


def hogg_kurt(x: npt.NDArray) -> float:
    """Calculatie Hogg's kurtosis coefficient.

    It is based on means of values between different percentiles (uncentered, unscaled).
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    hgc : float
        The value of Hogg's kurtosis coefficient.

    References
    ----------
    Hogg, R. V. (1972).
    More light on the kurtosis and related statistics.
    Journal of the American Statistical Association, 67(338):422-424.
    """
    p05, p50, p95 = np.nanquantile(x, [0.05, 0.5, 0.95])
    masked_p95 = np.where(x >= p95, x, np.nan)
    masked_p05 = np.where(x <= p05, x, np.nan)
    masked_p50g = np.where(x >= p50, x, np.nan)
    masked_p50l = np.where(x <= p50, x, np.nan)
    return float(
        (np.nanmean(masked_p95) - np.nanmean(masked_p05))
        / (np.nanmean(masked_p50g) - np.nanmean(masked_p50l))
    )


def crow_siddiqui_kurt(x: npt.NDArray) -> float:
    """Calculate Crow & Siddiqui kurtosis coefficient.

    It is based on quartiles and percentiles (uncentered, unscaled) and
    tries to compare two different measures of dispersion of the same
    sample.
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    csk : float
        The value of Crow & Siddiqui kurtosis coefficient.

    References
    ----------
    Crow, E. L.; Siddiqui, M. (1967).
    Robust estimation of location.
    Journal of the American Statistical Association, 62(318):353-389.
    """
    p025, p25, p75, p975 = np.nanquantile(x, [0.025, 0.25, 0.75, 0.975])
    return float((p975 - p025) / (p75 - p25))


def reza_ma_kurt(x: npt.NDArray) -> float:
    """Calculatie Reza & Ma kurtosis coefficient.

    It is based on hexadeciles (uncentered, unscaled) and is very
    similar to Moor's octile kurtosis.
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    rmk : float
        The value of Reza & Ma kurtosis coefficient.

    References
    ----------
    Reza, M. S.; Ma, J. (2016).
    ICA and PCA integrated feature extraction for classification.
    2016 IEEE 13th International Conference on Signal Processing (ICSP), 1083-1088.
    """
    h1, h7, h9, h15 = np.nanquantile(x, [0.0625, 0.4375, 0.5625, 0.9375])
    return float(((h15 - h9) + (h7 - h1)) / (h15 - h1))


def staudte_kurt(x: npt.NDArray) -> float:
    """Calculate Staudte kurtosis coefficient.

    It is based on inter-percentile ranges (uncentered, unscaled) and
    tries to compare two different measures of dispersion of the same
    sample.
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    sk : float
        The value of Staudte kurtosis coefficient.

    References
    ----------
    Staudte, R. G. (2017).
    Inference for quantile measures of kurtosis, peakedness, and tail weight.
    Communications in Statistics-Theory and Methods, 46(7), 3148-3163.
    """
    p10, p33, p66, p90 = np.nanquantile(x, [0.1, 1 / 3, 2 / 3, 0.9])
    return float((p90 - p10) / (p66 - p33))


def schmid_trede_peakedness(x: npt.NDArray) -> float:
    """Calculate Schmid and Trder measure of peakedness P.

    It is based on inter-percentile ranges (uncentered, unscaled) and
    tries to compare two different measures of dispersion of the same
    sample.
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    stp : float
        The value of measure of peakedness P.

    References
    ----------
    Schmid, F.; Trede, M. (2003).
    Simple tests for peakedness, fat tails and leptokurtosis based on quantiles.
    Computational Statistics and Data Analysis, 43, 1-12.
    """
    p125, p25, p75, p875 = np.nanquantile(x, [0.125, 0.25, 0.75, 0.875])
    return float((p875 - p125) / (p75 - p25))
