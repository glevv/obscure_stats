"""Module for measures of kurtosis."""


import numpy as np
from scipy import stats  # type: ignore[import-untyped]


def moors_kurt(x: np.ndarray) -> float:
    """Calculate Moor's vision of kurtosis, based on Z score.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Moor's kurtosis is desired.

    Returns
    -------
    mk : float or array_like.
        The value of Moor's kurtosis.

    References
    ----------
    Moors, J. J. A. (1986).
    The meaning of kurtosis: Darlington reexamined.
    The American Statistician, 40 (4): 283-284,
    """
    return np.nanvar(np.square(stats.zscore(x, nan_policy="omit"))) + 1


def moors_octile_kurt(x: np.ndarray) -> float:
    """Calculate Moors measure of kurtosis based on octiles (uncentered, unscaled).

    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Moor's octile kurtosis is desired.

    Returns
    -------
    mok : float or array_like.
        The value of Moor's octile kurtosis.

    References
    ----------
    Moors, J. J. A. (1988).
    A quantile alternative for kurtosis.
    Journal of the Royal Statistical Society. Series D, 37(1):25-32.
    """
    o1, o2, o3, o5, o6, o7 = np.nanquantile(
        x,
        [0.125, 0.25, 0.375, 0.625, 0.750, 0.875],
    )
    return ((o7 - o5) + (o3 - o1)) / (o6 - o2)


def hogg_kurt(x: np.ndarray) -> float:
    """Calculatie Hogg's kurtosis coefficient.

    It is based on means of values between different percentiles (uncentered, unscaled).
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Hogg's kurtosis coefficient is desired.s

    Returns
    -------
    hgc : float or array_like.
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
    return (np.nanmean(masked_p95) - np.nanmean(masked_p05)) / (
        np.nanmean(masked_p50g) - np.nanmean(masked_p50l)
    )


def crow_siddiqui_kurt(x: np.ndarray) -> float:
    """Calculate Crow & Siddiqui kurtosis coefficient.

    It is based on quartiles and percentiles (uncentered, unscaled).
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Crow & Siddiqui kurtosis coefficient is desired.

    Returns
    -------
    csk : float or array_like.
        The value of Crow & Siddiqui kurtosis coefficient.

    References
    ----------
    Crow, E. L. and Siddiqui, M. (1967).
    Robust estimation of location.
    Journal of the American Statistical Association, 62(318):353-389.
    """
    p025, p25, p75, p975 = np.nanquantile(x, [0.025, 0.25, 0.75, 0.975])
    return (p975 + p025) / (p75 - p25)


def reza_ma_kurt(x: np.ndarray) -> float:
    """Calculatie Reza & Ma kurtosis coefficient.

    It is based on hexadeciles (uncentered, unscaled).
    This measure should be more robust than moment based kurtosis.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Reza & Ma kurtosis coefficient is desired.

    Returns
    -------
    rmk : float or array_like.
        The value of Reza & Ma kurtosis coefficient.

    References
    ----------
    Reza, M.S., & Ma, J. (2016).
    ICA and PCA integrated feature extraction for classification.
    2016 IEEE 13th International Conference on Signal Processing (ICSP), 1083-1088.
    """
    h1, h7, h9, h15 = np.nanquantile(x, [0.0625, 0.4375, 0.5625, 0.9375])
    return ((h15 - h9) + (h7 - h1)) / (h15 - h1)
