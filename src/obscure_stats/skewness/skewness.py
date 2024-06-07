"""Module for measures of skewness."""

from __future__ import annotations

import numpy as np
from scipy import integrate, special, stats  # type: ignore[import-untyped]

from obscure_stats.central_tendency import half_sample_mode


def l_skew(x: np.ndarray) -> float:
    """Calculate standardized linear skewness.

    This measure is a 3rd linear moment, which is an
    alternative to conventional moments.
    The array will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    lsk : float.
        The value of L-Skewness.

    References
    ----------
    Hosking, J. R. M. (1990).
    L-moments: analysis and estimation of distributions
    using linear combinations of order statistics.
    Journal of the Royal Statistical Society, Series B. 52 (1): 105-124.
    """
    _x = np.sort(x, axis=None)
    n = len(_x)
    common = 1 / special.comb(n - 1, (0, 1, 2)) / n
    betas = [
        common[i] * np.nansum(special.comb(np.arange(i, n), i) * _x[i:])
        for i in range(3)
    ]
    l3 = 6 * betas[2] - 6 * betas[1] + betas[0]
    l2 = 2 * betas[1] - betas[0]
    return l3 / l2


def pearson_mode_skew(x: np.ndarray) -> float:
    """Calculate Pearson's mode skew coefficient.

    This measure could be unstable due mode instability.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    pmods : float
        The value of Pearson's mode skew coefficient.

    References
    ----------
    Pearson, E. S.; Hartley, H. O. (1966).
    Biometrika Tables for Statisticians, vols. I and II.
    Cambridge University Press, Cambridge.
    """
    mean = np.nanmean(x)
    mode = stats.mode(x)[0]
    std = np.nanstd(x)
    return (mean - mode) / std


def bickel_mode_skew(x: np.ndarray) -> float:
    """Calculate Robust Mode skew with half sample mode.

    This measure should be more stable than Pearson mode
    skewness coefficient.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    bms : float
        The value of Bickel's mode skew coefficient.

    References
    ----------
    Bickel, D. R. (2002).
    Robust estimators of the mode and skewness of continuous data.
    Computational Statistics & Data Analysis, Elsevier, 39(2), 153-163.
    """
    mode = half_sample_mode(x)
    return np.nanmean(np.sign(x - mode))


def pearson_median_skew(x: np.ndarray) -> float:
    """Calculatie Pearson's median skew coefficient.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    pmeds : float
        The value of Pearson's median skew coefficient.

    References
    ----------
    Pearson, E.S.; Hartley, H.O. (1966).
    Biometrika Tables for Statisticians, vols. I and II.
    Cambridge University Press, Cambridge.
    """
    mean = np.nanmean(x)
    median = np.nanmedian(x)
    std = np.nanstd(x)
    return 3 * (mean - median) / std


def medeen_skew(x: np.ndarray) -> float:
    """Calculate Medeen's skewness statistic.

    This measure is similar to Pearson median skewness coefficient
    but uses different normalization (mean absolute deviation from the
    median).

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    mss : float
        The value of Medeen's skewness statistic.

    References
    ----------
    Groeneveld, R.A.; Meeden, G. (1984).
    Measuring Skewness and Kurtosis.
    The Statistician. 33 (4): 391-399.
    """
    median = np.nanmedian(x)
    mean = np.nanmean(x)
    return (mean - median) / np.nanmean(np.abs(x - median))


def bowley_skew(x: np.ndarray) -> float:
    """Calculate Bowley's skewness coefficinet.

    Also known as Yule-Kendall skewness coefficient.
    It is based on quartiles (uncentered, unscaled) and compares the distance
    between the median and each of the two quartiles.
    This measure should be more robust than moment based skewness.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    bsk : float
        The value of Bowley's skewness coefficinet.

    References
    ----------
    Bowley, A. L. (1901).
    Elements of Statistics.
    P.S. King and Son, London.
    """
    q1, q2, q3 = np.nanquantile(x, [0.25, 0.5, 0.75])
    return (q3 + q1 - 2 * q2) / (q3 - q1)


def groeneveld_skew(x: np.ndarray) -> float:
    """Calculate Groeneveld's skewness coefficinet.

    It is based on quartiles (uncentered, unscaled).
    It is similar to Bowley skewness coefficient, but tries to
    reweight distance bwetwen median and quartiles separately.
    This measure should be more robust than moment based skewness.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    gsc : float
        The value of Groeneveld's skewness coefficinet.

    References
    ----------
    Groeneveld, R. A.; Meeden, G. (1984).
    Measuring Skewness and Kurtosis.
    The Statistician. 33 (4): 391-399.
    """
    q1, q2, q3 = np.nanquantile(x, [0.25, 0.5, 0.75])
    rs = (q3 + q1 - 2 * q2) / (q2 - q1)
    ls = (q3 + q1 - 2 * q2) / (q3 - q2)
    return rs if abs(rs) > abs(ls) else ls


def kelly_skew(x: np.ndarray) -> float:
    """Calculate Kelly's skewness coefficinet.

    It is based on deciles (uncentered, unscaled).
    Similar to Bowley skewness coefficient, but instead of
    quartiles, deciles are used.
    This measure should be more robust than moment based skewness.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    ksc : float
        The value of Kelly's skewness coefficinet.

    References
    ----------
    David, F. N.; Johnson, N. L. (1956).
    Some tests of significance with ordered variables.
    J. R. Stat. Soc. Ser. B Stat. Methodol. 18, 1-31.
    """
    d1, d5, d9 = np.nanquantile(x, [0.1, 0.5, 0.9])
    return (d9 + d1 - 2 * d5) / (d9 - d1)


def hossain_adnan_skew(x: np.ndarray) -> float:
    """Calculate Houssain and Adnan skewness coefficient.

    It is based on differences from the median, and is somewhar similar
    to Bickel mode skewness.
    This measure should be more robust than moment based skewness.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    has : float
        The value of Houssain and Adnan skewness coefficient.

    References
    ----------
    Hossain, M. F.; Adnan, M. A. S. A (2007).
    A New Approach to Determine the Asymmetry of a Distribution.
    Journal of Applied St atistical Science, Vol.15, pp. 127-134.
    """
    diff = x - np.nanmedian(x)
    return np.nanmean(diff) / np.nanmean(np.abs(diff))


def forhad_shorna_rank_skew(x: np.ndarray) -> float:
    """Calculate Forhad-Shorna coefficient of rank skewness.

    This measure is similar to Houssain and Adnan skewness coefficient,
    but uses differences in ranks instead of absolute differences.
    This measure should be more robust than moment based skewness.
    The array will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    fsrs : float
        The value of Forhad-Shorna coefficient of rank skewness.

    References
    ----------
    Shorna, U. S.; Hossain, M. (2019).
    A New Approach to Determine the Coefficient of Skewness and
    An Alternative Form of Boxplot.
    arXiv preprint arXiv:1908.06400.
    """
    _x = np.ravel(x)
    mr = (np.nanmin(_x) + np.nanmax(_x)) * 0.5
    arr = np.r_[_x, mr]
    arr_ranked = stats.rankdata(arr, method="min", nan_policy="omit")
    diff = arr_ranked[-1] - arr_ranked
    diff = diff[:-1]
    return np.nansum(diff) / np.nansum(np.abs(diff))


def _auc_skew_gamma(x: np.ndarray, dp: float, w: np.ndarray | float) -> float:
    """Calculate AUC skew."""
    n = int(1 / dp)
    half_n = n // 2
    qs = np.nanquantile(x, np.r_[np.linspace(0, 1, n), 0.5])
    med = qs[-1]
    qs = qs[:-1]
    qs_low = qs[:half_n]
    qs_high = qs[-half_n:]
    skews = (qs_low + qs_high - 2 * med) / (qs_high - qs_low) * w
    return integrate.trapezoid(skews, dx=dp)


def auc_skew_gamma(x: np.ndarray, dp: float = 0.01) -> float:
    """Calculate area under the curve of generalized Bowley skewness coefficients.

    This measure tries to combine multiple generalized Bowley skewness coefficients
    into one value.

    Parameters
    ----------
    x : array_like
        Input array.
    dp : float, default = 0.01
        Step used in calculating area under the curve (integrating).

    Returns
    -------
    aucbs : float
        The value of AUC Bowley skewness.

    References
    ----------
    Arachchige, C. N.; & Prendergast, L. A. (2019).
    Mean skewness measures.
    arXiv preprint arXiv:1912.06996.
    """
    w = 1.0
    return _auc_skew_gamma(x, dp, w)


def wauc_skew_gamma(x: np.ndarray, dp: float = 0.01) -> float:
    """
    Calculate weighted area under the curve of generalized Bowley skewness coefficients.

    This version use reweightning. It will assign bigger weights to the
    Bowley skewness coefficients calculated on percentiles far from the median.

    Parameters
    ----------
    x : array_like
        Input array.
    dp : float, default = 0.01
        Step used in calculating area under the curve (integrating).

    Returns
    -------
    waucbs : float
        The value of weighted AUC Bowley skewness.

    References
    ----------
    Arachchige, C. N.; & Prendergast, L. A. (2019).
    Mean skewness measures.
    arXiv preprint arXiv:1912.06996.
    """
    n = int(1 / dp)
    half_n = n // 2
    w = (np.arange(half_n) / half_n)[::-1]
    return _auc_skew_gamma(x, dp, w)


def cumulative_skew(x: np.ndarray) -> float:
    """
    Calculate cumulative measure of skewness.

    It is based on calculating the cumulative statistics of the Lorenz curve.
    The array will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    csc : float
        The value of cumulative skew.

    References
    ----------
    Schlemmer, M. (2022).
    A robust measure of skewness using cumulative statistic calculation.
    arXiv preprint arXiv:2209.10699.
    """
    _x = np.sort(x, axis=None)
    n = len(_x)
    p = np.nancumsum(_x)
    p = p / p[-1]
    r = np.arange(n)
    q = r / n
    d = q - p
    w = (2 * r - n) * 3 / n
    return np.sum(d * w) / np.sum(d)
