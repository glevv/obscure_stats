"""Module for measures of skewness."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import integrate, special, stats  # type: ignore[import-untyped]

from obscure_stats.central_tendency import half_sample_mode


def l_skew(x: npt.NDArray) -> float:
    """Calculate standardized linear skewness.

    This measure is a 3rd linear moment, which is an
    alternative to conventional moments.

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
    _x = np.sort(x)
    n = len(_x)
    common = 1 / special.comb(n - 1, (0, 1, 2)) / n
    betas = [
        common[i] * np.nansum(special.comb(np.arange(i, n), i) * _x[i:])
        for i in range(3)
    ]
    l3 = 6 * betas[2] - 6 * betas[1] + betas[0]
    l2 = 2 * betas[1] - betas[0]
    return float(l3 / l2)


def pearson_mode_skew(x: npt.NDArray) -> float:
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
    return float((mean - mode) / std)


def bickel_mode_skew(x: npt.NDArray) -> float:
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
    return float(np.nanmean(np.sign(np.asarray(x) - mode)))


def pearson_median_skew(x: npt.NDArray) -> float:
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
    return float(3.0 * (mean - median) / std)


def medeen_skew(x: npt.NDArray) -> float:
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
    return float((mean - median) / np.nanmean(np.abs(x - median)))


def bowley_skew(x: npt.NDArray) -> float:
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
    return float((q3 + q1 - 2 * q2) / (q3 - q1))


def groeneveld_range_skew(x: npt.NDArray) -> float:
    """Calculate Groeneveld's skewness coefficinet.

    This measure should be more robust than moment based skewness.
    This is the implementation of 'b4' skewness from the paper.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    grs : float
        The value of Groeneveld's range skewness coefficinet.

    References
    ----------
    Groeneveld, R. A.; Meeden, G. (1984).
    Measuring Skewness and Kurtosis.
    The Statistician. 33 (4): 391-399.
    """
    m = np.nanmedian(x)
    min_ = np.nanmin(x)
    max_ = np.nanmax(x)
    return float((max_ + min_ - 2 * m) / (max_ - min_))


def kelly_skew(x: npt.NDArray) -> float:
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
    return float((d9 + d1 - 2 * d5) / (d9 - d1))


def hossain_adnan_skew(x: npt.NDArray) -> float:
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
    return float(np.nanmean(diff) / np.nanmean(np.abs(diff)))


def forhad_shorna_rank_skew(x: npt.NDArray) -> float:
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
    arr_ranked = stats.rankdata(arr, method="average", nan_policy="omit")
    diff = arr_ranked[-1] - arr_ranked
    diff = diff[:-1]
    return float(np.nansum(diff) / np.nansum(np.abs(diff)))


def auc_skew_gamma(x: npt.NDArray, dp: float = 0.01) -> float:
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
    if dp <= 0:
        msg = "Parameter dp should be > 0."
        raise ValueError(msg)
    n = int(1 / dp)
    half_n = n // 2
    qs = np.nanquantile(x, np.r_[np.linspace(0, 1, n), 0.5])
    med = qs[-1]
    qs = qs[:-1]
    qs_low = qs[:half_n]
    qs_high = qs[-half_n:]
    skews = (qs_low + qs_high - 2 * med) / (qs_high - qs_low)
    return float(integrate.trapezoid(skews, dx=dp))


def left_quantile_weight(x: npt.NDArray, q: float = 0.25) -> float:
    """Calculate left quantile weight (LQW).

    It is based on inter-percentile ranges (uncentered, unscaled) of the
    left tail of the distribution.

    Parameters
    ----------
    x : array_like
        Input array.
    q : float, default = 0.25
        Quantile to use for the calculation, (0.0, 0.5)

    Returns
    -------
    lqw : float
        The value of left quantile weight.

    References
    ----------
    Brys, G.; Hubert, M.; Struyf, A. (2006).
    Robust measures of tail weight.
    Computational Statistics and Data Analysis 50(3), 733-759.
    """
    min_q, max_q = 0.0, 0.5
    if q <= min_q or q >= max_q:
        msg = "Parameter q should be in range (0, 0.5)."
        raise ValueError(msg)
    lower_quantile, q025, upper_quantile = np.nanquantile(
        x, [q * 0.5, 0.25, (1 - q) * 0.5]
    )
    return float(
        -(upper_quantile + lower_quantile - 2 * q025)
        / (upper_quantile - lower_quantile)
    )


def right_quantile_weight(x: npt.NDArray, q: float = 0.75) -> float:
    """Calculate right quantile weight (RQW).

    It is based on inter-percentile ranges (uncentered, unscaled) of the
    right tail of the distribution.

    Parameters
    ----------
    x : array_like
        Input array.
    q : float, default = 0.75
        Quantile to use for the calculation, (0.5, 1.0).

    Returns
    -------
    rqw : float
        The value of right quantile weight.

    References
    ----------
    Brys, G.; Hubert, M.; Struyf, A. (2006).
    Robust measures of tail weight.
    Computational Statistics and Data Analysis 50(3), 733-759.
    """
    min_q, max_q = 0.5, 1.0
    if q <= min_q or q >= max_q:
        msg = "Parameter q should be in range (0.5, 1.0)."
        raise ValueError(msg)
    lower_quantile, q075, upper_quantile = np.nanquantile(
        x, [1 - q * 0.5, 0.75, (1 + q) * 0.5]
    )
    return float(
        (lower_quantile + upper_quantile - 2 * q075) / (lower_quantile - upper_quantile)
    )


def cumulative_skew(x: npt.NDArray) -> float:
    """
    Calculate cumulative measure of skewness.

    It is based on calculating the cumulative statistics of the Lorenz curve.
    The array will be flatten before any calculations.

    This implementation shifts input array by min(x), so the Lorenz curve calculations
    would be correct.

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
    _x = np.sort(x)
    n = len(_x)
    q = np.nancumsum(_x) / np.nansum(_x)
    q = np.insert(q, 0, 0)
    r = np.arange(1, n + 1)
    r = np.insert(r, 0, 0)
    p = r / n
    d = p - q
    w = (2 * r - n) * 3 / n
    return float(np.sum(d * w) / np.sum(d))
