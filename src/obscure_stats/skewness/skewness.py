# Copyright (c) 2023 Hleb Levitski
# Licensed under the MIT License. See LICENSE in the project root for details.

"""Module for measures of skewness.

scipy is imported lazily inside functions: it takes ~0.3 s to import.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from obscure_stats.central_tendency import half_sample_mode


def pearson_mode_skew(x: npt.ArrayLike) -> float:
    """Calculate Pearson's mode skew coefficient.

    It is (mean - mode) / standard deviation. The sample mode is not defined
    for continuous data (all values are distinct), so the mode is estimated
    with the half sample mode (`half_sample_mode`), which also returns the most
    frequent value when it clearly dominates discrete data.
    Uses the population standard deviation (ddof=0).

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    pmods : float
        The value of Pearson's mode skew coefficient.

    References
    ----------
    Pearson, E. S.; Hartley, H. O. (1966).
    Biometrika Tables for Statisticians, vols. I and II.
    Cambridge University Press, Cambridge.

    Robertson, T.; Cryer, J. D. (1974).
    An Iterative Procedure for Estimating the Mode.
    Journal of the American Statistical Association, 69(348), 1012-1016.

    See Also
    --------
    obscure_stats.central_tendency.half_sample_mode - Half sample mode.
    obscure_stats.skewness.bickel_mode_skew - Mode skewness based on signs.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x)
    mode = half_sample_mode(x)
    std = np.nanstd(x)
    return float((mean - mode) / std)


def bickel_mode_skew(x: npt.ArrayLike) -> float:
    """Calculate Robust Mode skew with half sample mode.

    This measure should be more stable than Pearson mode
    skewness coefficient.

    Parameters
    ----------
    x : array_like
        Numeric values.

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
    x = np.asarray(x, dtype=np.float64)
    mode = half_sample_mode(x)
    return float(np.nanmean(np.sign(np.asarray(x) - mode)))


def pearson_median_skew(x: npt.ArrayLike) -> float:
    """Calculate Pearson's median skew coefficient.

    Parameters
    ----------
    x : array_like
        Numeric values.

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
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x)
    median = np.nanmedian(x)
    std = np.nanstd(x)
    return float(3.0 * (mean - median) / std)


def meeden_skew(x: npt.ArrayLike) -> float:
    """Calculate Meeden's skewness statistic.

    This measure is similar to Pearson median skewness coefficient
    but uses different normalization (mean absolute deviation from the
    median).

    Hossain and Adnan (2007) proposed the same statistic as
    sum(x - median) / sum(|x - median|).

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    mss : float
        The value of Meeden's skewness statistic.

    References
    ----------
    Groeneveld, R.A.; Meeden, G. (1984).
    Measuring Skewness and Kurtosis.
    The Statistician. 33 (4): 391-399.

    Hossain, M. F.; Adnan, M. A. S. A (2007).
    A New Approach to Determine the Asymmetry of a Distribution.
    Journal of Applied Statistical Science, Vol.15, pp. 127-134.
    """
    x = np.asarray(x, dtype=np.float64)
    median = np.nanmedian(x)
    mean = np.nanmean(x)
    return float((mean - median) / np.nanmean(np.abs(x - median)))


def bowley_skew(x: npt.ArrayLike) -> float:
    """Calculate Bowley's skewness coefficient.

    Also known as Yule-Kendall skewness coefficient.
    It is based on quartiles (uncentered, unscaled) and compares the distance
    between the median and each of the two quartiles.
    This measure should be more robust than moment based skewness.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    bsk : float
        The value of Bowley's skewness coefficient.

    References
    ----------
    Bowley, A. L. (1901).
    Elements of Statistics.
    P.S. King and Son, London.
    """
    x = np.asarray(x, dtype=np.float64)
    q1, q2, q3 = np.nanquantile(x, [0.25, 0.5, 0.75])
    return float((q3 + q1 - 2 * q2) / (q3 - q1))


def groeneveld_range_skew(x: npt.ArrayLike) -> float:
    """Calculate Groeneveld's skewness coefficient.

    This measure should be more robust than moment based skewness.
    This is the implementation of 'b4' skewness from the paper.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    grs : float
        The value of Groeneveld's range skewness coefficient.

    References
    ----------
    Groeneveld, R. A.; Meeden, G. (1984).
    Measuring Skewness and Kurtosis.
    The Statistician. 33 (4): 391-399.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmedian(x)
    min_ = np.nanmin(x)
    max_ = np.nanmax(x)
    return float((max_ + min_ - 2 * m) / (max_ - min_))


def kelly_skew(x: npt.ArrayLike) -> float:
    """Calculate Kelly's skewness coefficient.

    It is based on deciles (uncentered, unscaled).
    Similar to Bowley skewness coefficient, but instead of
    quartiles, deciles are used.
    This measure should be more robust than moment based skewness.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    ksc : float
        The value of Kelly's skewness coefficient.

    References
    ----------
    David, F. N.; Johnson, N. L. (1956).
    Some tests of significance with ordered variables.
    J. R. Stat. Soc. Ser. B Stat. Methodol. 18, 1-31.
    """
    x = np.asarray(x, dtype=np.float64)
    d1, d5, d9 = np.nanquantile(x, [0.1, 0.5, 0.9])
    return float((d9 + d1 - 2 * d5) / (d9 - d1))


def forhad_shorna_rank_skew(x: npt.ArrayLike) -> float:
    """Calculate Forhad-Shorna coefficient of rank skewness.

    This measure is similar to Meeden's skewness statistic (`meeden_skew`),
    but uses differences in ranks instead of absolute differences.
    This measure should be more robust than moment based skewness.
    The array will be flattened before any calculations.
    Ties are ranked by their average rank (the paper uses competition
    ranking), which keeps the statistic antisymmetric: f(-x) = -f(x).

    Parameters
    ----------
    x : array_like
        Numeric values.

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
    x = np.asarray(x, dtype=np.float64)
    # pyrefly: ignore [untyped-import]
    from scipy import stats  # noqa: PLC0415  (lazy, see module docstring)

    _x = np.ravel(x)
    mr = (np.nanmin(_x) + np.nanmax(_x)) * 0.5
    arr = np.r_[_x, mr]
    # The paper uses standard competition ("1224") ranking; average ranking
    # is used here so that the statistic changes sign when x does (with ties).
    arr_ranked = stats.rankdata(arr, method="average", nan_policy="omit")
    diff = arr_ranked[-1] - arr_ranked
    diff = diff[:-1]
    return float(np.nansum(diff) / np.nansum(np.abs(diff)))


def auc_skew_gamma(x: npt.ArrayLike, dp: float = 0.01) -> float:
    """Calculate area under the curve of generalized Bowley skewness coefficients.

    This measure tries to combine multiple generalized Bowley skewness coefficients
    into one value.

    Parameters
    ----------
    x : array_like
        Numeric values.
    dp : float, default = 0.01
        Grid step in p; the estimator averages J = 0.5 / dp generalized
        Bowley coefficients on a midpoint grid in (0, 0.5) (paper: J = 100).

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
    x = np.asarray(x, dtype=np.float64)
    if not 0 < dp < 0.25:
        msg = "Parameter dp should be in range (0, 0.25)."
        raise ValueError(msg)
    n_points = max(round(0.5 / dp), 1)
    # midpoint grid p_j = 0.5 (j - 1/2) / J excludes p = 0 (min / max)
    ps = (np.arange(n_points) + 0.5) * 0.5 / n_points
    qs = np.nanquantile(x, np.r_[ps, 0.5, 1.0 - ps], method="median_unbiased")
    qs_low, med, qs_high = qs[:n_points], qs[n_points], qs[n_points + 1 :]
    with np.errstate(divide="ignore", invalid="ignore"):
        skews = (qs_low + qs_high - 2 * med) / (qs_high - qs_low)
    skews = skews[np.isfinite(skews)]
    if skews.size == 0:
        return np.nan
    return float(np.mean(skews))


def left_quantile_weight(x: npt.ArrayLike, q: float = 0.25) -> float:
    """Calculate left quantile weight (LQW).

    It is based on inter-percentile ranges (uncentered, unscaled) of the
    left tail of the distribution.
    Brys et al. (2006) study q = 0.125 and q = 0.25; the smaller tail fraction
    has more power, the larger one a higher breakdown point.

    Parameters
    ----------
    x : array_like
        Numeric values.
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
    x = np.asarray(x, dtype=np.float64)
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


def right_quantile_weight(x: npt.ArrayLike, q: float = 0.75) -> float:
    """Calculate right quantile weight (RQW).

    It is based on inter-percentile ranges (uncentered, unscaled) of the
    right tail of the distribution.
    Brys et al. (2006) study q = 0.875 and q = 0.75; the larger quantile
    has more power, the smaller one a higher breakdown point.

    Parameters
    ----------
    x : array_like
        Numeric values.
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
    x = np.asarray(x, dtype=np.float64)
    min_q, max_q = 0.5, 1.0
    if q <= min_q or q >= max_q:
        msg = "Parameter q should be in range (0.5, 1.0)."
        raise ValueError(msg)
    lower_quantile, q075, upper_quantile = np.nanquantile(
        x, [1 - q * 0.5, 0.75, (1 + q) * 0.5]
    )
    return float(
        (lower_quantile + upper_quantile - 2 * q075) / (upper_quantile - lower_quantile)
    )


def cumulative_skew(x: npt.ArrayLike) -> float:
    """
    Calculate cumulative measure of skewness.

    It is based on calculating the cumulative statistics of the Lorenz curve.
    The array will be flattened before any calculations.

    This implementation shifts input array by min(x), so the Lorenz curve calculations
    would be correct. This is a deviation from the paper: results differ from
    the paper for strictly positive data.

    Parameters
    ----------
    x : array_like
        Numeric values.

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
    x = np.asarray(x, dtype=np.float64)
    _x = np.sort(np.ravel(x))
    _x = _x[np.isfinite(_x)]
    n = len(_x)
    if n == 0:
        return np.nan
    _x = _x - _x[0]
    q = np.nancumsum(_x) / np.nansum(_x)
    q = np.insert(q, 0, 0)
    r = np.arange(1, n + 1)
    r = np.insert(r, 0, 0)
    p = r / n
    d = p - q
    w = (2 * r - n) * 3 / n
    return float(np.sum(d * w) / np.sum(d))
