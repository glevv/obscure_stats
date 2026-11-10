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

    It is `(mean - mode) / standard deviation`: the mean minus the mode, divided by the
    standard deviation. The sample mode is not defined for continuous data (all values
    are distinct), so the mode is estimated with the half sample mode
    (`half_sample_mode`), which also returns the most frequent value when it clearly
    dominates discrete data. Uses the population standard deviation (ddof=0).

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of Pearson's mode skew coefficient.

    See Also
    --------
    obscure_stats.central_tendency.half_sample_mode : Half sample mode.
    obscure_stats.skewness.bickel_mode_skew : Mode skewness based on signs.

    References
    ----------
    Pearson, E. S.; Hartley, H. O. (1966).
    Biometrika Tables for Statisticians, vols. I and II.
    Cambridge University Press, Cambridge.

    Robertson, T.; Cryer, J. D. (1974).
    An Iterative Procedure for Estimating the Mode.
    Journal of the American Statistical Association, 69(348), 1012-1016.

    Examples
    --------
    >>> from obscure_stats.skewness import pearson_mode_skew
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(pearson_mode_skew(x), 4)
    0.5674
    """
    x = np.asarray(x, dtype=np.float64)
    mean = np.nanmean(x)
    mode = half_sample_mode(x)
    std = np.nanstd(x)
    return float((mean - mode) / std)


def bickel_mode_skew(x: npt.ArrayLike) -> float:
    """Calculate robust mode skewness with the half sample mode.

    It is the mean of `sign(x_i - mode)`, with the mode estimated by the half sample
    mode (`half_sample_mode`). In words: the share of values above the mode minus the
    share of values below it.

    This measure should be more stable than Pearson mode
    skewness coefficient.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of Bickel's mode skew coefficient.

    References
    ----------
    Bickel, D. R. (2002).
    Robust estimators of the mode and skewness of continuous data.
    Computational Statistics & Data Analysis, Elsevier, 39(2), 153-163.

    Examples
    --------
    >>> from obscure_stats.skewness import bickel_mode_skew
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(bickel_mode_skew(x), 4)
    0.25
    """
    x = np.asarray(x, dtype=np.float64)
    mode = half_sample_mode(x)
    return float(np.nanmean(np.sign(np.asarray(x) - mode)))


def pearson_median_skew(x: npt.ArrayLike) -> float:
    """Calculate Pearson's median skew coefficient.

    It is `3 (mean - median) / standard deviation`: three times the difference between
    the mean and the median, divided by the standard deviation. Uses the population
    standard deviation (ddof=0).

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of Pearson's median skew coefficient.

    References
    ----------
    Pearson, E.S.; Hartley, H.O. (1966).
    Biometrika Tables for Statisticians, vols. I and II.
    Cambridge University Press, Cambridge.

    Examples
    --------
    >>> from obscure_stats.skewness import pearson_median_skew
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(pearson_median_skew(x), 4)
    0.7943
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

    Hossain and Adnan (2007) proposed the same statistic as `sum(x - median) / sum(|x -
    median|)`: the sum of the deviations from the median divided by the sum of their
    absolute values.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of Meeden's skewness statistic.

    References
    ----------
    Groeneveld, R.A.; Meeden, G. (1984).
    Measuring Skewness and Kurtosis.
    The Statistician. 33 (4): 391-399.

    Hossain, M. F.; Adnan, M. A. S. A (2007).
    A New Approach to Determine the Asymmetry of a Distribution.
    Journal of Applied Statistical Science, Vol.15, pp. 127-134.

    Examples
    --------
    >>> from obscure_stats.skewness import meeden_skew
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(meeden_skew(x), 4)
    0.35
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
    float
        The value of Bowley's skewness coefficient.

    References
    ----------
    Bowley, A. L. (1901).
    Elements of Statistics.
    P.S. King and Son, London.

    Examples
    --------
    >>> from obscure_stats.skewness import bowley_skew
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(bowley_skew(x), 4)
    0.12
    """
    x = np.asarray(x, dtype=np.float64)
    q1, q2, q3 = np.nanquantile(x, [0.25, 0.5, 0.75])
    return float((q3 + q1 - 2 * q2) / (q3 - q1))


def groeneveld_range_skew(x: npt.ArrayLike) -> float:
    """Calculate Groeneveld's skewness coefficient.

    It is `(max + min - 2 median) / (max - min)`: the Bowley coefficient with the
    minimum and maximum in place of the quartiles. In words: twice the distance from the
    median to the midrange, relative to the range.

    This measure should be more robust than moment based skewness.
    This is the implementation of 'b4' skewness from the paper.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of Groeneveld's range skewness coefficient.

    References
    ----------
    Groeneveld, R. A.; Meeden, G. (1984).
    Measuring Skewness and Kurtosis.
    The Statistician. 33 (4): 391-399.

    Examples
    --------
    >>> from obscure_stats.skewness import groeneveld_range_skew
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(groeneveld_range_skew(x), 4)
    0.5
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
    float
        The value of Kelly's skewness coefficient.

    References
    ----------
    David, F. N.; Johnson, N. L. (1956).
    Some tests of significance with ordered variables.
    J. R. Stat. Soc. Ser. B Stat. Methodol. 18, 1-31.

    Examples
    --------
    >>> from obscure_stats.skewness import kelly_skew
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(kelly_skew(x), 4)
    0.3971
    """
    x = np.asarray(x, dtype=np.float64)
    d1, d5, d9 = np.nanquantile(x, [0.1, 0.5, 0.9])
    return float((d9 + d1 - 2 * d5) / (d9 - d1))


def forhad_shorna_rank_skew(x: npt.ArrayLike) -> float:
    """Calculate Forhad-Shorna coefficient of rank skewness.

    This measure is similar to Meeden's skewness statistic (`meeden_skew`), but uses
    differences in ranks instead of absolute differences. This measure should be more
    robust than moment based skewness. The array will be flattened before any
    calculations. Ties are ranked by their average rank (the paper uses competition
    ranking), which keeps the statistic antisymmetric: `f(-x) = -f(x)`, so negating the
    data negates the result.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of Forhad-Shorna coefficient of rank skewness.

    References
    ----------
    Shorna, U. S.; Hossain, M. (2019).
    A New Approach to Determine the Coefficient of Skewness and
    An Alternative Form of Boxplot.
    arXiv preprint arXiv:1908.06400.

    Examples
    --------
    >>> from obscure_stats.skewness import forhad_shorna_rank_skew
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(forhad_shorna_rank_skew(x), 4)
    0.75
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

    It is the mean of the generalized Bowley coefficients `(Q(1 - p) + Q(p) - 2 Q(0.5))
    / (Q(1 - p) - Q(p))` over an evenly spaced grid of p in (0, 0.5) with step dp, where
    Q is the sample quantile. In words: for each p, the sum of the upper and lower
    p-quantiles minus twice the median, divided by their difference, averaged over p;
    this is the area under the curve of these coefficients divided by the length of the
    interval (0.5).

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
    float
        The value of AUC Bowley skewness.

    References
    ----------
    Arachchige, C. N.; & Prendergast, L. A. (2019).
    Mean skewness measures.
    arXiv preprint arXiv:1912.06996.

    Examples
    --------
    >>> from obscure_stats.skewness import auc_skew_gamma
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(auc_skew_gamma(x), 4)
    0.2264
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


def cumulative_skew(x: npt.ArrayLike) -> float:
    """
    Calculate cumulative measure of skewness.

    It is based on calculating the cumulative statistics of the Lorenz curve.
    The array will be flattened before any calculations.

    This implementation shifts input array by `min(x)`, so the Lorenz curve calculations
    would be correct. This is a deviation from the paper: results differ from the paper
    for strictly positive data.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    float
        The value of cumulative skew.

    References
    ----------
    Schlemmer, M. (2022).
    A robust measure of skewness using cumulative statistic calculation.
    arXiv preprint arXiv:2209.10699.

    Examples
    --------
    >>> from obscure_stats.skewness import cumulative_skew
    >>> x = [1.0, 2.0, 2.5, 3.0, 4.5, 5.0, 7.0, 12.0]
    >>> round(cumulative_skew(x), 4)
    0.2703
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
