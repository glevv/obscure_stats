# Copyright (c) 2023 Hleb Levitski
# Licensed under the MIT License. See LICENSE in the project root for details.

"""Module for measures of dispersion.

All functions return raw values: no consistency (normalizing) constants
are applied.

scipy is imported lazily inside functions: it takes ~0.3 s to import.
"""

import numpy as np
import numpy.typing as npt


def studentized_range(x: npt.ArrayLike) -> float:
    """Calculate range normalized by standard deviation.

    Uses the population standard deviation (ddof=0).

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    sr : float
        The value of the studentized range.

    References
    ----------
    Student (1927).
    Errors of routine analysis.
    Biometrika. 19 (1/2): 151-164.
    """
    x = np.asarray(x, dtype=np.float64)
    std = np.nanstd(x)
    maximum = np.nanmax(x)
    minimum = np.nanmin(x)
    return float((maximum - minimum) / std)


def coefficient_of_lvariation(x: npt.ArrayLike) -> float:
    """Calculate linear coefficient of variation.

    L-CV is the L-scale (half of the Gini mean difference) divided
    by L-mean (the same as regular mean).

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    lcv : float
        The value of the linear coefficient of variation.

    References
    ----------
    Hosking, J. R. M. (1990).
    L-moments: analysis and estimation of distributions
    using linear combinations of order statistics.
    Journal of the Royal Statistical Society, Series B. 52 (1): 105-124.
    """
    x = np.asarray(x, dtype=np.float64)
    # pyrefly: ignore [untyped-import]
    from scipy import stats  # noqa: PLC0415  (lazy, see module docstring)

    l1, l2 = stats.lmoment(x, order=[1, 2], nan_policy="omit")
    return float(l2 / l1)


def coefficient_of_variation(x: npt.ArrayLike) -> float:
    """Calculate coefficient of variation (Standard deviation / Mean).

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    cv : float
        The value of the coefficient of variation.

    References
    ----------
    Brown, C. E. (1998).
    Coefficient of Variation.
    Applied Multivariate Statistics in Geohydrology and Related Sciences. Springer.
    """
    x = np.asarray(x, dtype=np.float64)
    return float(np.nanstd(x) / np.nanmean(x))


def robust_coefficient_of_variation(x: npt.ArrayLike) -> float:
    """Calculate robust coefficient of variation.

    It is based on median absolute deviation from the median, i.e. median
    absolute deviation from the median divided by the median.

    The raw value is returned, without any consistency constant.
    Multiply by 1.4826 to make the MAD consistent with the standard
    deviation of a normal distribution.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    rcv : float
        The value of the robust coefficient of variation.

    References
    ----------
    Reimann, C.; Filzmoser; P.; Garrett, R. G.; Dutter, R. (2008).
    Statistical Data Analysis Explained: Applied Environmental Statistics with R.
    John Wiley and Sons, New York.
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    med_abs_dev = np.nanmedian(np.abs(x - med))
    return float(med_abs_dev / med)


def quartile_coefficient_of_dispersion(x: npt.ArrayLike) -> float:
    """Calculate quartile coefficient of dispersion (IQR / Midhinge).

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    qcd : float
        The value of the quartile coefficient of dispersion.

    References
    ----------
    Shapiro, H. M. (2005).
    Practical flow cytometry.
    John Wiley & Sons.
    """
    x = np.asarray(x, dtype=np.float64)
    q1, q3 = np.nanquantile(x, [0.25, 0.75])
    return float(2 * (q3 - q1) / (q3 + q1))


def shapiro_coefficient_of_variation(x: npt.ArrayLike) -> float:
    """Calculate Shapiro's robust coefficient of variation (IQR / Median).

    The raw value is returned, without any consistency constant.
    Shapiro (2005) multiplies by 0.75 (~ 1 / 1.349) to make the IQR
    consistent with the standard deviation of a normal distribution.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    scv : float
        The value of Shapiro's coefficient of variation.

    References
    ----------
    Shapiro, H. M. (2005).
    Practical flow cytometry.
    John Wiley & Sons.
    """
    x = np.asarray(x, dtype=np.float64)
    q1, q2, q3 = np.nanquantile(x, [0.25, 0.5, 0.75])
    return float((q3 - q1) / q2)


def fisher_index_of_dispersion(x: npt.ArrayLike) -> float:
    """Calculate Fisher's index of dispersion.

    It is very similar to the coefficient of variation but uses unnormalized
    variation instead of the standard deviation.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    fi : float
        The value of the Fisher's index of dispersion.

    References
    ----------
    Fisher, R. A. (1925).
    Statistical methods for research workers.
    Hafner, New York.
    """
    x = np.asarray(x, dtype=np.float64)
    n = np.count_nonzero(~np.isnan(np.asarray(x, dtype=np.float64)))
    return float(n * np.nanvar(x) / np.nanmean(x))


def morisita_index_of_dispersion(x: npt.ArrayLike) -> float:
    """Calculate Morisita's index of dispersion.

    Morisita's index of dispersion is the scaled probability that two
    points chosen at random from the whole population are in the same sample.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    mi : float
        The value of the Morisita's index.

    References
    ----------
    Morisita, M. (1959).
    Measuring the dispersion and the analysis of distribution patterns.
    Memoirs of the Faculty of Science, Kyushu University Series e. Biol. 2: 215-235
    """
    x = np.asarray(x, dtype=np.float64)
    x_sum = np.nansum(x)
    n = np.count_nonzero(~np.isnan(np.asarray(x, dtype=np.float64)))
    return float(n * (np.nansum(np.square(x)) - x_sum) / (x_sum**2 - x_sum))


def standard_quantile_absolute_deviation(x: npt.ArrayLike) -> float:
    """Calculate standard quantile absolute deviation.

    This measure is a robust measure of dispersion, that has higher
    Gaussian efficiency, but lower breaking point than MAD.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    sqad : float
        The value of the standard quantile absolute deviation.

    References
    ----------
    Akinshin, A. (2022).
    Quantile absolute deviation.
    arXiv preprint arXiv:2208.13459.

    Notes
    -----
    The raw value is returned, without any consistency constant.
    Multiply by the finite-sample constant K_n from Table 3 of the paper to
    make it unbiased for the standard deviation of a normal distribution
    (K_n ~ 1 + 0.762/n + 0.967/n^2 for n > 100; K_n -> 1 as n -> inf).
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(np.count_nonzero(~np.isnan(x)))
    if n == 0:
        return np.nan
    med = np.nanmedian(x)
    # constant value that maximizes efficiency for normal distribution
    q = 0.6826894921370850  # stats.norm.cdf(1) - stats.norm.cdf(-1)
    return float(np.nanquantile(np.abs(x - med), q=q))


def shamos_estimator(x: npt.ArrayLike) -> float:
    """Calculate Shamos robust estimator of dispersion.

    This measure is complementary to Hodges-Lehmann-Sen estimator.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    se : float
        The value of Shamos estimator.

    References
    ----------
    Shamos, M. I. (1976).
    Geometry and statistics: Problems at the interface (p. 0032).
    Carnegie-Mellon University. Department of Computer Science.

    Notes
    -----
    This implementation enumerates all N(N-1)/2 pairs, so the time and memory
    complexity are N^2. It is best to not use it on large arrays.
    The raw value is returned, without any consistency constant.
    Multiply by 1.048358 to make it consistent with the standard deviation
    of a normal distribution.

    See Also
    --------
    obscure_stats.central_tendency.hodges_lehmann_sen_location - Hodges-Lehmann-Sen loc.
    """
    x = np.asarray(x, dtype=np.float64)
    _x = np.ravel(x)
    _x = _x[np.isfinite(_x)]

    n = len(_x)

    if n <= 1:
        return np.nan

    diffs = np.concatenate([np.abs(_x[i] - _x[i + 1 :]) for i in range(n - 1)])
    return float(np.median(diffs))


def coefficient_of_range(x: npt.ArrayLike) -> float:
    """Calculate coefficient of range (Range / Midrange).

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    cr : float
        The value of the range coefficient.

    References
    ----------
    Yadav, S. K., Singh, S.,  &  Gupta, R. (2019).
    Measures of Dispersion.
    In Biomedical Statistics (pp. 59-70). Springer, Singapore
    """
    x = np.asarray(x, dtype=np.float64)
    min_ = np.nanmin(x)
    max_ = np.nanmax(x)
    return float((max_ - min_) / (max_ + min_))


def gini_mean_difference(x: npt.ArrayLike) -> float:
    """Calculate Gini Mean Difference.

    Alternative measure of variability to the usual standard deviation.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    gmd : float
        The value of the Gini Mean Difference.

    References
    ----------
    Yitzhaki, S.; Schechtman, E. (2013).
    The Gini Methodology.
    Springer, New York.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.sort(x)
    x = x[np.isfinite(x)]
    # pyrefly: ignore [bad-argument-type]
    n = len(x)

    if n == 0:
        return np.nan
    if n == 1:
        return 0.0

    indices = np.arange(1, n + 1)
    # pyrefly: ignore [no-matching-overload, unsupported-operation]
    return float(2.0 * np.sum((2 * indices - n - 1) * x) / (n * (n - 1)))


def inter_expectile_range(x: npt.ArrayLike) -> float:
    """Calculate inter expectile range (IER).

    It is the same as IQR, but uses expectile instead of quantile.

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    ier : float
        The value of the inter expectile range.

    References
    ----------
    Bellini, F.; Mercuri, L.; Rroji, E. (2018)
    Implicit expectiles and measures of implied volatility.
    Quantitative Finance, 18(11), pp. 1851-1864.
    """
    x = np.asarray(x, dtype=np.float64)
    # pyrefly: ignore [untyped-import]
    from scipy import stats  # noqa: PLC0415  (lazy, see module docstring)

    _x = np.ravel(x)
    _x = _x[np.isfinite(_x)]
    if len(_x) <= 1:
        return np.nan
    return float(stats.expectile(_x, 0.75) - stats.expectile(_x, 0.25))


def length_of_shortest_half(x: npt.ArrayLike) -> float:
    """Calculate the length of the shortest half.

    It is the length of the shortest interval that contains floor(n/2) + 1
    observations, a very robust measure of scale (50% breakdown point).

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    lsh : float
        The value of the length of the shortest half.

    References
    ----------
    Rousseeuw, P. J.; Leroy, A. M. (1988).
    A robust scale estimator based on the shortest half.
    Statistica Neerlandica, 42(2), 103-116.

    Notes
    -----
    The raw value is returned, without any consistency constant.
    Divide by 1.349 (= 2 * Phi^(-1)(0.75)) to make it consistent with the
    standard deviation of a normal distribution.

    See Also
    --------
    obscure_stats.central_tendency.shorth - Mean of the shortest half.
    """
    x = np.asarray(x, dtype=np.float64)
    _x = np.ravel(x)
    _x = np.sort(_x[np.isfinite(_x)])
    n = len(_x)
    if n == 0:
        return np.nan
    h = n // 2
    return float(np.min(_x[h:] - _x[: n - h]))
