"""Module for measures of dispersion."""

import numpy as np
import numpy.typing as npt
from scipy import special, stats  # type: ignore[import-untyped]


def studentized_range(x: npt.NDArray) -> float:
    """Calculate range normalized by standard deviation.

    Parameters
    ----------
    x : array_like
        Input array.

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
    std = np.nanstd(x)
    maximum = np.nanmax(x)
    minimum = np.nanmin(x)
    return float((maximum - minimum) / std)


def coefficient_of_lvariation(x: npt.NDArray) -> float:
    """Calculate linear coefficient of variation.

    L-CV is the L-scale (half of mean absolute deviation) divided
    by L-mean (the same as regular mean).

    Parameters
    ----------
    x : array_like
        Input array.

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
    l1 = np.nanmean(x)
    _x = np.sort(x)
    n = len(_x)
    common = 1 / special.comb(n - 1, 1) / n
    beta_1 = common * np.nansum(special.comb(np.arange(1, n), 1) * _x[1:])
    l2 = 2 * beta_1 - l1
    return float(l2 / l1)


def coefficient_of_variation(x: npt.NDArray) -> float:
    """Calculate coefficient of variation (Standard deviation / Mean).

    Parameters
    ----------
    x : array_like
        Input array.

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
    return float(np.nanstd(x) / np.nanmean(x))


def robust_coefficient_of_variation(x: npt.NDArray) -> float:
    """Calculate robust coefficient of variation.

    It is based on median absolute deviation from the median, i.e. median
    absolute deviation from the median divided by the median.

    Parameters
    ----------
    x : array_like
        Input array.

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
    med = np.nanmedian(x)
    med_abs_dev = np.nanmedian(np.abs(x - med))
    return float(1.4826 * med_abs_dev / med)


def quartile_coefficient_of_dispersion(x: npt.NDArray) -> float:
    """Calculate quartile coefficient of dispersion (IQR / Midhinge).

    Parameters
    ----------
    x : array_like
        Input array.

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
    q1, q3 = np.nanquantile(x, [0.25, 0.75])
    return float(0.75 * (q3 - q1) / (q3 + q1))


def fisher_index_of_dispersion(x: npt.NDArray) -> float:
    """Calculate Fisher's index of dispersion.

    It is very similar to the coefficient of variation but uses unnormalized
    variation instead of the standard deviation.

    Parameters
    ----------
    x : array_like
        Input array.

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
    return float((len(x) - 1) * np.nanvar(x) / np.nanmean(x))


def morisita_index_of_dispersion(x: npt.NDArray) -> float:
    """Calculate Morisita's index of dispersion.

    Morisita's index of dispersion is the scaled probability that two
    points chosen at random from the whole population are in the same sample.

    Parameters
    ----------
    x : array_like
        Input array.

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
    x_sum = np.nansum(x)
    return float(len(x) * (np.nansum(np.square(x)) - x_sum) / (x_sum**2 - x_sum))


def standard_quantile_absolute_deviation(x: npt.NDArray) -> float:
    """Calculate standard quantile absolute deviation.

    This measure is a robust measure of dispersion, that has higher
    gaussian efficiency, but lower breaking point than MAD.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    sqad : float
        The value of the standard quantile absolute deviation.

    References
    ----------
    Akinshin, A. (2022).
    Quantile absolute deviation.
    arXiv preprint arXiv:2208.13459.
    """
    med = np.nanmedian(x)
    n = len(x)
    # finite sample correction
    k = 1.0 + 0.762 / n + 0.967 / n**2
    # constant value that maximizes efficiency for normal distribution
    q = 0.6826894921370850  # stats.norm.cdf(1) - stats.norm.cdf(-1)
    return float(k * np.nanquantile(np.abs(x - med), q=q))


def shamos_estimator(x: npt.NDArray) -> float:
    """Calculate Shamos robust estimator of dispersion.

    This measure is complementary to Hodges-Lehmann-Sen estimator.

    Parameters
    ----------
    x : array_like
        Input array.

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
    This implementation uses cartesian product, so the time and memory complexity
    are N^2. It is best to not use it on large arrays.

    See Also
    --------
    obscure_stats.central_tendency.hodges_lehmann_sen_location - Hodges-Lehmann-Sen loc.
    """
    # In the original paper authors suggest use only upper triangular
    # of the cartesian product, but in this implementation we use
    # whole matrix, which is equvalent.
    product = np.meshgrid(x, x, sparse=True)
    return float(np.nanmedian(np.abs(product[0] - product[1])))


def coefficient_of_range(x: npt.NDArray) -> float:
    """Calculate coefficient of range (Range / Midrange).

    Parameters
    ----------
    x : array_like
        Input array.

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
    min_ = np.nanmin(x)
    max_ = np.nanmax(x)
    return float((max_ - min_) / (max_ + min_))


def cole_index_of_dispersion(x: npt.NDArray) -> float:
    """Calculate Cole's index of dispersion.

    Higher values mean higher dispersion.
    This measure is invariant to multiplication.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    ci : float
        The value of the Cole's index of dispersion.

    References
    ----------
    Cole, L. C. (1946).
    A theory for analyzing contagiously distributed populations.
    Ecology. 27 (4): 329-341.
    """
    return float(np.nansum(np.square(x)) / np.nansum(x) ** 2)


def gini_mean_difference(x: npt.NDArray) -> float:
    """Calculate Gini Mean Difference.

    Alternative measure of variability to the usual standard deviation.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    gmd : float
        The value of the Gini Mean Difference.

    References
    ----------
    Yitzhaki, S.; Schechtman, E. (2013).
    The Gini Methodology.
    Springer, New York.

    Notes
    -----
    This implementation uses cartesian product, so the time and memory complexity
    are N^2. It is best to not use it on large arrays.
    """
    n = len(x)
    product = np.meshgrid(x, x, sparse=True)
    return float(np.nansum(np.abs(product[0] - product[1])) / (n * (n - 1)))


def inter_expectile_range(x: npt.NDArray) -> float:
    """Calculate inter expectile range (IER).

    It is the same as IQR, but uses expectile instead of quantile.

    Parameters
    ----------
    x : array_like
        Input array.

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
    _x = np.ravel(x)
    _x = _x[np.isfinite(_x)]
    if len(_x) <= 1:
        return np.nan
    return float(stats.expectile(_x, 0.75) - stats.expectile(_x, 0.25))
