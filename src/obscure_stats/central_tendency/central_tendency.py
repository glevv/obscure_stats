"""
Module for measures of central tendency
"""

import math

import numpy as np
from scipy import stats  # type: ignore


def midrange(x: np.ndarray) -> float:
    """
    Function for calculating midrange or midpoint, i.e. average between min and max.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose midrange is desired.

    Returns
    -------
    mr : float or array_like.
        The value of the midrange.

    References
    ----------
    Dodge, Y. (2003).
    The Oxford dictionary of Statistical Terms.
    Oxford University Press.
    """
    maximum = np.nanmax(x)
    minimum = np.nanmin(x)
    return (maximum + minimum) * 0.5


def midhinge(x: np.ndarray) -> float:
    """
    Function for calculating midhinge, i.e. average between 1st and 3rd quartile.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose midhinge is desired.

    Returns
    -------
    mh : float or array_like.
        The value of the midhinge.

    References
    ----------
    Tukey, J. W. (1977).
    Exploratory Data Analysis.
    Addison-Wesley.
    """
    q1, q3 = np.nanquantile(x, [0.25, 0.75])
    return (q3 + q1) * 0.5


def trimean(x: np.ndarray) -> float:
    """
    Function for calculating trimean, i.e average between 3 quartiles.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose trimean is desired.

    Returns
    -------
    tm : float or array_like.
        The value of the trimean.

    References
    ----------
    Tukey, J. W. (1977).
    Exploratory Data Analysis.
    Addison-Wesley.
    """
    q1, q2, q3 = np.nanquantile(x, [0.25, 0.5, 0.75])
    return 0.5 * q2 + 0.25 * q1 + 0.25 * q3


def contraharmonic_mean(x: np.ndarray) -> float:
    """
    Function for calculating contraharmonic mean.
    Contraharmonic mean is a function complementary to the harmonic mean.
    The contraharmonic mean is a special case of the Lehmer mean with p=2.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose contraharmonic mean is desired.

    Returns
    -------
    chm : float or array_like.
        The value of the contraharmonic mean.

    References
    ----------
    P. S. Bullen (1987).
    Handbook of means and their inequalities.
    Springer.
    """
    return np.nansum(np.square(x)) / np.nansum(x)


def midmean(x: np.ndarray) -> float:
    """
    Function for calculating interquartile mean, i.e mean inside interquartile range.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose interquartile mean is desired.

    Returns
    -------
    iqm : float or array_like.
        The value of the interquartile mean.

    References
    ----------
    Salkind, N. (2008).
    Encyclopedia of Research Design.
    SAGE.
    """
    q1, q3 = np.nanquantile(x, [0.25, 0.75])
    return np.nanmean(np.where((x >= q1) & (x <= q3), x, np.nan))


def hodges_lehmann_sen_location(x: np.ndarray) -> float:
    """
    Function for calculating Hodges-Lehmann-Sen robust location measure (pseudomedian).

    NOTE: this statistic uses cartesian product, so the time and memory complexity
    are N^2. It is best to not use it on large arrays.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    hls : float
        The value of Hodges-Lehmann-Sen estimator.

    References
    ----------
    Hodges, J. L.; Lehmann, E. L. (1963).
    Estimation of location based on ranks.
    Annals of Mathematical Statistics. 34 (2): 598-611.
    """
    product = np.meshgrid(x, x, sparse=True)
    return np.median(product[0] + product[1]) * 0.5


def trimmed_harrell_davis_median(x: np.ndarray) -> float:
    """
    Function for calculating Trimmed Harrell-Davis median estimator.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    thdm : float
        The value of Trimmed Harrell-Davis median.

    References
    ----------
    Akinshin, A. 2022.
    Trimmed Harrell-Davis quantile estimator based on
    the highest density interval of the given width.
    Communications in Statistics - Simulation and Computation, pp. 1-11.
    """
    xs = np.sort(x)
    n = len(x)
    n_calculated = 1 / n**0.5  # heuristic suggested by the author
    a_b = (n + 1) * 0.5
    hdi = (0.5 - n_calculated * 0.5, 0.5 + n_calculated * 0.5)
    hdi_cdf = stats.beta.cdf(hdi, a_b, a_b)
    i_start = int(math.floor(hdi[0] * n))
    i_end = int(math.ceil(hdi[1] * n))
    nums = np.arange(i_start, i_end + 1) / n
    nums[nums <= hdi[0]] = hdi[0]
    nums[nums >= hdi[1]] = hdi[1]
    cdfs = (stats.beta.cdf(nums, a_b, a_b) - hdi_cdf[0]) / (hdi_cdf[1] - hdi_cdf[0])
    W = cdfs[1:] - cdfs[:-1]
    return np.sum(xs[i_start:i_end] * W)
