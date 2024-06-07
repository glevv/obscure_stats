"""Module for measures of central tendency."""

import math

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


def midrange(x: np.ndarray) -> float:
    """Calculate midrange or midpoint, i.e. average between min and max.

    This measure could be noisy since it is based on minimum and maximum.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    mr : float
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
    """Calculate midhinge, i.e. average between 1st and 3rd quartile.

    This measure is more robust then average.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    mh : float
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
    """Calculate trimean, i.e weighted average between 3 quartiles.

    This measure is more robust then average.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    tm : float
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
    """Calculate contraharmonic mean.

    Contraharmonic mean is a function complementary to the harmonic mean.
    The contraharmonic mean is a special case of the Lehmer mean with p=2.
    Mostly used in signal processing (for example filters).

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    chm : float
        The value of the contraharmonic mean.

    References
    ----------
    Bullen, P. S. (1987).
    Handbook of means and their inequalities.
    Springer.
    """
    return np.nansum(np.square(x)) / np.nansum(x)


def midmean(x: np.ndarray) -> float:
    """Calculate interquartile mean, i.e mean inside interquartile range.

    This measure is more robust then average.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    iqm : float
        The value of the interquartile mean.

    References
    ----------
    Salkind, N. J. (2008).
    Encyclopedia of Research Design.
    SAGE Publications, Inc.
    """
    q1, q3 = np.nanquantile(x, [0.25, 0.75])
    return np.nanmean(np.where((x >= q1) & (x <= q3), x, np.nan))


def hodges_lehmann_sen_location(x: np.ndarray) -> float:
    """Calculate Hodges-Lehmann-Sen robust location measure (pseudomedian).

    This measure is more robust then average.

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

    Sen, P. K. (1963).
    On the Estimation of Relative Potency in Dilution (-Direct)
    Assays by Distribution-Free Methods.
    Biometrics 19, no. 4: 532-552.

    Notes
    -----
    This implementation uses cartesian product, so the time and memory complexity
    are N^2. It is best to not use it on large arrays.
    """
    # In the original paper authors suggest use only upper triangular
    # of the cartesian product, but in this implementation we use
    # whole matrix, which is equvalent.
    product = np.meshgrid(x, x, sparse=True)
    return np.nanmedian(product[0] + product[1]) * 0.5


def standard_trimmed_harrell_davis_quantile(x: np.ndarray, q: float = 0.5) -> float:
    """Calculate Standard Trimmed Harrell-Davis median estimator.

    This measure is very robust.
    It calculates weighted Harrel-Davies quantiles on only sqrt(N) samples,
    located in the region with the most probability mass.

    The array will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    q : float
        Quantile value in range (0, 1).

    Returns
    -------
    thdq : float
        The value of Trimmed Harrell-Davis quantile.

    References
    ----------
    Akinshin, A. (2022).
    Trimmed Harrell-Davis quantile estimator based on
    the highest density interval of the given width.
    Communications in Statistics - Simulation and Computation, pp. 1-11.

    See Also
    --------
    scipy.stats.mstats.hdquantiles - Harrell-Davis quantile estimates.
    """
    if q <= 0 or q >= 1:
        msg = "Parameter q should be in range (0, 1)."
        raise ValueError(msg)
    _x = np.sort(x, axis=None)
    _x = _x[np.isfinite(_x)]
    n = len(_x)
    if n == 0:
        return np.nan
    if n == 1:
        return _x[0]
    n_calculated = 1 / n**0.5  # heuristic suggested by the author
    a = (n + 1) * q
    b = (n + 1) * (1.0 - q)
    hdi = (max(0, q - n_calculated * 0.5), min(1, q + n_calculated * 0.5))
    hdi_cdf = stats.beta.cdf(hdi, a, b)
    i_start = int(math.floor(hdi[0] * n))
    i_end = int(math.ceil(hdi[1] * n))
    nums = np.arange(i_start, i_end + 1) / n
    nums[nums <= hdi[0]] = hdi[0]
    nums[nums >= hdi[1]] = hdi[1]
    cdfs = (stats.beta.cdf(nums, a, b) - hdi_cdf[0]) / (hdi_cdf[1] - hdi_cdf[0])
    w = cdfs[1:] - cdfs[:-1]
    return np.sum(_x[i_start:i_end] * w)


def half_sample_mode(x: np.ndarray) -> float:
    """Calculate half sample mode.

    This estimator is more stable than regular mode estimation,
    especially for floating point values.
    The array will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    hsm : float
        The value of half sample mode.

    References
    ----------
    Robertson, T.; Cryer, J. D. (1974).
    An Iterative Procedure for Estimating the Mode.
    Journal of the American Statistical Association, 69(348), 1012-1016.

    See Also
    --------
    scipy.stats.mode - Mode estimator.
    """
    # heavily inspired by https://github.com/cran/modeest/blob/master/R/hsm.R
    y = np.sort(x, axis=None)
    y = y[np.isfinite(y)]
    _corner_cases = (4, 3)  # for 4 samples and 3 samples
    while (ny := len(y)) >= _corner_cases[0]:
        half_y = ny // 2
        w_min = y[-1] - y[0]
        for i in range(ny - half_y):
            w = y[i + half_y - 1] - y[i]
            if w <= w_min:
                w_min = w
                j = i
        if w == 0:
            return y[j]
        y = y[j : (j + half_y - 1)]
    if len(y) == _corner_cases[1]:
        z = 2 * y[1] - y[0] - y[2]
        if z < 0:
            return np.mean(y[0:1])
        if z > 0:
            return np.mean(y[1:2])
        return y[1]
    return np.mean(y)


def tau_location(x: np.ndarray, c: float = 4.5) -> float:
    """Calculate Tau measure of location.

    This measure is very robust and has higher efficiency than median.

    Parameters
    ----------
    x : array_like
        Input array.
    c : float
        Constant that filter outliers.

    Returns
    -------
    tml : float
        The value of Tau location.

    References
    ----------
    Wilcox, R. (2012).
    Introduction to Robust Estimation Hypothesis Testing.
    3rd Edition, Academic Press, New York.
    """
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scaled_x = (x - med) / mad
    w = np.square(1.0 - np.square(scaled_x / c)) * (np.abs(scaled_x) <= c)
    return np.nansum(x * w) / np.nansum(w)
