# Copyright (c) 2023 Hleb Levitski
# Licensed under the MIT License. See LICENSE in the project root for details.

"""Module for measures of central tendency."""

import math

import numpy as np
import numpy.typing as npt

# pyrefly: ignore [untyped-import]
from scipy import optimize, stats


def midrange(x: npt.NDArray[np.number]) -> float:
    """Calculate midrange or midpoint, i.e. average between minimum and maximum.

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
    return float((maximum + minimum) * 0.5)


def midhinge(x: npt.NDArray[np.number]) -> float:
    """Calculate midhinge, i.e. average between 1st and 3rd quartile.

    This measure is more robust than average.

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
    return float((q3 + q1) * 0.5)


def trimean(x: npt.NDArray[np.number]) -> float:
    """Calculate trimean, i.e weighted average between 3 quartiles.

    This measure is more robust than average.

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
    return float(0.5 * q2 + 0.25 * q1 + 0.25 * q3)


def contraharmonic_mean(x: npt.NDArray[np.number]) -> float:
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
    return float(np.nansum(np.square(x)) / np.nansum(x))


def midmean(x: npt.NDArray[np.number]) -> float:
    """Calculate interquartile mean, i.e mean inside interquartile range.

    This measure is more robust than average.

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
    return float(np.nanmean(np.where((x >= q1) & (x <= q3), x, np.nan)))


def hodges_lehmann_sen_location(x: npt.NDArray[np.number]) -> float:
    """Calculate Hodges-Lehmann-Sen robust location measure (pseudomedian).

    This measure is more robust than average.

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
    This implementation enumerates all N(N-1)/2 pairs, so the time and memory
    complexity are N^2. It is best to not use it on large arrays.
    """
    _x = np.ravel(np.asarray(x, dtype=np.float64))
    _x = _x[~np.isnan(_x)]
    n = len(_x)

    if n == 0:
        return np.nan

    walsh = np.concatenate([_x[i] + _x[i:] for i in range(len(_x))])
    return float(np.median(walsh) * 0.5)


def _beta_hdi(a: float, b: float, width: float) -> tuple[float, float]:
    """Find highest density interval of given width for Beta(a, b).

    Port of getBetaHdi from Akinshin (2022), reference implementation.
    """
    eps = 1e-9

    if width > 1 - eps:
        return (0.0, 1.0)

    if a < 1 + eps and b < 1 + eps:
        dist = stats.beta(a, b)
        if dist.cdf(width) >= dist.sf(1.0 - width):
            return (0.0, width)
        return (1.0 - width, 1.0)

    if a < 1 + eps:
        return (0.0, width)

    if b < 1 + eps:
        return (1.0 - width, 1.0)

    mode = (a - 1) / (a + b - 2)
    lo, hi = max(0.0, mode - width), min(mode, 1.0 - width)
    pdf = stats.beta(a, b).pdf  # pyrefly: ignore [missing-attribute]

    def f(t: float) -> float:
        return pdf(t) - pdf(t + width)

    left = optimize.brentq(f, lo, hi, xtol=1e-12)
    return (left, left + width)  # pyrefly: ignore


def standard_trimmed_harrell_davis_quantile(
    x: npt.NDArray[np.number], q: float = 0.5
) -> float:
    """Calculate Standard Trimmed Harrell-Davis median estimator.

    This measure is very robust.
    It calculates weighted Harrel-Davies quantiles on only sqrt(N) samples,
    located in the region with the most probability mass.

    Parameters
    ----------
    x : array_like
        Input array.
    q : float, default = 0.5
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
    _x = np.sort(x)
    _x = _x[np.isfinite(_x)]
    n = len(_x)
    if n == 0:
        return np.nan
    if n == 1:
        return float(_x[0])
    width = 1 / n**0.5  # heuristic suggested by the author
    a = (n + 1) * q
    b = (n + 1) * (1.0 - q)
    hdi = _beta_hdi(a, b, width)
    hdi_cdf = stats.beta.cdf(hdi, a, b)
    i_start = math.floor(hdi[0] * n)
    i_end = math.ceil(hdi[1] * n)
    nums = np.arange(i_start, i_end + 1) / n
    nums[nums <= hdi[0]] = hdi[0]
    nums[nums >= hdi[1]] = hdi[1]
    cdfs = (stats.beta.cdf(nums, a, b) - hdi_cdf[0]) / (hdi_cdf[1] - hdi_cdf[0])
    w = cdfs[1:] - cdfs[:-1]
    return float(np.sum(_x[i_start:i_end] * w))


def half_sample_mode(x: npt.NDArray[np.number]) -> float:
    """Calculate half sample mode.

    This estimator is more stable than regular mode estimation,
    especially for floating point values.

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
    y = np.sort(x)
    y = y[np.isfinite(y)]
    while (ny := len(y)) >= 4:
        k = math.ceil(ny / 2) - 1
        widths = y[k:] - y[: ny - k]
        j = len(widths) - 1 - int(np.argmin(widths[::-1]))
        if widths[j] == 0:
            return float(y[j])
        y = y[j : j + k + 1]
    if len(y) == 3:
        z = 2 * y[1] - y[0] - y[2]
        if z < 0:
            return float(np.mean(y[0:2]))
        if z > 0:
            return float(np.mean(y[1:3]))
        return float(y[1])
    if len(y) == 0:
        return np.nan
    return float(np.mean(y))


def tau_location(x: npt.NDArray[np.number], c: float = 4.5) -> float:
    """Calculate Tau measure of location.

    This measure is very robust and has higher efficiency than median.

    Parameters
    ----------
    x : array_like
        Input array.
    c : float, default = 4.5
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
    if c <= 0:
        msg = "Parameter c should be strictly positive."
        raise ValueError(msg)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0:
        # more than half the sample is identical; that value is the location
        return float(med)
    scaled_x = (x - med) / mad
    w = np.square(1.0 - np.square(scaled_x / c)) * (np.abs(scaled_x) <= c)
    return float(np.nansum(x * w) / np.nansum(w))


def grenanders_m(x: npt.NDArray[np.number], p: float = 1.001, k: int = 2) -> float:
    """Calculate Grenander's Mode.

    This measure is a direct non-parametric estimation of the mode.
    It is not very robust to outliers.
    It is recommended to set k > 2p, but it is only possible if x has
    enough samples.

    Parameters
    ----------
    x : array_like
        Input array.
    p : float, default = 1.001
        Smoothing constant.
    k : int, default = 2
        The number of samples to filter.

    Returns
    -------
    gm : float
        The value of Grenander's Mode

    References
    ----------
    Grenander, U. (1965).
    Some direct estimates of the mode.
    Annals of Mathematical Statistics, 36, 131-138.
    """
    if p <= 1:
        msg = "Parameter p should be a float greater than 1."
        raise ValueError(msg)
    if (k <= 0) or (not isinstance(k, int)):
        msg = "Parameter k should be an integer between 1 and length of x."
        raise ValueError(msg)

    x_sort = np.sort(x).astype("float")
    x_sort = x_sort[np.isfinite(x_sort)]

    if len(x_sort) <= k:
        return np.nan

    # precalculate diffs
    diff = x_sort[k:] - x_sort[:-k]

    # if the diffs are constant - return the value
    if diff.sum() == 0.0:
        return float(x_sort[0])

    # zero diffs should have 'infinite' weight
    zero_mask = diff == 0.0
    if np.any(zero_mask):
        midpoints = 0.5 * (x_sort[k:][zero_mask] + x_sort[:-k][zero_mask])
        return float(np.mean(midpoints))

    # standard calculation for non-zero diffs
    weights = np.power(diff, -p)
    midpoints = 0.5 * (x_sort[k:] + x_sort[:-k])
    return float(np.sum(midpoints * weights) / np.sum(weights))


def gastwirth_location(x: npt.NDArray[np.number]) -> float:
    """Calculate Gastwirth's location estimator.

    This measure is more robust than average.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    gle : float
        The value of the Gastwirth's location.

    References
    ----------
    Gastwirth, J. L. (1966).
    On Robust Procedures.
    J. Amer. Statist. Assn., Vol. 61, pp. 929-948.
    """
    p33, p50, p66 = np.nanquantile(x, [1 / 3, 0.5, 2 / 3])
    return float(0.3 * p33 + 0.4 * p50 + 0.3 * p66)
