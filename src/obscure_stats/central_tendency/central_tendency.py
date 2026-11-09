# Copyright (c) 2023 Hleb Levitski
# Licensed under the MIT License. See LICENSE in the project root for details.

"""Module for measures of central tendency.

scipy is imported lazily inside functions: it takes ~0.3 s to import.
"""

import math

import numpy as np
import numpy.typing as npt


def midrange(x: npt.ArrayLike) -> float:
    """Calculate midrange or midpoint, i.e. average between minimum and maximum.

    This measure could be noisy since it is based on minimum and maximum.

    Parameters
    ----------
    x : array_like
        Numeric values.

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
    x = np.asarray(x, dtype=np.float64)
    maximum = np.nanmax(x)
    minimum = np.nanmin(x)
    return float((maximum + minimum) * 0.5)


def midhinge(x: npt.ArrayLike) -> float:
    """Calculate midhinge, i.e. average between 1st and 3rd quartile.

    This measure is more robust than average.

    Parameters
    ----------
    x : array_like
        Numeric values.

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
    x = np.asarray(x, dtype=np.float64)
    q1, q3 = np.nanquantile(x, [0.25, 0.75])
    return float((q3 + q1) * 0.5)


def trimean(x: npt.ArrayLike) -> float:
    """Calculate trimean, i.e weighted average between 3 quartiles.

    This measure is more robust than average.

    Parameters
    ----------
    x : array_like
        Numeric values.

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
    x = np.asarray(x, dtype=np.float64)
    q1, q2, q3 = np.nanquantile(x, [0.25, 0.5, 0.75])
    return float(0.5 * q2 + 0.25 * q1 + 0.25 * q3)


def contraharmonic_mean(x: npt.ArrayLike) -> float:
    """Calculate contraharmonic mean.

    Contraharmonic mean is a function complementary to the harmonic mean.
    The contraharmonic mean is a special case of the Lehmer mean with p=2.
    Mostly used in signal processing (for example filters).

    Parameters
    ----------
    x : array_like
        Numeric values.

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
    x = np.asarray(x, dtype=np.float64)
    return float(np.nansum(np.square(x)) / np.nansum(x))


def midmean(x: npt.ArrayLike) -> float:
    """Calculate interquartile mean, i.e mean inside interquartile range.

    This measure is more robust than average.

    Parameters
    ----------
    x : array_like
        Numeric values.

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
    x = np.asarray(x, dtype=np.float64)
    q1, q3 = np.nanquantile(x, [0.25, 0.75])
    return float(np.nanmean(np.where((x >= q1) & (x <= q3), x, np.nan)))


def hodges_lehmann_sen_location(x: npt.ArrayLike) -> float:
    """Calculate Hodges-Lehmann-Sen robust location measure (pseudomedian).

    This measure is more robust than average.

    Parameters
    ----------
    x : array_like
        Numeric values.

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
    This implementation enumerates all N(N+1)/2 Walsh averages (pairs i <= j),
    so the time and memory complexity are N^2. It is best to not use it on
    large arrays.
    """
    x = np.asarray(x, dtype=np.float64)
    _x = np.ravel(x)
    _x = _x[~np.isnan(_x)]
    n = len(_x)

    if n == 0:
        return np.nan

    walsh = np.concatenate([_x[i] + _x[i:] for i in range(n)])
    return float(np.median(walsh) * 0.5)


def _beta_hdi(a: float, b: float, width: float) -> tuple[float, float]:
    """Find highest density interval of given width for Beta(a, b).

    Port of getBetaHdi from Akinshin (2022), reference implementation.
    """
    # pyrefly: ignore [untyped-import]
    from scipy import optimize, stats  # noqa: PLC0415  (lazy, see module docstring)

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

    def f(t: float) -> float:
        """Return log pdf(t) - log pdf(t + width) (same sign as the pdf difference)."""
        if t <= 0.0:
            return -math.inf  # pdf(0) = 0 < pdf(width)
        if t + width >= 1.0:
            return math.inf  # pdf(1) = 0 < pdf(t)
        return (a - 1) * (math.log(t) - math.log(t + width)) + (b - 1) * (
            math.log1p(-t) - math.log1p(-(t + width))
        )

    left = optimize.brentq(f, lo, hi, xtol=1e-12)
    return (left, left + width)  # pyrefly: ignore


def standard_trimmed_harrell_davis_quantile(x: npt.ArrayLike, q: float = 0.5) -> float:
    """Calculate Standard Trimmed Harrell-Davis median estimator.

    This measure is very robust.
    It calculates weighted Harrell-Davis quantiles on only sqrt(N) samples,
    located in the region with the most probability mass.

    Parameters
    ----------
    x : array_like
        Numeric values.
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

    Notes
    -----
    The highest density interval of the Beta distribution has no closed form,
    so its endpoints are found numerically with Brent's method
    (`scipy.optimize.brentq`), as in the paper's reference implementation.
    This is a one-dimensional root search that defines a unique interval, not a
    fit to the data.
    """
    x = np.asarray(x, dtype=np.float64)
    # pyrefly: ignore [untyped-import]
    from scipy import stats  # noqa: PLC0415  (lazy, see module docstring)

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


def half_sample_mode(x: npt.ArrayLike) -> float:
    """Calculate half sample mode.

    This estimator is more stable than regular mode estimation,
    especially for floating point values.

    Parameters
    ----------
    x : array_like
        Numeric values.

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
    x = np.asarray(x, dtype=np.float64)
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


def tau_location(x: npt.ArrayLike, c: float = 4.5) -> float:
    """Calculate Tau measure of location.

    This measure is very robust and has higher efficiency than median.

    Parameters
    ----------
    x : array_like
        Numeric values.
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
    x = np.asarray(x, dtype=np.float64)
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


def grenanders_m(
    x: npt.ArrayLike,
    p: float | None = None,
    k: int | np.integer | None = None,
) -> float:
    """Calculate Grenander's mode.

    It is a direct (density-free) estimate of the mode: a weighted average of
    the midpoints of the k-spacings x(i+k) - x(i), with weights
    (x(i+k) - x(i))^(-p), so that short spacings (high density) dominate.

    For fixed p and k the estimate converges to the f^(p+1)-weighted mean
    int x f^(p+1) dx / int f^(p+1) dx (Grenander, 1965), which equals the mode
    only for symmetric unimodal distributions; it approaches the mode as p
    grows. By default p and k therefore grow with the sample size n, as
    recommended by Hall (1982): p = n^(2/7), which balances bias and variance,
    and k = ceil(n^(3/5)). Hall gives p only up to a constant factor; 1 is used.
    As p tends to infinity, the estimate tends to Venter's mode (the midpoint
    of the shortest k-spacing).

    Spacings of zero length get infinite weight: if there are any, the mean of
    their midpoints is returned (as in R's modeest).

    Parameters
    ----------
    x : array_like
        Numeric values.
    p : float, optional
        Weight exponent, greater than 1. Defaults to n^(2/7), where n is the
        number of finite observations.
    k : int, optional
        Order of the spacings (each interval contains k + 1 observations).
        Defaults to ceil(n^(3/5)). Values with 1 < p < k are recommended.

    Returns
    -------
    gm : float
        The value of Grenander's mode. NaN if there are at most k finite
        observations.

    References
    ----------
    Grenander, U. (1965).
    Some direct estimates of the mode.
    Annals of Mathematical Statistics, 36, 131-138.

    Hall, P. (1982).
    Asymptotic theory of Grenander's mode estimator.
    Zeitschrift fur Wahrscheinlichkeitstheorie und verwandte Gebiete,
    60, 315-334.

    Chacon, J. E. (2018).
    The modal age of statistics.
    arXiv preprint arXiv:1807.02789.
    """
    x = np.asarray(x, dtype=np.float64)
    if p is not None and p <= 1:
        msg = "Parameter p should be a float greater than 1."
        raise ValueError(msg)
    if k is not None and (
        isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0
    ):
        msg = "Parameter k should be an integer between 1 and length of x."
        raise ValueError(msg)

    x_sort = np.sort(x[np.isfinite(x)])
    n = len(x_sort)
    if p is None:
        p = n ** (2 / 7)  # Hall (1982): p ~ n^(2/7)
    if k is None:
        k = math.ceil(n**0.6)  # Hall (1982): "say k = n^(3/5)"

    if n <= k:
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

    weights = np.power(diff, -p)
    midpoints = 0.5 * (x_sort[k:] + x_sort[:-k])
    return float(np.sum(midpoints * weights) / np.sum(weights))


def gastwirth_location(x: npt.ArrayLike) -> float:
    """Calculate Gastwirth's location estimator.

    This measure is more robust than average.

    Parameters
    ----------
    x : array_like
        Numeric values.

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
    x = np.asarray(x, dtype=np.float64)
    p33, p50, p66 = np.nanquantile(x, [1 / 3, 0.5, 2 / 3])
    return float(0.3 * p33 + 0.4 * p50 + 0.3 * p66)


def shorth(x: npt.ArrayLike) -> float:
    """Calculate the shorth, the mean of the shortest half of the data.

    The shortest half is the shortest interval that contains floor(n/2) + 1
    observations. The shorth is very robust (50% breakdown point) but
    converges slowly (at rate n^(-1/3)).

    Parameters
    ----------
    x : array_like
        Numeric values.

    Returns
    -------
    sh : float
        The value of the shorth.

    References
    ----------
    Andrews, D. F.; Bickel, P. J.; Hampel, F. R.; Huber, P. J.;
    Rogers, W. H.; Tukey, J. W. (1972).
    Robust Estimates of Location: Survey and Advances.
    Princeton University Press.

    Notes
    -----
    If several intervals are equally short, the mean of their means is
    returned, which keeps the result independent of the order of the data.
    The least median of squares (LMS) location (Rousseeuw, 1984) is the
    midpoint of the endpoints of the same interval.

    See Also
    --------
    obscure_stats.dispersion.length_of_shortest_half - Length of the shortest half.
    """
    x = np.asarray(x, dtype=np.float64)
    _x = np.ravel(x)
    _x = np.sort(_x[np.isfinite(_x)])
    n = len(_x)
    if n == 0:
        return np.nan
    h = n // 2
    widths = _x[h:] - _x[: n - h]
    starts = np.flatnonzero(widths == np.min(widths))
    cs = np.r_[0.0, np.cumsum(_x)]
    means = (cs[starts + h + 1] - cs[starts]) / (h + 1)
    return float(np.mean(means))
