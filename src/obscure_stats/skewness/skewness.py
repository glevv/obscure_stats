"""Module for measures of skewness."""


import numpy as np
from scipy import stats  # type: ignore[import-untyped]


def pearson_mode_skew(x: np.ndarray) -> float:
    """Calculate Pearson's mode skew coefficient.

    This measure could be unstable due mode instability.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Pearson's mode skew coefficient is desired.

    Returns
    -------
    pmods : float or array_like.
        The value of Pearson's mode skew coefficient.

    References
    ----------
    Pearson, E. S. and Hartley, H. O. (1966).
    Biometrika Tables for Statisticians, vols. I and II.
    Cambridge University Press, Cambridge.
    """
    mean = np.nanmean(x)
    mode = stats.mode(x)[0]
    std = np.nanstd(x)
    return (mean - mode) / std


def pearson_median_skew(x: np.ndarray) -> float:
    """Calculatie Pearson's median skew coefficient.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Pearson's median skew coefficient is desired.

    Returns
    -------
    pmeds : float or array_like.
        The value of Pearson's median skew coefficient.

    References
    ----------
    Pearson, E.S. and Hartley, H.O. (1966).
    Biometrika Tables for Statisticians, vols. I and II.
    Cambridge University Press, Cambridge.
    """
    mean = np.nanmean(x)
    median = np.nanmedian(x)
    std = np.nanstd(x)
    return 3 * (mean - median) / std


def medeen_skew(x: np.ndarray) -> float:
    """Calculate Medeen's skewness statistic.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Medeen's skewness statistic is desired.

    Returns
    -------
    mss : float or array_like.
        The value of Medeen's skewness statistic.

    References
    ----------
    Groeneveld, R.A.; Meeden, G. (1984).
    Measuring Skewness and Kurtosis.
    The Statistician. 33 (4): 391-399.
    """
    median = np.nanmedian(x)
    mean = np.nanmean(x)
    return (mean - median) / np.mean(np.abs(x - median))


def bowley_skew(x: np.ndarray) -> float:
    """Calculate Bowley's skewness coefficinet.

    It is based on quartiles (uncentered, unscaled).
    This measure should be more robust than moment based skewness.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Bowley's skewness coefficinet is desired.

    Returns
    -------
    bsk : float or array_like.
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
    This measure should be more robust than moment based skewness.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Groeneveld's skewness coefficinet is desired.

    Returns
    -------
    bsk : float or array_like.
        The value of Groeneveld's skewness coefficinet.

    References
    ----------
    Groeneveld, R.A.; Meeden, G. (1984).
    Measuring Skewness and Kurtosis.
    The Statistician. 33 (4): 391-399.
    """
    q1, q2, q3 = np.nanquantile(x, [0.25, 0.5, 0.75])
    rs = (q3 + q1 - 2 * q2) / (q2 - q1)
    ls = (q3 + q1 - 2 * q2) / (q3 - q2)
    return rs if np.all(np.abs(rs) > np.abs(ls), axis=0) else ls


def kelly_skew(x: np.ndarray) -> float:
    """Calculate Kelly's skewness coefficinet.

    It is based on deciles (uncentered, unscaled).
    This measure should be more robust than moment based skewness.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Kelly's skewness coefficinet is desired.

    Returns
    -------
    ksc : float or array_like.
        The value of Kelly's skewness coefficinet.

    References
    ----------
    David, F. N. and Johnson, N. L., (1956).
    Some tests of significance with ordered variables.
    J. R. Stat. Soc. Ser. B Stat. Methodol. 18, 1-31.
    """
    d1, d5, d9 = np.nanquantile(x, [0.1, 0.5, 0.9])
    return (d9 + d1 - 2 * d5) / (d9 - d1)


def hossain_adnan_skew(x: np.ndarray) -> float:
    """Calculate Houssain and Adnan skewness coefficient.

    This measure should be more robust than moment based skewness.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Houssain and Adnan skewness coefficient
        is desired.

    Returns
    -------
    has : float or array_like.
        The value of Houssain and Adnan skewness coefficient.

    References
    ----------
    Hossain, M.F. and Adnan, M.A.S.A (2007).
    A New Approach to Determine the Asymmetry of a Distribution.
    Journal of Applied St atistical Science, Vol.15, pp. 127-134.
    """
    diff = x - np.nanmedian(x)
    return np.nanmean(diff) / np.nanmean(np.abs(diff))


def forhad_shorna_rank_skew(x: np.ndarray) -> float:
    """Calculate Forhad-Shorna coefficient of Rank Skewness.

    This measure should be more robust than moment based skewness.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose Forhad-Shorna coefficient of Rank Skewness
        is desired.

    Returns
    -------
    fsrs : float or array_like.
        The value of Forhad-Shorna coefficient of Rank Skewness.

    References
    ----------
    Shorna, U. S., & Hossain, M. (2019).
    A New Approach to Determine the Coefficient of Skewness and
    An Alternative Form of Boxplot.
    arXiv preprint arXiv:1908.06400.
    """
    mr = (np.nanmin(x) + np.nanmax(x)) * 0.5
    arr = np.r_[x, mr]
    arr_ranked = stats.rankdata(arr, method="min")
    diff = arr_ranked[-1] - arr_ranked
    diff = diff[:-1]
    return np.nansum(diff) / np.nansum(np.abs(diff))


def auc_skew_gamma(x: np.ndarray, dp: float = 0.01, *, weighted: bool = True) -> float:
    """Calculate Area under the curve of generalized Bowley skewness coefficients.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose AUC Bowley skewness is desired.
    dp : float, default = 0.01
        Step used in calculating area under the curve (integrating).
    weighted : bool, default = True
        Whether to use reweightning or not. This will assign bigger weight to the
        Bowley skewness coefficients calculated on percentiles far from the median.

    Returns
    -------
    aucbs : float or array_like.
        The value of AUC Bowley skewness.

    References
    ----------
    Arachchige, C. N., & Prendergast, L. A. (2019).
    Mean skewness measures.
    arXiv preprint arXiv:1912.06996.
    """
    n = int(1 / dp)
    half_n = n // 2
    w = (np.arange(half_n) / half_n)[::-1] if weighted else 1
    qs = np.nanquantile(x, np.r_[np.linspace(0, 1, n), 0.5])
    med = qs[-1]
    qs = qs[:-1]
    qs_low = qs[:half_n]
    qs_high = qs[-half_n:]
    skews = (qs_low + qs_high - 2 * med) / (qs_high - qs_low) * w
    return np.trapz(skews, dx=dp)
