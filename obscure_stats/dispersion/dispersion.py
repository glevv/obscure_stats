"""
Module for measures of dispersion
"""

import numpy as np
from scipy import stats  # type: ignore


def efficiency(x: np.ndarray) -> float:
    """
    Function for calculating array efficiency.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    eff : float or array_like.
        The value of the efficiency.

    References
    ----------
    Grubbs, Frank (1965).
    Statistical Measures of Accuracy for Riflemen and Missile Engineers. pp. 26-27.
    """
    return np.nanvar(x) / np.nanmean(x) ** 2


def studentized_range(x: np.ndarray) -> float:
    """
    Function for calculating range normalized by standard deviation.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    sr : float or array_like.
        The value of the studentized range.

    References
    ----------
    Student (1927).
    Errors of routine analysis.
    Biometrika. 19 (1/2): 151-164.
    """
    maximum = np.nanmax(x)
    minimum = np.nanmin(x)
    std = np.nanstd(x)
    return (maximum - minimum) / std


def coefficient_of_lvariation(x: np.ndarray) -> float:
    """
    Function for calculating linear coefficient of variation.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    lcv : float or array_like.
        The value of the linear coefficient of variation.

    References
    ----------
    Hosking, J.R.M. (1990).
    L-moments: analysis and estimation of distributions
    using linear combinations of order statistics.
    Journal of the Royal Statistical Society, Series B. 52 (1): 105-124.
    """
    l1 = np.nanmean(x)
    l2 = np.nanmean(np.abs(x - l1)) * 0.5
    return l2 / l1


def coefficient_of_variation(x: np.ndarray) -> float:
    """
    Function for calculating coefficient of variation.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    cv : float or array_like.
        The value of the coefficient of variation.

    References
    ----------
    Brown, C.E. (1998).
    Coefficient of Variation.
    Applied Multivariate Statistics in Geohydrology and Related Sciences. Springer.
    """
    return np.nanstd(x) / np.nanmean(x)


def quartile_coef_of_dispersion(x: np.ndarray) -> float:
    """
    Function for calculating Quartile Coefficient of Dispersion.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    qcd : float or array_like.
        The value of the quartile coefficient of dispersion.

    References
    ----------
    Bonett, D. G. (2006).
    Confidence interval for a coefficient of quartile variation.
    Computational Statistics & Data Analysis. 50 (11): 2953-2957.
    """
    q1, q3 = np.nanquantile(x, [0.25, 0.75])
    return (q3 - q1) / (q3 + q1)


def dispersion_ratio(x: np.ndarray) -> float:
    """
    Function for calculating dispersion ratio.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    dr : float or array_like.
        The value of the dispersion ratio.

    References
    ----------
    Soobramoney, J., Chifurira, R., & Zewotir, T. (2022)
    Selecting key features of online behaviour on South African informative websites
    prior to unsupervised machine learning.
    Statistics, Optimization & Information Computing.
    """
    return np.nanmean(x) / stats.gmean(x, nan_policy="omit")


def hoover_index(x: np.ndarray) -> float:
    """
    Function for calculating Hoover index.
    In general - measure of uniformity of the distribution.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    hi : float or array_like.
        The value of the Hoover index.

    References
    ----------
    Edgar Malone Hoover Jr. (1936).
    The Measurement of Industrial Localization.
    Review of Economics and Statistics, 18, No. 162-71.

    See also
    -------
    Gibbs M4 Index
    Robin Hood index
    Schutz index
    """
    return 0.5 * np.nansum(x - np.nanmean(x)) / np.nansum(x)


def lloyds_index(x: np.ndarray) -> float:
    """
    Function for calculating Lloyd's index of mean crowding.
    Lloyd's index of mean crowding (IMC) is the average number of other points
    contained in the sample unit that contains a randomly chosen point.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    li : float or array_like.
        The value of the Lloyd's index.

    References
    ----------
    Lloyd, M (1967).
    Mean crowding.
    J Anim Ecol. 36 (1): 1–30.
    """
    m = np.nanmean(x)
    s = np.nanvar(x)
    return m + s / (m - 1)


def morisita_index(x: np.ndarray) -> float:
    """
    Function for calculating Morisita's index of dispersion.
    Morisita's index of dispersion ( Im ) is the scaled probability
    that two points chosen at random from the whole population are in the same sample.

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    mi : float or array_like.
        The value of the Morisita's index.

    References
    ----------
    Morisita, M (1959).
    Measuring the dispersion and the analysis of distribution patterns.
    Memoirs of the Faculty of Science, Kyushu University Series e. Biol. 2: 215–235
    """
    x_sum = np.sum(x)
    return len(x) * (np.sum(np.square(x)) - x_sum) / (x_sum**2 - x_sum)


def sqad(x: np.ndarray) -> float:
    """
    Function for calculating Standard quantile absolute deviation

    Parameters
    ----------
    x : array_like
        Input array.

    Returns
    -------
    sqad : float or array_like.
        The value of the SQAD.

    References
    ----------
    Akinshin, A. (2022).
    Quantile absolute deviation.
    arXiv preprint arXiv:2208.13459.
    """
    med = np.nanmedian(x)
    return np.nanquantile(np.abs(x - med), q=0.682689492137086)
