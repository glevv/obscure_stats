"""
Module for association measures
"""

import warnings

import numpy as np
from scipy import stats  # type: ignore


def _check_const(x: np.ndarray) -> bool:
    """Checker for constant arrays."""
    return all(np.isclose(x, x[0]))


def chatterjeexi(x: np.ndarray, y: np.ndarray) -> float:
    """
    Function for calculating ξ correlation.
    Another variation of rank correlation which does not make any assumptions about
    underlying distributions of the variable.

    It ranges from 0 (variables are completely independent) to 1
    (one is a measurable function of the other).

    This implementation does not break ties at random, instead
    it break ties depending on order. This makes it dependent on
    data sorting, which could be useful in application like time
    series.

    NOTE: This measure is assymetric: (x, y) != (y, x).

    Parameters
    ----------
    x : array_like
        Measured values.
    y : array_like
        Target values.

    Returns
    -------
    xi : float.
        The value of the xi correlation coefficient.

    References
    ----------
    Chatterjee, S. (2021).
    A new coefficient of correlation.
    Journal of the American Statistical Association, 116(536), 2009-2022.
    """
    n = len(x)
    if _check_const(x) or _check_const(y):
        warnings.warn(
            "An input array is constant; the correlation coefficient is not defined."
        )
        return np.nan
    x_ranked = stats.rankdata(x, method="ordinal")
    y_forward_ranked = stats.rankdata(y, method="max")
    y_backward_ranked = stats.rankdata(np.negative(y), method="max")
    y_forward_ranked_ordered = y_forward_ranked[np.argsort(x_ranked)]
    nom = np.sum(np.abs(np.diff(y_forward_ranked_ordered)))
    denom = np.sum(y_backward_ranked * (n - y_backward_ranked)) / n**3
    xi = 1.0 - nom / (2 * n**2 * denom)
    return xi


def concordance_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Function for calculating concordance correlation coefficient.
    The main difference between Pearson's R and CCC is that CCC
    takes bias between variables into account.

    CCC measures the agreement between two variables, e.g.,
    to evaluate reproducibility or for inter-rater reliability.

    Parameters
    ----------
    x : array_like
        Measured values.
    y : array_like
        Reference values.

    Returns
    -------
    ccc : float.
        The value of the concordance correlation coefficient.

    References
    ----------
    Lawrence I-Kuei Lin (1989).
    A concordance correlation coefficient to evaluate reproducibility.
    Biometrics. 45 (1): 255-268.
    """
    if _check_const(x) or _check_const(y):
        warnings.warn(
            "An input array is constant; the correlation coefficient is not defined."
        )
        return np.nan
    std_x = np.std(x, ddof=0)
    std_y = np.std(y, ddof=0)
    w = std_y / std_x
    v = (np.mean(x) - np.mean(y)) ** 2 / (std_x * std_y) ** 0.5
    x_a = 2 / (v**2 + w + 1 / w)
    p = np.corrcoef(x, y)[0][1]
    ccc = p * x_a
    return ccc


def quadrant_count_ratio(
    x: np.ndarray, y: np.ndarray, exclusion_zone: bool = False
) -> float:
    """
    Function for calculating quadrant count ratio. It could be seen
    as simplified version of Pearson's R.

    This variation has an option for an exclusion zone.
    It is based on the standard error of the mean and will exlucde
    points that are in the range of mean+-sem.

    Parameters
    ----------
    x : array_like
        Measured values.
    y : array_like
        Reference values.
    exclusion_zone : bool
        Whether to use eclusion zone or not.

    Returns
    -------
    qcr : float.
        The value of the quadrant count ratio.

    References
    ----------
    Holmes, Peter (Autumn 2001).
    Correlation: From Picture to Formula.
    Teaching Statistics. 23 (3): 67-71.

    Hiraishi, M., Tanioka, K., & Shimokawa, T. (2021).
    Concordance rate of a four-quadrant plot for repeated measurements.
    BMC Medical Research Methodology, 21(1), 1-16.
    """
    if _check_const(x) or _check_const(y):
        warnings.warn(
            "An input array is constant; the correlation coefficient is not defined."
        )
        return np.nan
    _x = np.asarray(x)
    _y = np.asarray(y)
    mean_x = np.nanmean(_x)
    mean_y = np.nanmean(_y)
    n = len(x)
    if exclusion_zone:
        sem_x = np.std(_x, ddof=0) / n**0.5
        sem_y = np.std(_y, ddof=0) / n**0.5
    else:
        sem_x = 0
        sem_y = 0
    n_q1 = np.nansum((_x > mean_x + sem_x) & (_y > mean_y + sem_y))
    n_q2 = np.nansum((_x < mean_x - sem_x) & (_y > mean_y + sem_y))
    n_q3 = np.nansum((_x < mean_x - sem_x) & (_y < mean_y - sem_y))
    n_q4 = np.nansum((_x > mean_x + sem_x) & (_y < mean_y - sem_y))
    return (n_q1 + n_q3 - n_q2 - n_q4) / n


def zhangi(x: np.ndarray, y: np.ndarray) -> float:
    """
    Function for calculating I correlation proposed by Q. Zhang.
    This is a modification of Spearman and Chatterjee rank correlation coefficients.

    NOTE: This measure is assymetric: (x, y) != (y, x).

    Parameters
    ----------
    x : array_like
        Measured values.
    y : array_like
        Reference values.

    Returns
    -------
    i : float.
        The value of the Zhang I.

    References
    ----------
    Zhang, Q., 2023.
    On relationships between Chatterjee's and Spearman's correlation coefficients.
    arXiv preprint arXiv:2302.10131.
    """
    if _check_const(x) or _check_const(y):
        warnings.warn(
            "An input array is constant; the correlation coefficient is not defined."
        )
        return np.nan
    return max(
        abs(stats.spearmanr(x, y, nan_policy="omit")[1]),
        2.5**0.5 * chatterjeexi(x, y),
    )


def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Function for calculating Tanimoto similarity. It is very similar to Jaccard
    or Cosine similarity but differs in how dot product is normalized.

    Parameters
    ----------
    x : array_like
        Measured values.
    y : array_like
        Reference values.

    Returns
    -------
    td : float.
        The value of the tanimoto similarity measure

    References
    ----------
    Rogers DJ, Tanimoto TT, 1960.
    A Computer Program for Classifying Plants.
    Science. 132 (3434): 1115-8.
    """
    _x = np.asarray(x)
    _y = np.asarray(y)
    xy = np.nanmean(_x * _y)
    xx = np.nanmean(_x**2)
    yy = np.nanmean(_y**2)
    return xy / (xx + yy - xy)
