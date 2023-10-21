"""
Module for association measures
"""

import numpy as np
from scipy import stats  # type: ignore


def xi_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Function for calculating ξ correlation.
    Another variation of rank correlation.

    This implementation does not break ties at random, instead
    it break ties depending on order. This makes it dependent on
    data sorting, which could be useful in application like time
    series.

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

    Reference
    -------
    Chatterjee, S. (2021).
    A new coefficient of correlation.
    Journal of the American Statistical Association, 116(536), 2009-2022.
    """
    N = len(x)
    x_ranked = stats.rankdata(x, method="ordinal")
    y_forward_ranked = stats.rankdata(y, method="max")
    y_backward_ranked = stats.rankdata(-y, method="max")
    y_forward_ranked_ordered = y_forward_ranked[np.argsort(x_ranked)]
    nom = np.sum(np.abs(np.diff(y_forward_ranked_ordered)))
    denom = np.sum(y_backward_ranked * (N - y_backward_ranked)) / N**3
    xi = 1.0 - nom / (2 * N**2 * denom)
    return xi


def concordance_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Function for calculating concordance correlation coefficient.

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

    Reference
    -------
    Lawrence I-Kuei Lin (1989).
    A concordance correlation coefficient to evaluate reproducibility.
    Biometrics. 45 (1): 255-268.
    """
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
    Function for calculating quadrant count ratio.

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

    Reference
    -------
    Holmes, Peter (Autumn 2001).
    Correlation: From Picture to Formula.
    Teaching Statistics. 23 (3): 67-71.

    Hiraishi, M., Tanioka, K., & Shimokawa, T. (2021).
    Concordance rate of a four-quadrant plot for repeated measurements.
    BMC Medical Research Methodology, 21(1), 1-16.
    """
    mean_x = np.nanmean(x)
    mean_y = np.nanmean(y)
    if exclusion_zone:
        sem_x = np.std(x, ddof=0) / len(x) ** 0.5
        sem_y = np.std(y, ddof=0) / len(y) ** 0.5
    else:
        sem_x = 0
        sem_y = 0
    n_q1 = np.nansum(x > mean_x + sem_x & y > mean_y + sem_y)
    n_q2 = np.nansum(x < mean_x - sem_x & y > mean_y + sem_y)
    n_q3 = np.nansum(x < mean_x - sem_x & y < mean_y - sem_y)
    n_q4 = np.nansum(x > mean_x + sem_x & y < mean_y - sem_y)
    return (n_q1 + n_q3 - n_q2 - n_q4) / len(x)
