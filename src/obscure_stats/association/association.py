"""Module for association measures."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


def _check_arrays(x: np.ndarray, y: np.ndarray) -> bool:
    """Check arrays.

    - Lenghts of the arrays;
    - Constant input;
    - Contains inf.
    """
    if len(x) != len(y):
        warnings.warn(
            "Lenghts of the inputs do not match, please check the arrays.",
            stacklevel=2,
        )
        return True
    if all(np.isclose(x, x[0], equal_nan=False)) or all(
        np.isclose(y, y[0], equal_nan=False)
    ):
        warnings.warn(
            "One of the input arrays is constant;"
            " the correlation coefficient is not defined.",
            stacklevel=2,
        )
        return True
    if any(np.isinf(x)) or any(np.isinf(y)):
        warnings.warn(
            "One of the input arrays contains inf, please check the array.",
            stacklevel=2,
        )
        return True
    if (np.isnan(x).sum() >= len(x) - 1) or (np.isnan(y).sum() >= len(x) - 1):
        warnings.warn(
            "One of the input arrays has too many missing values,"
            " please check the arrays.",
            stacklevel=2,
        )
        return True
    return False


def _prep_arrays(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Prepare data for downstream task."""
    notnan = ~(np.isnan(x) | np.isnan(y))
    _x = np.asarray(x)
    _y = np.asarray(y)
    _x = _x[notnan]
    _y = _y[notnan]
    return _x, _y


def chatterjeexi(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Xi correlation coefficient.

    Another variation of rank correlation which does not make any assumptions about
    underlying distributions of the variable.

    It ranges from 0 (variables are completely independent) to 1
    (one is a measurable function of the other).

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

    References
    ----------
    Chatterjee, S. (2021).
    A new coefficient of correlation.
    Journal of the American Statistical Association, 116(536), 2009-2022.

    Notes
    -----
    This measure is assymetric: (x, y) != (y, x).
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    # heavily inspired by https://github.com/czbiohub-sf/xicor/issues/17#issue-965635013
    n = len(x)
    y_forward_ordered = y[np.argsort(x)]
    _, y_unique_indexes, y_counts = np.unique(
        y_forward_ordered, return_inverse=True, return_counts=True
    )
    right = np.cumsum(y_counts)[y_unique_indexes]
    left = np.cumsum(y_counts[::-1])[len(y_counts) - y_unique_indexes - 1]
    return 1.0 - 0.5 * np.sum(np.abs(np.diff(right))) / np.mean(left * (n - left))


def concordance_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate concordance correlation coefficient.

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
    Lin, L. I. (1989).
    A concordance correlation coefficient to evaluate reproducibility.
    Biometrics. 45 (1): 255-268.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    std_x = np.std(x, ddof=0)
    std_y = np.std(y, ddof=0)
    w = std_y / std_x
    v = (np.mean(x) - np.mean(y)) ** 2 / (std_x * std_y) ** 0.5
    x_a = 2 / (v**2 + w + 1 / w)
    p = np.corrcoef(x, y)[0][1]
    return p * x_a


def concordance_rate(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Calculate conventional concordance rate.

    Also known as quadrant count ratio.
    It could be seen as simplified version of Pearson's R.

    It differs from quadrant count ratio by adding and exclusion zone
    variation has an option for an exclusion zone. It is based on the
    standard error of the mean and will exlucde points that are in the
    range of mean+-sem.

    Parameters
    ----------
    x : array_like
        Measured values.
    y : array_like
        Reference values.

    Returns
    -------
    cr : float.
        The value of the quadrant count ratio.

    References
    ----------
    Holmes, P. (2001).
    Correlation: From Picture to Formula.
    Teaching Statistics. 23 (3): 67-71.

    See Also
    --------
    Quadrant count ratio.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    n = len(x)
    mean_x = np.sum(x) / n
    mean_y = np.sum(y) / n
    sem_x = np.std(x, ddof=0) / n**0.5
    sem_y = np.std(y, ddof=0) / n**0.5
    n_q1 = np.sum((x > mean_x + sem_x) & (y > mean_y + sem_y))
    n_q2 = np.sum((x < mean_x - sem_x) & (y > mean_y + sem_y))
    n_q3 = np.sum((x < mean_x - sem_x) & (y < mean_y - sem_y))
    n_q4 = np.sum((x > mean_x + sem_x) & (y < mean_y - sem_y))
    return (n_q1 + n_q3 - n_q2 - n_q4) / n


def symmetric_chatterjeexi(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate symmetric Xi correlation coefficient.

    Another variation of rank correlation which does not make any assumptions about
    underlying distributions of the variable.

    It ranges from 0 (variables are completely independent) to 1
    (one is a measurable function of the other).

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
    sxi : float.
        The value of the xi correlation coefficient.

    References
    ----------
    Chatterjee, S. (2021).
    A new coefficient of correlation.
    Journal of the American Statistical Association, 116(536), 2009-2022.

    See Also
    --------
    obscure_stats.associaton.chatterjeexi - Chatterjee Xi coefficient.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    n = len(x)
    # y ~ f(x)
    y_forward_ordered = y[np.argsort(x)]
    _, y_unique_indexes, y_counts = np.unique(
        y_forward_ordered, return_inverse=True, return_counts=True
    )
    right_xy = np.cumsum(y_counts)[y_unique_indexes]
    left_xy = np.cumsum(y_counts[::-1])[len(y_counts) - y_unique_indexes - 1]
    # x ~ f(y)
    x_forward_ordered = x[np.argsort(y)]
    _, x_unique_indexes, x_counts = np.unique(
        x_forward_ordered, return_inverse=True, return_counts=True
    )
    right_yx = np.cumsum(x_counts)[x_unique_indexes]
    left_yx = np.cumsum(x_counts[::-1])[len(x_counts) - x_unique_indexes - 1]
    # choose the highest from the two
    return 1.0 - min(
        0.5 * np.sum(np.abs(np.diff(right_xy))) / np.mean(left_xy * (n - left_xy)),
        0.5 * np.sum(np.abs(np.diff(right_yx))) / np.mean(left_yx * (n - left_yx)),
    )


def zhangi(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate I correlation coefficient proposed by Q. Zhang.

    This coefficient combines Spearman and Chatterjee rank correlation coefficients
    to get higher sensetivity to complex nonlinear relationships between variables.

    Parameters
    ----------
    x : array_like
        Measured values.
    y : array_like
        Reference values.

    Returns
    -------
    zi : float.
        The value of the Zhang I.

    References
    ----------
    Zhang, Q. (2023).
    On relationships between Chatterjee's and Spearman's correlation coefficients.
    arXiv preprint arXiv:2302.10131.

    Notes
    -----
    This measure is assymetric: (x, y) != (y, x).

    See Also
    --------
    scipy.stats.spearmanr - Spearman R coefficient.
    obscure_stats.associaton.chatterjeexi - Chatterjee Xi coefficient.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    return max(
        abs(stats.spearmanr(x, y, nan_policy="omit")[0]),
        2.5**0.5 * chatterjeexi(x, y),
    )


def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Tanimoto similarity.

    It is very similar to Jaccard or Cosine similarity but differs in how
    dot product is normalized.
    This version is designed for numeric values, instead of sets.

    Parameters
    ----------
    x : array_like
        Measured values.
    y : array_like
        Reference values.

    Returns
    -------
    ts : float.
        The value of the tanimoto similarity measure

    References
    ----------
    Rogers, D. J.; Tanimoto, T. T. (1960).
    A Computer Program for Classifying Plants.
    Science. 132 (3434): 1115-8.

    See Also
    --------
    Jaccard similarity
    Cosine similarity
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    xy = np.mean(x * y)
    xx = np.mean(x**2)
    yy = np.mean(y**2)
    return xy / (xx + yy - xy)
