"""
Module for measures of central tendency
"""

import numpy as np


def midrange(x: np.ndarray) -> float:
    """
    Function for calculating midrange.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose midrange is desired.

    Returns
    -------
    mr : float or array_like.
        The value of the midrange.

    Reference
    -------
    Dodge, Y. (2003).
    The Oxford dictionary of Statistical Terms.
    Oxford University Press.
    """
    maximum = np.nanmax(x)
    minimum = np.nanmin(x)
    return (maximum + minimum) * 0.5


def midhinge(x: np.ndarray) -> float:
    """
    Function for calculating midhinge.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose midhinge is desired.

    Returns
    -------
    mh : float or array_like.
        The value of the midhinge.

    Reference
    -------
    Tukey, J. W. (1977).
    Exploratory Data Analysis.
    Addison-Wesley.
    """
    q1, q3 = np.nanquantile(x, [0.25, 0.75])
    return (q3 + q1) * 0.5


def trimean(x: np.ndarray) -> float:
    """
    Function for calculating trimean.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose trimean is desired.

    Returns
    -------
    tm : float or array_like.
        The value of the trimean.

    Reference
    -------
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

    Reference
    -------
    P. S. Bullen (1987).
    Handbook of means and their inequalities.
    Springer.
    """
    return np.sum(x**2) / np.sum(x)


def midmean(x: np.ndarray) -> float:
    """
    Function for calculating interquartile mean.

    Parameters
    ----------
    x : array_like
        Array containing numbers whose interquartile mean is desired.

    Returns
    -------
    iqm : float or array_like.
        The value of the interquartile mean.

    Reference
    -------
    Salkind, N. (2008).
    Encyclopedia of Research Design.
    SAGE.
    """
    q1, q3 = np.nanquantile(x, [0.25, 0.75])
    return np.nanmean(np.where((x >= q1) & (x <= q3), x, np.nan))
