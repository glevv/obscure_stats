# Copyright (c) 2023 Hleb Levitski
# Licensed under the MIT License. See LICENSE in the project root for details.

"""Module for association measures."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

# pyrefly: ignore [untyped-import]
from scipy import stats

from obscure_stats.dispersion import gini_mean_difference


def _prepare_arrays(
    x: npt.ArrayLike, y: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Validate and prepare paired inputs; return None (after a warning) if unusable.

    The stages run once, in order:

    1. Convert both inputs to flattened float64 arrays (non-numeric input raises
       ValueError).
    2. Check the raw input: equal lengths, no infinite values.
    3. Drop the pairs with a missing value in x or y.
    4. Check what is left: at least two pairs, and neither variable constant.

    Warnings use stacklevel=3 so they point at the caller of the public function.
    """
    _x = np.ravel(np.asarray(x, dtype=np.float64))
    _y = np.ravel(np.asarray(y, dtype=np.float64))
    if len(_x) != len(_y):
        warnings.warn(
            "Lengths of the inputs do not match, please check the arrays.", stacklevel=3
        )
        return None
    if np.any(np.isinf(_x)) or np.any(np.isinf(_y)):
        warnings.warn(
            "One of the input arrays contains inf, please check the array.",
            stacklevel=3,
        )
        return None
    complete = ~(np.isnan(_x) | np.isnan(_y))
    _x, _y = _x[complete], _y[complete]
    if len(_x) <= 1:
        msg = (
            "One of the input arrays has too many missing values, please check the arrays."
            if not np.all(complete)
            else "Lengths of the inputs are too small, please check the arrays."
        )
        warnings.warn(msg, stacklevel=3)
        return None
    if np.all(_x == _x[0]) or np.all(_y == _y[0]):
        warnings.warn(
            "One of the input arrays is constant;"
            " the correlation coefficient is not defined.",
            stacklevel=3,
        )
        return None
    return _x, _y


def concordance_correlation(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate concordance correlation coefficient.

    The main difference between Pearson's R and CCC is that CCC
    takes bias between variables into account.

    CCC measures the agreement between two variables, e.g.,
    to evaluate reproducibility or for inter-rater reliability.

    This measure will be sensitive to any perturbation in the arrays,
    i.e. it is not addition or multiplication invariant.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the concordance correlation coefficient.

    References
    ----------
    Lin, L. I. (1989).
    A concordance correlation coefficient to evaluate reproducibility.
    Biometrics. 45 (1): 255-268.

    Examples
    --------
    >>> from obscure_stats.association import concordance_correlation
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(concordance_correlation(x, y), 4)
    0.8654
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    std_x = np.std(x, ddof=0)
    std_y = np.std(y, ddof=0)
    w = std_y / std_x
    v = (np.mean(x) - np.mean(y)) / (std_x * std_y) ** 0.5
    x_a = 2 / (v**2 + w + 1 / w)
    p = np.corrcoef(x, y)[0][1]
    return float(p * x_a)


def concordance_rate(
    x: npt.ArrayLike, y: npt.ArrayLike, *, exclusion: float = 0.0
) -> float:
    """Calculate four-quadrant concordance rate of paired changes.

    It is the share of paired changes that agree in direction (both increase
    or both decrease), as used to assess the trending ability of two
    measurement methods. The inputs are changes (e.g. differences between
    consecutive measurements), so the four quadrants are centered at zero.

    Pairs inside the central square exclusion zone, `max(|x|, |y|) <= exclusion`, are
    treated as noise and dropped. With the default of 0 only pairs where neither
    variable changed are dropped. A pair outside the zone where only one variable
    changed counts as not concordant.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric changes (e.g. differences between consecutive
        measurements).
    y : array_like
        Numeric changes (e.g. differences between consecutive
        measurements).
    exclusion : float, default = 0.0
        Half-width of the central square exclusion zone, in the units of
        x and y. There is no standard value: cardiac output studies use e.g.
        10-15% of the mean value or 0.5 L/min.

    Returns
    -------
    float
        The value of the concordance rate, in [0, 1]. NaN if no pairs remain
        outside the exclusion zone.

    See Also
    --------
    obscure_stats.association.fechner_correlation : Fechner correlation.

    References
    ----------
    Critchley, L. A.; Lee, A.; Ho, A. M. (2010).
    A critical review of the ability of continuous cardiac output monitors
    to measure trends in cardiac output.
    Anesthesia & Analgesia, 111(5), 1180-1192.

    Examples
    --------
    >>> from obscure_stats.association import concordance_rate
    >>> x = [1, -2, 3, -1, 2, 4]
    >>> y = [2, -1, -1, -3, 1, 3]
    >>> round(concordance_rate(x, y), 4)
    0.8333
    """
    if exclusion < 0:
        msg = "Parameter exclusion should be non-negative."
        raise ValueError(msg)
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    keep = np.maximum(np.abs(x), np.abs(y)) > exclusion
    if not np.any(keep):
        return np.nan
    return float(np.mean(x[keep] * y[keep] > 0))


def symmetric_chatterjee_xi(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate symmetric Xi correlation coefficient.

    Another variation of rank correlation which does not make any assumptions about
    underlying distributions of the variable.

    It ranges from 0 (variables are completely independent) to 1
    (one is a measurable function of the other). But a lot of the time the maximum
    value of the coefficient is lower than 1.

    This implementation does not break ties at random, instead
    it breaks ties depending on order. This makes it dependent on
    data sorting, which could be useful in application like time
    series.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the symmetric xi correlation coefficient.

    See Also
    --------
    scipy.stats.chatterjeexi : Chatterjee Xi coefficient.
    obscure_stats.association.normalized_chatterjee_xi : Normalized Xi coefficient.

    Notes
    -----
    A symmetric normalized version is `max(normalized_chatterjee_xi(x, y),
    normalized_chatterjee_xi(y, x))`, the larger of the two normalized coefficients.
    Without ties it equals `symmetric_chatterjee_xi * (n + 1) / (n - 2)`, this
    coefficient multiplied by n + 1 and divided by n - 2; with ties they differ
    slightly.

    References
    ----------
    Chatterjee, S. (2021).
    A new coefficient of correlation.
    Journal of the American Statistical Association, 116(536), 2009-2022.

    Examples
    --------
    >>> from obscure_stats.association import symmetric_chatterjee_xi
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(symmetric_chatterjee_xi(x, y), 4)
    0.3333
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    return float(max(stats.chatterjeexi(x, y)[0], stats.chatterjeexi(y, x)[0]))


def zhang_i(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate I correlation coefficient proposed by Q. Zhang.

    This coefficient combines Spearman and Chatterjee rank correlation coefficients
    to get higher sensitivity to complex nonlinear relationships between variables.
    The value is capped at 1 for use as a similarity measure; the original
    statistic is unbounded above.

    Like `symmetric_chatterjee_xi`, which it uses, it does not break ties at
    random: ties in x are broken by input order, so with ties the result
    depends on the order of the observations.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the Zhang I.

    See Also
    --------
    scipy.stats.spearmanr : Spearman R coefficient.
    obscure_stats.association.symmetric_chatterjee_xi : Chatterjee Xi coefficient.

    References
    ----------
    Zhang, Q. (2025).
    On the extensions of the Chatterjee-Spearman test.
    Journal of Nonparametric Statistics, 1-30.

    Examples
    --------
    >>> from obscure_stats.association import zhang_i
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(zhang_i(x, y), 4)
    0.881
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    spearman = abs(float(stats.spearmanr(x, y, nan_policy="omit")[0]))
    return min(1.0, max(spearman, 2.5**0.5 * symmetric_chatterjee_xi(x, y)))


def tanimoto_similarity(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate Tanimoto similarity.

    It is `xy / (xx + yy - xy)`, where `xy = mean(x_i y_i)`, `xx = mean(x_i^2)` and `yy
    = mean(y_i^2)`. In words: the mean product of x and y, divided by the sum of the
    mean squares of x and y minus that mean product.

    It is very similar to Jaccard or Cosine similarity but differs in how
    dot product is normalized.
    This version is designed for numeric values, instead of sets.

    This measure is not addition invariant.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the Tanimoto similarity measure.

    See Also
    --------
    obscure_stats.association.morisita_horn_similarity : Morisita-Horn similarity.
    scipy.spatial.distance.jaccard : Jaccard dissimilarity.
    scipy.spatial.distance.cosine : Cosine distance.

    References
    ----------
    Rogers, D. J.; Tanimoto, T. T. (1960).
    A Computer Program for Classifying Plants.
    Science. 132 (3434): 1115-8.

    Examples
    --------
    >>> from obscure_stats.association import tanimoto_similarity
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(tanimoto_similarity(x, y), 4)
    0.9414
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    xy = np.mean(x * y)
    xx = np.mean(x**2)
    yy = np.mean(y**2)
    return float(xy / (xx + yy - xy))


def blomqvist_beta(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate Blomqvist's beta.

    It is the mean of `sign((x_i - median(x)) (y_i - median(y)))`. In words: the share
    of pairs that lie on the same side of both medians minus the share that lie on
    opposite sides; pairs on a median count as zero.

    Also known as medial correlation. It is similar to Spearman Rho
    and Kendall Tau correlations, but has some advantages over them.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the Blomqvist's beta.

    See Also
    --------
    scipy.stats.spearmanr : Spearman R coefficient.
    scipy.stats.kendalltau : Kendall Tau coefficient.

    References
    ----------
    Blomqvist,  N. (1950).
    On a measure of dependence between two random variables.
    Annals of Mathematical Statistics, 21, 593-600.

    Schmid, F.; Schmidt, R. (2007).
    Nonparametric Inference on Multivariate Versions of
    Blomqvist's Beta and Related Measures of Tail Dependence.
    Metrika, 66(3), 323-354.

    Examples
    --------
    >>> from obscure_stats.association import blomqvist_beta
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(blomqvist_beta(x, y), 4)
    1.0
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    med_x = np.median(x)
    med_y = np.median(y)
    return float(np.mean(np.sign((x - med_x) * (y - med_y))))


def fechner_correlation(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate Fechner correlation.

    It is the mean of `sign(x_i - mean(x)) sign(y_i - mean(y))`. In words: the share of
    pairs whose deviations from the means have the same sign minus the share whose
    deviations have opposite signs.

    It is similar to Blomqvist beta and Pearson correlation.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the Fechner correlation.

    See Also
    --------
    obscure_stats.association.blomqvist_beta : Blomqvist beta.

    References
    ----------
    Latyshev, A.V.; Koldanov, P.A. (2014).
    Investigation of Connections Between Pearson and Fechner Correlations
    in Market Network: Experimental Study.
    Models, Algorithms, and Technologies for Network Analysis, 156, 175-182

    Examples
    --------
    >>> from obscure_stats.association import fechner_correlation
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(fechner_correlation(x, y), 4)
    1.0
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    avg_x = np.mean(x)
    avg_y = np.mean(y)
    return float(np.mean(np.sign(x - avg_x) * np.sign(y - avg_y)))


def winsorized_correlation(x: npt.ArrayLike, y: npt.ArrayLike, k: float = 0.1) -> float:
    """Calculate winsorized correlation coefficient.

    It is Pearson's correlation of the winsorized arrays: in each array the smallest and
    the largest k share of values are replaced by the nearest remaining value.

    This correlation is a robust alternative of the Pearson correlation.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.
    k : float, default = 0.1
        The percentages of values to winsorize on each side of the arrays.

    Returns
    -------
    float
        The value of the winsorized correlation.

    See Also
    --------
    scipy.stats.pearsonr : Pearson correlation coefficient.

    References
    ----------
    Wilcox, R. R. (1993).
    Some Results on a Winsorized Correlation Coefficient.
    British Journal of Mathematical and Statistical Psychology, 46, 339-349.

    Examples
    --------
    >>> from obscure_stats.association import winsorized_correlation
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(winsorized_correlation(x, y), 4)
    0.8964
    """
    if not 0 <= k < 0.5:
        msg = "Parameter k should be in range [0, 0.5)."
        raise ValueError(msg)
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    x_w = stats.mstats.winsorize(x, (k, k))
    y_w = stats.mstats.winsorize(y, (k, k))
    return float(np.corrcoef(x_w, y_w)[0, 1])


def rank_minrelation_coefficient(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate rank minrelation coefficient.

    This measure estimates `p(y > x)`, the probability that y is greater than x, when x
    and y are continuous random variables. In short, if a variable x exhibits a
    minrelation to y then, as x increases, y is likely to increase too.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the rank minrelation coefficient.

    See Also
    --------
    obscure_stats.association.concordance_rate : Concordance rate.
    obscure_stats.association.concordance_correlation : Concordance correlation.

    Notes
    -----
    This measure is asymmetric: `(x, y) != (y, x)`.

    References
    ----------
    Meyer, P. E. (2013).
    A Rank Minrelation-Majrelation Coefficient.
    arXiv preprint arXiv:1305.2038.

    Examples
    --------
    >>> from obscure_stats.association import rank_minrelation_coefficient
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(rank_minrelation_coefficient(x, y), 4)
    0.9923
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    n = len(x)
    m_sq = n**2  # Meyer (2013), eqs. 6-8 divide by m^2
    r_y = stats.rankdata(y)
    rank_x_inc = (stats.rankdata(x) ** 2) / m_sq - 0.5
    rank_y_inc = (r_y**2) / m_sq - 0.5
    # rankdata(-y) == n + 1 - rankdata(y) for the default "average" ties
    rank_y_dec = -((n + 1 - r_y) ** 2) / m_sq + 0.5
    lower = np.sum((-rank_x_inc < rank_y_inc) * (rank_x_inc + rank_y_inc) ** 2)
    higher = np.sum((rank_x_inc > rank_y_dec) * (rank_x_inc - rank_y_dec) ** 2)
    return float((lower - higher) / (lower + higher))


def tukey_correlation(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate Tukey's correlation coefficient.

    It is `(GMD(u + v)^2 - GMD(u - v)^2) / 4`, where `u = x / GMD(x)`, `v = y / GMD(y)`
    and GMD is the Gini mean difference. In words: scale both variables by their Gini
    mean difference, then take a quarter of the difference between the squared Gini mean
    differences of their sum and of their difference. This mirrors the identity `cov(u,
    v) = (var(u + v) - var(u - v)) / 4` with the Gini mean difference in place of the
    standard deviation. The result is clipped to [-1, 1].

    It is not quite as robust as rank correlations, but it is more
    efficient in Gaussian and near-Gaussian cases.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the Tukey's correlation coefficient.

    Notes
    -----
    This measure is asymmetric: `(x, y) != (y, x)`.

    References
    ----------
    Wainer, H.; Thissen, D. (1976).
    Three steps toward robust regression.
    Psychometrika, 41, 9-34.

    Examples
    --------
    >>> from obscure_stats.association import tukey_correlation
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(tukey_correlation(x, y), 4)
    0.8689
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    s_x = gini_mean_difference(x)
    s_y = gini_mean_difference(y)
    x_norm = x / s_x
    y_norm = y / s_y
    coef = 0.25 * (
        gini_mean_difference(x_norm + y_norm) ** 2
        - gini_mean_difference(x_norm - y_norm) ** 2
    )
    return float(max(min(coef, 1.0), -1.0))


def gaussian_rank_correlation(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate Gaussian rank correlation coefficient.

    The Gaussian rank correlation equals the usual correlation coefficient
    computed from the normal scores of the data.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the Gaussian rank correlation coefficient.

    References
    ----------
    Boudt, K.; Cornelissen, J.; & Croux, C. (2012).
    The Gaussian rank correlation estimator: robustness properties.
    Statistics and Computing, 22, 471-483.

    Examples
    --------
    >>> from obscure_stats.association import gaussian_rank_correlation
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(gaussian_rank_correlation(x, y), 4)
    0.8715
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    n = len(x)
    norm_factor = 1 / (n + 1)
    x_ranks_norm = stats.rankdata(x) * norm_factor
    y_ranks_norm = stats.rankdata(y) * norm_factor
    coef = np.sum(stats.norm.ppf(x_ranks_norm) * stats.norm.ppf(y_ranks_norm)) / np.sum(
        stats.norm.ppf(np.arange(1, n + 1) * norm_factor) ** 2
    )
    return float(max(min(coef, 1.0), -1.0))


def quantile_correlation(x: npt.ArrayLike, y: npt.ArrayLike, q: float = 0.5) -> float:
    """Calculate quantile correlation.

    This function measures linear association between two
    variables for a given quantile.

    The arrays will be flattened before any calculations.
    This implementation is dependent on the order of the arrays,
    which could be useful in application like time series.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.
    q : float, default = 0.5
        Quantile used in the calculations.

    Returns
    -------
    float
        The value of the quantile sample correlation.

    Notes
    -----
    This measure is asymmetric: `(x, y) != (y, x)`.

    References
    ----------
    Li, G.; Li, Y.; & Tsai, C. (2015).
    Quantile Correlations and Quantile Autoregressive Modeling.
    Journal of the American Statistical Association, 110(509), 246-261.

    Examples
    --------
    >>> from obscure_stats.association import quantile_correlation
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(quantile_correlation(x, y), 4)
    0.8729
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    return float(
        np.mean((q - (y < np.quantile(y, q=q))) * (x - np.mean(x)))
        / (((q - q**2) * np.var(x)) ** 0.5)
    )


def _chatterjee_xi_max(y: npt.ArrayLike) -> float:
    """Calculate xi(y, y), the largest value Chatterjee's xi can take for this y.

    Closed form from Dalitz et al. (2024), Theorem 2, with ties:
    1 - n (n - r_min) / (2 sum l_i (n - l_i)), where r_i = #{j: y_j <= y_i} and
    l_i = #{j: y_j >= y_i}. Equals `stats.chatterjeexi(y, y)` with one sort.
    """
    ys = np.sort(y)
    n = len(ys)
    starts = np.flatnonzero(np.r_[True, ys[1:] != ys[:-1]])  # first index of each value
    counts = np.diff(np.r_[starts, n])
    n_ge = n - starts  # l_i for each distinct value
    return float(1.0 - n * (n - counts[0]) / (2.0 * np.sum(counts * n_ge * (n - n_ge))))


def normalized_chatterjee_xi(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate normalized Xi correlation coefficient.

    Another variation of rank correlation which does not make any assumptions about
    underlying distributions of the variable.

    It ranges from 0 (variables are completely independent) to 1
    (one is a measurable function of the other). This variant normalizes Chatterjee Xi,
    so its maximum will always be 1.0.

    This implementation does not break ties at random, instead
    it breaks ties depending on order. This makes it dependent on
    data sorting, which could be useful in application like time
    series.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the normalized xi correlation coefficient.

    Notes
    -----
    This measure is asymmetric: `(x, y) != (y, x)`.

    References
    ----------
    Dalitz, C.; Arning, J.; Goebbels, S. (2024).
    A Simple Bias Reduction for Chatterjee's Correlation.
    J Stat Theory Pract 18, 51.

    Examples
    --------
    >>> from obscure_stats.association import normalized_chatterjee_xi
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(normalized_chatterjee_xi(x, y), 4)
    0.5
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    ratio = stats.chatterjeexi(x, y)[0] / _chatterjee_xi_max(y)
    # Dalitz et al. (2024), eq. 8
    return float(max(-1.0, ratio))


def morisita_horn_similarity(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate Morisita-Horn similarity.

    It is `2 sum(x_i y_i) / sum(x_i^2 m_y / m_x + y_i^2 m_x / m_y)`, where m_x and m_y
    are the means of x and y. In words: twice the sum of the products of x and y,
    divided by the sum of the squares of x weighted by `m_y / m_x` plus the sum of the
    squares of y weighted by `m_x / m_y`.

    It is very similar to Jaccard or Cosine similarity but differs in how
    dot product is normalized.
    This version is designed for numeric values, instead of sets.

    This measure is not addition invariant.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the Morisita-Horn similarity.

    See Also
    --------
    obscure_stats.association.tanimoto_similarity : Tanimoto Similarity.

    References
    ----------
    Magurran A.E. (2005).
    Biological diversity.
    Curr Biol 15:R116-8.

    Examples
    --------
    >>> from obscure_stats.association import morisita_horn_similarity
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(morisita_horn_similarity(x, y), 4)
    0.9768
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return float(
        np.sum(2 * x * y) / np.sum(x**2 * mean_y / mean_x + y**2 * mean_x / mean_y)
    )


def _rtd_sum(
    r1: npt.NDArray[np.float64], r2: npt.NDArray[np.float64], a: float
) -> float:
    """Calculate the unnormalized rank-turbulence sum (Dodds et al., eq. 7)."""
    return float((a + 1.0) / a * np.sum(np.abs(r1**-a - r2**-a) ** (1.0 / (a + 1.0))))


def rank_divergence(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    a: float = 2.0,
    *,
    normalize: bool = False,
) -> float:
    """Calculate Rank-Turbulence divergence.

    It is a rank based divergence measure. Rank 1 is assigned to the largest
    value (the most abundant type), as in the original paper; ties get
    average ranks.
    As a → 0, high ranked types are increasingly dampened relative to low ranked ones.
    At the other end of the dial, a → ∞, high rank types will dominate.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.
    a : float, default = 2
        Weighting parameter.
    normalize : bool, default = False
        If True, divide by the normalization constant from the paper
        (computed by treating the two systems as disjoint), so the result
        lies in [0, 1]. If False, return the unnormalized sum divided by
        the number of pairs.

    Returns
    -------
    float
        The value of the rank-turbulence divergence.

    References
    ----------
    Dodds, P.S.; Minot, J.R.; Arnold, M.V. (2023).
    Allotaxonometry and rank-turbulence divergence:
    a universal instrument for comparing complex systems.
    EPJ Data Sci. 12, 37.

    Examples
    --------
    >>> from obscure_stats.association import rank_divergence
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(rank_divergence(x, y), 4)
    0.4224
    """
    if a <= 0:
        msg = "Parameter a should be > 0."
        raise ValueError(msg)
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    r1 = stats.rankdata(-x)
    r2 = stats.rankdata(-y)
    d = _rtd_sum(r1, r2, a)
    if not normalize:
        return d / len(x)
    # absent types are tied below every present one
    pad = np.full(len(x), -np.inf)
    d1 = stats.rankdata(-np.r_[x, pad])
    d2 = stats.rankdata(-np.r_[pad, y])
    return d / _rtd_sum(d1, d2, a)


def gini_correlation(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate Gini correlation coefficient.

    It is `cov(X, G(Y)) / cov(X, F(X))`, where F and G are the distribution functions of
    X and Y: X enters with its values and Y only through its ranks. It lies between
    Pearson's (all values) and Spearman's (all ranks) correlation. Ties in y count as
    zero (mid-ranks).

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the Gini correlation, in [-1, 1].

    Notes
    -----
    This measure is asymmetric: `(x, y) != (y, x)`. Computed as the ratio of
    U-statistics `sum_{i<j} (x_i - x_j) sign(y_i - y_j) / sum_{i<j} |x_i - x_j|` in O(n
    log n). In words: over all pairs of observations, the differences in x, each signed
    by the direction of the difference in y, summed and divided by the sum of the
    absolute differences in x. For bivariate normal data it estimates Pearson's
    correlation.

    References
    ----------
    Schechtman, E.; Yitzhaki, S. (1987).
    A measure of association based on Gini's mean difference.
    Communications in Statistics - Theory and Methods, 16(1), 207-231.

    Examples
    --------
    >>> from obscure_stats.association import gini_correlation
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(gini_correlation(x, y), 4)
    0.881
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    n = len(x)
    # sum_j sign(y_i - y_j) = 2 * midrank(y_i) - n - 1
    num = np.sum(x * (2.0 * stats.rankdata(y) - n - 1.0))
    den = np.sum(np.sort(x) * (2.0 * np.arange(1, n + 1) - n - 1.0))
    return float(num / den)


def coefficient_of_monotone_association(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate coefficient of monotone association (CMA).

    It measures how well x predicts the ordering of y (monotone dependence), on the AUC
    scale: 1 means y is a non-decreasing function of x, 0.5 means no monotone
    association, 0 means a non-increasing function. For binary y it equals the area
    under the ROC curve (AUC) of x, and for continuous data it is `(rho_S + 1) / 2`,
    where rho_S is Spearman's correlation.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values (predictor).
    y : array_like
        Numeric values (outcome).

    Returns
    -------
    float
        The value of the coefficient of monotone association, in [0, 1].

    Notes
    -----
    This measure is asymmetric: `(x, y) != (y, x)`. `CMA = (AGC + 1) / 2`, with the
    asymmetric grade correlation `AGC = sum (Q_i - c)(R_i - c) / sum (R_i - c)^2`, where
    Q and R are the mid-ranks of x and y and `c = (n + 1) / 2`. In words: center both
    sets of ranks at their mean rank, sum the products of the centered ranks of x and y,
    divide by the sum of the squared centered ranks of y, then add one and halve, which
    maps the result from [-1, 1] to [0, 1].

    References
    ----------
    Walz, E.; Eberl, A.; Gneiting, T. (2025).
    Assessing Monotone Dependence: Area Under the Curve Meets Rank Correlation.
    arXiv preprint arXiv:2510.17994.

    Examples
    --------
    >>> from obscure_stats.association import coefficient_of_monotone_association
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(coefficient_of_monotone_association(x, y), 4)
    0.9405
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    c = (len(x) + 1) / 2
    q = stats.rankdata(x) - c
    r = stats.rankdata(y) - c
    agc = np.sum(q * r) / np.sum(r * r)
    return float((agc + 1.0) / 2.0)


def l1_concordance_correlation(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate L1 concordance correlation coefficient (rho_1).

    It measures agreement between two variables (e.g. two methods measuring the same
    subjects), like the concordance correlation coefficient, but with absolute instead
    of squared differences, which makes it robust to outliers: `1 - mean |x_i - y_i| /
    mean over all i, j of |x_i - y_j|`. In words: one minus the average absolute
    difference between paired values, divided by the average absolute difference between
    any value of x and any value of y.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    float
        The value of the L1 concordance correlation, at most 1 (1 means x = y);
        it is not bounded below by -1.

    See Also
    --------
    obscure_stats.association.concordance_correlation : Concordance correlation.

    Notes
    -----
    This is the paper's distribution-free estimator (following King and Chinchilli,
    2001). The double sum includes i = j and is computed in O(n log n). For bivariate
    normal data with equal means it estimates `1 - sqrt(1 - rho_c)`, one minus the
    square root of one minus rho_c, where rho_c is Lin's concordance correlation.

    References
    ----------
    Vallejos, R.; Osorio, F.; Ferrer, C. (2025).
    A new coefficient to measure agreement between continuous variables.
    arXiv preprint arXiv:2507.07913.

    Examples
    --------
    >>> from obscure_stats.association import l1_concordance_correlation
    >>> x = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> y = [2, 1, 4, 3, 7, 8, 6, 9]
    >>> round(l1_concordance_correlation(x, y), 4)
    0.5745
    """
    prepared = _prepare_arrays(x, y)
    if prepared is None:
        return np.nan
    x, y = prepared
    n = len(x)
    ys = np.sort(y)
    cs = np.r_[0.0, np.cumsum(ys)]
    k = np.searchsorted(ys, x)  # number of y below each x
    cross = np.sum(x * k - cs[k] + (cs[-1] - cs[k]) - x * (n - k))
    return float(1.0 - np.mean(np.abs(x - y)) / (cross / n**2))
