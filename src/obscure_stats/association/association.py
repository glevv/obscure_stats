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


def _is_constant(a: npt.NDArray[np.number]) -> bool:
    """Check whether all non-NaN values are exactly equal."""
    vals = np.ravel(a)
    vals = vals[~np.isnan(vals)]
    return vals.size > 1 and bool(np.all(vals == vals[0]))


def _check_arrays(x: npt.ArrayLike, y: npt.ArrayLike) -> bool:
    """Check arrays.

    - Equal lengths of the arrays;
    - Enough lengths of the arrays;
    - Constant input;
    - Contains inf;
    - Too many missing values.

    Inputs are converted to float64, so non-numeric input raises ValueError.
    Warnings use stacklevel=3 so they point at the caller of the public function.
    """
    _x = np.ravel(np.asarray(x, dtype=np.float64))
    _y = np.ravel(np.asarray(y, dtype=np.float64))
    if len(_x) != len(_y):
        warnings.warn(
            "Lengths of the inputs do not match, please check the arrays.", stacklevel=3
        )
        return True
    if len(_x) <= 1:
        warnings.warn(
            "Lengths of the inputs are too small, please check the arrays.",
            stacklevel=3,
        )
        return True
    if _is_constant(_x) or _is_constant(_y):
        warnings.warn(
            "One of the input arrays is constant;"
            " the correlation coefficient is not defined.",
            stacklevel=3,
        )
        return True
    if np.any(np.isinf(_x)) or np.any(np.isinf(_y)):
        warnings.warn(
            "One of the input arrays contains inf, please check the array.",
            stacklevel=3,
        )
        return True
    if (np.isnan(_x).sum() >= len(_x) - 1) or (np.isnan(_y).sum() >= len(_y) - 1):
        warnings.warn(
            "One of the input arrays has too many missing values,"
            " please check the arrays.",
            stacklevel=3,
        )
        return True
    return False


def _prep_arrays(
    x: npt.ArrayLike, y: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Prepare data for downstream task: float64, flattened, finite pairs only."""
    _x = np.ravel(np.asarray(x, dtype=np.float64))
    _y = np.ravel(np.asarray(y, dtype=np.float64))
    notnan = np.isfinite(_x) & np.isfinite(_y)
    _x = _x[notnan]
    _y = _y[notnan]
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

    Pairs inside the central square exclusion zone, max(|x|, |y|) <= exclusion,
    are treated as noise and dropped. With the default of 0 only pairs where
    neither variable changed are dropped. A pair outside the zone where only
    one variable changed counts as not concordant.

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
    cr : float.
        The value of the concordance rate, in [0, 1]. NaN if no pairs remain
        outside the exclusion zone.

    References
    ----------
    Critchley, L. A.; Lee, A.; Ho, A. M. (2010).
    A critical review of the ability of continuous cardiac output monitors
    to measure trends in cardiac output.
    Anesthesia & Analgesia, 111(5), 1180-1192.

    See Also
    --------
    obscure_stats.association.fechner_correlation - Fechner correlation.
    """
    if exclusion < 0:
        msg = "Parameter exclusion should be non-negative."
        raise ValueError(msg)
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
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
    it break ties depending on order. This makes it dependent on
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
    sxi : float.
        The value of the symmetric xi correlation coefficient.

    References
    ----------
    Chatterjee, S. (2021).
    A new coefficient of correlation.
    Journal of the American Statistical Association, 116(536), 2009-2022.

    See Also
    --------
    scipy.stats.chatterjee_xi - Chatterjee Xi coefficient.
    obscure_stats.association.normalized_chatterjee_xi - Normalized Xi coefficient.

    Notes
    -----
    A symmetric normalized version is max(normalized_chatterjee_xi(x, y),
    normalized_chatterjee_xi(y, x)). Without ties it equals
    symmetric_chatterjee_xi * (n + 1) / (n - 2); with ties they differ slightly.
    """
    if _check_arrays(x, y):
        return np.nan
    return float(
        max(
            stats.chatterjeexi(x, y, nan_policy="omit")[0],
            stats.chatterjeexi(y, x, nan_policy="omit")[0],
        )
    )


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
    zi : float.
        The value of the Zhang I.

    References
    ----------
    Zhang, Q. (2025).
    On the extensions of the Chatterjee-Spearman test.
    Journal of Nonparametric Statistics, 1-30.

    See Also
    --------
    scipy.stats.spearmanr - Spearman R coefficient.
    obscure_stats.association.symmetric_chatterjee_xi - Chatterjee Xi coefficient.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    return min(
        1.0,
        max(
            abs(stats.spearmanr(x, y, nan_policy="omit")[0]),
            2.5**0.5 * symmetric_chatterjee_xi(x, y),
        ),
    )


def tanimoto_similarity(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate Tanimoto similarity.

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
    ts : float.
        The value of the Tanimoto similarity measure

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
    return float(xy / (xx + yy - xy))


def blomqvist_beta(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate Blomqvist's beta.

    Also known as medial correlation. It is similar to Spearman Rho
    and Kendall Tau correlations, but have some advantages over them.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    bb : float.
        The value of the Blomqvist's beta.

    References
    ----------
    Blomqvist,  N. (1950).
    On a measure of dependence between two random variables.
    Annals of Mathematical Statistics, 21, 593-600.

    Schmid, F.; Schmidt, R. (2007).
    Nonparametric Inference on Multivariate Versions of
    Blomqvist's Beta and Related Measures of Tail Dependence.
    Metrika, 66(3), 323-354.

    See Also
    --------
    scipy.stats.spearmanr - Spearman R coefficient.
    scipy.stats.kendalltau - Kendall Tau coefficient.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    med_x = np.median(x)
    med_y = np.median(y)
    return float(np.mean(np.sign((x - med_x) * (y - med_y))))


def fechner_correlation(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate Fechner correlation.

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
    fc : float.
        The value of the Fechner correlation.

    References
    ----------
    Latyshev, A.V.; Koldanov, P.A. (2014).
    Investigation of Connections Between Pearson and Fechner Correlations
    in Market Network: Experimental Study.
    Models, Algorithms, and Technologies for Network Analysis, 156, 175-182

    See Also
    --------
    obscure_stats.association.blomqvist_beta - Blomqvist beta.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    avg_x = np.mean(x)
    avg_y = np.mean(y)
    return float(np.mean(np.sign(x - avg_x) * np.sign(y - avg_y)))


def winsorized_correlation(x: npt.ArrayLike, y: npt.ArrayLike, k: float = 0.1) -> float:
    """Calculate winsorized correlation coefficient.

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
    wcr : float.
        The value of the winsorized correlation.

    References
    ----------
    Wilcox, R. R. (1993).
    Some Results on a Winsorized Correlation Coefficient.
    British Journal of Mathematical and Statistical Psychology, 46, 339-349.

    See Also
    --------
    scipy.stats.pearsonr - Pearson correlation coefficient.
    """
    if not 0 <= k < 0.5:
        msg = "Parameter k should be in range [0, 0.5)."
        raise ValueError(msg)
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    x_w = stats.mstats.winsorize(x, (k, k))
    y_w = stats.mstats.winsorize(y, (k, k))
    return float(np.corrcoef(x_w, y_w)[0, 1])


def rank_minrelation_coefficient(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate rank minrelation coefficient.

    This measure estimates p(y > x) when x and y are continuous random variables.
    In short, if a variable x exhibits a minrelation to y then,
    as x increases, y is likely to increases too.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    rmc : float.
        The value of the rank minrelation coefficient.

    References
    ----------
    Meyer, P. E. (2013).
    A Rank Minrelation-Majrelation Coefficient.
    arXiv preprint arXiv:1305.2038.

    Notes
    -----
    This measure is asymmetric: (x, y) != (y, x).

    See Also
    --------
    Concordance rate.
    Concordance correlation coefficient.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
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
    tcc : float.
        The value of the Tukey's correlation coefficient.

    References
    ----------
    Wainer, H.; Thissen, D. (1976).
    Three steps toward robust regression.
    Psychometrika, 41, 9-34.

    Notes
    -----
    This measure is asymmetric: (x, y) != (y, x).
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
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
    grc : float.
        The value of the Gaussian rank correlation coefficient.

    References
    ----------
    Boudt, K.; Cornelissen, J.; & Croux, C. (2012).
    The Gaussian rank correlation estimator: robustness properties.
    Statistics and Computing, 22, 471-483.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
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
    qcor : float.
        The value of the quantile sample correlation.

    References
    ----------
    Li, G.; Li, Y.; & Tsai, C. (2015).
    Quantile Correlations and Quantile Autoregressive Modeling.
    Journal of the American Statistical Association, 110(509), 246-261.

    Notes
    -----
    This measure is asymmetric: (x, y) != (y, x).
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
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
    so it's maximum will always be 1.0.

    This implementation does not break ties at random, instead
    it break ties depending on order. This makes it dependent on
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
    nxi : float.
        The value of the normalized xi correlation coefficient.

    References
    ----------
    Dalitz, C.; Arning, J.; Goebbels, S. (2024).
    A Simple Bias Reduction for Chatterjee's Correlation.
    J Stat Theory Pract 18, 51.

    Notes
    -----
    This measure is asymmetric: (x, y) != (y, x).
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    ratio = stats.chatterjeexi(x, y)[0] / _chatterjee_xi_max(y)
    # Dalitz et al. (2024), eq. 8
    return float(max(-1.0, ratio))


def morisita_horn_similarity(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate Morisita-Horn similarity.

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
    mhs : float.
        The value of the Morisita-Horn similarity.

    References
    ----------
    Magurran A.E. (2005).
    Biological diversity.
    Curr Biol 15:R116-8.

    See Also
    --------
    obscure_stats.association.tanimoto_similarity - Tanimoto Similarity.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
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

    It is a rank based divergency measure. Rank 1 is assigned to the largest
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
    rtd : float.
        The value of the rank-turbulence divergence.

    References
    ----------
    Dodds, P.S.; Minot, J.R.; Arnold, M.V. (2023).
    Allotaxonometry and rank-turbulence divergence:
    a universal instrument for comparing complex systems.
    EPJ Data Sci. 12, 37.
    """
    if a <= 0:
        msg = "Parameter a should be > 0."
        raise ValueError(msg)
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
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

    It is cov(X, G(Y)) / cov(X, F(X)), where F and G are the distribution
    functions of X and Y: X enters with its values and Y only through its ranks.
    It lies between Pearson's (all values) and Spearman's (all ranks) correlation.
    Ties in y count as zero (mid-ranks).

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    gc : float.
        The value of the Gini correlation, in [-1, 1].

    References
    ----------
    Schechtman, E.; Yitzhaki, S. (1987).
    A measure of association based on Gini's mean difference.
    Communications in Statistics - Theory and Methods, 16(1), 207-231.

    Notes
    -----
    This measure is asymmetric: (x, y) != (y, x).
    Computed as the ratio of U-statistics sum_{i<j} (x_i - x_j) sign(y_i - y_j) /
    sum_{i<j} |x_i - x_j| in O(n log n). For bivariate normal data it estimates
    Pearson's correlation.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    n = len(x)
    # sum_j sign(y_i - y_j) = 2 * midrank(y_i) - n - 1
    num = np.sum(x * (2.0 * stats.rankdata(y) - n - 1.0))
    den = np.sum(np.sort(x) * (2.0 * np.arange(1, n + 1) - n - 1.0))
    return float(num / den)


def coefficient_of_monotone_association(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate coefficient of monotone association (CMA).

    It measures how well x predicts the ordering of y (monotone dependence),
    on the AUC scale: 1 means y is a non-decreasing function of x, 0.5 means no
    monotone association, 0 means a non-increasing function. For binary y it
    equals the area under the ROC curve (AUC) of x, and for continuous data it
    is (rho_S + 1) / 2, where rho_S is Spearman's correlation.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values (predictor).
    y : array_like
        Numeric values (outcome).

    Returns
    -------
    cma : float.
        The value of the coefficient of monotone association, in [0, 1].

    References
    ----------
    Walz, E.; Eberl, A.; Gneiting, T. (2025).
    Assessing Monotone Dependence: Area Under the Curve Meets Rank Correlation.
    arXiv preprint arXiv:2510.17994.

    Notes
    -----
    This measure is asymmetric: (x, y) != (y, x).
    CMA = (AGC + 1) / 2, with the asymmetric grade correlation
    AGC = sum (Q_i - c)(R_i - c) / sum (R_i - c)^2, where Q and R are the
    mid-ranks of x and y and c = (n + 1) / 2.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    c = (len(x) + 1) / 2
    q = stats.rankdata(x) - c
    r = stats.rankdata(y) - c
    agc = np.sum(q * r) / np.sum(r * r)
    return float((agc + 1.0) / 2.0)


def l1_concordance_correlation(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """Calculate L1 concordance correlation coefficient (rho_1).

    It measures agreement between two variables (e.g. two methods measuring the
    same subjects), like the concordance correlation coefficient, but with
    absolute instead of squared differences, which makes it robust to outliers:
    1 - mean |x_i - y_i| / mean over all i, j of |x_i - y_j|.

    The arrays will be flattened before any calculations.

    Parameters
    ----------
    x : array_like
        Numeric values.
    y : array_like
        Numeric values.

    Returns
    -------
    l1ccc : float.
        The value of the L1 concordance correlation, at most 1 (1 means x = y);
        it is not bounded below by -1.

    References
    ----------
    Vallejos, R.; Osorio, F.; Ferrer, C. (2025).
    A new coefficient to measure agreement between continuous variables.
    arXiv preprint arXiv:2507.07913.

    Notes
    -----
    This is the paper's distribution-free estimator (following King and
    Chinchilli, 2001). The double sum includes i = j and is computed in
    O(n log n). For bivariate normal data with equal means it estimates
    1 - sqrt(1 - rho_c), where rho_c is Lin's concordance correlation.

    See Also
    --------
    obscure_stats.association.concordance_correlation - Concordance correlation.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    n = len(x)
    ys = np.sort(y)
    cs = np.r_[0.0, np.cumsum(ys)]
    k = np.searchsorted(ys, x)  # number of y below each x
    cross = np.sum(x * k - cs[k] + (cs[-1] - cs[k]) - x * (n - k))
    return float(1.0 - np.mean(np.abs(x - y)) / (cross / n**2))
