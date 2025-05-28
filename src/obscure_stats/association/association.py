"""Module for association measures."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
from scipy import stats  # type: ignore[import-untyped]

from obscure_stats.dispersion import gini_mean_difference


def _check_arrays(x: npt.NDArray, y: npt.NDArray) -> bool:
    """Check arrays.

    - Equal lenghts of the arrays;
    - Enough lenghts of the arrays;
    - Constant input;
    - Contains inf.
    """
    _x = np.ravel(x)
    _y = np.ravel(y)
    if len(_x) != len(_y):
        warnings.warn(
            "Lenghts of the inputs do not match, please check the arrays.", stacklevel=2
        )
        return True
    if len(_x) <= 1:
        warnings.warn(
            "Lenghts of the inputs are too small, please check the arrays.",
            stacklevel=2,
        )
        return True
    if np.all(np.isclose(_x, np.nanmin(_x), equal_nan=False)) or np.all(
        np.isclose(_y, np.nanmin(_y), equal_nan=False)
    ):
        warnings.warn(
            "One of the input arrays is constant;"
            " the correlation coefficient is not defined.",
            stacklevel=2,
        )
        return True
    if np.any(np.isinf(_x)) or np.any(np.isinf(_y)):
        warnings.warn(
            "One of the input arrays contains inf, please check the array.",
            stacklevel=2,
        )
        return True
    if (np.isnan(_x).sum() >= len(_x) - 1) or (np.isnan(_y).sum() >= len(_y) - 1):
        warnings.warn(
            "One of the input arrays has too many missing values,"
            " please check the arrays.",
            stacklevel=2,
        )
        return True
    return False


def _prep_arrays(x: npt.NDArray, y: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """Prepare data for downstream task."""
    _x = np.ravel(x)
    _y = np.ravel(y)
    notnan = np.isfinite(_x) & np.isfinite(_y)
    _x = _x[notnan]
    _y = _y[notnan]
    return _x, _y


def chatterjee_xi(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate Xi correlation coefficient.

    Another variation of rank correlation which does not make any assumptions about
    underlying distributions of the variable.

    It ranges from 0 (variables are completely independent) to 1
    (one is a measurable function of the other). But a lot of the times the maximum
    value of the coefficient is lower than 1.

    This implementation does not break ties at random, instead
    it break ties depending on order. This makes it dependent on
    data sorting, which could be useful in application like time
    series.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

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
    if _check_arrays(x, y):
        return np.nan
    # heavily inspired by https://github.com/czbiohub-sf/xicor/issues/17#issue-965635013
    n = len(x)
    y_forward_ordered = y[np.argsort(x)]
    _, y_unique_indexes, y_counts = np.unique(
        y_forward_ordered, return_inverse=True, return_counts=True
    )
    right = np.cumsum(y_counts)[y_unique_indexes]
    left = np.cumsum(y_counts[::-1])[len(y_counts) - y_unique_indexes - 1]
    return float(
        1.0 - 0.5 * np.sum(np.abs(np.diff(right))) / np.mean(left * (n - left))
    )


def concordance_correlation(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate concordance correlation coefficient.

    The main difference between Pearson's R and CCC is that CCC
    takes bias between variables into account.

    CCC measures the agreement between two variables, e.g.,
    to evaluate reproducibility or for inter-rater reliability.

    This measure will be sensetive to any perturbation in the arrays,
    i.e. it is not addition or multiplication invariant.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

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
    if _check_arrays(x, y):
        return np.nan
    std_x = np.std(x, ddof=0)
    std_y = np.std(y, ddof=0)
    w = std_y / std_x
    v = (np.mean(x) - np.mean(y)) ** 2 / (std_x * std_y) ** 0.5
    x_a = 2 / (v**2 + w + 1 / w)
    p = np.corrcoef(x, y)[0][1]
    return float(p * x_a)


def concordance_rate(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate conventional concordance rate.

    Also known as quadrant count ratio.
    It could be seen as simplified version of Pearson's R.

    It differs from quadrant count ratio by adding and exclusion zone
    variation has an option for an exclusion zone. It is based on the
    standard error of the mean and will exlucde points that are in the
    range of mean+-sem.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

    Returns
    -------
    cr : float.
        The value of the concordance rate.

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
    if _check_arrays(x, y):
        return np.nan
    n = len(x)
    mean_x = np.sum(x) / n
    mean_y = np.sum(y) / n
    sem_x = np.std(x, ddof=0) / n**0.5
    sem_y = np.std(y, ddof=0) / n**0.5
    return float(
        (
            np.sum((x >= mean_x + sem_x) & (y >= mean_y + sem_y))
            - np.sum((x <= mean_x - sem_x) & (y >= mean_y + sem_y))
            + np.sum((x <= mean_x - sem_x) & (y <= mean_y - sem_y))
            - np.sum((x >= mean_x + sem_x) & (y <= mean_y - sem_y))
        )
        / n
    )


def symmetric_chatterjee_xi(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate symmetric Xi correlation coefficient.

    Another variation of rank correlation which does not make any assumptions about
    underlying distributions of the variable.

    It ranges from 0 (variables are completely independent) to 1
    (one is a measurable function of the other). But a lot of the times the maximum
    value of the coefficient is lower than 1.

    This implementation does not break ties at random, instead
    it break ties depending on order. This makes it dependent on
    data sorting, which could be useful in application like time
    series.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

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
    obscure_stats.associaton.chatterjee_xi - Chatterjee Xi coefficient.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    if _check_arrays(x, y):
        return np.nan
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
    return float(
        1.0
        - min(
            0.5 * np.sum(np.abs(np.diff(right_xy))) / np.mean(left_xy * (n - left_xy)),
            0.5 * np.sum(np.abs(np.diff(right_yx))) / np.mean(left_yx * (n - left_yx)),
        )
    )


def zhang_i(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate I correlation coefficient proposed by Q. Zhang.

    This coefficient combines Spearman and Chatterjee rank correlation coefficients
    to get higher sensetivity to complex nonlinear relationships between variables.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

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
    obscure_stats.associaton.symmetric_chatterjee_xi - Chatterjee Xi coefficient.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    if _check_arrays(x, y):
        return np.nan
    return float(
        min(
            1.0,
            max(
                abs(stats.spearmanr(x, y, nan_policy="omit")[0]),
                2.5**0.5 * symmetric_chatterjee_xi(x, y),
            ),
        )
    )


def tanimoto_similarity(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate Tanimoto similarity.

    It is very similar to Jaccard or Cosine similarity but differs in how
    dot product is normalized.
    This version is designed for numeric values, instead of sets.

    This measure is not adddition invariant.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

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
    if _check_arrays(x, y):
        return np.nan
    xy = np.mean(x * y)
    xx = np.mean(x**2)
    yy = np.mean(y**2)
    return float(xy / (xx + yy - xy))


def blomqvist_beta(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate Blomqvist's beta.

    Also known as medial correlation. It is similar to Spearman Rho
    and Kendall Tau correlations, but have some advantages over them.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

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
    if _check_arrays(x, y):
        return np.nan
    med_x = np.median(x)
    med_y = np.median(y)
    return float(np.mean(np.sign((x - med_x) * (y - med_y))))


def fechner_correlation(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate Fechner correlation.

    It is similar to Bloomqvist beta and Pearson correlation.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

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
    obscure_stats.associaton.blomqvist_beta - Bloomqvist beta.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    if _check_arrays(x, y):
        return np.nan
    avg_x = np.mean(x)
    avg_y = np.mean(y)
    return float(np.mean(np.sign(x - avg_x) * np.sign(y - avg_y)))


def winsorized_correlation(x: npt.NDArray, y: npt.NDArray, k: float = 0.1) -> float:
    """Calculate winsorized correlation coefficient.

    This correlation is a robust alternative of the Pearson correlation.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.
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
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    if _check_arrays(x, y):
        return np.nan
    x_w = stats.mstats.winsorize(x, (k, k))
    y_w = stats.mstats.winsorize(y, (k, k))
    return float(np.corrcoef(x_w, y_w)[0, 1])


def rank_minrelation_coefficient(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate rank minrelation coefficient.

    This measure estimates p(y > x) when x and y are continuous random variables.
    In short, if a variable x exhibits a minrelation to y then,
    as x increases, y is likely to increases too.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

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
    This measure is assymetric: (x, y) != (y, x).

    See Also
    --------
    Concordance rate.
    Concordance correlation coefficient.
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    if _check_arrays(x, y):
        return np.nan
    n_sq = len(x) ** 2 + 1
    rank_x_inc = (stats.rankdata(x) ** 2) / n_sq - 0.5
    rank_y_inc = (stats.rankdata(y) ** 2) / n_sq - 0.5
    rank_y_dec = -(stats.rankdata(-y) ** 2) / n_sq + 0.5
    lower = np.sum((-rank_x_inc < rank_y_inc) * (rank_x_inc + rank_y_inc) ** 2)
    higher = np.sum((rank_x_inc > rank_y_dec) * (rank_x_inc - rank_y_dec) ** 2)
    return float((lower - higher) / (lower + higher))


def tukey_correlation(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate Tukey's correlation coefficient.

    It is not quite as robust as rank correlations, but it is more
    efficient in Gaussian and near-Gaussian cases.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

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
    This measure is assymetric: (x, y) != (y, x).
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    if _check_arrays(x, y):
        return np.nan
    s_x = gini_mean_difference(x)
    s_y = gini_mean_difference(y)
    x_norm = x / s_x
    y_norm = y / s_y
    coef = 0.25 * (
        gini_mean_difference(x_norm + y_norm) ** 2
        - gini_mean_difference(x_norm - y_norm) ** 2
    )
    return float(max(min(coef, 1.0), -1.0))


def gaussain_rank_correlation(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate Gaussian rank correlation coefficient.

    The Gaussian rank correlation equals the usual correlation coefficient
    computed from the normal scores of the data.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

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
    if _check_arrays(x, y):
        return np.nan
    n = len(x)
    norm_factor = 1 / (n + 1)
    x_ranks_norm = stats.rankdata(x) * norm_factor
    y_ranks_norm = stats.rankdata(y) * norm_factor
    coef = np.sum(stats.norm.ppf(x_ranks_norm) * stats.norm.ppf(y_ranks_norm)) / np.sum(
        stats.norm.ppf(np.arange(1, n + 1) * norm_factor) ** 2
    )
    return float(max(min(coef, 1.0), -1.0))


def quantile_correlation(x: npt.NDArray, y: npt.NDArray, q: float = 0.5) -> float:
    """Calculate quantile correlation.

    This function measures linear association between two
    variables for a given quantile.

    The arrays will be flatten before any calculations.
    This implementation is dependent on the order of the arrays,
    which could be useful in application like time series.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.
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
    This measure is assymetric: (x, y) != (y, x).
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    if _check_arrays(x, y):
        return np.nan
    return float(
        np.mean((q - (y < np.quantile(y, q=q))) * (x - np.mean(x)))
        / (((q - q**2) * np.var(x)) ** 0.5)
    )


def normalized_chatterjee_xi(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate normalizd Xi correlation coefficient.

    Another variation of rank correlation which does not make any assumptions about
    underlying distributions of the variable.

    It ranges from 0 (variables are completely independent) to 1
    (one is a measurable function of the other). This variant normalizes Chatterjee Xi,
    so it's maximum will always be 1.0.

    This implementation does not break ties at random, instead
    it break ties depending on order. This makes it dependent on
    data sorting, which could be useful in application like time
    series.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

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
    This measure is assymetric: (x, y) != (y, x).
    """
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    if _check_arrays(x, y):
        return np.nan
    n = len(x)
    # y ~ f(x)
    y_forward_ordered = y[np.argsort(x)]
    _, y_unique_indexes, y_counts = np.unique(
        y_forward_ordered, return_inverse=True, return_counts=True
    )
    right_xy = np.cumsum(y_counts)[y_unique_indexes]
    left_xy = np.cumsum(y_counts[::-1])[len(y_counts) - y_unique_indexes - 1]
    # y ~ f(y)
    y_ordered = y[np.argsort(y)]
    _, y_unique_indexes, y_counts = np.unique(
        y_ordered, return_inverse=True, return_counts=True
    )
    right_yy = np.cumsum(y_counts)[y_unique_indexes]
    left_yy = np.cumsum(y_counts[::-1])[len(y_counts) - y_unique_indexes - 1]
    # divide one by another
    return float(
        max(
            -1,
            (
                1
                - 0.5
                * np.sum(np.abs(np.diff(right_xy)))
                / np.mean(left_xy * (n - left_xy))
            )
            / (
                1
                - 0.5
                * np.sum(np.abs(np.diff(right_yy)))
                / np.mean(left_yy * (n - left_yy)),
            ),
        )
    )


def morisita_horn_similarity(x: npt.NDArray, y: npt.NDArray) -> float:
    """Calculate Morisita-Horn similarity.

    It is very similar to Jaccard or Cosine similarity but differs in how
    dot product is normalized.
    This version is designed for numeric values, instead of sets.

    This measure is not adddition invariant.

    The arrays will be flatten before any calculations.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.

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
    if _check_arrays(x, y):
        return np.nan
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    return float(
        np.sum(2 * x * y) / np.sum(x**2 * mean_y / mean_x + y**2 * mean_x / mean_y)
    )


def rank_divergence(x: npt.NDArray, y: npt.NDArray, a: float = 2.0) -> float:
    """Calculate Rank-Turbulence divergence.

    It is a rank based divergency measure.
    As a → 0, high ranked types are increasingly dampened relative to low ranked ones.
    At the other end of the dial, a → ∞, high rank types will dominate.

    This measure is unnormalized.

    Parameters
    ----------
    x : array_like
        Input array.
    y : array_like
        Input array.
    a : float, default = 2
        Weighting parameter.

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
    if _check_arrays(x, y):
        return np.nan
    x, y = _prep_arrays(x, y)
    if _check_arrays(x, y):
        return np.nan
    if a <= 0:
        msg = "Parameter a should be > 0."
        raise ValueError(msg)
    return float(
        (a + 1.0)
        / a
        * np.mean(
            np.abs(1.0 / stats.rankdata(x) ** a - 1.0 / stats.rankdata(y) ** a)
            ** (1.0 / (a + 1.0))
        )
    )
