"""Regression tests pinning fixed bugs against independent reference formulas."""

from __future__ import annotations

import math
import typing
import warnings

import numpy as np
import numpy.typing as npt
import pytest

# pyrefly: ignore [untyped-import]
from scipy import stats

from obscure_stats.association import (
    blomqvist_beta,
    concordance_correlation,
    concordance_rate,
    morisita_horn_similarity,
    normalized_chatterjee_xi,
    rank_divergence,
    rank_minrelation_coefficient,
    symmetric_chatterjee_xi,
    winsorized_correlation,
)
from obscure_stats.association.association import _chatterjee_xi_max
from obscure_stats.central_tendency import (
    grenanders_m,
    half_sample_mode,
    hodges_lehmann_sen_location,
    tau_location,
)
from obscure_stats.dispersion import (
    fisher_index_of_dispersion,
    gini_mean_difference,
    morisita_index_of_dispersion,
    robust_coefficient_of_variation,
    shamos_estimator,
    shapiro_coefficient_of_variation,
    standard_quantile_absolute_deviation,
)
from obscure_stats.skewness import (
    auc_skew_gamma,
    cumulative_skew,
    left_quantile_weight,
    pearson_mode_skew,
    right_quantile_weight,
)
from obscure_stats.variation import (
    avdev,
    b_index,
    gibbs_m1,
    mcintosh_d,
    negative_extropy,
    renyi_entropy,
)


def _check(condition: bool, msg: str = "Result does not match reference.") -> None:  # noqa: FBT001
    """Raise ValueError with msg if the condition is false."""
    if not condition:
        raise ValueError(msg)


def _hsm_reference(x: npt.NDArray[np.float64]) -> float:
    """Calculate HSM. Port of R modeest::hsm (last-tie convention)."""
    y = np.sort(x)
    while len(y) >= 4:
        k = math.ceil(len(y) / 2) - 1
        widths = [y[i + k] - y[i] for i in range(len(y) - k)]
        j = max(i for i, w in enumerate(widths) if w == min(widths))
        if widths[j] == 0:
            return float(y[j])
        y = y[j : j + k + 1]
    if len(y) == 3:
        z = 2 * y[1] - y[0] - y[2]
        if z < 0:
            return float(y[:2].mean())
        if z > 0:
            return float(y[1:].mean())
        return float(y[1])
    return float(y.mean())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_hsm_matches_reference(seed: int) -> None:
    """Check if HSM consider every window and keep k  1 points per step."""
    x = np.random.default_rng(seed).gamma(2.0, size=257)
    _check(half_sample_mode(x) == pytest.approx(_hsm_reference(x)))


def test_hsm_three_points() -> None:
    """Check three-point case averages the two closest points."""
    _check(half_sample_mode(np.array([0.0, 1.0, 10.0])) == pytest.approx(0.5))
    _check(half_sample_mode(np.array([0.0, 9.0, 10.0])) == pytest.approx(9.5))


def test_hls_walsh_averages() -> None:
    """Test whether Walsh sums are computed over upper triangle, not the full matrix."""
    x = np.array([1.0, 2.0, 10.0])
    i, j = np.triu_indices(len(x))
    _check(
        hodges_lehmann_sen_location(x) == pytest.approx(np.median((x[i] + x[j]) / 2))
    )
    x = np.array([np.nan, np.nan])
    _check(np.isnan(hodges_lehmann_sen_location(x)))


def test_shamos_excludes_diagonal() -> None:
    """Test whether zero self-differences are excluded."""
    x = np.random.default_rng(3).normal(size=40)
    i, j = np.triu_indices(len(x), k=1)
    _check(shamos_estimator(x) == pytest.approx(np.median(np.abs(x[i] - x[j]))))


def test_gini_mean_difference_reference() -> None:
    """Test whether O(N log N) GMD equals the pairwise definition."""
    x = np.random.default_rng(4).normal(size=60)
    n = len(x)
    ref = np.abs(x[:, None] - x[None, :]).sum() / (n * (n - 1))
    _check(gini_mean_difference(x) == pytest.approx(ref))


def test_fisher_index_formula() -> None:
    """Test whether Fisher's index is sum((x - mean)^2) / mean."""
    x = np.random.default_rng(5).poisson(5, size=30).astype(float)
    ref = np.sum((x - x.mean()) ** 2) / x.mean()
    _check(fisher_index_of_dispersion(x) == pytest.approx(ref))


@pytest.mark.parametrize(
    "func",
    [
        fisher_index_of_dispersion,
        morisita_index_of_dispersion,
        gini_mean_difference,
        standard_quantile_absolute_deviation,
        cumulative_skew,
    ],
)
def test_nans_do_not_change_result(func: typing.Callable[..., float]) -> None:
    """Check if sample size must count only non-NaN observations."""
    x = np.random.default_rng(6).poisson(5, size=30).astype(float) + 1
    _check(func(np.r_[x, [np.nan] * 10]) == pytest.approx(func(x)))


@pytest.mark.parametrize("shift", [0.0, 0.5, 3.0])
def test_concordance_correlation_reference(shift: float) -> None:
    """Test whether Lin's CCC = 2 cov / (var_x  var_y  (mean_x - mean_y)^2)."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=200)
    y = x + rng.normal(scale=0.5, size=200) + shift
    ref = (
        2
        * np.cov(x, y, ddof=0)[0, 1]
        / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    )
    _check(concordance_correlation(x, y) == pytest.approx(ref))


def test_uniform_categories_max_variation() -> None:
    """Test whether B index and AVDEV equal 1 for perfectly even categories."""
    u = np.array(list("aabbccdd"))
    _check(b_index(u) == pytest.approx(1.0))
    _check(avdev(u) == pytest.approx(1.0))


def test_negative_extropy_constant() -> None:
    """Check that a single category must not produce NaN."""
    _check(negative_extropy(np.array(list("aaaa"))) == 0.0)


def test_quantile_weights_symmetric() -> None:
    """Test if symmetric data left and right tail weights agree in sign and size."""
    x = np.random.default_rng(8).normal(size=100_000)
    lqw, rqw = left_quantile_weight(x, 0.25), right_quantile_weight(x, 0.75)
    _check(rqw > 0)
    _check(lqw == pytest.approx(rqw, abs=0.02))


def test_auc_skew() -> None:
    """Check that symmetric data is ~0, right-skewed data positive and bounded."""
    rng = np.random.default_rng(9)
    _check(abs(auc_skew_gamma(rng.normal(size=20_000), dp=0.01)) < 0.05)
    res = auc_skew_gamma(rng.exponential(size=20_000))
    _check(0.2 < res <= 1.0)


def test_tau_location_zero_mad() -> None:
    """Test whether MAD == 0 must not produce NaN."""
    _check(tau_location(np.array([1.0, 1.0, 1.0, 1.0, 5.0])) == 1.0)


def test_cum_skew_all_nans() -> None:
    """Test whether cumulative skew would return nan for nan array."""
    _check(np.isnan(cumulative_skew(np.array([np.nan, np.nan]))))


def test_hsm_all_nans() -> None:
    """Test whether HSM would return nan for nan array."""
    _check(np.isnan(half_sample_mode(np.array([np.nan, np.nan]))))


def test_rank_divergence_rank_one_is_largest() -> None:
    """Test that rank 1 is the largest value (Dodds et al.), so top swaps dominate."""
    base = np.array([100.0, 3.0, 2.0, 1.0])
    top = rank_divergence(base, np.array([3.0, 100.0, 2.0, 1.0]))
    bottom = rank_divergence(base, np.array([100.0, 3.0, 1.0, 2.0]))
    _check(top > bottom, f"Top swap {top} should exceed bottom swap {bottom}.")


def test_rank_divergence_normalized() -> None:
    """Test that normalized RTD lies in [0, 1] and is 0 for identical input."""
    rng = np.random.default_rng(10)
    x, y = rng.random(50), rng.random(50)
    _check(0.0 <= rank_divergence(x, y, normalize=True) <= 1.0)
    _check(rank_divergence(x, x, normalize=True) == 0.0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: rank_divergence(np.ones(3), np.ones(3), a=0),
        lambda: winsorized_correlation(np.ones(3), np.ones(3), k=0.7),
    ],
)
def test_parameters_validated_before_array_checks(
    call: typing.Callable[[], float],
) -> None:
    """Test that invalid parameters raise even when the arrays are degenerate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="Parameter"):
            call()


def test_grenanders_m_accepts_numpy_int() -> None:
    """Test that numpy integers are accepted for k."""
    x = np.random.default_rng(11).normal(size=50)
    _check(grenanders_m(x, k=np.int64(3)) == pytest.approx(grenanders_m(x, k=3)))
    with pytest.raises(ValueError, match="Parameter k"):
        grenanders_m(x, k=True)


def test_renyi_entropy_min_entropy() -> None:
    """Test that infinite alpha returns min-entropy instead of crashing."""
    x = np.array(list("aaabbc"))
    _check(renyi_entropy(x, alpha=math.inf) == pytest.approx(1.0))
    _check(renyi_entropy(x, alpha=200) == pytest.approx(1.0, rel=1e-2))


@pytest.mark.parametrize("seed", range(5))
def test_pairwise_selection_matches_bruteforce(seed: int) -> None:
    """Test that exact selection equals the materialized-pairs definition."""
    rng = np.random.default_rng(seed)
    for x in (
        rng.normal(size=300),
        rng.integers(0, 5, size=257).astype(float),
        rng.standard_cauchy(size=199),
    ):
        n = len(x)
        walsh = np.concatenate([x[i] + x[i:] for i in range(n)])
        diffs = np.concatenate([np.abs(x[i] - x[i + 1 :]) for i in range(n - 1)])
        _check(hodges_lehmann_sen_location(x) == np.median(walsh) * 0.5)
        _check(shamos_estimator(x) == np.median(diffs))


def test_hls_keeps_inf_behaviour() -> None:
    """Test that infinite values still go through the exact (fallback) path."""
    _check(hodges_lehmann_sen_location(np.array([1.0, 2.0, np.inf])) == np.inf)


def test_degenerate_inputs_return_nan() -> None:
    """Test that empty or single-observation inputs return NaN without warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check(np.isnan(standard_quantile_absolute_deviation(np.array([np.nan]))))
        _check(np.isnan(mcintosh_d(np.array(["a"]))))
        _check(np.isnan(gini_mean_difference(np.array([]))))
    _check(gini_mean_difference(np.array([1.0])) == 0.0)


def test_rank_minrelation_reference() -> None:
    """Test against Meyer (2013) eq. 9 with the m^2 normalization."""
    rng = np.random.default_rng(14)
    x = rng.normal(size=40)
    y = x + rng.normal(size=40)
    m_sq = len(x) ** 2
    xa = stats.rankdata(x) ** 2 / m_sq - 0.5
    ya = stats.rankdata(y) ** 2 / m_sq - 0.5
    yd = 0.5 - stats.rankdata(-y) ** 2 / m_sq
    lower = np.sum((-xa < ya) * (xa + ya) ** 2)
    higher = np.sum((xa > yd) * (xa - yd) ** 2)
    ref = (lower - higher) / (lower + higher)
    _check(rank_minrelation_coefficient(x, y) == pytest.approx(ref))


def test_normalized_xi_lower_bound() -> None:
    """Test that normalized xi is clipped at -1 (Dalitz et al., eq. 8)."""
    x = np.arange(10.0)
    y = np.array([0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0, 0.0, 9.0])
    _check(normalized_chatterjee_xi(x, y) >= -1.0)


def test_auc_skew_matches_paper_estimator() -> None:
    """Test against eq. 8 of Arachchige & Prendergast (mean over a midpoint grid)."""
    x = np.random.default_rng(15).lognormal(size=500)
    j = 100
    ps = 0.5 * (np.arange(1, j + 1) - 0.5) / j
    lo = np.quantile(x, ps, method="median_unbiased")
    hi = np.quantile(x, 1 - ps, method="median_unbiased")
    med = np.quantile(x, 0.5, method="median_unbiased")
    ref = np.mean((hi + lo - 2 * med) / (hi - lo))
    _check(auc_skew_gamma(x, dp=0.005) == pytest.approx(ref))


@pytest.mark.parametrize("n", [2, 4, 10, 100, 150])
def test_sqad_has_no_consistency_constant(n: int) -> None:
    """Test that SQAD returns the raw 68.27% quantile of |x - median|."""
    x = np.random.default_rng(16).normal(size=n)
    raw = np.quantile(np.abs(x - np.median(x)), 0.6826894921370850)
    _check(standard_quantile_absolute_deviation(x) == pytest.approx(raw))


def test_robust_cvs_have_no_consistency_constant() -> None:
    """Test that robust CVs return raw MAD / median and IQR / median."""
    x = np.random.default_rng(17).lognormal(size=101)
    med = np.median(x)
    q1, q3 = np.quantile(x, [0.25, 0.75])
    _check(
        robust_coefficient_of_variation(x)
        == pytest.approx(np.median(np.abs(x - med)) / med)
    )
    _check(shapiro_coefficient_of_variation(x) == pytest.approx((q3 - q1) / med))


def test_chatterjee_xi_max_closed_form() -> None:
    """Test that the closed-form normalizer equals scipy's xi(y, y), with ties."""
    rng = np.random.default_rng(18)
    for y in (
        rng.normal(size=300),
        rng.integers(0, 4, size=300).astype(float),
        rng.integers(0, 300, size=300).astype(float),
    ):
        _check(_chatterjee_xi_max(y) == pytest.approx(stats.chatterjeexi(y, y)[0]))


def test_normalized_xi_uses_paired_sample() -> None:
    """Test that NaNs in x drop the pair from both numerator and normalizer."""
    rng = np.random.default_rng(19)
    x, y = rng.normal(size=60), rng.integers(0, 5, size=60).astype(float)
    x[:7] = np.nan
    ref = stats.chatterjeexi(x[7:], y[7:])[0] / stats.chatterjeexi(y[7:], y[7:])[0]
    _check(normalized_chatterjee_xi(x, y) == pytest.approx(max(-1.0, ref)))


def test_concordance_rate_hand_computed() -> None:
    """Test the four-quadrant concordance rate against hand-computed values."""
    x = np.array([1.0, 2.0, -1.0, -2.0, 3.0])
    y = np.array([1.0, 1.0, -1.0, 2.0, -0.5])
    # agreeing directions: (1, 1), (2, 1), (-1, -1) -> 3 of 5
    _check(concordance_rate(x, y) == pytest.approx(3 / 5))
    # zone max(|x|, |y|) <= 1 drops (1, 1) and (-1, -1): 1 of the remaining 3 agrees
    _check(concordance_rate(x, y, exclusion=1.0) == pytest.approx(1 / 3))


def test_concordance_rate_zero_changes() -> None:
    """Test that (0, 0) is dropped and a one-sided change is not concordant."""
    x = np.array([0.0, 5.0, 1.0, -1.0])
    y = np.array([0.0, 0.0, 2.0, -3.0])
    # (0, 0) dropped by the default zone; (5, 0) kept but not concordant
    _check(concordance_rate(x, y) == pytest.approx(2 / 3))


def test_concordance_rate_bounds_and_edge_cases() -> None:
    """Test the [0, 1] range, an empty result and parameter validation."""
    y = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    _check(concordance_rate(y, 2 * y) == 1.0)
    _check(concordance_rate(y, -y) == 0.0)
    _check(np.isnan(concordance_rate(y, 2 * y, exclusion=100.0)))
    with pytest.raises(ValueError, match="Parameter exclusion"):
        concordance_rate(y, y, exclusion=-1.0)


def test_documented_variation_relationships() -> None:
    """Test the formulas in the Notes linking Gibbs M1, Renyi and McIntosh D."""
    rng = np.random.default_rng(20)
    c = rng.choice(np.array(list("abcde")), size=200, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    n = len(c)
    m1 = gibbs_m1(c)
    _check(renyi_entropy(c, alpha=2) == pytest.approx(-math.log2(1 - m1)))
    _check(
        mcintosh_d(c) == pytest.approx((1 - math.sqrt(1 - m1)) * n / (n - math.sqrt(n)))
    )


def test_documented_symmetric_xi_relationship() -> None:
    """Test that symmetric normalized xi is xi * (n + 1) / (n - 2) without ties."""
    rng = np.random.default_rng(21)
    x = rng.normal(size=150)
    y = x**2 + rng.normal(size=150)
    n = len(x)
    sym_norm = max(normalized_chatterjee_xi(x, y), normalized_chatterjee_xi(y, x))
    _check(sym_norm == pytest.approx(symmetric_chatterjee_xi(x, y) * (n + 1) / (n - 2)))


def test_pearson_mode_skew_uses_half_sample_mode() -> None:
    """Test that continuous data uses the half sample mode, not the minimum."""
    x = np.random.default_rng(22).lognormal(size=500)
    ref = (x.mean() - half_sample_mode(x)) / x.std()
    _check(pearson_mode_skew(x) == pytest.approx(ref))
    _check(pearson_mode_skew(x) != pytest.approx((x.mean() - x.min()) / x.std()))


def test_pearson_mode_skew_discrete_mode() -> None:
    """Test that a dominant repeated value is used as the mode."""
    x = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 7.0, 9.0])
    _check(pearson_mode_skew(x) == pytest.approx((x.mean() - 2.0) / x.std()))


def test_grenanders_m_default_rule() -> None:
    """Test that defaults follow Hall (1982): p = n^(2/7), k = ceil(n^(3/5))."""
    x = np.random.default_rng(23).lognormal(size=500)
    n = len(x)
    ref = grenanders_m(x, p=n ** (2 / 7), k=math.ceil(n**0.6))
    _check(grenanders_m(x) == pytest.approx(ref))


def test_grenanders_m_targets_the_mode() -> None:
    """Test that defaults estimate the mode, not the f^(p+1)-weighted mean."""
    x = np.random.default_rng(24).lognormal(0.0, 0.5, size=20_000)
    mode = math.exp(-0.25)  # 0.779; the old defaults converged to ~0.94
    _check(abs(grenanders_m(x) - mode) < 0.08)


def test_grenanders_m_tiny_samples() -> None:
    """Test that samples too small for the default window return NaN."""
    _check(np.isnan(grenanders_m(np.array([1.0]))))
    _check(np.isnan(grenanders_m(np.array([1.0, 2.0]))))
    _check(np.isfinite(grenanders_m(np.array([1.0, 2.0, 4.0]))))


def test_association_converts_integers_to_float() -> None:
    """Test that large integer inputs do not overflow (they did in x**2)."""
    x = np.array([4_000_000_000, 4_000_000_001, 4_000_000_005, 4_000_000_002])
    y = x[::-1].copy()
    res_int = morisita_horn_similarity(x, y)
    res_float = morisita_horn_similarity(x.astype(float), y.astype(float))
    _check(res_int == pytest.approx(res_float))


def test_association_non_numeric_input() -> None:
    """Test that non-numeric input raises a clear ValueError."""
    with pytest.raises(ValueError, match="could not convert"):
        blomqvist_beta(["a", "b", "c"], [1.0, 2.0, 3.0])


def test_association_warnings_point_at_caller() -> None:
    """Test that input warnings are attributed to the caller, not the library."""
    with pytest.warns(UserWarning, match="is constant") as rec:
        blomqvist_beta(np.ones(5), np.arange(5.0))
    _check(rec[0].filename == __file__, f"Warning attributed to {rec[0].filename}")
