"""Tests pinning the measures added in 0.8.0 to their papers' formulas and values."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

# pyrefly: ignore [untyped-import]
from scipy import stats

from obscure_stats.association import (
    coefficient_of_monotone_association,
    gini_correlation,
    l1_concordance_correlation,
)
from obscure_stats.central_tendency import shorth
from obscure_stats.dispersion import (
    length_of_shortest_half,
)


def _check(condition: bool, msg: str = "Result does not match reference.") -> None:  # noqa: FBT001
    """Raise ValueError with msg if the condition is false."""
    if not condition:
        raise ValueError(msg)


def _tied_pair(seed: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return a small sample with ties in both x and y."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 5, 25).astype(np.float64)
    y = x + rng.integers(0, 3, 25).astype(np.float64)
    return x, y


def test_l1_concordance_correlation_reference() -> None:
    """Test the Bland-Altman example against R L1pack::l1ccc (U-statistic)."""
    large = np.array(
        [
            494,
            395,
            516,
            434,
            476,
            557,
            413,
            442,
            650,
            433,
            417,
            656,
            267,
            478,
            178,
            423,
            427,
        ],
        dtype=float,
    )
    mini = np.array(
        [
            512,
            430,
            520,
            428,
            500,
            600,
            364,
            380,
            658,
            445,
            432,
            626,
            260,
            477,
            259,
            350,
            451,
        ],
        dtype=float,
    )
    _check(
        l1_concordance_correlation(large, mini) == pytest.approx(0.76416850, abs=1e-8)
    )
    _check(l1_concordance_correlation(large, large) == 1.0)


def test_coefficient_of_monotone_association() -> None:
    """Test that CMA is the AUC for binary y and (Spearman + 1) / 2 without ties."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    y_bin = (x + rng.normal(size=60) > 0).astype(float)
    pos, neg = x[y_bin == 1], x[y_bin == 0]
    auc = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    _check(coefficient_of_monotone_association(x, y_bin) == pytest.approx(auc))
    y = x + rng.normal(size=60)
    rho = stats.spearmanr(x, y)[0]
    _check(coefficient_of_monotone_association(x, y) == pytest.approx((rho + 1) / 2))


def test_gini_correlation() -> None:
    """Test the U-statistic ratio of Schechtman & Yitzhaki with ties."""
    x, y = _tied_pair(8)
    n = len(x)
    num = sum(
        (x[i] - x[j]) * np.sign(y[i] - y[j]) for i in range(n) for j in range(i + 1, n)
    )
    den = sum(abs(x[i] - x[j]) for i in range(n) for j in range(i + 1, n))
    _check(gini_correlation(x, y) == pytest.approx(num / den))
    z = np.random.default_rng(9).normal(size=50)
    _check(gini_correlation(z, np.exp(z)) == pytest.approx(1.0))


def test_shorth_and_length_of_shortest_half() -> None:
    """Test a hand-computed example with two equally short halves."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    # windows of 3: [1,2,3] (width 2, mean 2), [2,3,4] (width 2, mean 3), [3,4,10]
    _check(shorth(x) == pytest.approx(2.5))
    _check(length_of_shortest_half(x) == pytest.approx(2.0))
    _check(shorth(-x) == pytest.approx(-2.5))
    _check(np.isnan(shorth(np.array([]))))
