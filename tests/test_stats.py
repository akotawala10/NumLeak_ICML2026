"""Tests for Wilson CIs, bootstrap Pearson CI, and Benjamini-Hochberg FDR."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_leak.metrics import (
    benjamini_hochberg,
    bootstrap_pearson_ci,
    exact_match_rate,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# Wilson-score
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "k,n,alpha",
    [(1, 10, 0.05), (5, 10, 0.05), (0, 10, 0.05), (10, 10, 0.05),
     (30, 100, 0.05), (99, 100, 0.01)],
)
def test_wilson_interval_contains_point_estimate(k, n, alpha):
    lo, hi = wilson_interval(k, n, alpha=alpha)
    p = k / n
    assert lo <= p <= hi


def test_wilson_zero_n_returns_nans():
    lo, hi = wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_wilson_bounds_stay_in_unit_interval():
    for k in range(0, 11):
        lo, hi = wilson_interval(k, 10)
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# Bootstrap Pearson
# ---------------------------------------------------------------------------


def test_bootstrap_pearson_high_correlation_interval_tight_and_positive():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    y = x + rng.standard_normal(200) * 0.1  # r ≈ 0.99
    r, lo, hi = bootstrap_pearson_ci(x, y, n_iters=300, seed=0)
    assert r > 0.9
    assert lo > 0.8 and hi <= 1.0


def test_bootstrap_pearson_no_relationship_includes_zero():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    y = rng.standard_normal(200)
    r, lo, hi = bootstrap_pearson_ci(x, y, n_iters=300, seed=1)
    assert lo < 0 < hi


def test_bootstrap_pearson_too_few_returns_nans():
    r, lo, hi = bootstrap_pearson_ci(np.array([1.0]), np.array([2.0]))
    assert np.isnan(r) and np.isnan(lo) and np.isnan(hi)


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------


def test_bh_rejects_all_zeros():
    q, rej = benjamini_hochberg([0.0, 0.0, 0.0], alpha=0.05)
    assert rej.all()
    assert np.allclose(q, 0.0)


def test_bh_rejects_none_when_all_large():
    q, rej = benjamini_hochberg([0.9, 0.8, 0.7], alpha=0.05)
    assert not rej.any()


def test_bh_partial_rejection():
    # 4 tests; with alpha=0.05, the i-th smallest is rejected if
    # p(i) <= i * 0.05 / 4. So p ≤ [0.0125, 0.025, 0.0375, 0.05].
    pvals = [0.001, 0.02, 0.04, 0.5]
    q, rej = benjamini_hochberg(pvals, alpha=0.05)
    # 0.001 → q = 0.004 ✓; 0.02 → q = 0.04 ✓; 0.04 → q = 0.0533 ✗; 0.5 → q = 0.5 ✗
    assert list(rej) == [True, True, False, False]


def test_bh_handles_nan():
    q, rej = benjamini_hochberg([0.001, float("nan"), 0.5])
    assert rej[0] == True
    assert not rej[1]  # NaN p → not rejected
    assert not rej[2]
    assert np.isnan(q[1])


# ---------------------------------------------------------------------------
# exact_match_rate with CI
# ---------------------------------------------------------------------------


def test_exact_match_rate_with_ci_attaches_wilson_bounds():
    df = pd.DataFrame({
        "model_name": ["m"] * 20,
        "factor": ["HML"] * 20,
        "variant": ["A"] * 20,
        "within_25bps": [1.0] * 6 + [0.0] * 14,
    })
    out = exact_match_rate(df, tol_bps=25, by=["model_name"], with_ci=True)
    row = out.iloc[0]
    assert row["within_25bps"] == pytest.approx(6 / 20)
    assert row["within_25bps_lo"] < row["within_25bps"] < row["within_25bps_hi"]
    assert row["n"] == 20
