"""Test the factor-shuffle null against the specific failure mode it exists to detect."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_leak import controls, metrics


def _enriched_df(estimates: dict[str, dict[str, float]],
                 truth: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a minimally-enriched DataFrame for controls testing.

    estimates[factor][month] = model estimate
    truth[factor][month] = Ken French truth
    """
    rows: list[dict] = []
    for factor, est_by_month in estimates.items():
        for month, est in est_by_month.items():
            rows.append({
                "model_name": "m",
                "factor": factor,
                "variant": "A",
                "month": month,
                "parsed_estimate": est,
                "truth": truth[factor][month],
                "error": None,
            })
    df = pd.DataFrame(rows)
    tol = 0.25
    df["within_25bps"] = (df["parsed_estimate"] - df["truth"]).abs() < tol
    return df


def test_factor_shuffle_null_flags_always_zero_predictor():
    """The crucial test: a model that always says ~0 for HML gets a
    high within-tol rate because truth for HML is often near zero. The
    factor-shuffle null, however, should be LOW for HML (because the
    model's non-HML estimates aren't near zero), exposing that HML recall
    is coincidence, not memorization."""
    months = [f"2000-{m:02d}" for m in range(1, 13)]
    truth = {
        "HML": {m: 0.1 for m in months},      # truth is consistently near 0
        "Mkt-RF": {m: 5.0 for m in months},   # truth is large positive
    }
    estimates = {
        # HML: model always predicts 0 → all within 25bps of 0.1 (yes, 10bps).
        "HML": {m: 0.0 for m in months},
        # Mkt-RF: model makes an accurate guess, say 5.0 on the nose.
        "Mkt-RF": {m: 5.0 for m in months},
    }
    df = _enriched_df(estimates, truth)

    out = controls.chance_rate_factor_shuffle(df, tol_bps=25, n_iters=200)
    by_factor = out.set_index("factor")

    # For HML: observed within-tol = 1.0 (always-0 estimate hits near-0 truth).
    assert by_factor.loc["HML", "observed_rate"] == pytest.approx(1.0)
    # But the null (pair Mkt-RF estimate=5.0 with HML truth=0.1) should be
    # very low (|5.0 - 0.1| = 4.9 > 0.25). Within-tol rate under null ≈ 0.
    assert by_factor.loc["HML", "chance_rate"] < 0.1, (
        "factor-shuffle should reveal that HML recall is coincidence"
    )


def test_factor_shuffle_null_respects_real_recall():
    """Conversely, when the model really does know each factor's truth,
    the null (which pairs the wrong factor's estimate with the right
    factor's truth) should be low, so observed > null."""
    months = [f"2000-{m:02d}" for m in range(1, 13)]
    truth = {
        "HML": {m: -2.0 + 0.1 * i for i, m in enumerate(months)},
        "Mkt-RF": {m: 4.0 - 0.2 * i for i, m in enumerate(months)},
    }
    # Model knows each factor perfectly.
    estimates = {f: {m: v for m, v in mv.items()} for f, mv in truth.items()}
    df = _enriched_df(estimates, truth)

    out = controls.chance_rate_factor_shuffle(df, tol_bps=25, n_iters=200)
    by_factor = out.set_index("factor")
    for f in ("HML", "Mkt-RF"):
        assert by_factor.loc[f, "observed_rate"] == pytest.approx(1.0)
        # The null pairs the wrong factor's estimate with this factor's
        # truth; magnitudes differ by several percent so within-tol is ~0.
        assert by_factor.loc[f, "chance_rate"] < 0.3


def test_factor_shuffle_skips_groups_with_fewer_than_two_factors():
    months = [f"2000-{m:02d}" for m in range(1, 13)]
    df = _enriched_df(
        {"HML": {m: 0.0 for m in months}},
        {"HML": {m: 0.1 for m in months}},
    )
    out = controls.chance_rate_factor_shuffle(df, tol_bps=25, n_iters=50)
    assert out.empty
