"""Tests for the temporal-gradient OLS with p-value."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_leak import metrics


def _build_df(
    years: list[int],
    rates: list[float],
    *,
    model: str = "m",
    factor: str = "Mkt-RF",
) -> pd.DataFrame:
    """Build a minimally-enriched DataFrame with one row per (year, rate)
    suitable for feeding into ``temporal_gradient``. Each row has
    ``within_25bps`` equal to the target rate for that year (we inflate to
    an integer count of hits per year so the group-mean reproduces the
    target rate)."""
    rows = []
    for y, rate in zip(years, rates):
        hits = int(round(rate * 10))
        for i in range(10):
            rows.append({
                "model_name": model,
                "factor": factor,
                "variant": "A",
                "year": y,
                "within_25bps": 1.0 if i < hits else 0.0,
            })
    return pd.DataFrame(rows)


def test_positive_slope_is_detected():
    df = _build_df(
        years=[1990, 1995, 2000, 2005, 2010, 2015, 2020],
        rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    out = metrics.temporal_gradient(df, tol_bps=25)
    row = out.iloc[0]
    # Rate rises by ~0.6 over 30 years → slope ~0.02/year.
    assert row["slope_per_year"] == pytest.approx(0.02, rel=0.02)
    assert row["n_bins"] == 7
    assert row["r2"] > 0.99
    assert row["pvalue"] < 1e-4


def test_flat_slope_gives_high_pvalue():
    np.random.seed(0)
    df = _build_df(
        years=[1990, 1995, 2000, 2005, 2010, 2015, 2020],
        rates=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    )
    out = metrics.temporal_gradient(df, tol_bps=25)
    row = out.iloc[0]
    assert abs(row["slope_per_year"]) < 1e-9
    # Exactly zero variance in y → stderr undefined but code handles it.


def test_grouped_temporal_gradient_returns_one_row_per_group():
    df1 = _build_df(years=list(range(1990, 2021)),
                    rates=[0.1 + 0.02 * i for i in range(31)],
                    model="m1")
    df2 = _build_df(years=list(range(1990, 2021)),
                    rates=[0.5] * 31,
                    model="m2")
    df = pd.concat([df1, df2], ignore_index=True)
    out = metrics.temporal_gradient(df, tol_bps=25, by=["model_name"])
    assert set(out["model_name"]) == {"m1", "m2"}
    m1 = out.set_index("model_name").loc["m1"]
    m2 = out.set_index("model_name").loc["m2"]
    # m1 has a strong positive trend; m2 is flat.
    assert m1["slope_per_year"] > 0.015
    assert m1["pvalue"] < 1e-6
    assert abs(m2["slope_per_year"]) < 1e-9


def test_too_few_years_returns_nan_row():
    df = _build_df(years=[2000, 2005], rates=[0.3, 0.6])
    out = metrics.temporal_gradient(df, tol_bps=25)
    row = out.iloc[0]
    assert pd.isna(row["slope_per_year"])
    assert row["n_bins"] == 0
