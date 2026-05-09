"""Tests for the metrics module.

Uses the pilot's dry-run (mock endpoints, ±30bps noise) to generate a small
in-memory JSONL and verify the full enrich + metrics pipeline.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_leak import ff_loader, metrics
from factor_leak.probe import (
    MockEndpoint,
    ProbeQuery,
    run_sweep,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "ff"


def _make_truth_responder(noise: float, seed: int = 0):
    rng = random.Random(seed)
    truth = ff_loader.load_all_factors(DATA_DIR)

    def _responder(prompt: str) -> str:
        import re

        mm = re.search(r"\b(January|February|March|April|May|June|July|"
                       r"August|September|October|November|December)\s+(\d{4})\b", prompt)
        if not mm:
            return "I don't know."
        from factor_leak.probe import MONTH_NAMES

        month = pd.Period(year=int(mm.group(2)),
                          month=MONTH_NAMES.index(mm.group(1)) + 1, freq="M")
        factor_map = {"market excess return": "Mkt-RF", "size": "SMB",
                      "value": "HML", "profitability": "RMW",
                      "investment": "CMA", "momentum": "Mom"}
        for phrase, key in factor_map.items():
            if phrase in prompt.lower():
                try:
                    v = truth.loc[month, key]
                except KeyError:
                    return "I don't know."
                if pd.isna(v):
                    return "I don't know."
                return f"{float(v) + rng.uniform(-noise, noise):.3f}"
        return "I don't know."

    return _responder


@pytest.fixture
def pilot_df(tmp_path: Path) -> pd.DataFrame:
    """Small mock sweep with predictable noise, enriched and returned."""
    endpoints = {
        "model_clean": MockEndpoint("model_clean", _make_truth_responder(noise=0.02, seed=1)),  # ±2bps
        "model_noisy": MockEndpoint("model_noisy", _make_truth_responder(noise=0.40, seed=2)),  # ±40bps
    }
    months = ["1987-10", "2008-10", "2020-03", "1995-05", "2001-09", "2014-07"]
    queries = [
        ProbeQuery(m, f, v, month)
        for m in ["model_clean", "model_noisy"]
        for f in ["Mkt-RF", "HML"]
        for v in ["A"]
        for month in months
    ]
    out = tmp_path / "sweep.jsonl"
    run_sweep(queries, endpoints, out, max_workers=1)
    raw = metrics.load_results(out)
    return metrics.enrich(raw, DATA_DIR)


# ---------------------------------------------------------------------------
# load_results + enrich
# ---------------------------------------------------------------------------


def test_enrich_adds_expected_columns(pilot_df: pd.DataFrame):
    required = {
        "truth", "parsed_estimate", "abs_error", "dir_correct",
        "within_5bps", "within_10bps", "within_25bps", "year",
    }
    assert required <= set(pilot_df.columns)
    assert pilot_df["truth"].notna().all()
    assert pilot_df["parsed_estimate"].notna().all()


def test_clean_model_hits_all_tolerances(pilot_df: pd.DataFrame):
    clean = pilot_df[pilot_df["model_name"] == "model_clean"]
    # ±2bps noise — easily within all tolerance bands.
    assert (clean["within_5bps"] == True).all()
    assert (clean["within_10bps"] == True).all()
    assert (clean["within_25bps"] == True).all()


def test_noisy_model_has_lower_tight_recall(pilot_df: pd.DataFrame):
    clean_rate = pilot_df[pilot_df["model_name"] == "model_clean"]["within_5bps"].mean()
    noisy_rate = pilot_df[pilot_df["model_name"] == "model_noisy"]["within_5bps"].mean()
    assert clean_rate > noisy_rate


# ---------------------------------------------------------------------------
# Scalar / grouped metrics
# ---------------------------------------------------------------------------


def test_exact_match_rate_scalar_vs_grouped(pilot_df: pd.DataFrame):
    overall = metrics.exact_match_rate(pilot_df, 25)
    assert 0.0 <= overall <= 1.0
    by_model = metrics.exact_match_rate(pilot_df, 25, by=["model_name"])
    assert set(by_model["model_name"]) == {"model_clean", "model_noisy"}


def test_directional_accuracy_high_for_low_noise(pilot_df: pd.DataFrame):
    by = metrics.directional_accuracy(pilot_df, by=["model_name"])
    clean = by.set_index("model_name").loc["model_clean", "dir_correct"]
    # All months chosen are nontrivial returns → sign should always match for clean model.
    assert clean == pytest.approx(1.0)


def test_calibration_correlation_positive_and_bounded(pilot_df: pd.DataFrame):
    r = metrics.calibration(pilot_df)
    assert -1.0 <= r <= 1.0
    assert r > 0.9  # mock perfectly tracks truth + tiny noise


# ---------------------------------------------------------------------------
# Temporal gradient
# ---------------------------------------------------------------------------


def test_temporal_gradient_returns_per_group(pilot_df: pd.DataFrame):
    grad = metrics.temporal_gradient(pilot_df, tol_bps=25, by=["model_name"])
    assert set(grad["model_name"]) == {"model_clean", "model_noisy"}
    assert {"slope_per_year", "intercept", "n_bins", "r2"} <= set(grad.columns)


# ---------------------------------------------------------------------------
# Famous-month concentration
# ---------------------------------------------------------------------------


def test_famous_concentration_splits_correctly(pilot_df: pd.DataFrame):
    famous = ["1987-10", "2008-10", "2020-03"]
    out = metrics.famous_concentration(pilot_df, famous, tol_bps=25, by=["model_name"])
    assert set(out["model_name"]) == {"model_clean", "model_noisy"}
    assert {"famous_rate", "other_rate", "lift"} <= set(out.columns)
    # Clean model has no rate differential (all perfect).
    clean = out.set_index("model_name").loc["model_clean"]
    assert clean["famous_rate"] == pytest.approx(clean["other_rate"])


# ---------------------------------------------------------------------------
# Headline table
# ---------------------------------------------------------------------------


def test_headline_table_has_one_row_per_model_factor(pilot_df: pd.DataFrame):
    table = metrics.headline_table(pilot_df)
    # 2 models x 2 factors.
    assert len(table) == 4
    assert {"model_name", "factor", "within_5bps", "within_25bps",
            "dir_correct", "pearson_r", "n_parsed"} <= set(table.columns)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_enrich_handles_api_errors(tmp_path: Path):
    """Rows with errors should keep NaN for derived stats."""
    def _fail(prompt: str) -> str:
        raise RuntimeError("upstream 500")

    ep = {"flaky": MockEndpoint("flaky", _fail)}
    q = ProbeQuery("flaky", "Mkt-RF", "A", "2020-03")
    out = tmp_path / "err.jsonl"
    run_sweep([q], ep, out, max_workers=1)
    raw = metrics.load_results(out)
    enriched = metrics.enrich(raw, DATA_DIR)
    row = enriched.iloc[0]
    assert row["error"] is not None
    assert pd.isna(row["parsed_estimate"])
    assert pd.isna(row["abs_error"])
    assert pd.isna(row["dir_correct"])


# ---------------------------------------------------------------------------
# Variant C: comparative accuracy
# ---------------------------------------------------------------------------


def test_comparative_accuracy_when_model_picks_first_month(tmp_path: Path):
    """A model that always names the first month in the prompt should be right
    whenever the first month is truly higher."""

    def _always_pick_first(prompt: str) -> str:
        # The prompt reads "Between <Month1 Year1> and <Month2 Year2>, ..." —
        # just echo the first month back.
        import re

        m = re.search(r"Between\s+([A-Za-z]+\s+\d{4})\s+and", prompt)
        return m.group(1) if m else "I don't know."

    ep = {"first_picker": MockEndpoint("first_picker", _always_pick_first)}
    pairs = [
        ("2020-03", "2008-10"),  # 2020-03 HML truly lower
        ("2008-10", "2020-03"),  # 2008-10 HML higher
    ]
    queries = [ProbeQuery("first_picker", "HML", "C", m1, month2=m2) for m1, m2 in pairs]
    out = tmp_path / "compC.jsonl"
    run_sweep(queries, ep, out, max_workers=1)
    raw = metrics.load_results(out)
    enriched = metrics.enrich(raw, DATA_DIR)
    # Check that the comparative_correct column is populated and the accuracy
    # equals the fraction of pairs where month1 really is higher.
    assert enriched["comparative_correct"].notna().all()
    acc = metrics.comparative_accuracy(enriched)
    # Of the two pairs, exactly one has month1 higher.
    assert acc == pytest.approx(0.5)
