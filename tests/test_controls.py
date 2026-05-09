"""Tests for controls module."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from factor_leak import controls, ff_loader, metrics
from factor_leak.probe import MockEndpoint, ProbeQuery, run_sweep

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "ff"


def test_synthetic_factor_prompt_mentions_factor_name_and_month():
    p = controls.synthetic_factor_prompt("2020-03")
    assert controls.SYNTHETIC_FACTOR_NAME in p
    assert "March 2020" in p
    assert "decimal percentage" in p


def test_illiquid_fund_prompt_mentions_fund_name_and_month():
    p = controls.illiquid_fund_prompt("1987-10")
    assert controls.ILLIQUID_FUND_NAME in p
    assert "October 1987" in p


def test_control_prompt_dispatch():
    assert "Gleason-Zeta" in controls.control_prompt("synthetic_factor", "2020-03")
    assert "Holbrooke-Mansfield" in controls.control_prompt("illiquid_fund", "2020-03")


def test_control_prompt_rejects_unknown_kind():
    import pytest

    with pytest.raises(ValueError):
        controls.control_prompt("nope", "2020-03")


def test_permutation_chance_rate_bounds_real_recall(tmp_path: Path):
    """Under the null (random-shuffled answers), the chance rate is below the
    true recall rate when the model actually knows something."""
    truth = ff_loader.load_all_factors(DATA_DIR)

    # A model that "knows" HML perfectly — truth plus ±2bps jitter.
    rng = random.Random(13)

    def _responder(prompt: str) -> str:
        import re

        mm = re.search(r"\b(January|February|March|April|May|June|July|"
                       r"August|September|October|November|December)\s+(\d{4})\b", prompt)
        from factor_leak.probe import MONTH_NAMES

        month = pd.Period(year=int(mm.group(2)),
                          month=MONTH_NAMES.index(mm.group(1)) + 1, freq="M")
        v = truth.loc[month, "HML"]
        return f"{float(v) + rng.uniform(-0.02, 0.02):.3f}"

    ep = {"oracle": MockEndpoint("oracle", _responder)}
    months = [str(pd.Period(f"{y}-06", freq="M")) for y in range(1990, 2015)]
    queries = [ProbeQuery("oracle", "HML", "A", m) for m in months]
    out = tmp_path / "sweep.jsonl"
    run_sweep(queries, ep, out, max_workers=1)
    enriched = metrics.enrich(metrics.load_results(out), DATA_DIR)

    true_recall = metrics.exact_match_rate(enriched, tol_bps=25)
    chance = controls.chance_rate_permutation(enriched, tol_bps=25, n_iters=200, seed=0)
    assert len(chance) == 1
    chance_rate = chance.iloc[0]["chance_rate"]
    assert true_recall > chance_rate + 0.2, (
        f"oracle should beat chance by a wide margin, got true={true_recall} vs chance={chance_rate}"
    )
