"""Smoke tests for plotting code."""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import pandas as pd
import pytest

from factor_leak import ff_loader, metrics, plots
from factor_leak.probe import MockEndpoint, ProbeQuery, run_sweep

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "ff"


@pytest.fixture(scope="module")
def enriched_df(tmp_path_factory) -> pd.DataFrame:
    truth = ff_loader.load_all_factors(DATA_DIR)

    def _responder(prompt: str) -> str:
        import re

        mm = re.search(
            r"\b(January|February|March|April|May|June|July|"
            r"August|September|October|November|December)\s+(\d{4})\b",
            prompt,
        )
        if not mm:
            return "I don't know."
        from factor_leak.probe import MONTH_NAMES

        month = pd.Period(
            year=int(mm.group(2)),
            month=MONTH_NAMES.index(mm.group(1)) + 1,
            freq="M",
        )
        factor_map = {
            "market excess return": "Mkt-RF",
            "size": "SMB",
            "value": "HML",
            "profitability": "RMW",
            "investment": "CMA",
            "momentum": "Mom",
        }
        for phrase, key in factor_map.items():
            if phrase in prompt.lower():
                try:
                    v = truth.loc[month, key]
                except KeyError:
                    return "I don't know."
                if pd.isna(v):
                    return "I don't know."
                return f"{float(v) + random.uniform(-0.3, 0.3):.3f}"
        return "I don't know."

    random.seed(42)
    endpoints = {name: MockEndpoint(name, _responder) for name in ["mA", "mB"]}
    months = [str(pd.Period(f"{y}-06", freq="M")) for y in [1990, 2000, 2010, 2020]]
    queries = [
        ProbeQuery(m, f, "A", month)
        for m in endpoints
        for f in ["Mkt-RF", "SMB", "HML"]
        for month in months
    ]
    out = tmp_path_factory.mktemp("sweep") / "sweep.jsonl"
    run_sweep(queries, endpoints, out, max_workers=1)
    return metrics.enrich(metrics.load_results(out), DATA_DIR)


def test_recall_heatmap_renders(enriched_df: pd.DataFrame, tmp_path: Path):
    fig = plots.recall_heatmap(enriched_df, tol_bps=25)
    out = tmp_path / "heatmap"
    saved = plots.save_figure(fig, out)
    assert saved.exists() and saved.stat().st_size > 0


def test_temporal_gradient_plot_renders(enriched_df: pd.DataFrame, tmp_path: Path):
    fig = plots.temporal_gradient_plot(enriched_df, tol_bps=25)
    out = tmp_path / "temporal"
    saved = plots.save_figure(fig, out)
    assert saved.exists() and saved.stat().st_size > 0


def test_temporal_plot_factor_filter_honored(enriched_df: pd.DataFrame, tmp_path: Path):
    fig = plots.temporal_gradient_plot(enriched_df, tol_bps=25, factor="HML")
    saved = plots.save_figure(fig, tmp_path / "temporal_hml")
    assert saved.exists()
