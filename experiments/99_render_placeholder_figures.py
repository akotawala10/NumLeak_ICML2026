"""Render placeholder figures so the paper compiles with embedded images.

These are generated from synthetic recall patterns — NOT real sweep data.
Re-run experiments/00_pilot.py (and the full sweep) to replace with real
results before submission.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from factor_leak import plots


def build_mock_enriched(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    models = [
        "gpt-4.1", "gpt-4o-mini",
        "claude-sonnet-4.6", "claude-haiku-4.5",
        "llama-3.3-70b", "deepseek-v3",
    ]
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    factor_recall = {"Mkt-RF": 0.55, "SMB": 0.30, "HML": 0.48,
                     "RMW": 0.18, "CMA": 0.16, "Mom": 0.38}
    model_skill = {"gpt-4.1": 1.20, "gpt-4o-mini": 0.80,
                   "claude-sonnet-4.6": 1.10, "claude-haiku-4.5": 0.75,
                   "llama-3.3-70b": 0.95, "deepseek-v3": 1.00}

    rows: list[dict] = []
    for m in models:
        for f in factors:
            for y in range(1990, 2023):
                year_factor = 0.5 + 0.5 * (y - 1990) / 32.0
                p25 = min(0.95, factor_recall[f] * model_skill[m] * year_factor)
                p5 = p25 * 0.35
                p10 = p25 * 0.65
                for _ in range(4):
                    hit25 = rng.random() < p25
                    rows.append({
                        "model_name": m,
                        "factor": f,
                        "variant": "A",
                        "month": f"{y}-06",
                        "year": y,
                        "within_5bps": float(hit25 and rng.random() < (p5 / max(p25, 1e-9))),
                        "within_10bps": float(hit25 and rng.random() < (p10 / max(p25, 1e-9))),
                        "within_25bps": float(hit25),
                    })
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = REPO_ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    df = build_mock_enriched()

    fig1 = plots.recall_heatmap(df, tol_bps=25,
                                title="Fig 1 (PLACEHOLDER — synthetic recall pattern)")
    plots.save_figure(fig1, out_dir / "fig1_recall_heatmap")
    plt.close(fig1)

    fig2 = plots.temporal_gradient_plot(
        df, tol_bps=25,
        title="Fig 2 (PLACEHOLDER — synthetic temporal gradient)")
    plots.save_figure(fig2, out_dir / "fig2_temporal_gradient")
    plt.close(fig2)

    print("wrote:")
    for p in sorted(out_dir.glob("fig*.pdf")):
        print(" ", p.relative_to(REPO_ROOT), p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
