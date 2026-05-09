"""Generate Fig 3: calibration scatter plot of Sonnet's Mkt-RF recall vs. truth.

Hero figure: shows r=0.98 visually. Points are colored by pre/near/post-cutoff
to emphasize that memorization is uniform across bucket.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from factor_leak import metrics, plots

FIG_DIR = REPO_ROOT / "figures"


def main() -> None:
    raw = metrics.load_results(REPO_ROOT / "experiments/results/sweep.jsonl")
    raw = raw[raw["error"].isna()].copy()
    df = metrics.enrich(raw, REPO_ROOT / "data/ff")

    sub = df[
        (df["variant"] == "A")
        & (df["model_name"] == "claude-sonnet-4.6")
        & (df["factor"] == "Mkt-RF")
    ].dropna(subset=["parsed_estimate", "truth"])

    fig, ax = plt.subplots(figsize=(4.0, 4.0))

    # Color-code by cutoff bucket
    palette = {"pre": "#1f77b4", "near": "#ff7f0e", "post": "#d62728"}
    for bucket, color in palette.items():
        mask = sub["bucket"] == bucket
        if mask.any():
            ax.scatter(sub.loc[mask, "truth"], sub.loc[mask, "parsed_estimate"],
                       c=color, label=f"{bucket} (n={int(mask.sum())})",
                       alpha=0.7, s=35, edgecolor="white", linewidth=0.4)

    # 45-degree line
    lo = min(sub["truth"].min(), sub["parsed_estimate"].min()) - 1
    hi = max(sub["truth"].max(), sub["parsed_estimate"].max()) + 1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1, label="perfect recall")

    # Fitted linear regression line
    x = sub["truth"].to_numpy()
    y = sub["parsed_estimate"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    xs = np.array([lo, hi])
    ax.plot(xs, slope * xs + intercept, color="grey", linewidth=1,
            label=f"OLS (slope={slope:.2f})")

    ax.set_xlabel("Kenneth French truth (%)")
    ax.set_ylabel("Sonnet 4.6 estimate (%)")
    ax.set_title(f"Sonnet 4.6 × Mkt-RF: $r$ = {r:.3f}, $n$ = {len(sub)}")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    plots.save_figure(fig, FIG_DIR / "fig3_calibration_sonnet_mktrf")
    plt.close(fig)
    print(f"wrote {FIG_DIR}/fig3_calibration_sonnet_mktrf.pdf (n={len(sub)}, r={r:.3f})")


if __name__ == "__main__":
    main()
