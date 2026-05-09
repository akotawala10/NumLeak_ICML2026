"""Figure: transmission scatter — sentiment vs truth Mkt-RF and vs
model's recalled Mkt-RF, for Sonnet (n=77) and Opus (n=40), with
OLS lines and slopes annotated.

Output: figures/figA9_transmission_scatter.{pdf,png}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.parse import parse_numeric  # noqa: E402

FIG = REPO / "figures"
RES = REPO / "experiments" / "results"


def main() -> None:
    raw = [json.loads(l) for l in (RES / "transmission.jsonl").open() if l.strip()]
    df = pd.DataFrame(raw)
    df["sentiment"] = df["response"].map(
        lambda x: parse_numeric(x) if x else None)
    df = df.dropna(subset=["sentiment", "truth_mktrf", "recall_estimate"])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))

    palette = {"claude-sonnet-4.6": "#1f77b4",
               "claude-opus-4.7":   "#2ca02c"}
    label = {"claude-sonnet-4.6": "Sonnet 4.6",
             "claude-opus-4.7":   "Opus 4.7"}

    for ax, xcol, xlab in zip(
        axes,
        ["truth_mktrf", "recall_estimate"],
        ["truth Mkt-RF (%)", "model's recalled Mkt-RF (%)"]):

        for model, grp in df.groupby("model_name"):
            x = grp[xcol].values; y = grp["sentiment"].values
            n = len(grp)
            slope, intercept = np.polyfit(x, y, 1)
            r = float(np.corrcoef(x, y)[0, 1])
            color = palette[model]
            ax.scatter(x, y, color=color, alpha=0.55, s=22,
                       edgecolor="white", linewidth=0.4)
            x_line = np.array([x.min(), x.max()])
            ax.plot(x_line, intercept + slope * x_line, "-", color=color,
                    linewidth=1.6, alpha=0.9,
                    label=f"{label[model]}: $\\beta{{=}}{slope:+.3f}$, "
                          f"$r{{=}}{r:+.2f}$, $n{{=}}{n}$")

        ax.axhline(0, color="#999", linewidth=0.5)
        ax.axvline(0, color="#999", linewidth=0.5)
        ax.set_xlabel(xlab)
        ax.set_ylabel("model's sentiment estimate ($-$1 to $+$1)")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    axes[0].set_title("Sentiment vs ground truth", fontsize=10)
    axes[1].set_title("Sentiment vs model's own recall estimate", fontsize=10)

    fig.tight_layout()
    out_pdf = FIG / "figA9_transmission_scatter.pdf"
    out_png = FIG / "figA9_transmission_scatter.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    print(f"wrote {out_pdf}\nwrote {out_png}")


if __name__ == "__main__":
    main()
