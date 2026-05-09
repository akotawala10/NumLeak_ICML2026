"""One-off: regenerate fig3_cross_model_mktrf.pdf showing only Opus,
Sonnet, GPT-5.4 (drop Haiku) so the body figure is consistent with
Tab. 1's 3-seed pooled Haiku value of r=0.27 (vs. the single-seed
main-sweep r=0.68 we'd otherwise need to footnote).
Reuses helpers in 40_cross_model_figures.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

# Load helpers from sibling script (avoid touching it)
spec = importlib.util.spec_from_file_location(
    "fig40", REPO / "experiments" / "40_cross_model_figures.py")
fig40 = importlib.util.module_from_spec(spec); sys.modules["fig40"] = fig40
spec.loader.exec_module(fig40)

# 3-model panel (drop Haiku)
PANEL = [
    ("Opus 4.7",   "claude-opus-4.7",   "#2ca02c"),
    ("Sonnet 4.6", "claude-sonnet-4.6", "#1f77b4"),
    ("GPT-5.4",    "gpt-5.4",           "#d62728"),
]


def main():
    by_model = fig40.load_mktrf_by_model()
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0))
    for (label, key, color), ax in zip(PANEL, axes):
        df = by_model.get(key)
        if df is None or len(df) < 3:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            continue
        r = float(np.corrcoef(df["truth"], df["estimate"])[0, 1])
        n = len(df)
        w25 = float(((df["estimate"] - df["truth"]).abs() <= 0.25).mean())
        lo = min(df["truth"].min(), df["estimate"].min()) - 1
        hi = max(df["truth"].max(), df["estimate"].max()) + 1
        lo, hi = min(lo, -25), max(hi, 20)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.5)
        ax.scatter(df["truth"], df["estimate"], s=22, color=color,
                   edgecolor="white", linewidth=0.4, alpha=0.9)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_title(f"{label}\n$r$={r:+.3f}, w25={w25:.0%}, $n$={n}",
                     fontsize=9)
        ax.set_xlabel("KF truth (\\%)", fontsize=8)
        ax.set_ylabel("model estimate (\\%)", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=7)
    fig.tight_layout()
    out = REPO / "figures" / "fig3_cross_model_mktrf.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
