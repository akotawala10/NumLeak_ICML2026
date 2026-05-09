"""Figures for the paper.

Two headline plots (handoff §8):

- ``recall_heatmap`` — Fig 1: within-tolerance recall by (factor × model).
- ``temporal_gradient_plot`` — Fig 2: annual recall rate over time per model.

Each function returns the matplotlib ``Figure`` and does not save to disk
automatically — callers choose the output path and format (PDF for the paper,
PNG for notebooks).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import metrics


# Column order matches factor_leak.ff_loader canonical order.
_FACTOR_ORDER = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def _pivot_recall(
    df: pd.DataFrame, tol_bps: int
) -> pd.DataFrame:
    rates = metrics.exact_match_rate(
        df[df["variant"].isin(["A", "B"])], tol_bps, by=["model_name", "factor"]
    )
    col = f"within_{tol_bps}bps"
    pivot = rates.pivot(index="model_name", columns="factor", values=col)
    present = [f for f in _FACTOR_ORDER if f in pivot.columns]
    return pivot[present]


def recall_heatmap(
    df: pd.DataFrame,
    tol_bps: int = 25,
    title: str | None = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """Fig 1: heatmap of within-``tol_bps`` recall, rows = model, cols = factor.

    Cell annotations show the rate as a percentage.
    """
    pivot = _pivot_recall(df, tol_bps)
    fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 2, 0.6 * len(pivot) + 1.5))
    mat = pivot.to_numpy(dtype=float)
    im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                label = "—"
                color = "black"
            else:
                label = f"{v * 100:.0f}%"
                color = "white" if v < 0.5 else "black"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"recall rate (|r̂−r| < {tol_bps} bps)")
    ax.set_xlabel("factor")
    ax.set_ylabel("model")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def temporal_gradient_plot(
    df: pd.DataFrame,
    tol_bps: int = 25,
    factor: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Fig 2: annual within-``tol_bps`` recall rate vs. year, one line per model.

    If ``factor`` is given, restrict to that factor. Otherwise aggregate
    across all factors.
    """
    col = f"within_{tol_bps}bps"
    sub = df[df["variant"].isin(["A", "B"])].dropna(subset=[col]).copy()
    if factor is not None:
        sub = sub[sub["factor"] == factor]

    annual = (
        sub.groupby(["model_name", "year"])[col].mean().reset_index()
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for model, frame in annual.groupby("model_name"):
        frame = frame.sort_values("year")
        ax.plot(frame["year"], frame[col], marker="o", markersize=3, linewidth=1.2, label=model)

    ax.set_xlabel("year")
    ax.set_ylabel(f"recall rate (|r̂−r| < {tol_bps} bps)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=False)
    if title:
        ax.set_title(title)
    elif factor:
        ax.set_title(f"Temporal recall gradient — {factor}")
    else:
        ax.set_title("Temporal recall gradient — all factors pooled")
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, out_path: Path | str, dpi: int = 200) -> Path:
    """Save as PDF (for paper) and PNG (for notebook preview) side-by-side."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    png_path = out_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=dpi)
    return pdf_path
