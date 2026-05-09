"""Generate appendix-quality figures.

Outputs (to figures/):
- figA1_calibration_grid.pdf      — 2×6 scatter of parsed vs truth, all cells
- figA2_mktrf_timeseries.pdf      — Ken French truth line + Sonnet dots
- figA3_variant_a_vs_d.pdf        — A and D estimates vs truth, Sonnet Mkt-RF
- figA4_refusal_heatmap.pdf       — 2×6×3 parse rate per (model, factor, variant)
- figA5_error_distributions.pdf   — per-factor abs-error histograms, Sonnet+Haiku

All figures set DPI, tight_layout and use consistent colors.
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

from factor_leak import metrics  # noqa: E402
from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402

FIG = REPO / "figures"
DATA = REPO / "data" / "ff"
RES = REPO / "experiments" / "results"

MODEL_ORDER = ["claude-sonnet-4.6", "claude-haiku-4.5"]
MODEL_LABEL = {"claude-sonnet-4.6": "Sonnet 4.6",
               "claude-haiku-4.5":  "Haiku 4.5"}
FACTOR_ORDER = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

BUCKET_COLOR = {"pre": "#1f77b4", "near": "#ff7f0e", "post": "#d62728"}


def _dedup(records):
    latest = {}
    for r in records:
        k = r["key"]
        prev = latest.get(k)
        if prev is None:
            latest[k] = r
        elif prev["error"] is not None and r["error"] is None:
            latest[k] = r
        elif (prev["error"] is None) == (r["error"] is None) and r["ts"] > prev["ts"]:
            latest[k] = r
    return list(latest.values())


def load_sweep():
    raw = [json.loads(l) for l in (RES / "sweep.jsonl").open() if l.strip()]
    raw = _dedup(raw)
    df = pd.DataFrame([
        {"model_name": r["query"]["model_name"],
         "factor": r["query"]["factor"],
         "variant": r["query"]["variant"],
         "month": r["query"]["month"],
         "month2": r["query"].get("month2"),
         "prompt": r["prompt"],
         "response": r["response"],
         "error": r["error"],
         "latency_s": r["latency_s"],
         "ts": r["ts"],
         "key": r["key"],
         "input_tokens": r.get("input_tokens", 0),
         "output_tokens": r.get("output_tokens", 0)}
        for r in raw
    ])
    df = df[df["error"].isna()].copy()
    return metrics.enrich(df, DATA)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"wrote {FIG}/{name}.pdf (+ .png)")


# =========================================================================
# A1 — calibration scatter grid (all 12 cells)
# =========================================================================
def fig_calibration_grid(df: pd.DataFrame) -> plt.Figure:
    sub = df[df["variant"] == "A"].dropna(subset=["parsed_estimate", "truth"])
    fig, axes = plt.subplots(2, 6, figsize=(14.0, 5.2), sharex=False, sharey=False)
    plt.subplots_adjust(wspace=0.35, hspace=0.28, left=0.06, right=0.99,
                        top=0.92, bottom=0.12)
    for i, model in enumerate(MODEL_ORDER):
        for j, factor in enumerate(FACTOR_ORDER):
            ax = axes[i, j]
            cell = sub[(sub["model_name"] == model) & (sub["factor"] == factor)]
            if len(cell) < 3:
                ax.set_visible(True)
                ax.text(0.5, 0.5, f"n={len(cell)}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="grey")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                for bucket, color in BUCKET_COLOR.items():
                    m = cell["bucket"] == bucket
                    if m.any():
                        ax.scatter(cell.loc[m, "truth"],
                                   cell.loc[m, "parsed_estimate"],
                                   c=color, s=14, alpha=0.75,
                                   edgecolor="white", linewidth=0.3)
                lo = min(cell["truth"].min(), cell["parsed_estimate"].min()) - 1
                hi = max(cell["truth"].max(), cell["parsed_estimate"].max()) + 1
                lim_lo = min(lo, -15); lim_hi = max(hi, 15)
                ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--",
                        linewidth=0.6, alpha=0.5)
                ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
                r = float(np.corrcoef(cell["truth"], cell["parsed_estimate"])[0, 1])
                ax.text(0.03, 0.97,
                        f"$r$={r:+.2f}\n$n$={len(cell)}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.0))
                ax.tick_params(axis="both", labelsize=7)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.25, linewidth=0.4)
            if i == 0:
                ax.set_title(factor, fontsize=10)
            if j == 0:
                ax.set_ylabel(MODEL_LABEL[model] + "\nestimate (%)", fontsize=9)
            if i == 1:
                ax.set_xlabel("truth (%)", fontsize=9)

    # single legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=c, markeredgecolor="white", markersize=7,
                      label=b) for b, c in BUCKET_COLOR.items()]
    handles.append(Line2D([0], [0], color="k", linestyle="--",
                          label="perfect recall"))
    fig.legend(handles=handles, loc="lower center",
               ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.00))
    fig.suptitle("Variant A: model estimate vs Kenneth French truth, all 12 cells",
                 fontsize=11, y=0.98)
    return fig


# =========================================================================
# A2 — Mkt-RF time series (truth line + Sonnet dots)
# =========================================================================
def fig_mktrf_timeseries(df: pd.DataFrame) -> plt.Figure:
    truth_df = load_all_factors(DATA)
    # Probe window: 1963-07 through 2026-02 (RMW/CMA coverage start).
    window_start = pd.Period("1963-07", freq="M")
    truth = truth_df["Mkt-RF"].dropna()
    truth = truth[truth.index >= window_start]
    truth_years = truth.index.to_timestamp()

    sonnet = df[(df["variant"] == "A")
                & (df["model_name"] == "claude-sonnet-4.6")
                & (df["factor"] == "Mkt-RF")
                ].dropna(subset=["parsed_estimate", "truth"]).copy()
    sonnet["t"] = pd.PeriodIndex(sonnet["month"], freq="M").to_timestamp()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 4.8),
                                         sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})
    ax_top.plot(truth_years, truth.values, color="#1f77b4", linewidth=0.7,
                alpha=0.6, label="Kenneth French truth")
    ax_top.scatter(sonnet["t"], sonnet["parsed_estimate"],
                   s=26, color="#d62728", edgecolor="white", linewidth=0.4,
                   alpha=0.9, zorder=5, label="Sonnet 4.6 estimate")
    ax_top.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax_top.set_ylabel("Mkt-RF monthly return (%)")
    ax_top.set_title("Sonnet 4.6 recalls the Mkt-RF time series across 63 years",
                     fontsize=11)
    ax_top.legend(loc="lower left", fontsize=9, frameon=False)
    ax_top.grid(True, alpha=0.3)

    # bottom: per-point residual |estimate - truth|
    sonnet = sonnet.sort_values("t")
    resid = (sonnet["parsed_estimate"] - sonnet["truth"]).abs()
    ax_bot.vlines(sonnet["t"], 0, resid, color="#d62728", linewidth=1.0, alpha=0.75)
    ax_bot.axhline(0.25, color="grey", linestyle="--", linewidth=0.6,
                   label="25 bps tolerance")
    ax_bot.set_ylabel("|estimate $-$ truth|\n(%)")
    ax_bot.set_xlabel("probe month")
    ax_bot.legend(loc="upper right", fontsize=8, frameon=False)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# =========================================================================
# A3 — Variant A vs Variant D (CoT breaks memorization)
# =========================================================================
def fig_variant_a_vs_d() -> plt.Figure:
    # Load sweep A (Sonnet Mkt-RF) + Variant D (Sonnet Mkt-RF)
    truth_df = load_all_factors(DATA)

    a_raw = [json.loads(l) for l in (RES / "sweep.jsonl").open() if l.strip()]
    a_raw = _dedup(a_raw)
    a = pd.DataFrame([
        {"month": r["query"]["month"], "response": r["response"]}
        for r in a_raw
        if r["error"] is None
        and r["query"]["model_name"] == "claude-sonnet-4.6"
        and r["query"]["factor"] == "Mkt-RF"
        and r["query"]["variant"] == "A"
    ])
    a["est_A"] = a["response"].map(parse_numeric)

    d_rows = [json.loads(l) for l in (RES / "variant_d.jsonl").open() if l.strip()]
    d = pd.DataFrame([
        {"month": r["month"], "response": r["response"]}
        for r in d_rows
        if r.get("error") is None and r["model_name"] == "claude-sonnet-4.6"
    ])
    d["est_D"] = d["response"].map(parse_numeric)

    merged = a.merge(d, on="month", how="inner",
                     suffixes=("_A", "_D")).dropna(subset=["est_A", "est_D"])
    merged["truth"] = truth_df["Mkt-RF"].reindex(
        pd.PeriodIndex(merged["month"], freq="M")).values

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.0, 4.5),
                                   gridspec_kw={"width_ratios": [1, 1]})

    # Left: Variant A vs Variant D calibration against truth
    lo = min(merged["truth"].min(), merged["est_A"].min(),
             merged["est_D"].min()) - 1
    hi = max(merged["truth"].max(), merged["est_A"].max(),
             merged["est_D"].max()) + 1
    axL.plot([lo, hi], [lo, hi], "k--", linewidth=0.6, alpha=0.5)
    axL.scatter(merged["truth"], merged["est_A"], s=28, color="#2ca02c",
                edgecolor="white", linewidth=0.4, alpha=0.85,
                label=f"Variant A (direct)")
    axL.scatter(merged["truth"], merged["est_D"], s=28, color="#d62728",
                edgecolor="white", linewidth=0.4, alpha=0.85,
                label=f"Variant D (CoT)")
    rA = np.corrcoef(merged["truth"], merged["est_A"])[0, 1]
    rD = np.corrcoef(merged["truth"], merged["est_D"])[0, 1]
    axL.set_xlim(lo, hi); axL.set_ylim(lo, hi); axL.set_aspect("equal")
    axL.set_xlabel("Kenneth French truth (%)")
    axL.set_ylabel("Sonnet 4.6 estimate (%)")
    axL.set_title(f"A vs D on matched Mkt-RF months ($n$={len(merged)})")
    axL.legend(loc="upper left", fontsize=9, frameon=False,
               title=f"$r_A$={rA:+.2f}\n$r_D$={rD:+.2f}",
               title_fontsize=9)
    axL.grid(True, alpha=0.25)

    # Right: per-month paired error |A - truth| vs |D - truth|
    err_A = (merged["est_A"] - merged["truth"]).abs()
    err_D = (merged["est_D"] - merged["truth"]).abs()
    hi_err = max(err_A.max(), err_D.max()) + 0.5
    axR.plot([0, hi_err], [0, hi_err], "k--", linewidth=0.6, alpha=0.5)
    axR.scatter(err_A, err_D, s=28, color="#9467bd",
                edgecolor="white", linewidth=0.4, alpha=0.85)
    axR.axvline(0.25, color="grey", linestyle=":", linewidth=0.5)
    axR.axhline(0.25, color="grey", linestyle=":", linewidth=0.5)
    axR.set_xlim(0, hi_err); axR.set_ylim(0, hi_err); axR.set_aspect("equal")
    axR.set_xlabel("|Variant A estimate $-$ truth| (%)")
    axR.set_ylabel("|Variant D estimate $-$ truth| (%)")
    axR.set_title("Per-month abs error: D is strictly worse")
    axR.grid(True, alpha=0.25)
    # count points above/below 45
    above = int((err_D > err_A).sum())
    below = int((err_D < err_A).sum())
    axR.text(0.97, 0.03,
             f"D worse than A: {above}/{above+below} months",
             transform=axR.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(facecolor="white", edgecolor="lightgrey", pad=2.5))
    fig.suptitle("Chain-of-thought degrades recall: Variant A (direct) vs Variant D (CoT)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


# =========================================================================
# A4 — Refusal heatmap (3 variants)
# =========================================================================
def fig_refusal_heatmap(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.2))
    for i, v in enumerate(["A", "B", "C"]):
        ax = axes[i]
        sub = df[df["variant"] == v].copy()
        if v == "C":
            sub["answered"] = sub["parsed_month"].notna()
        else:
            sub["answered"] = sub["parsed_estimate"].notna()
        tab = (sub.groupby(["model_name", "factor"])
                  .agg(rate=("answered", "mean")).reset_index())
        pivot = tab.pivot(index="model_name", columns="factor", values="rate")
        pivot = pivot.reindex(index=MODEL_ORDER, columns=FACTOR_ORDER)
        mat = pivot.to_numpy()
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v_ = mat[r, c]
                if np.isnan(v_):
                    label = "—"; color = "black"
                else:
                    label = f"{v_*100:.0f}"
                    color = "white" if (v_ < 0.25 or v_ > 0.85) else "black"
                ax.text(c, r, label, ha="center", va="center",
                        color=color, fontsize=9)
        ax.set_xticks(range(len(FACTOR_ORDER)))
        ax.set_xticklabels(FACTOR_ORDER, fontsize=9, rotation=0)
        if i == 0:
            ax.set_yticks(range(len(MODEL_ORDER)))
            ax.set_yticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=9)
        else:
            ax.set_yticks([])
        ax.set_title(f"Variant {v}", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, label="parse rate")
    fig.suptitle("Fraction of queries that produced a committal answer",
                 fontsize=11, y=1.03)
    return fig


# =========================================================================
# A5 — Error distribution ridge (optional, small multiples)
# =========================================================================
def fig_error_distributions(df: pd.DataFrame) -> plt.Figure:
    sub = df[df["variant"] == "A"].dropna(subset=["parsed_estimate", "truth"]).copy()
    sub["abs_err"] = (sub["parsed_estimate"] - sub["truth"]).abs()
    fig, axes = plt.subplots(2, 6, figsize=(13.5, 4.0), sharex=True, sharey=True)
    for i, model in enumerate(MODEL_ORDER):
        for j, factor in enumerate(FACTOR_ORDER):
            ax = axes[i, j]
            cell = sub[(sub["model_name"] == model) & (sub["factor"] == factor)]
            if len(cell) < 3:
                ax.set_xticks([]); ax.set_yticks([])
                continue
            ax.hist(cell["abs_err"], bins=np.arange(0, 14, 0.5),
                    color="#4c72b0", alpha=0.85, edgecolor="white", linewidth=0.3)
            ax.axvline(0.25, color="red", linestyle="--", linewidth=0.7,
                       alpha=0.7)
            w25 = float((cell["abs_err"] <= 0.25).mean())
            median = float(cell["abs_err"].median())
            ax.text(0.97, 0.95,
                    f"w25={w25:.0%}\nmed={median:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=1.5))
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.2, linewidth=0.4)
            if i == 0:
                ax.set_title(factor, fontsize=10)
            if j == 0:
                ax.set_ylabel(MODEL_LABEL[model], fontsize=9)
            if i == 1:
                ax.set_xlabel("|estimate $-$ truth| (%)", fontsize=8)
    fig.suptitle("Absolute-error distributions per (model, factor) — red dashed = 25 bps",
                 fontsize=11, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main() -> None:
    df = load_sweep()

    save(fig_calibration_grid(df), "figA1_calibration_grid")
    save(fig_mktrf_timeseries(df), "figA2_mktrf_timeseries")
    save(fig_variant_a_vs_d(),      "figA3_variant_a_vs_d")
    save(fig_refusal_heatmap(df),   "figA4_refusal_heatmap")
    save(fig_error_distributions(df), "figA5_error_distributions")


if __name__ == "__main__":
    main()
