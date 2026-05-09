"""Rebuild fig2_temporal_gradient.pdf to include all 9 models on Mkt-RF.

Two-panel design:
- Left: per-model OLS slope of within-25bps on months-to-cutoff with
  95% CI from heteroskedasticity-robust SE. Forest plot showing all
  slopes overlap zero — the "no cutoff gradient" claim, clean.
- Right: per-model |error| (model est. - truth, abs value) vs
  months-to-cutoff, with a per-model rolling-mean line. Visualizes
  why the slopes overlap zero: error doesn't trend with proximity to
  cutoff for any model.

Llama-3.1-8B excluded (0/40 parsed, no recall to plot).

Output: figures/fig2_temporal_gradient.{pdf,png}
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

from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402

FIG = REPO / "figures"
DATA = REPO / "data" / "ff"
RES = REPO / "experiments" / "results"


# (display, internal, cutoff, color, source_file, source_kind)
MODELS = [
    ("Sonnet 4.6",    "claude-sonnet-4.6",  "2025-03", "#1f77b4", "sweep.jsonl",            "main"),
    ("Haiku 4.5",     "claude-haiku-4.5",   "2025-07", "#ff7f0e", "sweep.jsonl",            "main"),
    ("Opus 4.7",      "claude-opus-4.7",    "2026-01", "#2ca02c", "opus_baselines.jsonl",   "baseline"),
    ("GPT-5.4",       "gpt-5.4",            "2025-08", "#d62728", "gpt54_baselines.jsonl",  "baseline"),
    ("GPT-5.4-mini",  "gpt-5.4-mini",       "2025-08", "#9467bd", "more_baselines.jsonl",   "baseline"),
    ("GPT-5.4-nano",  "gpt-5.4-nano",       "2025-08", "#8c564b", "more_baselines.jsonl",   "baseline"),
    ("DeepSeek-V3.2", "deepseek-v3.2-azure","2025-07", "#e377c2", "more_baselines.jsonl",   "baseline"),
    ("Llama-3.3-70B", "llama-3.3-70b-groq", "2024-12", "#7f7f7f", "llama_baselines.jsonl",  "baseline"),
]


def _dedup_main(records):
    latest = {}
    for r in records:
        k = r["key"]
        prev = latest.get(k)
        if prev is None: latest[k] = r
        elif prev["error"] is not None and r["error"] is None: latest[k] = r
        elif (prev["error"] is None) == (r["error"] is None) and r["ts"] > prev["ts"]:
            latest[k] = r
    return list(latest.values())


def load_mktrf(internal: str, src_file: str, src_kind: str) -> pd.DataFrame:
    path = RES / src_file
    if not path.exists(): return pd.DataFrame()
    raw = [json.loads(l) for l in path.open() if l.strip()]
    if src_kind == "main":
        raw = _dedup_main(raw)
        rows = [{"month": r["query"]["month"], "response": r["response"],
                 "error": r["error"]}
                for r in raw
                if r["query"]["model_name"] == internal
                and r["query"]["factor"] == "Mkt-RF"
                and r["query"]["variant"] == "A"]
    else:
        rows = [{"month": r["month"], "response": r["response"],
                 "error": r["error"]}
                for r in raw
                if r["model_name"] == internal and r.get("probe") == "mktrf"]
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["parsed"] = df["response"].map(lambda x: parse_numeric(x) if x else None)
    return df.dropna(subset=["parsed"])


def main() -> None:
    truth = load_all_factors(DATA)["Mkt-RF"]

    rows: list[dict] = []
    for display, internal, cutoff_str, color, src_file, src_kind in MODELS:
        df = load_mktrf(internal, src_file, src_kind)
        if df.empty: continue
        cutoff = pd.Period(cutoff_str, freq="M")
        df = df.copy()
        df["period"] = df["month"].apply(lambda m: pd.Period(m, freq="M"))
        df["truth"] = df["period"].apply(lambda p: float(truth.get(p, np.nan)))
        df = df.dropna(subset=["truth"])
        df["months_to_cutoff"] = df["period"].apply(lambda p: (cutoff - p).n)
        df["within25"] = ((df["parsed"] - df["truth"]).abs() <= 0.25).astype(float)
        df["abs_err"] = (df["parsed"] - df["truth"]).abs()
        df["display"] = display
        df["color"] = color
        rows.append(df[["display", "color", "months_to_cutoff", "within25",
                        "abs_err"]])

    big = pd.concat(rows, ignore_index=True)

    # ---------- Panel A: forest plot of slopes -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5),
                             gridspec_kw={"width_ratios": [1.0, 1.5]})
    ax1 = axes[0]
    summary: list[dict] = []
    for display in [m[0] for m in MODELS]:
        sub = big[big["display"] == display]
        if len(sub) < 5: continue
        x = sub["months_to_cutoff"].values
        y = sub["within25"].values
        # OLS with HC-robust SE (without statsmodels): use residual SE
        x_c = x - x.mean()
        slope = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
        intercept = y.mean() - slope * x.mean()
        residuals = y - (intercept + slope * x)
        n = len(x)
        # Robust HC0 SE on slope
        sxx = (x_c ** 2).sum()
        var_slope = ((x_c ** 2 * residuals ** 2).sum()) / (sxx ** 2)
        se = np.sqrt(var_slope)
        # Express slope per decade for readability
        slope_per_decade = slope * 120  # 120 months/decade
        se_per_decade = se * 120
        ci_lo = slope_per_decade - 1.96 * se_per_decade
        ci_hi = slope_per_decade + 1.96 * se_per_decade
        summary.append({"display": display,
                        "color": sub["color"].iloc[0],
                        "slope": slope_per_decade,
                        "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n})

    summary = sorted(summary, key=lambda d: -d["slope"])
    y_pos = np.arange(len(summary))
    for y_p, s in zip(y_pos, summary):
        ax1.errorbar(s["slope"], y_p,
                     xerr=[[s["slope"] - s["ci_lo"]], [s["ci_hi"] - s["slope"]]],
                     fmt="o", color=s["color"], markersize=7,
                     elinewidth=1.4, capsize=3.5)
    ax1.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{s['display']} (n={s['n']})" for s in summary],
                        fontsize=8.5)
    ax1.set_xlabel("OLS slope: $\\Delta$within-25bps recall per decade pre-cutoff",
                   fontsize=9)
    ax1.set_title("Per-model cutoff slope (95% HC0 CI)", fontsize=10)
    ax1.grid(True, axis="x", alpha=0.3)

    # ---------- Panel B: |error| vs distance, smoothed -----------------
    ax2 = axes[1]
    ax2.axvspan(-6, 6, color="grey", alpha=0.08, label="±6 months of cutoff")
    ax2.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)

    # Sort summary by abs slope so top of legend is most-monotone (debug aid)
    for s in summary:
        sub = big[big["display"] == s["display"]].sort_values("months_to_cutoff")
        if len(sub) < 5: continue
        # Rolling mean over a 60-month window (5-year smoother)
        sub_r = sub.copy()
        sub_r = sub_r.sort_values("months_to_cutoff").reset_index(drop=True)
        # Use bin-mean instead of rolling for irregular x-spacing
        bin_w = 60
        sub_r["bin"] = (sub_r["months_to_cutoff"] // bin_w).astype(int)
        agg = sub_r.groupby("bin")["abs_err"].agg(["mean", "count"]).reset_index()
        agg = agg[agg["count"] >= 3]
        if agg.empty: continue
        agg["x"] = agg["bin"] * bin_w + bin_w / 2
        ax2.plot(agg["x"], agg["mean"], "-", color=s["color"], alpha=0.85,
                 linewidth=1.5, label=s["display"])
        # Faint scatter underneath
        ax2.scatter(sub["months_to_cutoff"], sub["abs_err"], color=s["color"],
                    s=6, alpha=0.15)

    ax2.set_xlabel("months to model's training cutoff  (positive = in training)",
                   fontsize=9)
    ax2.set_ylabel("|model estimate $-$ truth| (Mkt-RF, %)", fontsize=9)
    ax2.set_title("Per-month error vs cutoff distance (5y rolling mean)",
                  fontsize=10)
    ax2.set_ylim(0, 9.0)
    ax2.set_xlim(-30, 760)
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")

    fig.tight_layout()
    out_pdf = FIG / "fig2_temporal_gradient.pdf"
    out_png = FIG / "fig2_temporal_gradient.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print()
    print("Per-model slope summary:")
    for s in summary:
        print(f"  {s['display']:18s}: slope={s['slope']:+.3f}/decade "
              f"CI=[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}], n={s['n']}")


if __name__ == "__main__":
    main()
