"""Cross-model figures reflecting Opus + GPT-5.4 data.

Produces three files in figures/:
- fig3_cross_model_mktrf.pdf : 2x2 Mkt-RF calibration scatter, one panel
                                per model (Opus, Sonnet, Haiku, GPT-5.4).
                                Replaces the single-panel Sonnet-only Fig 3.
- figA6_cross_model_probes.pdf : 4x4 scatter grid across the four baseline
                                 probes (rows) and four models (cols).
                                 Visualizes the appendix Table H.
- figA2_mktrf_timeseries.pdf   : (overwrites existing) Mkt-RF truth line with
                                 both Sonnet and Opus estimates overlaid.
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
import yfinance as yf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.metrics import enrich  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402

FIG = REPO / "figures"
DATA = REPO / "data" / "ff"
RES = REPO / "experiments" / "results"

# Display order & colors (top → bottom: strongest → weakest recall on Mkt-RF)
MODEL_DISPLAY = [
    ("Opus 4.7",   "claude-opus-4.7",   "#2ca02c"),
    ("Sonnet 4.6", "claude-sonnet-4.6", "#1f77b4"),
    ("Haiku 4.5",  "claude-haiku-4.5",  "#ff7f0e"),
    ("GPT-5.4",    "gpt-5.4",           "#d62728"),
]
PROBE_DISPLAY = [
    ("Mkt-RF",   "mktrf"),
    ("S&P 500",  "snp500"),
    ("NASDAQ",   "nasdaq"),
    ("Blind",    "blind"),
]


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


def load_mktrf_by_model() -> dict[str, pd.DataFrame]:
    """Return {model_name: DataFrame(month, truth, estimate)} for Mkt-RF.

    Sonnet + Haiku come from the main sweep (sweep.jsonl Variant A Mkt-RF).
    Opus + GPT-5.4 come from their respective baseline JSONLs.
    """
    out: dict[str, pd.DataFrame] = {}

    # Sonnet + Haiku from main sweep
    raw = [json.loads(l) for l in (RES / "sweep.jsonl").open() if l.strip()]
    raw = _dedup(raw)
    df_main = pd.DataFrame([
        {"model_name": r["query"]["model_name"], "factor": r["query"]["factor"],
         "variant": r["query"]["variant"], "month": r["query"]["month"],
         "response": r["response"], "error": r["error"], "key": r["key"]}
        for r in raw
    ])
    df_main = df_main[(df_main["error"].isna())
                      & (df_main["factor"] == "Mkt-RF")
                      & (df_main["variant"] == "A")].copy()
    df_main["estimate"] = df_main["response"].map(lambda r: parse_numeric(r) if r else None)
    truth = load_all_factors(DATA)["Mkt-RF"]
    df_main["month_p"] = pd.PeriodIndex(df_main["month"], freq="M")
    df_main["truth"] = truth.reindex(df_main["month_p"]).values
    for m in ["claude-sonnet-4.6", "claude-haiku-4.5"]:
        sub = df_main[df_main["model_name"] == m].dropna(subset=["estimate", "truth"])
        out[m] = sub[["month", "truth", "estimate"]].copy()

    # Opus from opus_baselines.jsonl (probe=="mktrf")
    opus_rows = [json.loads(l) for l in (RES / "opus_baselines.jsonl").open() if l.strip()]
    opus_df = pd.DataFrame([r for r in opus_rows if r.get("error") is None
                            and r["probe"] == "mktrf"])
    opus_df["estimate"] = opus_df["response"].map(lambda r: parse_numeric(r) if r else None)
    out["claude-opus-4.7"] = (opus_df.dropna(subset=["estimate", "truth"])
                              [["month", "truth", "estimate"]].copy())

    # GPT-5.4 from gpt54_baselines.jsonl (probe=="mktrf")
    gpt_rows = [json.loads(l) for l in (RES / "gpt54_baselines.jsonl").open() if l.strip()]
    gpt_df = pd.DataFrame([r for r in gpt_rows if r.get("error") is None
                           and r["probe"] == "mktrf"])
    gpt_df["estimate"] = gpt_df["response"].map(lambda r: parse_numeric(r) if r else None)
    out["gpt-5.4"] = (gpt_df.dropna(subset=["estimate", "truth"])
                      [["month", "truth", "estimate"]].copy())
    return out


def load_factor_cells(factors: list[str]) -> dict[tuple[str, str], pd.DataFrame]:
    """Return {(model, factor): df(month, truth, estimate)} for all 4 models.

    Sonnet/Haiku come from the main sweep (Variant A). Opus/GPT-5.4 come
    from their respective factor-fill JSONLs (SMB/HML/RMW/CMA/Mom) plus
    their baseline JSONLs (Mkt-RF).
    """
    out: dict[tuple[str, str], pd.DataFrame] = {}
    truth_all = load_all_factors(DATA)

    # Sonnet + Haiku from main sweep
    raw = [json.loads(l) for l in (RES / "sweep.jsonl").open() if l.strip()]
    raw = _dedup(raw)
    df_main = pd.DataFrame([
        {"model_name": r["query"]["model_name"], "factor": r["query"]["factor"],
         "variant": r["query"]["variant"], "month": r["query"]["month"],
         "response": r["response"], "error": r["error"], "key": r["key"]}
        for r in raw
    ])
    df_main = df_main[df_main["error"].isna() & (df_main["variant"] == "A")].copy()
    df_main["estimate"] = df_main["response"].map(lambda r: parse_numeric(r) if r else None)
    for (m, f), sub in df_main.groupby(["model_name", "factor"]):
        if f not in factors: continue
        s = sub.dropna(subset=["estimate"]).copy()
        s["truth"] = [float(truth_all[f].get(pd.Period(x, freq="M"), float("nan")))
                      for x in s["month"]]
        s = s.dropna(subset=["truth"])
        out[(m, f)] = s[["month", "truth", "estimate"]].copy()

    # Opus + GPT-5.4 Mkt-RF from baselines
    for model, path in [
        ("claude-opus-4.7", RES / "opus_baselines.jsonl"),
        ("gpt-5.4",         RES / "gpt54_baselines.jsonl"),
    ]:
        rows = [json.loads(l) for l in path.open() if l.strip()]
        df = pd.DataFrame([r for r in rows if r.get("error") is None
                           and r["probe"] == "mktrf"])
        df["estimate"] = df["response"].map(parse_numeric)
        df = df.dropna(subset=["estimate"]).copy()
        df["truth"] = [float(truth_all["Mkt-RF"].get(pd.Period(m, freq="M"), float("nan")))
                       for m in df["month"]]
        df = df.dropna(subset=["truth"])
        out[(model, "Mkt-RF")] = df[["month", "truth", "estimate"]].copy()

    # Opus + GPT-5.4 other factors from factor-fill
    for model, path in [
        ("claude-opus-4.7", RES / "opus_factors.jsonl"),
        ("gpt-5.4",         RES / "gpt54_factors.jsonl"),
    ]:
        rows = [json.loads(l) for l in path.open() if l.strip()]
        df = pd.DataFrame([r for r in rows if r.get("error") is None])
        df["estimate"] = df["response"].map(parse_numeric)
        df = df.dropna(subset=["estimate"]).copy()
        for factor, grp in df.groupby("factor"):
            if factor not in factors: continue
            g = grp.copy()
            g["truth"] = [float(truth_all[factor].get(pd.Period(m, freq="M"), float("nan")))
                          for m in g["month"]]
            g = g.dropna(subset=["truth"])
            out[(model, factor)] = g[["month", "truth", "estimate"]].copy()

    return out


def fig3_factor_grid(cells: dict[tuple[str, str], pd.DataFrame],
                     factors: list[str]) -> plt.Figure:
    """N x 4 grid: rows = factors, cols = 4 models. Used for the
    SMB/HML factor-specificity panel — Mkt-RF row is intentionally
    omitted to avoid duplicating fig:calibration / fig:cross-probes."""
    n_rows = len(factors)
    n_cols = len(MODEL_DISPLAY)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.7 * n_cols, 2.7 * n_rows + 0.4),
                             sharex=False, sharey=False)
    for i, factor in enumerate(factors):
        for j, (model_label, model_key, color) in enumerate(MODEL_DISPLAY):
            ax = axes[i, j]
            df = cells.get((model_key, factor))
            if df is None or len(df) < 3:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                r = float(np.corrcoef(df["truth"], df["estimate"])[0, 1])
                w25 = float(((df["estimate"] - df["truth"]).abs() <= 0.25).mean())
                lo = min(df["truth"].min(), df["estimate"].min()) - 1
                hi = max(df["truth"].max(), df["estimate"].max()) + 1
                lo, hi = min(lo, -15), max(hi, 15)
                ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.5, alpha=0.5)
                ax.scatter(df["truth"], df["estimate"], s=14, color=color,
                           edgecolor="white", linewidth=0.3, alpha=0.85)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
                ax.text(0.03, 0.97,
                        f"$r$={r:+.2f}\nw25={w25:.0%}\n$n$={len(df)}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=7.5,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.75, pad=1.3))
                ax.grid(True, alpha=0.22, linewidth=0.4)
                ax.tick_params(axis="both", labelsize=7)
            if i == 0:
                ax.set_title(model_label, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{factor}\nestimate (%)", fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("truth (%)", fontsize=9)
    fig.tight_layout()
    return fig


def fig3_cross_model_mktrf(by_model: dict[str, pd.DataFrame]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.0))
    for (label, key, color), ax in zip(MODEL_DISPLAY, axes.flat):
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
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.5,
                label="perfect recall")
        ax.scatter(df["truth"], df["estimate"], s=26, color=color,
                   edgecolor="white", linewidth=0.4, alpha=0.9)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_title(f"{label} — $r$={r:+.3f}, w25={w25:.0%}, $n$={n}",
                     fontsize=10)
        ax.set_xlabel("Kenneth French truth (%)", fontsize=9)
        ax.set_ylabel("model estimate (%)", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=8)
    fig.suptitle("Mkt-RF calibration across four frontier models (Variant A)",
                 fontsize=11, y=1.00)
    fig.tight_layout()
    return fig


def load_all_probes() -> pd.DataFrame:
    """Unified dataframe of (model, probe, month, truth, estimate) across
    the four baseline probes for the two Anthropic baseline runs (Opus, GPT)
    and the 13_snp500_baseline / 14_blind_query / 15_nasdaq for Sonnet+Haiku."""
    rows: list[dict] = []

    # Opus: all four probes in one file
    for r in (json.loads(l) for l in (RES / "opus_baselines.jsonl").open() if l.strip()):
        if r.get("error"): continue
        rows.append({
            "model": "claude-opus-4.7", "probe": r["probe"],
            "month": r["month"], "truth": r["truth"],
            "estimate": parse_numeric(r["response"]) if r.get("response") else None,
        })

    # GPT-5.4: all four probes in one file
    for r in (json.loads(l) for l in (RES / "gpt54_baselines.jsonl").open() if l.strip()):
        if r.get("error"): continue
        rows.append({
            "model": "gpt-5.4", "probe": r["probe"],
            "month": r["month"], "truth": r["truth"],
            "estimate": parse_numeric(r["response"]) if r.get("response") else None,
        })

    # Sonnet + Haiku S&P 500 from 13_
    truth_sp = (yf.download("^GSPC", start="1963-07-01", end="2026-03-01",
                            interval="1mo", auto_adjust=False, progress=False)
                [("Close", "^GSPC")].dropna().pct_change().dropna() * 100)
    truth_sp.index = pd.PeriodIndex(truth_sp.index, freq="M")
    for r in (json.loads(l) for l in (RES / "snp500_baseline.jsonl").open() if l.strip()):
        if r.get("error"): continue
        tt = float(truth_sp.get(pd.Period(r["month"], freq="M"), float("nan")))
        rows.append({
            "model": r["model_name"], "probe": "snp500",
            "month": r["month"], "truth": tt,
            "estimate": parse_numeric(r["response"]) if r.get("response") else None,
        })

    # Sonnet + Haiku NASDAQ from 15_
    truth_nd = (yf.download("^IXIC", start="1971-02-01", end="2026-03-01",
                            interval="1mo", auto_adjust=False, progress=False)
                [("Close", "^IXIC")].dropna().pct_change().dropna() * 100)
    truth_nd.index = pd.PeriodIndex(truth_nd.index, freq="M")
    for r in (json.loads(l) for l in (RES / "nasdaq_baseline.jsonl").open() if l.strip()):
        if r.get("error"): continue
        tt = float(truth_nd.get(pd.Period(r["month"], freq="M"), float("nan")))
        rows.append({
            "model": r["model_name"], "probe": "nasdaq",
            "month": r["month"], "truth": tt,
            "estimate": parse_numeric(r["response"]) if r.get("response") else None,
        })

    # Sonnet blind from 14_
    truth_mktrf = load_all_factors(DATA)["Mkt-RF"]
    for r in (json.loads(l) for l in (RES / "blind_query.jsonl").open() if l.strip()):
        if r.get("error"): continue
        tt = float(truth_mktrf.get(pd.Period(r["month"], freq="M"), float("nan")))
        rows.append({
            "model": r["model_name"], "probe": "blind",
            "month": r["month"], "truth": tt,
            "estimate": parse_numeric(r["response"]) if r.get("response") else None,
        })

    # Sonnet + Haiku Mkt-RF from main sweep
    raw = [json.loads(l) for l in (RES / "sweep.jsonl").open() if l.strip()]
    raw = _dedup(raw)
    for r in raw:
        q = r["query"]
        if r.get("error") or q["factor"] != "Mkt-RF" or q["variant"] != "A":
            continue
        if q["model_name"] not in ("claude-sonnet-4.6", "claude-haiku-4.5"):
            continue
        tt = float(truth_mktrf.get(pd.Period(q["month"], freq="M"), float("nan")))
        rows.append({
            "model": q["model_name"], "probe": "mktrf",
            "month": q["month"], "truth": tt,
            "estimate": parse_numeric(r["response"]) if r.get("response") else None,
        })

    return pd.DataFrame(rows).dropna(subset=["estimate", "truth"])


def figA6_cross_model_probes(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(4, 4, figsize=(11.5, 11.0), sharex=False, sharey=False)
    for i, (probe_label, probe_key) in enumerate(PROBE_DISPLAY):
        for j, (model_label, model_key, color) in enumerate(MODEL_DISPLAY):
            ax = axes[i, j]
            sub = df[(df["model"] == model_key) & (df["probe"] == probe_key)]
            if len(sub) < 3:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                r = float(np.corrcoef(sub["truth"], sub["estimate"])[0, 1])
                w25 = float(((sub["estimate"] - sub["truth"]).abs() <= 0.25).mean())
                lo = min(sub["truth"].min(), sub["estimate"].min()) - 1
                hi = max(sub["truth"].max(), sub["estimate"].max()) + 1
                lo = min(lo, -25); hi = max(hi, 20)
                ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.5, alpha=0.5)
                ax.scatter(sub["truth"], sub["estimate"], s=14, color=color,
                           edgecolor="white", linewidth=0.3, alpha=0.85)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
                ax.text(0.03, 0.97,
                        f"$r$={r:+.3f}\nw25={w25:.0%}\n$n$={len(sub)}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.75, pad=1.3))
                ax.grid(True, alpha=0.2, linewidth=0.4)
                ax.tick_params(axis="both", labelsize=7)
            if i == 0:
                ax.set_title(model_label, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{probe_label}\nestimate (%)", fontsize=10)
            if i == 3:
                ax.set_xlabel("truth (%)", fontsize=9)
    fig.suptitle("Cross-model recall on four probes for U.S. equity returns",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    return fig


def figA2_mktrf_timeseries(by_model: dict[str, pd.DataFrame]) -> plt.Figure:
    # Probe window: 1963-07 onward, clip to window so the plot isn't cluttered.
    truth_full = load_all_factors(DATA)["Mkt-RF"].dropna()
    win_start = pd.Period("1963-07", freq="M")
    truth = truth_full[truth_full.index >= win_start]
    truth_x = truth.index.to_timestamp()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 5.2),
                                         sharex=True,
                                         gridspec_kw={"height_ratios": [3, 1]})
    ax_top.plot(truth_x, truth.values, color="#1f77b4", linewidth=0.7,
                alpha=0.55, label="Kenneth French Mkt-RF truth")
    # Sonnet
    son = by_model.get("claude-sonnet-4.6")
    if son is not None and len(son):
        t = pd.PeriodIndex(son["month"], freq="M").to_timestamp()
        ax_top.scatter(t, son["estimate"], s=22, color="#1f77b4",
                       edgecolor="white", linewidth=0.35, alpha=0.85,
                       zorder=5, label=f"Sonnet 4.6 (n={len(son)})", marker="o")
    # Opus
    opus = by_model.get("claude-opus-4.7")
    if opus is not None and len(opus):
        t = pd.PeriodIndex(opus["month"], freq="M").to_timestamp()
        ax_top.scatter(t, opus["estimate"], s=28, color="#2ca02c",
                       edgecolor="white", linewidth=0.4, alpha=0.9,
                       zorder=6, label=f"Opus 4.7 (n={len(opus)})", marker="D")
    ax_top.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax_top.set_ylabel("Mkt-RF monthly return (%)")
    ax_top.set_title("Sonnet 4.6 and Opus 4.7 recall the Mkt-RF series across 63 years",
                     fontsize=11)
    ax_top.legend(loc="lower left", fontsize=9, frameon=False)
    ax_top.grid(True, alpha=0.3)

    # bottom panel: |estimate - truth| per month for each model (overlaid stems)
    if son is not None and len(son):
        t = pd.PeriodIndex(son["month"], freq="M").to_timestamp()
        ax_bot.vlines(t, 0, (son["estimate"] - son["truth"]).abs(),
                      color="#1f77b4", linewidth=0.8, alpha=0.6)
    if opus is not None and len(opus):
        t = pd.PeriodIndex(opus["month"], freq="M").to_timestamp()
        ax_bot.vlines(t, 0, (opus["estimate"] - opus["truth"]).abs(),
                      color="#2ca02c", linewidth=1.0, alpha=0.85)
    ax_bot.axhline(0.25, color="grey", linestyle="--", linewidth=0.6,
                   label="25 bps tolerance")
    ax_bot.set_ylabel("|estimate $-$ truth|\n(%)")
    ax_bot.set_xlabel("probe month")
    ax_bot.legend(loc="upper right", fontsize=8, frameon=False)
    ax_bot.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"wrote {FIG}/{name}.pdf")


def main() -> None:
    mktrf_by_model = load_mktrf_by_model()
    for k, df in mktrf_by_model.items():
        print(f"  {k}: n={len(df)}")

    save(fig3_cross_model_mktrf(mktrf_by_model), "fig3_cross_model_mktrf")
    save(figA2_mktrf_timeseries(mktrf_by_model), "figA2_mktrf_timeseries")

    # 2x4 factor grid covering SMB and HML across all 4 models.
    # Mkt-RF row dropped — already shown in fig3_cross_model_mktrf
    # (main body) and figA6_cross_model_probes (probe-substitution).
    factors_for_grid = ["SMB", "HML"]
    cells = load_factor_cells(factors_for_grid)
    save(fig3_factor_grid(cells, factors_for_grid), "fig3_factor_grid")

    print("loading all probes ...")
    all_probes = load_all_probes()
    print(f"  total rows: {len(all_probes)}")
    save(figA6_cross_model_probes(all_probes), "figA6_cross_model_probes")


if __name__ == "__main__":
    main()
