"""Rebuild fig1_recall_heatmap.pdf with Opus and GPT-5.4 rows added.

Those two models were probed only on Mkt-RF (not on SMB/HML/RMW/CMA/Mom),
so their non-Mkt-RF cells are rendered as hatched "not probed" rather than
zero-valued. Opus and GPT-5.4 rows bookend the Sonnet+Haiku rows.
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

FACTOR_ORDER = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


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


def sweep_heatmap() -> dict[tuple[str, str], float]:
    """within-25bps recall per (model, factor) for Sonnet + Haiku on
    Variants A∪B from the main sweep (matching the original plot)."""
    raw = [json.loads(l) for l in (RES / "sweep.jsonl").open() if l.strip()]
    raw = _dedup(raw)
    df = pd.DataFrame([
        {"model_name": r["query"]["model_name"], "factor": r["query"]["factor"],
         "variant": r["query"]["variant"], "month": r["query"]["month"],
         "response": r["response"], "error": r["error"]}
        for r in raw
    ])
    df = df[df["error"].isna() & df["variant"].isin(["A", "B"])].copy()
    df["est"] = df["response"].map(lambda r: parse_numeric(r) if r else None)
    truth = load_all_factors(DATA)
    out: dict[tuple[str, str], float] = {}
    for (m, f), grp in df.groupby(["model_name", "factor"]):
        s = grp.dropna(subset=["est"]).copy()
        if len(s) == 0: continue
        s["truth"] = [float(truth[f].get(pd.Period(x, freq="M"), float("nan")))
                      for x in s["month"]]
        s = s.dropna(subset=["truth"])
        if len(s) == 0: continue
        w25 = float(((s["est"] - s["truth"]).abs() <= 0.25).mean())
        out[(m, f)] = w25
    return out


def opus_gpt_allfactors() -> dict[tuple[str, str], float]:
    """within-25bps per (model, factor) for Opus and GPT-5.4.

    Mkt-RF comes from the cross-probe baseline JSONLs;
    SMB/HML/RMW/CMA/Mom come from the factor-fill JSONLs.
    """
    truth_all = load_all_factors(DATA)
    out: dict[tuple[str, str], float] = {}
    # Mkt-RF from baselines
    for model_name, path in [
        ("claude-opus-4.7", RES / "opus_baselines.jsonl"),
        ("gpt-5.4",         RES / "gpt54_baselines.jsonl"),
    ]:
        rows = [json.loads(l) for l in path.open() if l.strip()]
        df = pd.DataFrame([r for r in rows if r.get("error") is None
                           and r["probe"] == "mktrf"])
        df["est"] = df["response"].map(parse_numeric)
        df = df.dropna(subset=["est"]).copy()
        df["truth"] = [float(truth_all["Mkt-RF"].get(pd.Period(m, freq="M"), float("nan")))
                       for m in df["month"]]
        df = df.dropna(subset=["truth"])
        out[(model_name, "Mkt-RF")] = float(((df["est"] - df["truth"]).abs() <= 0.25).mean())
    # Other 5 factors from factor-fill runs
    for model_name, path in [
        ("claude-opus-4.7", RES / "opus_factors.jsonl"),
        ("gpt-5.4",         RES / "gpt54_factors.jsonl"),
    ]:
        rows = [json.loads(l) for l in path.open() if l.strip()]
        df = pd.DataFrame([r for r in rows if r.get("error") is None])
        df["est"] = df["response"].map(parse_numeric)
        df = df.dropna(subset=["est"]).copy()
        for factor, grp in df.groupby("factor"):
            g = grp.copy()
            g["truth"] = [float(truth_all[factor].get(pd.Period(m, freq="M"), float("nan")))
                          for m in g["month"]]
            g = g.dropna(subset=["truth"])
            if len(g) == 0: continue
            out[(model_name, factor)] = float(((g["est"] - g["truth"]).abs() <= 0.25).mean())
    return out


def main() -> None:
    sweep = sweep_heatmap()
    extras = opus_gpt_allfactors()

    row_models = [
        ("Opus 4.7",   "claude-opus-4.7"),
        ("Sonnet 4.6", "claude-sonnet-4.6"),
        ("Haiku 4.5",  "claude-haiku-4.5"),
        ("GPT-5.4",    "gpt-5.4"),
    ]
    mat = np.full((len(row_models), len(FACTOR_ORDER)), np.nan)
    for i, (_, mk) in enumerate(row_models):
        for j, fk in enumerate(FACTOR_ORDER):
            if mk in ("claude-opus-4.7", "gpt-5.4"):
                mat[i, j] = extras.get((mk, fk), np.nan)
            else:
                mat[i, j] = sweep.get((mk, fk), np.nan)

    fig, ax = plt.subplots(figsize=(1.05 * len(FACTOR_ORDER) + 2.2,
                                    0.7 * len(row_models) + 1.2))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    for i, (_, mk) in enumerate(row_models):
        for j, fk in enumerate(FACTOR_ORDER):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        color="#555555", fontsize=10, zorder=3)
            else:
                color = "white" if v < 0.5 else "black"
                ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                        color=color, fontsize=9, zorder=3)
    ax.set_xticks(range(len(FACTOR_ORDER)))
    ax.set_xticklabels(FACTOR_ORDER)
    ax.set_yticks(range(len(row_models)))
    ax.set_yticklabels([r[0] for r in row_models])
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("within-25 bps recall")
    ax.set_xlabel("factor")
    ax.set_ylabel("model")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_recall_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig1_recall_heatmap.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"wrote {FIG}/fig1_recall_heatmap.pdf (+ .png)")


if __name__ == "__main__":
    main()
