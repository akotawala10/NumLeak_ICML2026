"""Dump every number the appendix needs, as a single stdout report.

Reads:
- experiments/results/sweep.jsonl     -> refusal rates, per-cell cutoff grad, famous lift
- experiments/results/variant_d.jsonl -> CoT numbers
- experiments/results/variant_e_temperature.jsonl -> T=1 stability
- experiments/results/multiseed.jsonl -> per-seed robustness
- experiments/results/controls.jsonl  -> per-(model, kind) breakdown

No side effects. Intended to be run once; output is pasted into the appendix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak import metrics  # noqa: E402
from factor_leak.constants import FAMOUS_MONTHS  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402

RES = REPO / "experiments" / "results"
DATA = REPO / "data" / "ff"


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
    return df


def section(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    df = load_sweep()
    enriched = metrics.enrich(df, DATA)

    # -----------------------------------------------------------------------
    section("A. Refusal / parse rate per (model, factor, variant)")
    # Parse rate = fraction of A/B/C queries that produced a non-null parsed
    # estimate (or a valid C-pick). Anything else counts as a refusal / non-
    # commit.
    for v in ["A", "B", "C"]:
        sub = enriched[enriched["variant"] == v]
        if v == "C":
            ans = sub["parsed_month"].notna()
        else:
            ans = sub["parsed_estimate"].notna()
        sub = sub.assign(answered=ans)
        tab = (sub.groupby(["model_name", "factor"])
                  .agg(n=("answered", "size"),
                       parse_rate=("answered", "mean"))
                  .reset_index())
        print(f"\n-- Variant {v} --")
        with pd.option_context("display.width", 140,
                               "display.float_format", "{:.3f}".format):
            print(tab.to_string(index=False))

    # -----------------------------------------------------------------------
    section("B. Per-cell cutoff-gradient OLS with BH-FDR")
    grad = metrics.temporal_gradient(
        enriched, tol_bps=25,
        by=["model_name", "factor"], x="months_to_cutoff",
    )
    qvals, rej = metrics.benjamini_hochberg(grad["pvalue"].to_numpy(), alpha=0.05)
    grad["q_fdr"] = qvals
    grad["fdr_reject"] = rej
    with pd.option_context("display.width", 160,
                           "display.max_columns", None,
                           "display.float_format", "{:.4f}".format):
        print(grad.to_string(index=False))

    # -----------------------------------------------------------------------
    section("C. Famous-month lift per cell")
    a = enriched[enriched["variant"] == "A"]
    fm = metrics.famous_concentration(a, FAMOUS_MONTHS, tol_bps=25,
                                      by=["model_name", "factor"])
    with pd.option_context("display.width", 140,
                           "display.float_format", "{:.3f}".format):
        print(fm.to_string(index=False))

    # -----------------------------------------------------------------------
    section("D. Variant D (CoT) — Mkt-RF only")
    vd_path = RES / "variant_d.jsonl"
    if vd_path.exists():
        vd = [json.loads(l) for l in vd_path.open() if l.strip()]
        df_d = pd.DataFrame([r for r in vd if r.get("error") is None])
        df_d["parsed"] = df_d["response"].map(lambda r: parse_numeric(r) if r else None)

        from factor_leak.ff_loader import load_all_factors
        truth_df = load_all_factors(DATA)
        # Ensure month column alignment
        df_d["month_p"] = pd.PeriodIndex(df_d["month"], freq="M")
        truth_vals = truth_df["Mkt-RF"].reindex(df_d["month_p"]).values
        df_d["truth"] = truth_vals
        df_d["abs_err"] = (df_d["parsed"] - df_d["truth"]).abs()
        df_d["within_25bps"] = df_d["abs_err"] <= 0.25

        for m, frame in df_d.groupby("model_name"):
            n_total = len(frame)
            n_parsed = int(frame["parsed"].notna().sum())
            parse_rate = n_parsed / n_total if n_total else float("nan")
            sub = frame.dropna(subset=["parsed", "truth"])
            w25 = sub["within_25bps"].mean() if len(sub) else float("nan")
            if len(sub) >= 3:
                r = float(np.corrcoef(sub["parsed"], sub["truth"])[0, 1])
            else:
                r = float("nan")
            print(f"  {m:22s}  n={n_total:3d}  parse_rate={parse_rate:.3f}  "
                  f"within_25bps={w25:.3f}  r={r:+.3f}")

    # -----------------------------------------------------------------------
    section("E. Variant E (T=1) — Mkt-RF only")
    ve_path = RES / "variant_e_temperature.jsonl"
    if ve_path.exists():
        ve = [json.loads(l) for l in ve_path.open() if l.strip()]
        df_e = pd.DataFrame([r for r in ve if r.get("error") is None])
        df_e["parsed"] = df_e["response"].map(lambda r: parse_numeric(r) if r else None)

        from factor_leak.ff_loader import load_all_factors
        truth_df = load_all_factors(DATA)
        df_e["month_p"] = pd.PeriodIndex(df_e["month"], freq="M")
        df_e["truth"] = truth_df["Mkt-RF"].reindex(df_e["month_p"]).values
        df_e["abs_err"] = (df_e["parsed"] - df_e["truth"]).abs()
        df_e["within_25bps"] = df_e["abs_err"] <= 0.25

        for m, frame in df_e.groupby("model_name"):
            n_total = len(frame)
            sub = frame.dropna(subset=["parsed", "truth"])
            w25 = sub["within_25bps"].mean() if len(sub) else float("nan")
            if len(sub) >= 3:
                r = float(np.corrcoef(sub["parsed"], sub["truth"])[0, 1])
            else:
                r = float("nan")
            # within-draw spread: pair draw 0 vs draw 1 for same month
            wide = (frame.dropna(subset=["parsed"])
                         .pivot_table(index="month", columns="draw",
                                      values="parsed", aggfunc="first"))
            if {0, 1}.issubset(wide.columns):
                pairs = wide.dropna(subset=[0, 1])
                spread = (pairs[0] - pairs[1]).abs()
                agree25 = float((spread <= 0.25).mean())
                mean_spread = float(spread.mean())
                n_pairs = len(pairs)
            else:
                agree25 = float("nan")
                mean_spread = float("nan")
                n_pairs = 0
            print(f"  {m:22s}  n_draws={n_total:3d}  within_25bps={w25:.3f}  "
                  f"r={r:+.3f}  agree_pairs={n_pairs:3d}  "
                  f"P(|spread|<=25bps)={agree25:.3f}  mean|spread|={mean_spread:.3f}%")

    # -----------------------------------------------------------------------
    section("F. Multi-seed robustness (Mkt-RF, seeds 1/2/3)")
    ms_path = RES / "multiseed.jsonl"
    if ms_path.exists():
        ms = [json.loads(l) for l in ms_path.open() if l.strip()]
        df_m = pd.DataFrame([r for r in ms if r.get("error") is None])
        df_m["parsed"] = df_m["response"].map(lambda r: parse_numeric(r) if r else None)

        from factor_leak.ff_loader import load_all_factors
        truth_df = load_all_factors(DATA)
        df_m["month_p"] = pd.PeriodIndex(df_m["month"], freq="M")
        df_m["truth"] = truth_df["Mkt-RF"].reindex(df_m["month_p"]).values
        df_m["abs_err"] = (df_m["parsed"] - df_m["truth"]).abs()
        df_m["within_25bps"] = df_m["abs_err"] <= 0.25

        out = []
        for (model, seed), frame in df_m.groupby(["model_name", "seed"]):
            sub = frame.dropna(subset=["parsed", "truth"])
            n = len(sub)
            if n < 3:
                r = float("nan")
            else:
                r = float(np.corrcoef(sub["parsed"], sub["truth"])[0, 1])
            w25 = sub["within_25bps"].mean() if n else float("nan")
            sign = ((np.sign(sub["parsed"]) == np.sign(sub["truth"]))
                    .mean() if n else float("nan"))
            out.append({"model": model, "seed": seed, "n": n,
                        "pearson_r": r, "within_25bps": w25, "sign": sign})
        for row in sorted(out, key=lambda r: (r["model"], r["seed"])):
            print(f"  {row['model']:22s}  seed={row['seed']}  n={row['n']:3d}  "
                  f"r={row['pearson_r']:+.3f}  within_25bps={row['within_25bps']:.3f}  "
                  f"sign={row['sign']:.3f}")

    # -----------------------------------------------------------------------
    section("G. Controls C2 / C3 per (model, kind)")
    cpath = RES / "controls.jsonl"
    if cpath.exists():
        rows = [json.loads(l) for l in cpath.open() if l.strip()]
        dfc = pd.DataFrame(rows)
        dfc["parsed"] = dfc["response"].map(lambda r: parse_numeric(r) if r else None)
        out = (dfc.groupby(["model_name", "kind"])
                 .agg(n=("parsed", "size"),
                      parse_rate=("parsed", lambda s: s.notna().mean()))
                 .reset_index())
        with pd.option_context("display.width", 140,
                               "display.float_format", "{:.3f}".format):
            print(out.to_string(index=False))


if __name__ == "__main__":
    main()
