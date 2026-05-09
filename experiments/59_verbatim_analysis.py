"""Analyze verbatim_extraction.jsonl produced by 59_verbatim_extraction.py.

Reports per-(model, template) cell:
  - n_queries
  - refusal_rate (fraction with n_parsed == 0)
  - mean/median rows_emitted
  - mean/median consecutive verbatim run from start_month
  - total verbatim hits / total parseable rows (verbatim hit rate)
  - max consecutive run observed across all queries in the cell
  - count of queries with consecutive run >= 3 (Carlini-style memorized-sequence threshold)

Also reports cross-template aggregates per model and the headline
panel-wide finding.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.ff_loader import load_all_factors  # noqa: E402

JSONL = REPO / "experiments/results/verbatim_extraction.jsonl"
DATA = REPO / "data/ff"


PANEL_ORDER = [
    "claude-opus-4.7",
    "claude-sonnet-4.6",
    "claude-haiku-4.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "deepseek-v3.2-azure",
    "llama-3.3-70b-groq",
    "llama-3.1-8b-groq",
]
TEMPLATE_ORDER = ["A_csv", "B_framed", "C_markdown"]


def main():
    if not JSONL.exists():
        sys.exit(f"missing {JSONL}")

    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["refused"] = df["n_parsed"] == 0
    df["had_short_run"] = df["consecutive_run"] >= 3
    df["had_long_run"] = df["consecutive_run"] >= 6

    print(f"Total queries:    {len(df)}")
    print(f"Total errors:     {df['error'].notna().sum()}")
    print(f"Refusals:         {df['refused'].sum()} ({df['refused'].mean():.1%})")
    print(f"Any verbatim hit: {(df['n_verbatim'] > 0).sum()}")
    print(f"Run ≥3 (Carlini): {df['had_short_run'].sum()}")
    print(f"Run ≥6 (long):    {df['had_long_run'].sum()}")
    print()

    # Per-(model, template) cell
    print("=" * 130)
    print(f"{'Model':24s} {'Template':12s} {'n':>4s} {'refuse':>7s} {'avg_rows':>9s} "
          f"{'avg_run':>8s} {'max_run':>8s} {'n>=3':>5s} {'verbatim_rate':>14s}")
    print("=" * 130)
    for model in PANEL_ORDER:
        for tname in TEMPLATE_ORDER:
            sub = df[(df["model_name"] == model) & (df["template"] == tname)]
            if len(sub) == 0:
                continue
            refuse = float(sub["refused"].mean())
            avg_rows = float(sub["n_parsed"].mean())
            avg_run = float(sub["consecutive_run"].mean())
            max_run = int(sub["consecutive_run"].max())
            n_ge3 = int((sub["consecutive_run"] >= 3).sum())
            v_total = int(sub["n_verbatim"].sum())
            n_total = int(sub["n_parsed"].sum())
            v_rate = (v_total / n_total) if n_total > 0 else 0.0
            print(f"{model:24s} {tname:12s} {len(sub):>4d} {refuse:>7.1%} "
                  f"{avg_rows:>9.1f} {avg_run:>8.2f} {max_run:>8d} {n_ge3:>5d} "
                  f"{v_total:>5d}/{n_total:<5d} ({v_rate:.1%})")
        print("-" * 130)

    print()
    print("Per-model cross-template aggregates")
    print("=" * 110)
    print(f"{'Model':24s} {'n':>4s} {'refuse':>7s} {'avg_run':>8s} {'max_run':>8s} "
          f"{'n>=3':>5s} {'n>=6':>5s} {'verbatim_rate':>14s}")
    print("=" * 110)
    summary_rows = []
    for model in PANEL_ORDER:
        sub = df[df["model_name"] == model]
        if len(sub) == 0:
            continue
        refuse = float(sub["refused"].mean())
        avg_run = float(sub["consecutive_run"].mean())
        max_run = int(sub["consecutive_run"].max())
        n_ge3 = int((sub["consecutive_run"] >= 3).sum())
        n_ge6 = int((sub["consecutive_run"] >= 6).sum())
        v_total = int(sub["n_verbatim"].sum())
        n_total = int(sub["n_parsed"].sum())
        v_rate = (v_total / n_total) if n_total > 0 else 0.0
        print(f"{model:24s} {len(sub):>4d} {refuse:>7.1%} "
              f"{avg_run:>8.2f} {max_run:>8d} {n_ge3:>5d} {n_ge6:>5d} "
              f"{v_total:>5d}/{n_total:<5d} ({v_rate:.1%})")
        summary_rows.append({"model": model, "n_queries": len(sub),
                             "refuse_rate": refuse, "avg_run": avg_run,
                             "max_run": max_run, "n_ge3_runs": n_ge3,
                             "n_ge6_runs": n_ge6,
                             "verbatim_rate": v_rate, "n_verbatim": v_total,
                             "n_total_rows": n_total})

    # Save analysis summary
    out = JSONL.parent / "verbatim_extraction_summary.json"
    out.write_text(json.dumps(summary_rows, indent=2))
    print(f"\nWrote summary to {out.relative_to(REPO)}")

    # Show example "longest run" cases per model for evaluation
    print()
    print("Longest-consecutive-run examples per model")
    print("=" * 110)
    for model in PANEL_ORDER:
        sub = df[df["model_name"] == model]
        if len(sub) == 0:
            continue
        best = sub.loc[sub["consecutive_run"].idxmax()]
        if int(best["consecutive_run"]) == 0:
            continue
        print(f"\n{model} — run={best['consecutive_run']} starting {best['start_month']} "
              f"({best['template']}):")
        print(f"  prompt prefix: ...{best['prompt'][-60:]!r}")
        resp = (best["response"] or "")[:300].replace("\n", " ⏎ ")
        print(f"  response: {resp}")


if __name__ == "__main__":
    main()
