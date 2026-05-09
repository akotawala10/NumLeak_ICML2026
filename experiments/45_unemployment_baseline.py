"""Cross-domain replication: U.S. monthly unemployment rate (FRED
UNRATE). Reviewer concern was 'Fama-French only' -- this is the
audit framework applied to a different domain (macro / labor),
different sign convention (always positive level), and different
canonical source (BLS via FRED, not Ken French's data library).

We use a Variant-A-style prompt asking for a single-decimal level
in percent. Truth: FRED UNRATE monthly (already in
data/macro/UNRATE.csv).

If recall on UNRATE is high on Sonnet/Opus, the audit framework is
domain-portable and our Mkt-RF result generalizes; if recall is
low, this is itself a finding (Mkt-RF is a privileged series).

Output: results/unrate_baseline.jsonl + console summary.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

from factor_leak.cost import cost_from_usage  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "unrate_baseline.jsonl"
TRUTH_CSV = REPO / "data" / "macro" / "UNRATE.csv"


def unrate_prompt(month: str) -> str:
    return (
        f"What was the U.S. civilian unemployment rate "
        f"(seasonally adjusted, BLS series UNRATE) in "
        f"{format_month_human(month)}? Answer with a single decimal "
        f"percent (e.g., 4.3 for 4.3%) and nothing else."
    )


def load_truth() -> pd.Series:
    df = pd.read_csv(TRUTH_CSV)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.dropna(subset=["UNRATE"])
    s = pd.Series(df["UNRATE"].values,
                  index=pd.PeriodIndex(df["observation_date"], freq="M"))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-months", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+",
                    default=["claude-sonnet-4.6", "claude-opus-4.7"])
    ap.add_argument("--start", default="1980-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--budget-usd", type=float, default=1.00)
    ap.add_argument("--rpm", type=float, default=60.0)
    args = ap.parse_args()

    truth = load_truth()
    pool = [str(p) for p in truth.index
            if args.start <= str(p) <= args.end]
    rng = random.Random(args.seed)
    months = sorted(rng.sample(pool, args.n_months))
    print(f"UNRATE probe: {len(months)} months × {len(args.models)} models "
          f"= {len(months)*len(args.models)} queries")

    eps = {e.name: e for e in default_endpoints() if e.name in args.models}
    missing = set(args.models) - eps.keys()
    if missing: sys.exit(f"no endpoint for {sorted(missing)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    spend = 0.0
    records: list[dict] = []
    aborted = False
    min_interval = 60.0 / max(args.rpm, 1.0)

    with OUT.open("a") as f:
        for model in args.models:
            if aborted: break
            ep = eps[model]
            for m in months:
                tt = float(truth.get(pd.Period(m, freq="M"), float("nan")))
                if not (tt == tt): continue
                prompt = unrate_prompt(m)
                t0 = time.perf_counter()
                try:
                    resp = ep(prompt, max_tokens=24)
                    text, err = resp.text, None
                    in_tok, out_tok = resp.input_tokens, resp.output_tokens
                except Exception as e:  # noqa: BLE001
                    text, err = None, f"{type(e).__name__}: {e}"
                    in_tok = out_tok = 0
                usd = cost_from_usage(model, in_tok, out_tok)
                spend += usd
                rec = {
                    "model_name": model, "month": m, "truth": tt,
                    "prompt": prompt, "response": text, "error": err,
                    "input_tokens": in_tok, "output_tokens": out_tok,
                    "latency_s": time.perf_counter() - t0, "usd": usd,
                    "ts": time.time(),
                }
                f.write(json.dumps(rec) + "\n")
                records.append(rec)
                if spend > args.budget_usd:
                    print(f"  ABORT: ${spend:.4f} > ${args.budget_usd:.2f}")
                    aborted = True
                    break
                elapsed = time.perf_counter() - t0
                if min_interval - elapsed > 0:
                    time.sleep(min_interval - elapsed)

    print(f"\nspent ${spend:.4f} over {len(records)} queries")

    df = pd.DataFrame(records)
    if df.empty: return
    df["parsed"] = df["response"].map(lambda r: parse_numeric(r) if r else None)
    df["abs_err"] = (df["parsed"] - df["truth"]).abs()
    print("\n=== UNRATE baseline summary ===")
    for model, grp in df.groupby("model_name"):
        n = len(grp)
        p = grp.dropna(subset=["parsed", "truth"])
        np_ = len(p)
        pr = np_ / n if n else float("nan")
        if np_ >= 3:
            r = float(np.corrcoef(p["parsed"], p["truth"])[0, 1])
        else:
            r = float("nan")
        w25 = (p["abs_err"] <= 0.25).mean() if np_ else float("nan")
        w50 = (p["abs_err"] <= 0.50).mean() if np_ else float("nan")
        print(f"  {model:22s} n={n} parsed={np_} parse={pr:.2f} "
              f"r={r:+.3f} w25={w25:.3f} w50={w50:.3f}")


if __name__ == "__main__":
    main()
