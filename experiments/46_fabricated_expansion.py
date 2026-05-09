"""Expanded fabricated-series control: 5 fictional factor names ×
4 frontier models × 12 months = 240 queries.

Reviewer concern on the original n=24: underpowered. The point of
the test is "if the model recalls Mkt-RF because it has memorized
the Fama-French series specifically, fabricated factor names with
identical prompt structure should produce no parseable answers."
We expand to n=240 to give the test real statistical weight.

Fabricated names are constructed to (a) sound plausible and
(b) match no entity we can find in public corpora.

Output: results/fabricated_expanded.jsonl + console summary.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

from factor_leak.cost import cost_from_usage  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "fabricated_expanded.jsonl"

FABRICATED_NAMES = [
    "Gleason-Zeta volatility-conditioned residual factor",
    "Holbrooke-Mansfield Opportunity Fund III (2007 vintage)",
    "Brennan-Iyer mean-reversion premium factor",
    "Northrop-Calloway long-horizon dispersion factor",
    "Pemberton-Yi cross-sectional liquidity premium factor",
]


def fabricated_prompt(name: str, month: str) -> str:
    return (
        f"What was the monthly return of the {name} in "
        f"{format_month_human(month)}? "
        f"Answer with a signed decimal percentage "
        f"(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        f"and nothing else."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["claude-opus-4.7", "claude-sonnet-4.6",
                             "claude-haiku-4.5", "gpt-5.4"])
    ap.add_argument("--n-months", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--budget-usd", type=float, default=2.50)
    ap.add_argument("--rpm", type=float, default=60.0)
    args = ap.parse_args()

    eps = {e.name: e for e in default_endpoints() if e.name in args.models}
    missing = set(args.models) - eps.keys()
    if missing: sys.exit(f"no endpoint for {sorted(missing)}")

    rng = random.Random(args.seed)
    pool = [str(p) for p in pd.period_range("1985-01", "2024-12", freq="M")]
    months = sorted(rng.sample(pool, args.n_months))
    print(f"sampling {len(months)} months: {months}")
    print(f"{len(FABRICATED_NAMES)} names × {len(args.models)} models × "
          f"{len(months)} months = "
          f"{len(FABRICATED_NAMES)*len(args.models)*len(months)} queries")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    spend = 0.0
    records: list[dict] = []
    aborted = False
    min_interval = 60.0 / max(args.rpm, 1.0)

    with OUT.open("a") as f:
        for model in args.models:
            if aborted: break
            ep = eps[model]
            for name in FABRICATED_NAMES:
                if aborted: break
                for m in months:
                    prompt = fabricated_prompt(name, m)
                    t0 = time.perf_counter()
                    try:
                        resp = ep(prompt, max_tokens=64)
                        text, err = resp.text, None
                        in_tok, out_tok = resp.input_tokens, resp.output_tokens
                    except Exception as e:  # noqa: BLE001
                        text, err = None, f"{type(e).__name__}: {e}"
                        in_tok = out_tok = 0
                    usd = cost_from_usage(model, in_tok, out_tok)
                    spend += usd
                    rec = {
                        "model_name": model, "name": name, "month": m,
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
    df["parsed"] = df["response"].map(
        lambda r: parse_numeric(r) if r else None)
    df["committal"] = df["parsed"].notna()

    print("\n=== fabricated-series summary ===")
    overall_n = len(df)
    overall_p = df["committal"].sum()
    print(f"  overall: n={overall_n}, parsed={overall_p}, "
          f"parse_rate={overall_p/overall_n:.3f}")
    for model, grp in df.groupby("model_name"):
        n = len(grp)
        p = grp["committal"].sum()
        print(f"    {model:22s} n={n} parsed={p} parse_rate={p/n:.3f}")
    print()
    for name, grp in df.groupby("name"):
        n = len(grp)
        p = grp["committal"].sum()
        print(f"    {name[:60]:60s} n={n} parsed={p} parse_rate={p/n:.3f}")


if __name__ == "__main__":
    main()
