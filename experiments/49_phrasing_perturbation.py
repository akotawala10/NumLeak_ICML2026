"""Phrasing-perturbation control: does Mkt-RF recall hold under
different surface forms of the same content question?

Reviewer concern: r=0.98 on Sonnet/Opus could be specific to the
exact Variant-A template ("What was the monthly return of the
Fama-French [factor] factor in [month]? Answer with a signed
decimal percentage..."). If reworded the model might not retrieve
the same value.

Test: same months (n=30, fresh sample) on Sonnet and Opus under
three reworded prompts that hold content fixed but vary surface
form. Compute r and within-25bps for each phrasing variant and
compare to the main-sweep Variant-A baseline (Sonnet r=0.98, Opus
r=0.99).

Output: results/phrasing_perturbation.jsonl + console summary.
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
from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "phrasing_perturbation.jsonl"
DATA = REPO / "data" / "ff"

# Three reworded Variant-A surfaces. All ask for the same content
# (Mkt-RF monthly return) and constrain output to a signed decimal
# percentage so the existing parser handles them.

PHRASINGS = {
    "terse": (
        "Mkt-RF (Fama-French market excess) monthly return in "
        "{month_human}? Reply with a signed decimal percentage "
        "(e.g., -3.12 or +1.50) and nothing else."
    ),
    "imperative": (
        "Tell me the Fama-French Mkt-RF return for {month_human}. "
        "Answer with a signed decimal percentage (e.g., -3.12 or "
        "+1.50) and nothing else."
    ),
    "conversational": (
        "Do you remember what the Fama-French Mkt-RF factor returned "
        "in {month_human}? Give your best estimate as a signed "
        "decimal percentage (e.g., -3.12 or +1.50), nothing else."
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["claude-sonnet-4.6", "claude-opus-4.7"])
    ap.add_argument("--n-months", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--start", default="1980-01")
    ap.add_argument("--end", default="2020-12")
    ap.add_argument("--budget-usd", type=float, default=1.50)
    ap.add_argument("--rpm", type=float, default=60.0)
    args = ap.parse_args()

    eps = {e.name: e for e in default_endpoints() if e.name in args.models}
    missing = set(args.models) - eps.keys()
    if missing: sys.exit(f"no endpoint for {sorted(missing)}")

    truth = load_all_factors(DATA)["Mkt-RF"]
    pool = [str(p) for p in truth.index
            if args.start <= str(p) <= args.end]
    rng = random.Random(args.seed)
    months = sorted(rng.sample(pool, args.n_months))
    print(f"sampled {len(months)} months from "
          f"{args.start}..{args.end}")
    print(f"{len(PHRASINGS)} phrasings × {len(args.models)} models × "
          f"{len(months)} months = "
          f"{len(PHRASINGS)*len(args.models)*len(months)} queries")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    spend = 0.0
    records: list[dict] = []
    aborted = False
    min_interval = 60.0 / max(args.rpm, 1.0)

    with OUT.open("a") as f:
        for model in args.models:
            if aborted: break
            ep = eps[model]
            for phrasing, template in PHRASINGS.items():
                if aborted: break
                for m in months:
                    truth_val = float(truth.get(pd.Period(m, freq="M"),
                                                float("nan")))
                    if not (truth_val == truth_val):
                        continue
                    prompt = template.format(
                        month_human=format_month_human(m))
                    t0 = time.perf_counter()
                    try:
                        resp = ep(prompt, max_tokens=48)
                        text, err = resp.text, None
                        in_tok, out_tok = resp.input_tokens, resp.output_tokens
                    except Exception as e:  # noqa: BLE001
                        text, err = None, f"{type(e).__name__}: {e}"
                        in_tok = out_tok = 0
                    usd = cost_from_usage(model, in_tok, out_tok)
                    spend += usd
                    rec = {
                        "model_name": model, "phrasing": phrasing,
                        "month": m, "truth": truth_val,
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
    df["abs_err"] = (df["parsed"] - df["truth"]).abs()

    print("\n=== phrasing-perturbation summary ===")
    for (model, phrasing), grp in df.groupby(["model_name", "phrasing"]):
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
        print(f"  {model:22s} {phrasing:14s} n={n} parsed={np_} "
              f"parse={pr:.2f} r={r:+.3f} w25={w25:.2f} w50={w50:.2f}")

    print("\nMain-sweep baseline: Sonnet r=0.98 (n=77), Opus r=0.99 (n=40)")


if __name__ == "__main__":
    main()
