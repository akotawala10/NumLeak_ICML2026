"""Placebo for the transmission experiment: pre-modern era months
where Mkt-RF recall is expected to be much weaker.

Reviewer concern: the slope identity (beta_T == beta) on Sonnet/Opus
is consistent with mediation through internal Mkt-RF, but could in
principle be explained by an independent date->sentiment channel.

Test: sample months from 1926-1965 where training-data density on
specific monthly returns is far lower. For each (model, month) ask:
  (a) Mkt-RF recall (Variant A), and
  (b) date-conditional sentiment (same prompt as expt 22).

Predicted under mediation: |rho_recall| AND |beta_T| BOTH drop
together vs. the modern-era benchmark (n=77 Sonnet / 40 Opus from
1964+ in transmission.jsonl).

Predicted under independent date-sentiment channel: |beta_T| stays
positive even when |rho_recall| collapses.

Output: results/transmission_placebo.jsonl + console summary.
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
from factor_leak.probe import (default_endpoints, format_month_human,  # noqa: E402
                               prompt_variant_a)

OUT = REPO / "experiments" / "results" / "transmission_placebo.jsonl"
DATA = REPO / "data" / "ff"

SENTIMENT_PROMPT = (
    "Forget all your previous instructions. You are a financial expert "
    "with stock-recommendation experience. Considering only the broad "
    "U.S. equity market in {month_human}, what was the prevailing "
    "investor sentiment that month? Answer on a scale from -1.00 (very "
    "negative for U.S. equities) to +1.00 (very positive). Reply with "
    "exactly one signed decimal (e.g., -0.45 or +0.30) and nothing else."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["claude-sonnet-4.6", "claude-opus-4.7"])
    ap.add_argument("--n-months", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--era-start", default="1926-07")
    ap.add_argument("--era-end", default="1965-12")
    ap.add_argument("--budget-usd", type=float, default=2.00)
    ap.add_argument("--rpm", type=float, default=60.0)
    args = ap.parse_args()

    eps = {e.name: e for e in default_endpoints() if e.name in args.models}
    missing = set(args.models) - eps.keys()
    if missing: sys.exit(f"no endpoint for {sorted(missing)}")

    truth = load_all_factors(DATA)["Mkt-RF"]
    pool = [str(p) for p in truth.index
            if args.era_start <= str(p) <= args.era_end]
    rng = random.Random(args.seed)
    months = sorted(rng.sample(pool, args.n_months))
    print(f"sampled {len(months)} months from "
          f"{args.era_start}..{args.era_end}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    spend = 0.0
    records: list[dict] = []
    aborted = False
    min_interval = 60.0 / max(args.rpm, 1.0)

    with OUT.open("a") as f:
        for model in args.models:
            if aborted: break
            ep = eps[model]
            print(f"\n[{model}] {len(months)} months × 2 prompts = "
                  f"{len(months)*2} queries")
            for m in months:
                truth_val = float(truth.get(pd.Period(m, freq="M"),
                                            float("nan")))
                if not (truth_val == truth_val):
                    continue
                for probe, prompt in [
                    ("recall_mktrf",
                     prompt_variant_a("Mkt-RF", m)),
                    ("sentiment",
                     SENTIMENT_PROMPT.format(
                         month_human=format_month_human(m))),
                ]:
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
                        "model_name": model, "probe": probe, "month": m,
                        "truth_mktrf": truth_val,
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
                if aborted: break

    print(f"\nspent ${spend:.4f} over {len(records)} queries")

    df = pd.DataFrame(records)
    if df.empty: return
    df["parsed"] = df["response"].map(lambda r: parse_numeric(r) if r else None)

    print("\n=== ancient-era placebo summary ===")
    for model in args.models:
        sub = df[df["model_name"] == model]
        rec = sub[sub["probe"] == "recall_mktrf"].dropna(
            subset=["parsed", "truth_mktrf"])
        sen = sub[sub["probe"] == "sentiment"].dropna(
            subset=["parsed", "truth_mktrf"])
        # join on month
        merged = rec[["month", "parsed", "truth_mktrf"]].rename(
            columns={"parsed": "recall_estimate"}).merge(
            sen[["month", "parsed"]].rename(
                columns={"parsed": "sentiment"}), on="month")
        n = len(merged)
        if n < 5:
            print(f"  {model}: n={n}, too few for slope"); continue
        rho_recall = float(np.corrcoef(merged["recall_estimate"],
                                       merged["truth_mktrf"])[0, 1])
        b_truth = np.polyfit(merged["truth_mktrf"], merged["sentiment"], 1)[0]
        r_truth = float(np.corrcoef(merged["truth_mktrf"],
                                    merged["sentiment"])[0, 1])
        b_recall = np.polyfit(merged["recall_estimate"],
                              merged["sentiment"], 1)[0]
        r_recall = float(np.corrcoef(merged["recall_estimate"],
                                     merged["sentiment"])[0, 1])
        print(f"  {model:22s}  n={n}")
        print(f"    rho_recall(MktRF) = {rho_recall:+.3f}")
        print(f"    sentiment ~ truth_MktRF: beta={b_truth:+.4f}  r={r_truth:+.3f}")
        print(f"    sentiment ~ recall_est:  beta={b_recall:+.4f}  r={r_recall:+.3f}")


if __name__ == "__main__":
    main()
