"""EXP-T2: prompt-level date-scramble control for the transmission probe.

The existing transmission probe (22_transmission_estimate.py) elicits
date-anchored sentiment in [-1, +1] for each month and computes:
  beta_T = OLS slope of sentiment ~ truth Mkt-RF.

The placebo (44_transmission_placebo.py) varies the *era* (1965-2020 vs
1926-1965) to discriminate recall-mediated from narrative-mediated
transmission. The remaining limitation flagged in App. limitations is
that within the modern era, the sentiment-truth covariance could be
year-level co-occurrence rather than month-level date-conditional.

This script tests that by re-running the same sentiment prompt with the
date in the prompt SHIFTED by +6 months. We then compare:

  beta_T              = sentiment(orig)      ~ truth(orig)   [known from 22_]
  beta_T_scram->orig  = sentiment(scrambled) ~ truth(orig)
  beta_T_scram->shift = sentiment(scrambled) ~ truth(shift)

Predictions:
  - Month-level signal: sentiment(scrambled) tracks truth(shift), not truth(orig).
    -> beta_T_scram->orig collapses; beta_T_scram->shift ≈ beta_T.
  - Year-level signal: sentiment(scrambled) still tracks truth(orig).
    -> beta_T_scram->orig persists.

Skips months where shift would land past the model's training cutoff.
Writes to experiments/results/transmission_scrambled.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

from factor_leak.constants import MODEL_CUTOFFS  # noqa: E402
from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "transmission_scrambled.jsonl"
SRC = REPO / "experiments" / "results" / "transmission.jsonl"
DATA = REPO / "data" / "ff"


def shift_month(month_str: str, delta_months: int) -> str:
    """Return the YYYY-MM that is delta_months later than month_str."""
    p = pd.Period(month_str, freq="M") + delta_months
    return f"{p.year:04d}-{p.month:02d}"


def in_window(month_str: str, model: str, buffer_months: int = 1) -> bool:
    """True if month_str is at least ``buffer_months`` before the model's cutoff."""
    if model not in MODEL_CUTOFFS:
        return True
    cutoff = pd.Period(MODEL_CUTOFFS[model], freq="M")
    p = pd.Period(month_str, freq="M")
    return (cutoff - p).n >= buffer_months


def build_prompt(month_str: str) -> str:
    """Same template as 22_transmission_estimate.py, with date substituted."""
    human = format_month_human(month_str)
    return (
        "Forget all your previous instructions. You are a financial expert "
        "with stock-recommendation experience. Considering only the broad "
        f"U.S. equity market in {human}, what was the prevailing investor "
        "sentiment that month? Answer on a scale from -1.00 (very negative "
        "for U.S. equities) to +1.00 (very positive). Reply with exactly "
        "one signed decimal (e.g., -0.45 or +0.30) and nothing else."
    )


def parse_sentiment(text: str | None) -> float | None:
    if text is None:
        return None
    v = parse_numeric(text)
    if v is None:
        return None
    if -1.0 <= v <= 1.0:
        return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift-months", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    truth_df = load_all_factors(DATA)
    truth = truth_df["Mkt-RF"]

    src_recs = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(src_recs)} records from {SRC.relative_to(REPO)}")

    # Build query plan: (model, original_month, shifted_month)
    plan = []
    skipped = []
    for r in src_recs:
        if r.get("error"):
            continue
        model = r["model_name"]
        orig = r["month"]
        shifted = shift_month(orig, args.shift_months)
        # Skip if shifted month is post-cutoff (would create a fake post-cutoff query)
        if not in_window(shifted, model):
            skipped.append((model, orig, shifted))
            continue
        # Skip if shifted month has no truth value
        try:
            t_shift = float(truth.get(pd.Period(shifted, freq="M"), float("nan")))
        except KeyError:
            skipped.append((model, orig, shifted))
            continue
        if pd.isna(t_shift):
            skipped.append((model, orig, shifted))
            continue
        plan.append({"model": model, "orig_month": orig, "shifted_month": shifted,
                     "truth_orig": r["truth_mktrf"], "truth_shift": t_shift,
                     "recall_orig": r.get("recall_estimate")})

    if args.limit:
        plan = plan[: args.limit]

    print(f"plan: {len(plan)} queries; {len(skipped)} skipped (post-cutoff or missing truth)")
    from collections import Counter
    print(f"  by model: {Counter(p['model'] for p in plan)}")
    print(f"  shift: +{args.shift_months} months")
    if args.dry_run:
        return

    endpoints = {e.name: e for e in default_endpoints()
                 if e.name in {p["model"] for p in plan}}
    missing = {p["model"] for p in plan} - endpoints.keys()
    if missing:
        sys.exit(f"missing endpoints: {sorted(missing)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("a") as f:
        for i, p in enumerate(plan):
            ep = endpoints[p["model"]]
            prompt = build_prompt(p["shifted_month"])
            t0 = time.perf_counter()
            try:
                resp = ep(prompt, max_tokens=args.max_tokens)
                text = resp.text
                err = None
                in_tok = resp.input_tokens
                out_tok = resp.output_tokens
            except Exception as e:  # noqa: BLE001
                text = None
                err = f"{type(e).__name__}: {e}"
                in_tok = out_tok = 0
            dt = time.perf_counter() - t0
            sentiment = parse_sentiment(text)
            rec = {
                "model_name": p["model"],
                "orig_month": p["orig_month"],
                "shifted_month": p["shifted_month"],
                "truth_orig_mktrf": p["truth_orig"],
                "truth_shift_mktrf": p["truth_shift"],
                "recall_orig": p["recall_orig"],
                "prompt": prompt,
                "response": text,
                "sentiment": sentiment,
                "error": err,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_s": dt,
                "ts": time.time(),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            written += 1
            if (i + 1) % 20 == 0 or i + 1 == len(plan):
                print(f"  [{i+1:>4d}/{len(plan)}] {p['model']} {p['orig_month']}->{p['shifted_month']} "
                      f"sentiment={sentiment} err={err}")

    print(f"\nwrote {written} records to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
