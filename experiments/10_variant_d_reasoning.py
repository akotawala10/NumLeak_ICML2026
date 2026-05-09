"""Variant D: ask the same question with a reasoning/chain-of-thought prefix.

Tests whether explicit step-by-step thinking changes recall on Mkt-RF
(the cell with the strongest baseline signal). Runs only on Mkt-RF for
both Anthropic models to keep the budget tight.

Prompt: ``Think step-by-step about historical Fama-French factor returns,
then answer: what was the monthly return of the Fama-French {factor} factor
in {Month Year}? End with a signed decimal percentage only.''

Expected max_tokens is larger (384) to allow reasoning tokens before the
final number. Parser extracts the last number in the response.
"""
from __future__ import annotations

import argparse
import sys
import time
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from factor_leak.ff_loader import FACTOR_LONG_NAMES, load_all_factors  # noqa: E402
from factor_leak.probe import (  # noqa: E402
    MockEndpoint, ProbeQuery, _append_jsonl, default_endpoints, load_cache_keys,
    format_month_human,
)
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.cost import cost_from_usage  # noqa: E402

OUT = REPO_ROOT / "experiments" / "results" / "variant_d.jsonl"


def reasoning_prompt(factor: str, month: str) -> str:
    long_name = FACTOR_LONG_NAMES[factor]
    return (
        f"Think step-by-step about historical Fama-French factor returns, "
        f"then answer: what was the monthly return of the Fama-French "
        f"{long_name} factor in {format_month_human(month)}? "
        f"End your response with a signed decimal percentage "
        f"(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) only."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["claude-haiku-4.5", "claude-sonnet-4.6"])
    ap.add_argument("--factor", default="Mkt-RF")
    ap.add_argument("--months", nargs="+", default=None,
                    help="default: reuse the same 77 Mkt-RF months from the main sweep")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    # Reuse the same months Sonnet probed on Mkt-RF so results are
    # directly comparable to Variant A.
    if args.months is None:
        import pandas as pd
        # Grab months with ground truth + that were in main sweep
        sweep_jsonl = REPO_ROOT / "experiments" / "results" / "sweep.jsonl"
        months = set()
        with sweep_jsonl.open() as f:
            for line in f:
                r = json.loads(line)
                q = r["query"]
                if q["factor"] == args.factor and q["variant"] == "A" and r["error"] is None:
                    months.add(q["month"])
        args.months = sorted(months)
    print(f"Variant D: {len(args.models)} models × {args.factor} × {len(args.months)} months = "
          f"{len(args.models) * len(args.months)} queries")

    # Cost preview
    total_q = len(args.models) * len(args.months)
    # crude estimate: 60 tokens in, 250 tokens out (CoT inflates output)
    est_cost = 0.0
    for m in args.models:
        from factor_leak.cost import MODEL_PRICING
        p = MODEL_PRICING.get(m, {"input": 0, "output": 0})
        est_cost += len(args.months) * (60 * p["input"] + 250 * p["output"]) / 1_000_000
    print(f"Rough cost estimate: ${est_cost:.3f}")
    if not args.yes and input("proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
        sys.exit("aborted")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    endpoints = {e.name: e for e in default_endpoints() if e.name in args.models}

    # Build cache from existing records.
    existing = set()
    if OUT.exists():
        with OUT.open() as f:
            for line in f:
                r = json.loads(line)
                k = f"{r['model_name']}|{r['factor']}|{r['month']}"
                if r["error"] is None:
                    existing.add(k)

    spent = 0.0
    for model in args.models:
        ep = endpoints[model]
        for month in args.months:
            key = f"{model}|{args.factor}|{month}"
            if key in existing:
                continue
            prompt = reasoning_prompt(args.factor, month)
            t0 = time.perf_counter()
            try:
                resp = ep(prompt, max_tokens=args.max_tokens)
                latency = time.perf_counter() - t0
                record = {
                    "model_name": model, "factor": args.factor, "month": month,
                    "prompt": prompt,
                    "response": resp.text,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "latency_s": latency, "error": None, "ts": time.time(),
                }
                usd = cost_from_usage(model, resp.input_tokens, resp.output_tokens)
                spent += usd
            except Exception as exc:
                latency = time.perf_counter() - t0
                record = {
                    "model_name": model, "factor": args.factor, "month": month,
                    "prompt": prompt, "response": None,
                    "input_tokens": 0, "output_tokens": 0,
                    "latency_s": latency,
                    "error": f"{type(exc).__name__}: {exc}",
                    "ts": time.time(),
                }
            _append_jsonl(OUT, record)
            print(f"  [{month}] {model}: {'OK' if record['error'] is None else 'ERR'} "
                  f"(cum ${spent:.3f})", flush=True)

    print(f"\nDone. Total spent: ${spent:.4f}.  Results in {OUT}")


if __name__ == "__main__":
    main()
