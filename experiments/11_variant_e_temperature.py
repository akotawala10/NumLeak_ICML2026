"""Variant E: temperature>0 robustness check.

Tests whether Mkt-RF recall is stable under sampling noise or collapses
when the model isn't pinned to greedy decoding. Same Sonnet × Mkt-RF
months as the main sweep, at T=1.0, two independent draws per month.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from factor_leak.probe import (  # noqa: E402
    AnthropicEndpoint, _append_jsonl, prompt_variant_a,
)
from factor_leak.cost import cost_from_usage  # noqa: E402

OUT = REPO_ROOT / "experiments" / "results" / "variant_e_temperature.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", default="Mkt-RF")
    ap.add_argument("--models", nargs="+",
                    default=["claude-sonnet-4.6"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--draws", type=int, default=2,
                    help="independent samples per month at T>0")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    # Reuse same months as main sweep.
    sweep_jsonl = REPO_ROOT / "experiments" / "results" / "sweep.jsonl"
    months = set()
    with sweep_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            q = r["query"]
            if (q["factor"] == args.factor and q["variant"] == "A"
                    and r["error"] is None
                    and q["model_name"] in args.models):
                months.add(q["month"])
    months = sorted(months)
    total = len(args.models) * len(months) * args.draws
    print(f"Variant E: {len(args.models)} models × {len(months)} months × "
          f"{args.draws} draws at T={args.temperature} = {total} queries")
    if not args.yes and input("proceed? [y/N] ").strip().lower() not in {"y","yes"}:
        sys.exit("aborted")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    endpoints = {
        "claude-sonnet-4.6": AnthropicEndpoint(name="claude-sonnet-4.6",
                                                model="claude-sonnet-4-6",
                                                temperature=args.temperature),
        "claude-haiku-4.5": AnthropicEndpoint(name="claude-haiku-4.5",
                                               model="claude-haiku-4-5-20251001",
                                               temperature=args.temperature),
    }

    existing = set()
    if OUT.exists():
        with OUT.open() as f:
            for line in f:
                r = json.loads(line)
                k = f"{r['model_name']}|{r['factor']}|{r['month']}|{r['draw']}"
                if r["error"] is None:
                    existing.add(k)

    spent = 0.0
    for model in args.models:
        ep = endpoints[model]
        for month in months:
            for draw in range(args.draws):
                key = f"{model}|{args.factor}|{month}|{draw}"
                if key in existing:
                    continue
                prompt = prompt_variant_a(args.factor, month)
                t0 = time.perf_counter()
                try:
                    resp = ep(prompt, max_tokens=48)
                    lat = time.perf_counter() - t0
                    rec = {
                        "model_name": model, "factor": args.factor,
                        "month": month, "draw": draw,
                        "temperature": args.temperature, "prompt": prompt,
                        "response": resp.text,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "latency_s": lat, "error": None, "ts": time.time(),
                    }
                    spent += cost_from_usage(model, resp.input_tokens, resp.output_tokens)
                except Exception as exc:
                    lat = time.perf_counter() - t0
                    rec = {
                        "model_name": model, "factor": args.factor,
                        "month": month, "draw": draw,
                        "temperature": args.temperature, "prompt": prompt,
                        "response": None, "input_tokens": 0, "output_tokens": 0,
                        "latency_s": lat,
                        "error": f"{type(exc).__name__}: {exc}",
                        "ts": time.time(),
                    }
                _append_jsonl(OUT, rec)
    print(f"\nDone. Spent ${spent:.4f}. Results at {OUT}")


if __name__ == "__main__":
    main()
