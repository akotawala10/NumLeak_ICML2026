"""Camera-ready multi-seed extension.

Reviewer 2 noted multi-seed robustness was only shown for Sonnet/Haiku on
Mkt-RF. This driver extends multi-seed coverage to Opus 4.7 and GPT-5.4 on
Mkt-RF, and adds SMB + Mom (a size factor and a momentum factor) on all four
frontier models so the within-family selectivity claim also gets multi-seed
support.

Design:
- 40 random months per (seed, factor); seeds {1, 2, 3}; deterministic hash.
- Dedup key is (seed, model, factor, month) — fixes a latent bug in
  12_multiseed.py where the key omitted factor.
- Hard cost cap: aborts mid-run if estimated spend exceeds --max-cost.

Expected cost at full scope:
- Opus 4.7  : 3 factors x 3 seeds x 40 months = 360 q at ~$0.0022 = $0.79
- Sonnet 4.6: 2 new factors x 3 seeds x 40 months = 240 q at ~$0.00035 = $0.08
- Haiku 4.5 : 2 new factors x 3 seeds x 40 months = 240 q at ~$0.00041 = $0.10
- GPT-5.4   : 3 factors x 3 seeds x 40 months = 360 q at ~$0.0011 = $0.40
- Total: ~$1.37 (cap is $5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from factor_leak.cost import cost_from_usage  # noqa: E402
from factor_leak.probe import (  # noqa: E402
    _append_jsonl, default_endpoints, prompt_variant_a,
)

OUT = REPO_ROOT / "experiments" / "results" / "camera_ready_multiseed.jsonl"


def sample_months(seed: int, factor: str, n: int = 40) -> list[str]:
    """Deterministic month sample per (seed, factor).

    Mkt-RF/SMB/HML/RMW/CMA defined on 1963-07..2022-12; Mom defined on
    1927-01..2022-12 but we restrict to the FF5 window for uniform support.
    """
    import pandas as pd
    all_months = pd.period_range("1963-07", "2022-12", freq="M").astype(str).tolist()
    rng = random.Random(
        int.from_bytes(
            hashlib.sha1(f"camera_ready|{factor}|{seed}".encode()).digest()[:4],
            "big",
        )
    )
    return sorted(rng.sample(all_months, n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument(
        "--models", nargs="+",
        default=["claude-opus-4.7", "claude-sonnet-4.6",
                 "claude-haiku-4.5", "gpt-5.4"],
    )
    ap.add_argument("--factors", nargs="+", default=["Mkt-RF", "SMB", "Mom"])
    ap.add_argument("--n-months", type=int, default=40)
    ap.add_argument("--max-cost", type=float, default=5.0,
                    help="Hard USD cap; aborts mid-run if exceeded.")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    total = len(args.seeds) * len(args.models) * len(args.factors) * args.n_months
    print(f"Scope: {len(args.seeds)} seeds x {len(args.models)} models x "
          f"{len(args.factors)} factors x {args.n_months} months = {total} queries")
    print(f"Cost cap: ${args.max_cost:.2f}")
    if not args.yes and input("proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
        sys.exit("aborted")

    existing: set[tuple] = set()
    if OUT.exists():
        with OUT.open() as f:
            for line in f:
                r = json.loads(line)
                if r["error"] is None:
                    existing.add(
                        (r["seed"], r["model_name"], r["factor"], r["month"])
                    )
    if existing:
        print(f"Resuming: {len(existing)} successful records already on disk.")

    endpoints = {e.name: e for e in default_endpoints() if e.name in args.models}
    missing = set(args.models) - set(endpoints)
    if missing:
        sys.exit(f"Unknown endpoints: {sorted(missing)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    spent = 0.0
    n_done = 0
    n_skipped = 0
    t_start = time.time()

    for seed in args.seeds:
        for factor in args.factors:
            months = sample_months(seed, factor, args.n_months)
            for model in args.models:
                ep = endpoints[model]
                for month in months:
                    key = (seed, model, factor, month)
                    if key in existing:
                        n_skipped += 1
                        continue
                    if spent >= args.max_cost:
                        print(f"\nCost cap ${args.max_cost:.2f} reached "
                              f"(spent ${spent:.4f}); halting.")
                        _summary(n_done, n_skipped, spent, t_start)
                        return
                    prompt = prompt_variant_a(factor, month)
                    t0 = time.perf_counter()
                    try:
                        resp = ep(prompt, max_tokens=48)
                        rec = {
                            "seed": seed, "model_name": model,
                            "factor": factor, "month": month,
                            "prompt": prompt, "response": resp.text,
                            "input_tokens": resp.input_tokens,
                            "output_tokens": resp.output_tokens,
                            "latency_s": time.perf_counter() - t0,
                            "error": None, "ts": time.time(),
                        }
                        q_cost = cost_from_usage(
                            model, resp.input_tokens, resp.output_tokens
                        )
                        spent += q_cost
                    except Exception as exc:
                        rec = {
                            "seed": seed, "model_name": model,
                            "factor": factor, "month": month,
                            "prompt": prompt, "response": None,
                            "input_tokens": 0, "output_tokens": 0,
                            "latency_s": time.perf_counter() - t0,
                            "error": f"{type(exc).__name__}: {exc}",
                            "ts": time.time(),
                        }
                    _append_jsonl(OUT, rec)
                    n_done += 1
                    if n_done % 20 == 0:
                        print(f"  {n_done:4d} done, ${spent:.4f} spent, "
                              f"{time.time()-t_start:.0f}s")

    _summary(n_done, n_skipped, spent, t_start)


def _summary(n_done: int, n_skipped: int, spent: float, t_start: float) -> None:
    print()
    print(f"New queries: {n_done}")
    print(f"Cached: {n_skipped}")
    print(f"Spent: ${spent:.4f}")
    print(f"Wall: {time.time()-t_start:.1f}s")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
