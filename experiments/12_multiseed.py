"""Multi-seed robustness: rerun the headline Mkt-RF probe with different
random month samples to check that Pearson $r{=}0.98$ isn't an artifact
of seed=42.

Samples 40 random Mkt-RF months in 1963–2022 with seeds {1, 2, 3}, runs
Variant A on both Anthropic models, reports per-seed Pearson / within-25bps
/ directional accuracy. Expected cost ~$1-2 total.
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
from factor_leak.ff_loader import load_all_factors, ground_truth  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import (  # noqa: E402
    _append_jsonl, default_endpoints, prompt_variant_a,
)

OUT = REPO_ROOT / "experiments" / "results" / "multiseed.jsonl"


def sample_months(seed: int, n: int = 40) -> list[str]:
    """40 random Mkt-RF months in 1963-07..2022-12 for the given seed.
    Mkt-RF is defined for all of this window; no availability filter needed.
    """
    import pandas as pd
    all_months = pd.period_range("1963-07", "2022-12", freq="M").astype(str).tolist()
    rng = random.Random(
        int.from_bytes(hashlib.sha1(f"multiseed|{seed}".encode()).digest()[:4], "big")
    )
    return sorted(rng.sample(all_months, n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--models", nargs="+",
                    default=["claude-haiku-4.5", "claude-sonnet-4.6"])
    ap.add_argument("--factor", default="Mkt-RF")
    ap.add_argument("--n-months", type=int, default=40)
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    total = len(args.seeds) * len(args.models) * args.n_months
    print(f"Multi-seed: {len(args.seeds)} seeds × {len(args.models)} models × "
          f"{args.n_months} Mkt-RF months = {total} queries")
    if not args.yes and input("proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
        sys.exit("aborted")

    # Dedup cache: (seed, model, month) tuples
    existing = set()
    if OUT.exists():
        with OUT.open() as f:
            for line in f:
                r = json.loads(line)
                if r["error"] is None:
                    existing.add((r["seed"], r["model_name"], r["month"]))

    endpoints = {e.name: e for e in default_endpoints() if e.name in args.models}
    OUT.parent.mkdir(parents=True, exist_ok=True)

    spent = 0.0
    for seed in args.seeds:
        months = sample_months(seed, args.n_months)
        for model in args.models:
            ep = endpoints[model]
            for month in months:
                if (seed, model, month) in existing:
                    continue
                prompt = prompt_variant_a(args.factor, month)
                t0 = time.perf_counter()
                try:
                    resp = ep(prompt, max_tokens=48)
                    rec = {
                        "seed": seed, "model_name": model,
                        "factor": args.factor, "month": month,
                        "prompt": prompt, "response": resp.text,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "latency_s": time.perf_counter() - t0,
                        "error": None, "ts": time.time(),
                    }
                    spent += cost_from_usage(model, resp.input_tokens, resp.output_tokens)
                except Exception as exc:
                    rec = {
                        "seed": seed, "model_name": model,
                        "factor": args.factor, "month": month,
                        "prompt": prompt, "response": None,
                        "input_tokens": 0, "output_tokens": 0,
                        "latency_s": time.perf_counter() - t0,
                        "error": f"{type(exc).__name__}: {exc}",
                        "ts": time.time(),
                    }
                _append_jsonl(OUT, rec)
    print(f"Done. Spent ${spent:.4f}")


if __name__ == "__main__":
    main()
