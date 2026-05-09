"""Full probe sweep with cutoff-aware sampling.

Per-cell sampling, for each (model, factor):
    50% from pre-cutoff months (deep in training data)
    25% from near-cutoff months (within ±6 months of the model's cutoff)
    25% from post-cutoff months (past the cutoff, not in training data)
    plus the 20 famous months (deduped)

The pre/near/post classification is per-model (``factor_leak.constants.
cutoff_bucket``), so the same calendar month can be "pre" for gpt-4o-mini
(cutoff 2023-10) and "near" or "post" for claude-haiku-4.5 (cutoff
2025-07).

For variant C (comparative) we sample 60 month pairs per (model, factor)
drawn from the full pool.

Total query volume at defaults:
    6 models × 6 factors × (120 months × 2 variants + 60 pairs) = 10,800
    Budget-cut: --months-per-cell 60 --pairs-per-cell 30 → ~5,400 queries

Usage:
    python experiments/01_full_sweep.py --dry-run
    python experiments/01_full_sweep.py                          # all 6 models
    python experiments/01_full_sweep.py --models gpt-4o-mini claude-haiku-4.5
    python experiments/01_full_sweep.py --months-per-cell 60 --pairs-per-cell 30
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path


def _stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed from the given parts.

    Python's builtin ``hash(str)`` is salted per-process (PYTHONHASHSEED),
    so we can't use it for cross-invocation reproducibility. A short
    SHA-1 prefix is cheap and stable.
    """
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha1(s).digest()[:4], "big")

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Load .env (if present) BEFORE any SDK client is created; shell env wins.
from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from factor_leak.constants import (  # noqa: E402
    FAMOUS_MONTHS,
    MODEL_CUTOFFS,
    NEAR_CUTOFF_BAND,
    cutoff_bucket,
)
from factor_leak.cost import estimate_sweep_cost, format_sweep_cost  # noqa: E402
from factor_leak.ff_loader import FACTOR_LONG_NAMES, load_all_factors  # noqa: E402
from factor_leak.probe import (  # noqa: E402
    BudgetExceeded,
    MockEndpoint,
    ProbeQuery,
    default_endpoints,
    load_cache_keys,
    run_sweep,
)


DATA_DIR = REPO_ROOT / "data" / "ff"
# Pilot and full-sweep share this single JSONL so pilot queries carry
# over to the full sweep without being re-run (handoff §3.4 cost note).
DEFAULT_OUT = REPO_ROOT / "experiments" / "results" / "sweep.jsonl"

DEFAULT_MODELS = ["gpt-4.1", "gpt-4o-mini",
                  "claude-sonnet-4.6", "claude-haiku-4.5",
                  "llama-3.3-70b", "deepseek-v3"]
DEFAULT_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

# Full history window — we partition this per-model into pre/near/post
# buckets at query-build time using ``cutoff_bucket``. The lower bound is
# 1963-07 so RMW/CMA are defined; earlier months are added via
# ``FAMOUS_MONTHS`` for Mkt-RF/SMB/HML which have pre-1963 data.
FULL_WINDOW = ("1963-07", "2026-02")


def _available_months(factor: str, window: tuple[str, str]) -> list[str]:
    """Return months in ``window`` for which ``factor`` has a non-NaN value."""
    wide = load_all_factors(DATA_DIR)
    period_range = pd.period_range(window[0], window[1], freq="M")
    out: list[str] = []
    for p in period_range:
        if p in wide.index and pd.notna(wide.loc[p, factor]):
            out.append(str(p))
    return out


def build_month_pool(
    factor: str,
    model: str,
    n_pre: int,
    n_near: int,
    n_post: int,
    famous: list[str],
    rng: random.Random,
) -> list[str]:
    """Build a cutoff-aware month pool for one (factor × model) cell.

    We split the full 1963-07..2026-02 window into three buckets based on
    the model's published training cutoff, then take a prefix from each
    bucket's shuffled pool:

    - ``pre``: cutoff − month > 6 months (training data territory)
    - ``near``: |cutoff − month| ≤ 6 months (transition band)
    - ``post``: cutoff − month < −6 months (past the cutoff — held out)

    Prefix-stable sampling (shuffle once, take first k) is critical so
    that pilot pools are strict subsets of full-sweep pools.
    """
    all_months = _available_months(factor, FULL_WINDOW)
    pre_pool = [m for m in all_months
                if cutoff_bucket(model, m) == "pre" and m not in FAMOUS_MONTHS]
    near_pool = [m for m in all_months
                 if cutoff_bucket(model, m) == "near" and m not in FAMOUS_MONTHS]
    post_pool = [m for m in all_months
                 if cutoff_bucket(model, m) == "post" and m not in FAMOUS_MONTHS]
    famous_present = [m for m in famous
                      if pd.notna(load_all_factors(DATA_DIR).loc[pd.Period(m, freq="M"), factor])]

    rng.shuffle(pre_pool)
    rng.shuffle(near_pool)
    rng.shuffle(post_pool)
    pre = pre_pool[: min(n_pre, len(pre_pool))]
    near = near_pool[: min(n_near, len(near_pool))]
    post = post_pool[: min(n_post, len(post_pool))]
    return sorted(set(famous_present + pre + near + post))


def build_queries(
    models: list[str],
    factors: list[str],
    variants: list[str],
    months_per_cell: int,
    pairs_per_cell: int,
    seed: int,
) -> list[ProbeQuery]:
    """Full-sweep sampling plan. Deterministic under ``seed``.

    Splits each cell's ``months_per_cell`` budget as:
        50% pre-cutoff, 25% near-cutoff, 25% post-cutoff,
    plus the first ``n_famous`` entries of ``FAMOUS_MONTHS`` (deduped).
    """
    n_total = months_per_cell
    n_famous = min(n_total // 6, len(FAMOUS_MONTHS))  # ~16% famous
    remaining = n_total - n_famous
    n_pre = int(round(0.5 * remaining))
    n_near = int(round(0.25 * remaining))
    n_post = remaining - n_pre - n_near
    famous_subset = FAMOUS_MONTHS[:n_famous]

    queries: list[ProbeQuery] = []
    for m in models:
        for f in factors:
            # Per-(model, factor) seeding so the pool for a given cell is
            # independent of the factor ordering. This is what makes pilot
            # pools a strict subset of full-sweep pools.
            rng = random.Random(_stable_seed(seed, m, f))
            pool = build_month_pool(f, m, n_pre, n_near, n_post, famous_subset, rng)
            for month in pool:
                for v in variants:
                    if v in ("A", "B"):
                        queries.append(ProbeQuery(m, f, v, month))
            if "C" in variants and len(pool) >= 2:
                rng_c = random.Random(_stable_seed(seed, m, f, "C"))
                seen_pairs: set[tuple[str, str]] = set()
                attempts = 0
                while len(seen_pairs) < pairs_per_cell and attempts < pairs_per_cell * 20:
                    a, b = rng_c.sample(pool, 2)
                    key = tuple(sorted([a, b]))
                    if key in seen_pairs:
                        attempts += 1
                        continue
                    seen_pairs.add(key)
                    queries.append(ProbeQuery(m, f, "C", month=a, month2=b))
                    attempts += 1
    return queries


def build_mock_endpoints(names: list[str]):
    """Fakes that return '+0.00' — dry-run only; no semantics."""
    def _no_op(prompt, *args, **kwargs):  # accept optional max_tokens
        return "+0.00"
    return {n: MockEndpoint(n, _no_op) for n in names}


def summarize_queue(queries: list[ProbeQuery], already_cached: int = 0) -> None:
    per_model = Counter(q.model_name for q in queries)
    per_variant = Counter(q.variant for q in queries)
    per_factor = Counter(q.factor for q in queries)
    print(f"\n=== sweep plan: {len(queries)} queries "
          f"({already_cached} already cached — will be skipped) ===")
    print("  by model:   ", dict(per_model))
    print("  by variant: ", dict(per_variant))
    print("  by factor:  ", dict(per_factor))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--factors", nargs="+", default=DEFAULT_FACTORS)
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--months-per-cell", type=int, default=120)
    ap.add_argument("--pairs-per-cell", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="Hard cap on actual cumulative spend (USD). Sweep "
                         "aborts mid-flight if exceeded; cache is preserved. "
                         "If omitted, defaults to 1.5x the pre-flight estimate.")
    ap.add_argument("--no-budget", action="store_true",
                    help="DISABLE the hard budget cap entirely (dangerous).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Use mock endpoints — no API calls, no cost.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the confirmation prompt.")
    ap.add_argument("--pilot", action="store_true",
                    help="Pilot preset: --months-per-cell 20 --pairs-per-cell 0 "
                         "--variants A B --factors Mkt-RF HML "
                         "--models gpt-4o-mini claude-haiku-4.5. "
                         "Writes to the shared cache, so full sweep reuses these.")
    args = ap.parse_args()

    if args.pilot:
        # Only overwrite fields the user hasn't explicitly set. We compare
        # against the argparse defaults so `--pilot --models foo bar` keeps
        # the user's model list instead of snapping back to the preset.
        if args.months_per_cell == ap.get_default("months_per_cell"):
            args.months_per_cell = 20
        if args.pairs_per_cell == ap.get_default("pairs_per_cell"):
            args.pairs_per_cell = 0
        if args.variants == ap.get_default("variants"):
            args.variants = ["A", "B"]
        if args.factors == ap.get_default("factors"):
            args.factors = ["Mkt-RF", "HML"]
        if args.models == ap.get_default("models"):
            args.models = ["claude-haiku-4.5", "claude-sonnet-4.6"]

    bad_factors = set(args.factors) - set(FACTOR_LONG_NAMES)
    if bad_factors:
        sys.exit(f"unknown factors: {sorted(bad_factors)}")

    # Dry-runs MUST NOT pollute the shared cache JSONL: their mock
    # responses would later be treated as legitimate "already cached"
    # results and the real sweep would silently skip thousands of
    # queries. Route dry-runs to a per-invocation temp file.
    if args.dry_run and args.out == DEFAULT_OUT:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            prefix="factor_leak_dryrun_",
            suffix=".jsonl",
            delete=False,
        )
        args.out = Path(tmp.name)
        tmp.close()
        print(f"(dry-run output redirected to {args.out})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cached = load_cache_keys(args.out) if args.out.exists() else set()

    queries_all = build_queries(
        args.models, args.factors, args.variants,
        args.months_per_cell, args.pairs_per_cell, args.seed,
    )
    remaining = [q for q in queries_all if q.key not in cached]
    summarize_queue(queries_all, already_cached=len(queries_all) - len(remaining))

    cost_summary = estimate_sweep_cost(remaining)
    print()
    print(format_sweep_cost(cost_summary))

    # Determine the hard budget cap for this run.
    if args.no_budget:
        budget_cap: float | None = None
    elif args.budget_usd is not None:
        budget_cap = args.budget_usd
    else:
        # Safety default: 1.5x the estimate, rounded up to cents.
        budget_cap = round(cost_summary["total_usd"] * 1.5 + 0.005, 2)

    if budget_cap is not None:
        print(f"Hard budget cap:   ${budget_cap:.2f}  "
              f"(sweep auto-aborts if actual spend reaches this)")
    else:
        print("Hard budget cap:   DISABLED (--no-budget)")

    if (args.budget_usd is not None
            and cost_summary["total_usd"] > args.budget_usd):
        sys.exit(
            f"\nestimated cost ${cost_summary['total_usd']:.2f} exceeds "
            f"--budget-usd ${args.budget_usd:.2f}; aborting before any call."
        )

    if not args.dry_run and not args.yes and remaining:
        cap_str = f"${budget_cap:.2f}" if budget_cap is not None else "DISABLED"
        ans = input(
            f"\nRun {len(remaining)} live queries "
            f"(est ${cost_summary['total_usd']:.2f}, hard cap {cap_str})? [y/N] "
        ).strip().lower()
        if ans not in {"y", "yes"}:
            print("aborted.")
            return

    if args.dry_run:
        endpoints = build_mock_endpoints(args.models)
    else:
        endpoints = {e.name: e for e in default_endpoints() if e.name in args.models}
        missing = set(args.models) - endpoints.keys()
        if missing:
            sys.exit(f"no endpoint definition for models: {sorted(missing)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    def _progress(done: int, total: int) -> None:
        if done == total or done % max(total // 50, 1) == 0:
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float("inf")
            print(f"  [{done}/{total}] {rate:.1f} q/s  ETA {eta:.0f}s", flush=True)

    try:
        new = run_sweep(
            queries_all, endpoints, args.out,
            max_workers=args.max_workers,
            progress=_progress,
            budget_usd=budget_cap,
        )
    except BudgetExceeded as exc:
        print(f"\n!! BUDGET CAP HIT: {exc}")
        sys.exit(2)

    # Tally the true spend from recorded token usage.
    from factor_leak.cost import cost_from_usage
    actual = sum(
        cost_from_usage(r.query.model_name, r.input_tokens, r.output_tokens)
        for r in new
    )
    print(f"ran {len(new)} new queries (resumed; results in {args.out})")
    print(f"actual cumulative spend this run: ${actual:.4f}")

    errors = sum(1 for r in new if r.error is not None)
    if errors:
        print(f"!! {errors} queries errored; rerun this script to retry them.")


if __name__ == "__main__":
    main()
