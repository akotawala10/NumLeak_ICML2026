"""Run Controls 2 & 3 — synthetic factor + illiquid PE fund probes.

A healthy factor-recall claim depends on LLMs *not* hallucinating numbers for
series they cannot know. This script issues the same variant-A numeric-answer
prompt for a made-up factor and a made-up illiquid fund, across the same
months used by the main sweep. We report:

- parse rate (fraction of answers that yielded a number)
- mean absolute magnitude of those numbers (how big are the hallucinations?)

If parse rate is low → recall signal is clean.
If parse rate is high → the paper must caveat that models freely guess
numeric answers, and the "recall" is partly hallucinating into the right
neighborhood.

Usage (identical flags to 00_pilot):
    python experiments/05_controls.py --dry-run
    python experiments/05_controls.py --models gpt-4o-mini claude-haiku-4.5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from factor_leak import controls  # noqa: E402
from factor_leak.env import load_dotenv  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import MockEndpoint, default_endpoints  # noqa: E402

# Load API keys from repo-root .env.
load_dotenv(REPO_ROOT / ".env")

DEFAULT_OUT = REPO_ROOT / "experiments" / "results" / "controls.jsonl"


def _run_one(endpoint, kind: str, month: str) -> dict:
    prompt = controls.control_prompt(kind, month)
    t0 = time.perf_counter()
    input_tokens = 0
    output_tokens = 0
    try:
        ep_resp = endpoint(prompt, max_tokens=48)
        response = ep_resp.text
        input_tokens = ep_resp.input_tokens
        output_tokens = ep_resp.output_tokens
        error = None
    except Exception as exc:  # noqa: BLE001
        response = None
        error = f"{type(exc).__name__}: {exc}"
    return {
        "model_name": endpoint.name,
        "kind": kind,
        "month": month,
        "prompt": prompt,
        "response": response,
        "error": error,
        "latency_s": time.perf_counter() - t0,
        "ts": time.time(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def build_mock_endpoints(names: list[str]):
    """Mock endpoints that decline (return a fixed no-answer) — expected behavior."""
    def _decline(prompt):
        return ("I don't have reliable data on that specific series. "
                "Without access to confidential fund records I cannot give a number.")
    return {n: MockEndpoint(n, _decline) for n in names}


def summarize(records: list[dict]) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        print("(no records)")
        return
    print(f"\n=== controls summary: {len(df)} queries ===")
    for (model, kind), frame in df.groupby(["model_name", "kind"]):
        ok = frame[frame["error"].isna()]
        parsed = ok["response"].map(lambda r: parse_numeric(r) if r else None)
        parsed_rate = parsed.notna().mean() if len(ok) else float("nan")
        mag = parsed.dropna().abs().mean() if parsed.notna().any() else float("nan")
        print(
            f"  {model:25s} {kind:20s}  n={len(frame):3d} "
            f"parse={parsed_rate:5.1%}  mean|ans|={mag if pd.notna(mag) else float('nan'):.3f}%"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["gpt-4o-mini", "claude-haiku-4.5"])
    ap.add_argument("--months", nargs="+",
                    default=["1987-10", "1995-05", "2001-09", "2008-10",
                             "2014-07", "2020-03"])
    ap.add_argument("--kinds", nargs="+",
                    default=["synthetic_factor", "illiquid_fund"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        endpoints = build_mock_endpoints(args.models)
    else:
        endpoints = {e.name: e for e in default_endpoints() if e.name in args.models}
        missing = set(args.models) - endpoints.keys()
        if missing:
            sys.exit(f"no endpoint definition for models: {sorted(missing)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Controls are small (2 kinds × ~6 months × 2 models = 24 queries). Skip
    # threading to keep output order deterministic during inspection.
    total = len(args.models) * len(args.kinds) * len(args.months)
    print(f"controls: {total} queries")
    records: list[dict] = []
    with args.out.open("a") as f:
        for model in args.models:
            for kind in args.kinds:
                for month in args.months:
                    rec = _run_one(endpoints[model], kind, month)
                    f.write(json.dumps(rec) + "\n")
                    records.append(rec)
    summarize(records)


if __name__ == "__main__":
    main()
