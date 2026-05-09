"""Probe Opus 4.7 and GPT-5.4 on SMB/HML/RMW/CMA/Mom to fill the
missing Table 1 / Fig 1 cells.

Mirrors Variant A on 40 months per (model, factor). Sonnet and Haiku
already cover these factors via the main sweep. This fills the hatched
"n/a" cells in Fig 1 for Opus and GPT-5.4.

Output: results/opus_factors.jsonl, results/gpt54_factors.jsonl +
console summary. Budget-capped.
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
from factor_leak.ff_loader import load_all_factors, FACTOR_LONG_NAMES  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

RES = REPO / "experiments" / "results"
DATA = REPO / "data" / "ff"

MODELS = ["claude-opus-4.7", "gpt-5.4"]
FACTORS = ["SMB", "HML", "RMW", "CMA", "Mom"]


def variant_a_prompt(factor: str, month: str) -> str:
    long_name = FACTOR_LONG_NAMES[factor]
    return (
        f"What was the monthly return of the Fama-French {long_name} factor "
        f"in {format_month_human(month)}? "
        f"Answer with a signed decimal percentage "
        f"(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        f"and nothing else."
    )


def sample_months(factor: str, n: int, seed: int) -> list[str]:
    s = load_all_factors(DATA)[factor].dropna()
    pool = [str(p) for p in s.index]
    rng = random.Random(seed)
    return sorted(rng.sample(pool, n))


def out_path(model: str) -> Path:
    tag = "opus" if "opus" in model else ("gpt54" if "gpt" in model else model)
    return RES / f"{tag}_factors.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-months", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--factors", nargs="+", default=FACTORS)
    ap.add_argument("--budget-usd", type=float, default=1.50)
    args = ap.parse_args()

    eps = {e.name: e for e in default_endpoints() if e.name in args.models}
    missing = set(args.models) - eps.keys()
    if missing:
        sys.exit(f"no endpoint wired for {sorted(missing)}")

    truth = load_all_factors(DATA)
    total_spend = 0.0
    records: list[dict] = []
    aborted = False

    for model in args.models:
        if aborted: break
        ep = eps[model]
        out = out_path(model)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            for factor in args.factors:
                if aborted: break
                months = sample_months(factor, args.n_months, args.seed)
                print(f"[{model} × {factor}] {len(months)} queries")
                for m in months:
                    tt = float(truth[factor].get(pd.Period(m, freq="M"),
                                                 float("nan")))
                    prompt = variant_a_prompt(factor, m)
                    t0 = time.perf_counter()
                    try:
                        resp = ep(prompt, max_tokens=48)
                        text, err = resp.text, None
                        in_tok, out_tok = resp.input_tokens, resp.output_tokens
                    except Exception as e:  # noqa: BLE001
                        text, err = None, f"{type(e).__name__}: {e}"
                        in_tok = out_tok = 0
                    usd = cost_from_usage(model, in_tok, out_tok)
                    total_spend += usd
                    rec = {
                        "model_name": model, "factor": factor, "month": m,
                        "truth": tt, "prompt": prompt,
                        "response": text, "error": err,
                        "input_tokens": in_tok, "output_tokens": out_tok,
                        "latency_s": time.perf_counter() - t0,
                        "usd": usd, "ts": time.time(),
                    }
                    f.write(json.dumps(rec) + "\n")
                    records.append(rec)
                    if total_spend > args.budget_usd:
                        print(f"  ABORT: ${total_spend:.3f} > budget "
                              f"${args.budget_usd:.3f}")
                        aborted = True
                        break

    print(f"\nspent ${total_spend:.4f} over {len(records)} queries")

    df = pd.DataFrame(records)
    if df.empty: return
    df["parsed"] = df["response"].map(lambda r: parse_numeric(r) if r else None)
    df["abs_err"] = (df["parsed"] - df["truth"]).abs()
    print("\n=== summary ===")
    for (m, f), grp in df.groupby(["model_name", "factor"]):
        n = len(grp)
        p = grp.dropna(subset=["parsed", "truth"])
        np_ = len(p)
        pr = np_ / n if n else float("nan")
        w25 = (p["abs_err"] <= 0.25).mean() if np_ else float("nan")
        r = float(np.corrcoef(p["parsed"], p["truth"])[0, 1]) if np_ >= 3 else float("nan")
        sign_ok = ((np.sign(p["parsed"]) == np.sign(p["truth"])).mean()
                   if np_ else float("nan"))
        print(f"  {m:22s} × {f:6s}: n={n} parsed={np_} parse={pr:.2f} "
              f"r={r:+.3f} w25={w25:.3f} sign={sign_ok:.3f}")


if __name__ == "__main__":
    main()
