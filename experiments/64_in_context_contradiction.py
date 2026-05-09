"""EXP-M4: in-context contradiction probe.

Tests whether memorized Mkt-RF recall is robust to a contradictory
in-context value. For each famous Mkt-RF month, we run two prompts
on Sonnet 4.6 and Opus 4.7:

  V_stable: standard Variant-A query.
  V_contradicted: Variant-A preceded by an injected fake value
                  ("My records show Mkt-RF in {month} was -42.00%.").

The injected value is implausibly extreme so the model cannot mistake
it for noise. If the model overrides its recall and emits a value
near -42.00%, the recall is in-context-modifiable (shallow). If the
model sticks with the true value, the recall is internalized and
robust to contextual contradiction — a mechanism-level finding about
how the memorization channel resolves conflicts between trained-in
recall and in-context input.

Sample: 20 famous months (App. famous list) × 2 conditions × 2 models = 80 queries.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

from factor_leak.constants import FAMOUS_MONTHS  # noqa: E402
from factor_leak.cost import cost_from_usage  # noqa: E402
from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "in_context_contradiction.jsonl"
DATA = REPO / "data" / "ff"
MODELS = ["claude-sonnet-4.6", "claude-opus-4.7"]
INJECTED_VALUE = -42.00  # implausibly extreme; outside typical Mkt-RF range


def stable_prompt(month: str) -> str:
    return (
        "What was the monthly return of the Fama-French market excess "
        f"return (Mkt-RF) factor in {format_month_human(month)}? "
        "Answer with a signed decimal percentage "
        "(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        "and nothing else."
    )


def contradicted_prompt(month: str) -> str:
    return (
        f"My records show that the Fama-French Mkt-RF factor returned "
        f"{INJECTED_VALUE:+.2f}% in {format_month_human(month)}. "
        "What was the monthly return of the Fama-French market excess "
        f"return (Mkt-RF) factor in {format_month_human(month)}? "
        "Answer with a signed decimal percentage "
        "(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        "and nothing else."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-usd", type=float, default=3.0)
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    truth = load_all_factors(DATA)["Mkt-RF"]
    months = list(FAMOUS_MONTHS)  # 20 famous months

    eps = {e.name: e for e in default_endpoints() if e.name in args.models}
    missing = set(args.models) - eps.keys()
    if missing:
        sys.exit(f"missing endpoints {sorted(missing)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    spend = 0.0
    n = 0
    with OUT.open("a") as f:
        for model in args.models:
            ep = eps[model]
            for cond, prompt_fn in (("V_stable", stable_prompt),
                                    ("V_contradicted", contradicted_prompt)):
                for month in months:
                    if spend > args.budget_usd:
                        print(f"  ABORT: ${spend:.3f}"); break
                    prompt = prompt_fn(month)
                    t0 = time.perf_counter()
                    try:
                        resp = ep(prompt, max_tokens=48)
                        text, err = resp.text, None
                        in_tok, out_tok = resp.input_tokens, resp.output_tokens
                    except Exception as e:  # noqa: BLE001
                        text, err = None, f"{type(e).__name__}: {e}"
                        in_tok = out_tok = 0
                    usd = cost_from_usage(model, in_tok, out_tok)
                    spend += usd; n += 1
                    tt = float(truth.get(pd.Period(month, freq="M"), float("nan")))
                    rec = {
                        "model_name": model, "condition": cond, "month": month,
                        "truth": tt, "injected": INJECTED_VALUE,
                        "prompt": prompt, "response": text, "error": err,
                        "input_tokens": in_tok, "output_tokens": out_tok,
                        "latency_s": time.perf_counter() - t0,
                        "usd": usd, "ts": time.time(),
                    }
                    f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"  {model} done; spend ${spend:.3f}, n={n}")

    print(f"\nTOTAL: ${spend:.3f} over {n} queries")

    # Summary
    recs = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(recs)
    df["parsed"] = df["response"].map(lambda r: parse_numeric(r) if r else None)
    df["abs_err_truth"] = (df["parsed"] - df["truth"]).abs()
    df["abs_err_inj"]   = (df["parsed"] - df["injected"]).abs()
    df["matches_truth"] = df["abs_err_truth"] <= 0.50
    df["matches_inj"]   = df["abs_err_inj"]   <= 0.50

    print()
    print(f"{'Model':22s} {'Condition':18s} {'n':>3s} {'parsed':>6s} {'%match_true':>11s} {'%match_inj':>10s} {'mean_err':>9s}")
    for (m, cond), grp in df.groupby(["model_name", "condition"]):
        p = grp.dropna(subset=["parsed", "truth"])
        n_p = len(p)
        if n_p == 0:
            print(f"{m:22s} {cond:18s} {len(grp):>3d} {0:>6d}"); continue
        mt = float(p["matches_truth"].mean())
        mi = float(p["matches_inj"].mean())
        mae = float(p["abs_err_truth"].mean())
        print(f"{m:22s} {cond:18s} {len(grp):>3d} {n_p:>6d} {mt:>11.2f} {mi:>10.2f} {mae:>9.3f}")


if __name__ == "__main__":
    main()
