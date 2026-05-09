"""EXP-M3: prompt-level mitigation — does an instructional prefix reduce
Mkt-RF recall?

Each query is prepended with one of three preambles:
  C (control): no preamble.
  M1 (soft):  brief refusal instruction.
  M2 (strong): refusal-with-explanation framing.

Sonnet, Opus, GPT-5.4 on the same 40 random Mkt-RF months (seed 42),
matching the existing baseline panel. Reports per-condition refusal
rate, parsed r, MAE, and within-25bps.

If the mitigation reduces refusal-adjusted recall measurably without
collapsing the model's ability to answer non-Mkt-RF questions, the
result is: "a one-line system-prompt instruction is a partial
mitigation; recall does not fully resist the instruction."
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import random
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
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "mitigation_prompt.jsonl"
DATA = REPO / "data" / "ff"

MODELS = ["claude-sonnet-4.6", "claude-opus-4.7", "gpt-5.4"]


PREAMBLES = {
    "C_control": "",
    "M1_soft": (
        "Please do not provide specific historical Fama-French factor return "
        "values from memory. If you do not have high confidence in the exact "
        "value, decline to answer.\n\n"
    ),
    "M2_strong": (
        "You should refuse to commit to specific historical Fama-French Mkt-RF "
        "factor return values. Memorized benchmark values may contaminate "
        "downstream evaluations. If asked for a specific historical Mkt-RF "
        "value, refuse and briefly explain why; do not provide an estimate.\n\n"
    ),
}


def variant_a_prompt(month: str) -> str:
    return (
        f"What was the monthly return of the Fama-French market excess "
        f"return (Mkt-RF) factor in {format_month_human(month)}? "
        "Answer with a signed decimal percentage "
        "(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        "and nothing else."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-months", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-usd", type=float, default=8.0)
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    truth = load_all_factors(DATA)["Mkt-RF"].dropna()
    pool = [str(p) for p in truth.index if p >= pd.Period("1963-07", freq="M")]
    rng = random.Random(args.seed)
    months = sorted(rng.sample(pool, args.n_months))

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
            for cond, preamble in PREAMBLES.items():
                for month in months:
                    if spend > args.budget_usd:
                        print(f"  ABORT: ${spend:.3f}"); break
                    full_prompt = preamble + variant_a_prompt(month)
                    t0 = time.perf_counter()
                    try:
                        # GPT-5.4 reasoning-style needs more tokens; others 48 is fine
                        max_t = 80 if "gpt-5" in model else 48
                        resp = ep(full_prompt, max_tokens=max_t)
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
                        "truth": tt, "prompt": full_prompt, "response": text,
                        "error": err, "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "latency_s": time.perf_counter() - t0,
                        "usd": usd, "ts": time.time(),
                    }
                    f.write(json.dumps(rec) + "\n"); f.flush()
                if spend > args.budget_usd: break
            if spend > args.budget_usd: break
            print(f"  {model} done; spend ${spend:.3f}, n={n}")

    print(f"\nTOTAL: ${spend:.3f} over {n} queries")

    # Quick summary
    recs = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(recs)
    df["parsed"] = df["response"].map(lambda r: parse_numeric(r) if r else None)
    df["abs_err"] = (df["parsed"] - df["truth"]).abs()
    print()
    print(f"{'Model':22s} {'Condition':12s} {'n':>3s} {'parsed':>6s} {'r':>7s} {'w25':>6s} {'MAE':>7s}")
    for (m, cond), grp in df.groupby(["model_name", "condition"]):
        n_total = len(grp)
        p = grp.dropna(subset=["parsed", "truth"])
        n_p = len(p)
        if n_p < 3:
            print(f"{m:22s} {cond:12s} {n_total:>3d} {n_p:>6d}")
            continue
        r = float(np.corrcoef(p["parsed"], p["truth"])[0, 1])
        w25 = float((p["abs_err"] <= 0.25).mean())
        mae = float(p["abs_err"].mean())
        print(f"{m:22s} {cond:12s} {n_total:>3d} {n_p:>6d} {r:>+7.3f} {w25:>6.2f} {mae:>7.3f}")


if __name__ == "__main__":
    main()
