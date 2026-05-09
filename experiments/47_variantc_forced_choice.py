"""Forced-choice Variant-C rerun on the SAME pairs used in the
main sweep, to remove the parse-rate selection-bias concern.

Reviewer concern: Sonnet's Variant-C parse rate on Mkt-RF was
~67% (vs. 88% on Variant A). The 52.9% rank accuracy was computed
on the parsed subset; if the model preferentially refuses on
hard pairs, the headline accuracy is biased upward.

Fix: rerun the same month pairs with a strict forced-choice
prompt that includes the two month strings and asks the model to
return EXACTLY one of them. This drives parse rate near 100%, so
the resulting accuracy is unbiased by selection.

We rerun on Sonnet × Mkt-RF (the cell where rank-value decoupling
is reported) and Opus × Mkt-RF (top-tier sanity check).

Output: results/variantc_forced_choice.jsonl + console summary.
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

from factor_leak.cost import cost_from_usage  # noqa: E402
from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "variantc_forced_choice.jsonl"
DATA = REPO / "data" / "ff"
SWEEP = REPO / "experiments" / "results" / "sweep.jsonl"


def forced_choice_prompt(m1: str, m2: str) -> str:
    s1, s2 = format_month_human(m1), format_month_human(m2)
    return (
        f"Between {s1} and {s2}, which one had the higher monthly return "
        f"for the Fama-French market excess return (Mkt-RF) factor? "
        f"Reply with exactly one of the following two strings, "
        f"with no other text: \"{s1}\" or \"{s2}\". "
        f"You must pick one even if uncertain."
    )


def parse_choice(text: str, m1: str, m2: str) -> str | None:
    """Return 'm1', 'm2', or None."""
    if not text: return None
    t = text.strip().strip('"').strip("'").lower()
    s1 = format_month_human(m1).lower()
    s2 = format_month_human(m2).lower()
    in1 = s1 in t
    in2 = s2 in t
    if in1 and not in2: return "m1"
    if in2 and not in1: return "m2"
    if in1 and in2:
        # Both mentioned — take the LAST one (mirrors endorsement-aware
        # parser's strong-endorse + last-mentioned fallback).
        return "m1" if t.rfind(s1) > t.rfind(s2) else "m2"
    return None


def load_pairs(model: str) -> list[tuple[str, str]]:
    pairs = []
    for line in SWEEP.open():
        r = json.loads(line)
        q = r.get("query") or {}
        if (q.get("model_name") == model
                and q.get("factor") == "Mkt-RF"
                and q.get("variant") == "C"):
            pairs.append((q["month"], q["month2"]))
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["claude-sonnet-4.6", "claude-opus-4.7"])
    ap.add_argument("--budget-usd", type=float, default=2.00)
    ap.add_argument("--rpm", type=float, default=60.0)
    args = ap.parse_args()

    eps = {e.name: e for e in default_endpoints() if e.name in args.models}
    missing = set(args.models) - eps.keys()
    if missing: sys.exit(f"no endpoint for {sorted(missing)}")

    truth = load_all_factors(DATA)["Mkt-RF"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    spend = 0.0
    records: list[dict] = []
    aborted = False
    min_interval = 60.0 / max(args.rpm, 1.0)

    with OUT.open("a") as f:
        for model in args.models:
            if aborted: break
            ep = eps[model]
            pairs = load_pairs(model)
            if not pairs:
                print(f"[{model}] no Variant-C Mkt-RF pairs found in sweep.jsonl")
                continue
            print(f"[{model}] {len(pairs)} pairs")
            for m1, m2 in pairs:
                t1 = float(truth.get(pd.Period(m1, freq="M"), float("nan")))
                t2 = float(truth.get(pd.Period(m2, freq="M"), float("nan")))
                if not (t1 == t1 and t2 == t2):
                    continue
                truth_winner = "m1" if t1 > t2 else "m2"
                prompt = forced_choice_prompt(m1, m2)
                t0 = time.perf_counter()
                try:
                    resp = ep(prompt, max_tokens=32)
                    text, err = resp.text, None
                    in_tok, out_tok = resp.input_tokens, resp.output_tokens
                except Exception as e:  # noqa: BLE001
                    text, err = None, f"{type(e).__name__}: {e}"
                    in_tok = out_tok = 0
                usd = cost_from_usage(model, in_tok, out_tok)
                spend += usd
                choice = parse_choice(text, m1, m2)
                rec = {
                    "model_name": model, "month1": m1, "month2": m2,
                    "truth1": t1, "truth2": t2,
                    "truth_winner": truth_winner, "model_choice": choice,
                    "correct": (choice == truth_winner) if choice else None,
                    "prompt": prompt, "response": text, "error": err,
                    "input_tokens": in_tok, "output_tokens": out_tok,
                    "latency_s": time.perf_counter() - t0, "usd": usd,
                    "ts": time.time(),
                }
                f.write(json.dumps(rec) + "\n")
                records.append(rec)
                if spend > args.budget_usd:
                    print(f"  ABORT: ${spend:.4f} > ${args.budget_usd:.2f}")
                    aborted = True
                    break
                elapsed = time.perf_counter() - t0
                if min_interval - elapsed > 0:
                    time.sleep(min_interval - elapsed)

    print(f"\nspent ${spend:.4f} over {len(records)} queries")

    df = pd.DataFrame(records)
    if df.empty: return

    print("\n=== forced-choice Variant-C summary ===")
    for model, grp in df.groupby("model_name"):
        n = len(grp)
        parsed = grp.dropna(subset=["model_choice"])
        np_ = len(parsed)
        pr = np_ / n if n else float("nan")
        if np_ > 0:
            acc = (parsed["correct"] == True).mean()
            # CI on accuracy
            from math import sqrt
            se = sqrt(acc * (1 - acc) / np_) if np_ else float("nan")
            lo, hi = max(0, acc - 1.96 * se), min(1, acc + 1.96 * se)
        else:
            acc, lo, hi = float("nan"), float("nan"), float("nan")
        print(f"  {model:22s} n={n} parsed={np_} parse_rate={pr:.3f} "
              f"acc={acc:.3f} 95%CI=[{lo:.3f},{hi:.3f}]")


if __name__ == "__main__":
    main()
