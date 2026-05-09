"""Positive numerical-recall baseline — S&P 500 monthly price returns.

Reviewer concern: "Sonnet might recall ANY prominent financial series,
not specifically Fama-French. Show that."

Probes Sonnet 4.6 and Haiku 4.5 on S&P 500 monthly returns using the same
Variant-A numeric-answer template. Ground truth from Yahoo Finance (^GSPC)
monthly close-to-close arithmetic returns in percent. Sample = 40 random
months drawn from the 1985-02 to 2026-02 window (^GSPC coverage on YF).

Output: results/snp500_baseline.jsonl + console summary.
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
import yfinance as yf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import (  # noqa: E402
    default_endpoints, format_month_human,
)

OUT = REPO / "experiments" / "results" / "snp500_baseline.jsonl"
MODELS = ["claude-sonnet-4.6", "claude-haiku-4.5"]


def snp500_prompt(month: str) -> str:
    return (
        f"What was the monthly return of the S&P 500 index in "
        f"{format_month_human(month)}? "
        f"Answer with a signed decimal percentage "
        f"(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        f"and nothing else."
    )


def load_truth() -> pd.Series:
    sp = yf.download("^GSPC", start="1963-07-01", end="2026-03-01",
                     interval="1mo", auto_adjust=False, progress=False)
    close = sp[("Close", "^GSPC")].dropna()
    ret = close.pct_change().dropna() * 100
    ret.index = pd.PeriodIndex(ret.index, freq="M")
    return ret


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-months", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    truth = load_truth()
    rng = random.Random(args.seed)
    months = sorted(rng.sample(sorted(str(p) for p in truth.index),
                               args.n_months))
    print(f"S&P500 probe: {len(months)} months × {len(args.models)} models "
          f"= {len(months)*len(args.models)} queries")

    endpoints = {e.name: e for e in default_endpoints() if e.name in args.models}
    missing = set(args.models) - endpoints.keys()
    if missing:
        sys.exit(f"no endpoint for {sorted(missing)} (check .env)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with OUT.open("a") as f:
        for model in args.models:
            ep = endpoints[model]
            for m in months:
                prompt = snp500_prompt(m)
                t0 = time.perf_counter()
                try:
                    resp = ep(prompt, max_tokens=48)
                    text = resp.text
                    err = None
                    in_tok = resp.input_tokens
                    out_tok = resp.output_tokens
                except Exception as e:  # noqa: BLE001
                    text = None
                    err = f"{type(e).__name__}: {e}"
                    in_tok = out_tok = 0
                rec = {
                    "model_name": model, "month": m, "prompt": prompt,
                    "response": text, "error": err,
                    "input_tokens": in_tok, "output_tokens": out_tok,
                    "latency_s": time.perf_counter() - t0,
                    "ts": time.time(),
                }
                f.write(json.dumps(rec) + "\n")
                records.append(rec)

    # Summary
    df = pd.DataFrame(records)
    df["parsed"] = df["response"].map(lambda r: parse_numeric(r) if r else None)
    df["truth"] = df["month"].map(
        lambda m: float(truth.get(pd.Period(m, freq="M"), float("nan")))
    )
    df["abs_err"] = (df["parsed"] - df["truth"]).abs()
    print("\n=== S&P500 baseline summary ===")
    for model, grp in df.groupby("model_name"):
        n = len(grp)
        parsed = grp.dropna(subset=["parsed", "truth"])
        n_parsed = len(parsed)
        parse_rate = n_parsed / n if n else float("nan")
        w25 = (parsed["abs_err"] <= 0.25).mean() if n_parsed else float("nan")
        w100 = (parsed["abs_err"] <= 1.0).mean() if n_parsed else float("nan")
        if n_parsed >= 3:
            r = float(np.corrcoef(parsed["parsed"], parsed["truth"])[0, 1])
        else:
            r = float("nan")
        sign = ((np.sign(parsed["parsed"]) == np.sign(parsed["truth"])).mean()
                if n_parsed else float("nan"))
        print(f"  {model}: n={n} parsed={n_parsed} parse_rate={parse_rate:.2f} "
              f"r={r:+.3f} within_25bps={w25:.3f} within_100bps={w100:.3f} "
              f"sign={sign:.3f}")


if __name__ == "__main__":
    main()
