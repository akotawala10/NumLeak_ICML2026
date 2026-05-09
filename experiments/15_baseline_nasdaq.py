"""Secondary positive-recall baseline — NASDAQ Composite monthly returns.

S&P 500 turned out to be r=0.99 correlated with Mkt-RF on the sample
months, so it mainly tested label invariance. NASDAQ Composite is
still equity but tech-heavy; its correlation with Mkt-RF is closer to
0.85-0.9 over the available window. A genuinely distinct series from
Mkt-RF.
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
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "nasdaq_baseline.jsonl"


def nasdaq_prompt(month: str) -> str:
    return (
        f"What was the monthly return of the NASDAQ Composite index in "
        f"{format_month_human(month)}? "
        f"Answer with a signed decimal percentage "
        f"(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        f"and nothing else."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-months", type=int, default=40)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--models", nargs="+",
                    default=["claude-sonnet-4.6", "claude-haiku-4.5"])
    args = ap.parse_args()

    nd = yf.download("^IXIC", start="1971-02-01", end="2026-03-01",
                     interval="1mo", auto_adjust=False, progress=False)
    close = nd[("Close", "^IXIC")].dropna()
    ret = close.pct_change().dropna() * 100
    ret.index = pd.PeriodIndex(ret.index, freq="M")

    rng = random.Random(args.seed)
    months = sorted(rng.sample(sorted(str(p) for p in ret.index), args.n_months))
    print(f"NASDAQ probe: {len(months)} months × {len(args.models)} models "
          f"= {len(months)*len(args.models)} queries")

    endpoints = {e.name: e for e in default_endpoints() if e.name in args.models}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    with OUT.open("a") as f:
        for model in args.models:
            ep = endpoints[model]
            for m in months:
                prompt = nasdaq_prompt(m)
                t0 = time.perf_counter()
                try:
                    resp = ep(prompt, max_tokens=48)
                    text, err = resp.text, None
                    in_tok, out_tok = resp.input_tokens, resp.output_tokens
                except Exception as e:  # noqa: BLE001
                    text, err = None, f"{type(e).__name__}: {e}"
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

    df = pd.DataFrame(records)
    df["parsed"] = df["response"].map(lambda r: parse_numeric(r) if r else None)
    df["truth"] = df["month"].map(
        lambda m: float(ret.get(pd.Period(m, freq="M"), float("nan")))
    )
    df["abs_err"] = (df["parsed"] - df["truth"]).abs()

    # Also compute corr between NASDAQ truth and Mkt-RF truth on these months
    from factor_leak.ff_loader import load_all_factors
    mkt_rf = load_all_factors(REPO / "data" / "ff")["Mkt-RF"]
    mkt_vals = [float(mkt_rf.get(pd.Period(m, freq="M"), float("nan"))) for m in months]
    nd_vals = [float(ret.get(pd.Period(m, freq="M"), float("nan"))) for m in months]
    ok = [(a, b) for a, b in zip(mkt_vals, nd_vals) if not (np.isnan(a) or np.isnan(b))]
    if len(ok) >= 3:
        a = np.array([x[0] for x in ok]); b = np.array([x[1] for x in ok])
        r_series = float(np.corrcoef(a, b)[0, 1])
        print(f"\ncorr(NASDAQ, Mkt-RF) on sample: r={r_series:.3f}, n={len(ok)}")

    print("\n=== NASDAQ baseline summary ===")
    for model, grp in df.groupby("model_name"):
        n = len(grp)
        p = grp.dropna(subset=["parsed", "truth"])
        n_p = len(p)
        parse_rate = n_p / n if n else float("nan")
        w25 = (p["abs_err"] <= 0.25).mean() if n_p else float("nan")
        w100 = (p["abs_err"] <= 1.0).mean() if n_p else float("nan")
        r = float(np.corrcoef(p["parsed"], p["truth"])[0, 1]) if n_p >= 3 else float("nan")
        sign = ((np.sign(p["parsed"]) == np.sign(p["truth"])).mean()
                if n_p else float("nan"))
        print(f"  {model}: n={n} parsed={n_p} parse_rate={parse_rate:.2f} "
              f"r={r:+.3f} within_25bps={w25:.3f} within_100bps={w100:.3f} "
              f"sign={sign:.3f}")


if __name__ == "__main__":
    main()
