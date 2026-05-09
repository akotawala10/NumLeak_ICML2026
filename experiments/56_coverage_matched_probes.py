"""EXP-S2: coverage-matched non-FF series recall.

Probes Sonnet 4.6 and Opus 4.7 on three high-public-coverage series
with disjoint training-data path from Fama-French:

  (1) Gold spot price monthly return    (yfinance GC=F)
  (2) WTI crude oil monthly return      (yfinance CL=F)
  (3) US 10-year Treasury yield monthly change in basis points
                                        (yfinance ^TNX, monthly close-to-close)

Variant-A-style prompt: name the series + month, request signed
percentage (or signed bps for 10Y). 30 random months from 2000-08 to
2024-12 (intersection of yfinance histories), seed 2027.

Output: results/coverage_matched.jsonl + console summary.
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
load_dotenv(REPO / ".env", override=True)

from factor_leak.cost import cost_from_usage  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "coverage_matched.jsonl"


def build_truth_gold_oil(ticker: str) -> pd.Series:
    """Monthly close-to-close return in pp."""
    df = yf.download(ticker, start="2000-01-01", end="2025-01-01",
                     progress=False, auto_adjust=False)
    monthly = df["Close"].resample("ME").last().dropna()
    ret = monthly.pct_change().dropna() * 100.0
    # Reindex by YYYY-MM string, force a 1-D Series.
    arr = ret.values.reshape(-1)
    s = pd.Series(arr, index=ret.index.strftime("%Y-%m"))
    return s


def build_truth_tnx() -> pd.Series:
    """Monthly change in 10Y yield, in basis points."""
    df = yf.download("^TNX", start="1985-01-01", end="2025-01-01",
                     progress=False, auto_adjust=False)
    monthly = df["Close"].resample("ME").last().dropna()
    chg_pp = monthly.diff().dropna()
    arr = (chg_pp.values * 100.0).reshape(-1)  # convert pp → bps
    s = pd.Series(arr, index=chg_pp.index.strftime("%Y-%m"))
    return s


PROMPTS = {
    "gold": (
        "What was the monthly price return of spot gold (LBMA London "
        "PM fix or comparable benchmark) in {m}? Answer with a signed "
        "decimal percentage (e.g., -3.12 for a 3.12% loss, +1.50 for a "
        "1.50% gain) and nothing else."
    ),
    "oil": (
        "What was the monthly price return of WTI crude oil (front-month "
        "futures or NYMEX cash-settled spot) in {m}? Answer with a signed "
        "decimal percentage (e.g., -3.12 for a 3.12% loss, +1.50 for a "
        "1.50% gain) and nothing else."
    ),
    "tnx": (
        "What was the monthly change in the U.S. 10-year Treasury "
        "constant-maturity yield in {m}? Answer with a signed integer "
        "in basis points (e.g., -25 for a 25bp decline, +18 for a 18bp "
        "increase) and nothing else."
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-months", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--models", nargs="+",
                    default=["claude-sonnet-4.6", "claude-opus-4.7"])
    ap.add_argument("--start", default="2000-08")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--budget-usd", type=float, default=10.00)
    ap.add_argument("--rpm", type=float, default=60.0)
    args = ap.parse_args()

    print("Loading truth series via yfinance ...", flush=True)
    truth = {
        "gold": build_truth_gold_oil("GC=F"),
        "oil":  build_truth_gold_oil("CL=F"),
        "tnx":  build_truth_tnx(),
    }
    for k, s in truth.items():
        print(f"  {k}: {len(s)} months {s.index[0]}..{s.index[-1]}")

    pool = [m for m in truth["gold"].index
            if args.start <= m <= args.end and m in truth["oil"].index and m in truth["tnx"].index]
    rng = random.Random(args.seed)
    months = sorted(rng.sample(pool, args.n_months))
    print(f"Sampled {len(months)} months: {months[0]}..{months[-1]}")

    eps = {e.name: e for e in default_endpoints() if e.name in args.models}
    missing = set(args.models) - eps.keys()
    if missing:
        sys.exit(f"no endpoint for {sorted(missing)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    spend = 0.0
    n_done = 0
    aborted = False
    min_interval = 60.0 / max(args.rpm, 1.0)
    n_total = len(months) * len(args.models) * len(PROMPTS)
    print(f"Total queries: {n_total}; budget ${args.budget_usd:.2f}")

    with OUT.open("a") as f:
        for series, tmpl in PROMPTS.items():
            if aborted:
                break
            for model in args.models:
                if aborted:
                    break
                ep = eps[model]
                for m in months:
                    tt_raw = truth[series].get(m)
                    if tt_raw is None or pd.isna(tt_raw):
                        continue
                    tt = float(tt_raw)
                    prompt = tmpl.format(m=format_month_human(m))
                    t0 = time.perf_counter()
                    try:
                        resp = ep(prompt, max_tokens=24)
                        text, err = resp.text, None
                        in_tok, out_tok = resp.input_tokens, resp.output_tokens
                    except Exception as e:  # noqa: BLE001
                        text, err = None, f"{type(e).__name__}: {e}"
                        in_tok = out_tok = 0
                    usd = cost_from_usage(model, in_tok, out_tok)
                    spend += usd
                    rec = {
                        "model_name": model, "series": series, "month": m, "truth": tt,
                        "prompt": prompt, "response": text, "error": err,
                        "input_tokens": in_tok, "output_tokens": out_tok,
                        "latency_s": time.perf_counter() - t0, "usd": usd,
                        "ts": time.time(),
                    }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    n_done += 1
                    if n_done % 20 == 0:
                        print(f"  [{n_done}/{n_total}] ${spend:.4f}", flush=True)
                    if spend > args.budget_usd:
                        print(f"  ABORT: ${spend:.4f} > ${args.budget_usd:.2f}")
                        aborted = True
                        break
                    elapsed = time.perf_counter() - t0
                    if min_interval - elapsed > 0:
                        time.sleep(min_interval - elapsed)

    print(f"\nspent ${spend:.4f} over {n_done} queries\n")

    # Summary
    rows = []
    for line in OUT.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error") or rec.get("response") is None:
            rows.append({**rec, "parsed": None})
            continue
        text = rec["response"].strip()
        # 10Y yield: parse signed integer; gold/oil: signed decimal percent
        if rec["series"] == "tnx":
            cleaned = text.replace("bp", "").replace("bps", "").replace("basis points", "").strip().split()[0] if text else ""
            try:
                v = float(cleaned.replace("+", "").replace(",", ""))
            except (ValueError, IndexError):
                v = None
        else:
            v = parse_numeric(text)
        rows.append({**rec, "parsed": v})

    df = pd.DataFrame(rows)
    if df.empty:
        return

    print("=" * 92)
    print(f"{'series':6s} {'model':24s} {'n':>4s} {'parse':>6s} {'r':>8s} {'w25':>6s} {'sign':>6s} {'MAE':>7s}")
    print("=" * 92)
    for (series, model), grp in df.groupby(["series", "model_name"]):
        n = len(grp)
        ok = grp.dropna(subset=["parsed", "truth"])
        np_ = len(ok)
        pr = np_ / n if n else float("nan")
        if np_ >= 3:
            r = float(np.corrcoef(ok["parsed"], ok["truth"])[0, 1])
            w25 = float(((ok["parsed"] - ok["truth"]).abs() <= 0.25).mean())
            nz = ok["truth"] != 0
            sign = float((np.sign(ok["parsed"][nz]) == np.sign(ok["truth"][nz])).mean()) if nz.any() else float("nan")
            mae = float((ok["parsed"] - ok["truth"]).abs().mean())
        else:
            r = w25 = sign = mae = float("nan")
        print(f"{series:6s} {model:24s} {n:>4d} {pr:>6.2f} {r:>8.3f} {w25:>6.3f} {sign:>6.3f} {mae:>7.3f}")


if __name__ == "__main__":
    main()
