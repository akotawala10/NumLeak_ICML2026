"""Run GPT-5.5 on the existing protocol so it appears alongside GPT-5.4
in Tab. 1, App. headline_full, and App. baselines.

Mirrors 16_gpt54_mktrf.py (Mkt-RF + S&P 500 + NASDAQ + blind probes)
and 18_fill_factors.py (5 non-Mkt-RF factors). Same seeds, same prompts,
same answer format, so results are directly comparable to GPT-5.4.

Outputs:
  gpt55_baselines.jsonl  — Mkt-RF, S&P 500, NASDAQ, blind on GPT-5.5
  gpt55_factors.jsonl    — SMB, HML, RMW, CMA, Mom on GPT-5.5
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

from factor_leak.cost import cost_from_usage  # noqa: E402
from factor_leak.ff_loader import load_all_factors, FACTOR_LONG_NAMES  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

RES = REPO / "experiments" / "results"
DATA = REPO / "data" / "ff"
MODEL = "gpt-5.5"
BASELINES_OUT = RES / "gpt55_baselines.jsonl"
FACTORS_OUT = RES / "gpt55_factors.jsonl"

# GPT-5.5 is a reasoning model: it consumes ~200-400 reasoning tokens
# before emitting the visible answer. max_output_tokens must be large
# enough to cover both. 1024 leaves headroom for harder months.
MAX_OUT_TOKENS = 1024


# ---------------------------------------------------------------------------
# Prompt builders (mirrored from 16_gpt54_mktrf.py and 18_fill_factors.py)
# ---------------------------------------------------------------------------


def mktrf_prompt(month: str) -> str:
    return (
        "What was the monthly return of the Fama-French market excess "
        f"return (Mkt-RF) factor in {format_month_human(month)}? "
        "Answer with a signed decimal percentage "
        "(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        "and nothing else."
    )


def snp500_prompt(month: str) -> str:
    return (
        "What was the monthly return of the S&P 500 index in "
        f"{format_month_human(month)}? "
        "Answer with a signed decimal percentage "
        "(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        "and nothing else."
    )


def nasdaq_prompt(month: str) -> str:
    return (
        "What was the monthly return of the NASDAQ Composite index in "
        f"{format_month_human(month)}? "
        "Answer with a signed decimal percentage "
        "(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        "and nothing else."
    )


def blind_prompt(month: str) -> str:
    return (
        "What was the monthly return of the broad U.S. stock market "
        "in excess of the one-month Treasury bill rate, in "
        f"{format_month_human(month)}? "
        "Answer with a signed decimal percentage "
        "(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        "and nothing else."
    )


def factor_prompt(factor: str, month: str) -> str:
    long_name = FACTOR_LONG_NAMES[factor]
    return (
        f"What was the monthly return of the Fama-French {long_name} factor "
        f"in {format_month_human(month)}? "
        "Answer with a signed decimal percentage "
        "(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        "and nothing else."
    )


PROBE_BUILDERS = {
    "mktrf":  mktrf_prompt,
    "snp500": snp500_prompt,
    "nasdaq": nasdaq_prompt,
    "blind":  blind_prompt,
}

FACTORS = ["SMB", "HML", "RMW", "CMA", "Mom"]


# ---------------------------------------------------------------------------
# Truth + sampling
# ---------------------------------------------------------------------------


def truth_for_probe(probe: str, months: list[str]) -> list[float]:
    if probe in ("mktrf", "blind"):
        s = load_all_factors(DATA)["Mkt-RF"]
        return [float(s.get(pd.Period(m, freq="M"), float("nan"))) for m in months]
    if probe == "snp500":
        sp = yf.download("^GSPC", start="1963-07-01", end="2026-03-01",
                         interval="1mo", auto_adjust=False, progress=False)
        close = sp[("Close", "^GSPC")].dropna()
        ret = close.pct_change().dropna() * 100
        ret.index = pd.PeriodIndex(ret.index, freq="M")
        return [float(ret.get(pd.Period(m, freq="M"), float("nan"))) for m in months]
    if probe == "nasdaq":
        nd = yf.download("^IXIC", start="1971-02-01", end="2026-03-01",
                         interval="1mo", auto_adjust=False, progress=False)
        close = nd[("Close", "^IXIC")].dropna()
        ret = close.pct_change().dropna() * 100
        ret.index = pd.PeriodIndex(ret.index, freq="M")
        return [float(ret.get(pd.Period(m, freq="M"), float("nan"))) for m in months]
    raise ValueError(probe)


def sample_baseline_months(probe: str, n: int, seed: int) -> list[str]:
    """Match 16_gpt54_mktrf.py sampling exactly so months align."""
    if probe in ("mktrf", "blind"):
        s = load_all_factors(DATA)["Mkt-RF"].dropna()
        pool = [str(p) for p in s.index if p >= pd.Period("1963-07", freq="M")]
    elif probe == "snp500":
        pool = [f"{y}-{m:02d}" for y in range(1985, 2024) for m in range(1, 13)]
    elif probe == "nasdaq":
        pool = [f"{y}-{m:02d}" for y in range(1971, 2024) for m in range(1, 13)]
    else:
        raise ValueError(probe)
    rng = random.Random(seed)
    return sorted(rng.sample(pool, n))


def sample_factor_months(factor: str, n: int, seed: int) -> list[str]:
    s = load_all_factors(DATA)[factor].dropna()
    pool = [str(p) for p in s.index]
    rng = random.Random(seed)
    return sorted(rng.sample(pool, n))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_query(ep, prompt: str) -> tuple[str | None, str | None, int, int, float]:
    t0 = time.perf_counter()
    try:
        resp = ep(prompt, max_tokens=MAX_OUT_TOKENS)
        return resp.text, None, resp.input_tokens, resp.output_tokens, time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", 0, 0, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-months", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--skip-factors", action="store_true")
    args = ap.parse_args()

    eps = {e.name: e for e in default_endpoints()}
    if MODEL not in eps:
        sys.exit(f"no endpoint for {MODEL!r}")
    ep = eps[MODEL]

    BASELINES_OUT.parent.mkdir(parents=True, exist_ok=True)
    total_spend = 0.0
    n_total = 0

    # ---------- Baselines: Mkt-RF + S&P 500 + NASDAQ + blind ----------
    if not args.skip_baselines:
        with BASELINES_OUT.open("a") as f:
            for probe in ("mktrf", "snp500", "nasdaq", "blind"):
                months = sample_baseline_months(probe, args.n_months, args.seed)
                truths = truth_for_probe(probe, months)
                builder = PROBE_BUILDERS[probe]
                print(f"[{MODEL} × {probe}] {len(months)} queries")
                for month, tt in zip(months, truths):
                    if total_spend > args.budget_usd:
                        print(f"  ABORT: ${total_spend:.3f} > budget"); break
                    prompt = builder(month)
                    text, err, in_tok, out_tok, dt = run_query(ep, prompt)
                    usd = cost_from_usage(MODEL, in_tok, out_tok)
                    total_spend += usd; n_total += 1
                    rec = {
                        "model_name": MODEL, "probe": probe, "month": month,
                        "truth": tt, "prompt": prompt, "response": text,
                        "error": err, "input_tokens": in_tok,
                        "output_tokens": out_tok, "latency_s": dt,
                        "usd": usd, "ts": time.time(),
                    }
                    f.write(json.dumps(rec) + "\n"); f.flush()
                if total_spend > args.budget_usd: break
        print(f"baselines done: ${total_spend:.4f} / {n_total} queries")

    # ---------- Factors: SMB, HML, RMW, CMA, Mom ----------
    if not args.skip_factors and total_spend <= args.budget_usd:
        truth_df = load_all_factors(DATA)
        with FACTORS_OUT.open("a") as f:
            for factor in FACTORS:
                if total_spend > args.budget_usd: break
                months = sample_factor_months(factor, args.n_months, args.seed)
                print(f"[{MODEL} × {factor}] {len(months)} queries")
                for month in months:
                    if total_spend > args.budget_usd:
                        print(f"  ABORT: ${total_spend:.3f} > budget"); break
                    tt = float(truth_df[factor].get(pd.Period(month, freq="M"),
                                                     float("nan")))
                    prompt = factor_prompt(factor, month)
                    text, err, in_tok, out_tok, dt = run_query(ep, prompt)
                    usd = cost_from_usage(MODEL, in_tok, out_tok)
                    total_spend += usd; n_total += 1
                    rec = {
                        "model_name": MODEL, "factor": factor, "month": month,
                        "truth": tt, "prompt": prompt, "response": text,
                        "error": err, "input_tokens": in_tok,
                        "output_tokens": out_tok, "latency_s": dt,
                        "usd": usd, "ts": time.time(),
                    }
                    f.write(json.dumps(rec) + "\n"); f.flush()

    print(f"\nTOTAL: ${total_spend:.4f} over {n_total} queries")

    # ---------- Quick summary ----------
    if BASELINES_OUT.exists():
        recs = [json.loads(l) for l in BASELINES_OUT.read_text().splitlines() if l.strip()]
        recs = [r for r in recs if r["model_name"] == MODEL and not r.get("error")]
        if recs:
            df = pd.DataFrame(recs)
            df["parsed"] = df["response"].map(lambda r: parse_numeric(r) if r else None)
            df["abs_err"] = (df["parsed"] - df["truth"]).abs()
            print("\n=== GPT-5.5 baselines ===")
            for probe, grp in df.groupby("probe"):
                p = grp.dropna(subset=["parsed", "truth"])
                if len(p) < 3: continue
                r = float(np.corrcoef(p["parsed"], p["truth"])[0, 1])
                w25 = (p["abs_err"] <= 0.25).mean()
                sign = (np.sign(p["parsed"]) == np.sign(p["truth"])).mean()
                print(f"  {probe:8s} n={len(p)} r={r:+.3f} w25={w25:.3f} sign={sign:.3f}")
    if FACTORS_OUT.exists():
        recs = [json.loads(l) for l in FACTORS_OUT.read_text().splitlines() if l.strip()]
        recs = [r for r in recs if r["model_name"] == MODEL and not r.get("error")]
        if recs:
            df = pd.DataFrame(recs)
            df["parsed"] = df["response"].map(lambda r: parse_numeric(r) if r else None)
            df["abs_err"] = (df["parsed"] - df["truth"]).abs()
            print("\n=== GPT-5.5 factors ===")
            for factor, grp in df.groupby("factor"):
                p = grp.dropna(subset=["parsed", "truth"])
                if len(p) < 3: continue
                r = float(np.corrcoef(p["parsed"], p["truth"])[0, 1])
                w25 = (p["abs_err"] <= 0.25).mean()
                sign = (np.sign(p["parsed"]) == np.sign(p["truth"])).mean()
                print(f"  {factor:6s} n={len(p)} r={r:+.3f} w25={w25:.3f} sign={sign:.3f}")


if __name__ == "__main__":
    main()
