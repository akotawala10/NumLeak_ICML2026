"""Probe gpt-5.4-mini, gpt-5.4-nano (Azure Cognitive Services Responses
API) and DeepSeek-V3.2 (Azure AI Foundry) on Mkt-RF + S&P 500 + NASDAQ
+ blind probes.

Companion to experiments/19_llama_baselines.py — fills in the OpenAI
mid/small tier (mini, nano) so we have a 3-tier capability curve on
GPT to mirror Anthropic Opus/Sonnet/Haiku and Llama 70B/8B, and adds
DeepSeek as a 5th provider for cross-provider breadth.

Output: results/more_baselines.jsonl + console summary.
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
from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "more_baselines.jsonl"
DATA = REPO / "data" / "ff"
MODELS = ["gpt-5.4-mini", "gpt-5.4-nano", "deepseek-v3.2-azure"]


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


PROBE_BUILDERS = {
    "mktrf":  mktrf_prompt,
    "snp500": snp500_prompt,
    "nasdaq": nasdaq_prompt,
    "blind":  blind_prompt,
}


def truth_for_probe(probe: str, months: list[str]) -> list[float]:
    if probe in ("mktrf", "blind"):
        s = load_all_factors(DATA)["Mkt-RF"]
        return [float(s.get(pd.Period(m, freq="M"), float("nan"))) for m in months]
    if probe == "snp500":
        sp = yf.download("^GSPC", start="1963-07-01", end="2026-03-01",
                         interval="1mo", auto_adjust=False, progress=False)
        ret = sp[("Close", "^GSPC")].dropna().pct_change().dropna() * 100
        ret.index = pd.PeriodIndex(ret.index, freq="M")
        return [float(ret.get(pd.Period(m, freq="M"), float("nan"))) for m in months]
    if probe == "nasdaq":
        nd = yf.download("^IXIC", start="1971-02-01", end="2026-03-01",
                         interval="1mo", auto_adjust=False, progress=False)
        ret = nd[("Close", "^IXIC")].dropna().pct_change().dropna() * 100
        ret.index = pd.PeriodIndex(ret.index, freq="M")
        return [float(ret.get(pd.Period(m, freq="M"), float("nan"))) for m in months]
    raise ValueError(probe)


def sample_months(probe: str, n: int, seed: int) -> list[str]:
    if probe in ("mktrf", "blind"):
        s = load_all_factors(DATA)["Mkt-RF"].dropna()
        pool = [str(p) for p in s.index if p >= pd.Period("1963-07", freq="M")]
    elif probe == "snp500":
        sp = yf.download("^GSPC", start="1963-07-01", end="2026-03-01",
                         interval="1mo", auto_adjust=False, progress=False)
        ret = sp[("Close", "^GSPC")].dropna().pct_change().dropna()
        ret.index = pd.PeriodIndex(ret.index, freq="M")
        pool = [str(p) for p in ret.index]
    elif probe == "nasdaq":
        nd = yf.download("^IXIC", start="1971-02-01", end="2026-03-01",
                         interval="1mo", auto_adjust=False, progress=False)
        ret = nd[("Close", "^IXIC")].dropna().pct_change().dropna()
        ret.index = pd.PeriodIndex(ret.index, freq="M")
        pool = [str(p) for p in ret.index]
    else:
        raise ValueError(probe)
    rng = random.Random(seed)
    return sorted(rng.sample(pool, n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-months", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--probes", nargs="+",
                    default=["mktrf", "snp500", "nasdaq", "blind"])
    ap.add_argument("--budget-usd", type=float, default=2.00,
                    help="Hard cap on cumulative spend.")
    ap.add_argument("--rpm", type=float, default=60.0,
                    help="Soft requests-per-minute cap (Azure tier limits).")
    args = ap.parse_args()

    eps = {e.name: e for e in default_endpoints() if e.name in args.models}
    missing = set(args.models) - eps.keys()
    if missing:
        sys.exit(f"no endpoint for {sorted(missing)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    spend = 0.0
    records: list[dict] = []
    aborted = False
    min_interval = 60.0 / max(args.rpm, 1.0)

    with OUT.open("a") as f:
        for model in args.models:
            if aborted: break
            ep = eps[model]
            for probe in args.probes:
                if aborted: break
                months = sample_months(probe, args.n_months, args.seed)
                truths = truth_for_probe(probe, months)
                builder = PROBE_BUILDERS[probe]
                print(f"[{model} × {probe}] {len(months)} queries")
                for m, truth in zip(months, truths):
                    t0 = time.perf_counter()
                    prompt = builder(m)
                    try:
                        resp = ep(prompt, max_tokens=48)
                        text, err = resp.text, None
                        in_tok, out_tok = resp.input_tokens, resp.output_tokens
                    except Exception as e:  # noqa: BLE001
                        text, err = None, f"{type(e).__name__}: {e}"
                        in_tok = out_tok = 0
                    usd = cost_from_usage(model, in_tok, out_tok)
                    spend += usd
                    rec = {
                        "model_name": model, "probe": probe, "month": m,
                        "truth": truth, "prompt": prompt,
                        "response": text, "error": err,
                        "input_tokens": in_tok, "output_tokens": out_tok,
                        "latency_s": time.perf_counter() - t0,
                        "usd": usd, "ts": time.time(),
                    }
                    f.write(json.dumps(rec) + "\n")
                    records.append(rec)
                    if spend > args.budget_usd:
                        print(f"  ABORT: ${spend:.4f} > ${args.budget_usd:.2f}")
                        aborted = True
                        break
                    elapsed = time.perf_counter() - t0
                    sleep_for = min_interval - elapsed
                    if sleep_for > 0:
                        time.sleep(sleep_for)

    print(f"\nspent ${spend:.4f} over {len(records)} queries")

    df = pd.DataFrame(records)
    if df.empty: return
    df["parsed"] = df["response"].map(lambda r: parse_numeric(r) if r else None)
    df["abs_err"] = (df["parsed"] - df["truth"]).abs()
    print("\n=== summary ===")
    for (m, p), grp in df.groupby(["model_name", "probe"]):
        n = len(grp)
        ok = grp.dropna(subset=["parsed", "truth"])
        nok = len(ok)
        pr = nok / n if n else float("nan")
        w25 = (ok["abs_err"] <= 0.25).mean() if nok else float("nan")
        w100 = (ok["abs_err"] <= 1.0).mean() if nok else float("nan")
        r = float(np.corrcoef(ok["parsed"], ok["truth"])[0, 1]) if nok >= 3 else float("nan")
        sgn = ((np.sign(ok["parsed"]) == np.sign(ok["truth"])).mean()
               if nok else float("nan"))
        print(f"  {m:22s} × {p:7s}: n={n} parsed={nok} parse={pr:.2f} "
              f"r={r:+.3f} w25={w25:.3f} w100={w100:.3f} sign={sgn:.3f}")


if __name__ == "__main__":
    main()
