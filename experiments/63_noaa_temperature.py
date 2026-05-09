"""EXP-D1: cross-domain replication on NOAA monthly average contiguous US
temperature.

Tests whether the recall channel extends beyond finance to a clearly
different public numeric series. Truth is sourced from NOAA NCEI Climate
at a Glance (national tavg time series), a widely-mirrored climate
record that is independent of any FF / equity training data.

Sample: 30 random months 1980-2023 (seed 2027).
Models: Sonnet 4.6, Opus 4.7, GPT-5.4.

A model that has memorized "the kind of numeric public data we study"
should also recall this series; a model whose channel is
finance-specific should not.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

from factor_leak.cost import cost_from_usage  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "noaa_temperature.jsonl"
TRUTH_CSV = REPO / "data" / "noaa_us_tavg.csv"
MODELS = ["claude-sonnet-4.6", "claude-opus-4.7", "gpt-5.4"]

# NOAA NCEI Climate at a Glance — Contiguous US (110), monthly avg temp,
# 1895-present. JSON time-series; we ask for a wide range so the truth
# table covers the full sample window.
NOAA_URL = (
    "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
    "national/time-series/110/tavg/all/1/1980-2024/data.json"
)


def load_truth() -> pd.Series:
    """Return monthly avg US tavg in degrees F, indexed by YYYY-MM string."""
    if TRUTH_CSV.exists():
        df = pd.read_csv(TRUTH_CSV)
        return pd.Series(df["tavg_f"].values, index=df["month"].values)
    print(f"fetching NOAA truth from {NOAA_URL} ...")
    r = requests.get(NOAA_URL, timeout=30)
    r.raise_for_status()
    js = r.json()
    # NOAA returns {"data": {"YYYYMM": {"value": "47.5", "anomaly": "..."}, ...}}
    rows = []
    for ym, payload in js["data"].items():
        try:
            tavg = float(payload["value"])
        except (KeyError, ValueError):
            continue
        # ym is "YYYYMM" (no separator)
        month_str = f"{ym[:4]}-{ym[4:6]}"
        rows.append({"month": month_str, "tavg_f": tavg})
    df = pd.DataFrame(rows).sort_values("month")
    TRUTH_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRUTH_CSV, index=False)
    print(f"wrote {len(df)} rows to {TRUTH_CSV.relative_to(REPO)}")
    return pd.Series(df["tavg_f"].values, index=df["month"].values)


def temp_prompt(month: str) -> str:
    return (
        "What was the average temperature for the contiguous United States "
        f"in {format_month_human(month)}? Answer with a single signed "
        "decimal in degrees Fahrenheit (e.g., 32.5) and nothing else."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-months", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--budget-usd", type=float, default=3.0)
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    truth = load_truth()
    pool = [m for m in truth.index if "1980-01" <= m <= "2023-12"]
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
            for month in months:
                if spend > args.budget_usd:
                    print(f"  ABORT: ${spend:.3f}"); break
                tt = float(truth[month])
                prompt = temp_prompt(month)
                t0 = time.perf_counter()
                try:
                    max_t = 80 if "gpt-5" in model else 32
                    resp = ep(prompt, max_tokens=max_t)
                    text, err = resp.text, None
                    in_tok, out_tok = resp.input_tokens, resp.output_tokens
                except Exception as e:  # noqa: BLE001
                    text, err = None, f"{type(e).__name__}: {e}"
                    in_tok = out_tok = 0
                usd = cost_from_usage(model, in_tok, out_tok)
                spend += usd; n += 1
                rec = {
                    "model_name": model, "month": month, "truth_f": tt,
                    "prompt": prompt, "response": text, "error": err,
                    "input_tokens": in_tok, "output_tokens": out_tok,
                    "latency_s": time.perf_counter() - t0,
                    "usd": usd, "ts": time.time(),
                }
                f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"  {model} done; spend ${spend:.3f}, n={n}")

    print(f"\nTOTAL: ${spend:.3f} over {n} queries")

    recs = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(recs)
    df["parsed"] = df["response"].map(lambda r: parse_numeric(r) if r else None)
    df["abs_err"] = (df["parsed"] - df["truth_f"]).abs()
    print()
    print(f"{'Model':22s} {'n':>3s} {'parsed':>6s} {'r':>7s} {'w0.5degF':>9s} {'MAE_F':>7s}")
    for m, grp in df.groupby("model_name"):
        n_total = len(grp)
        p = grp.dropna(subset=["parsed", "truth_f"])
        n_p = len(p)
        if n_p < 3:
            print(f"{m:22s} {n_total:>3d} {n_p:>6d}")
            continue
        r = float(np.corrcoef(p["parsed"], p["truth_f"])[0, 1])
        w05 = float((p["abs_err"] <= 0.5).mean())
        mae = float(p["abs_err"].mean())
        print(f"{m:22s} {n_total:>3d} {n_p:>6d} {r:>+7.3f} {w05:>9.2f} {mae:>7.3f}")


if __name__ == "__main__":
    main()
