"""Recent-release / post-existence holdout for Mkt-RF recall.

Compares Opus 4.7 and Sonnet 4.6 on:
  - Pre-cutoff sample: existing 40-month Variant-A historical Mkt-RF
    results (re-used from opus_baselines.jsonl / Sonnet main sweep).
  - Post-cutoff sample: 14 months from 2025-01 onward (post most
    plausible model training cutoffs).

Saves raw + processed; emits a small LaTeX table.

We frame this as a "post-existence holdout" rather than a definitive
training-cutoff probe: we do not claim a specific cutoff date, only that
the post-cutoff months are unlikely to have appeared in training.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

from factor_leak.cost import cost_from_usage  # noqa: E402
from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

DATA = REPO / "data" / "ff"
OUT_RAW = REPO / "experiments" / "results" / "post_cutoff_holdout.jsonl"
OUT_CSV = REPO / "experiments" / "results" / "post_cutoff_holdout.csv"
OUT_TEX = REPO / "tables" / "post_cutoff_holdout.tex"

MODELS = ["claude-opus-4.7", "claude-sonnet-4.6"]
POST_CUTOFF_FROM = "2025-01"


def variant_a_prompt(month: str) -> str:
    return (
        f"What was the monthly return of the Fama-French market excess "
        f"return (Mkt-RF) factor in {format_month_human(month)}? "
        "Answer with a signed decimal percentage "
        "(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        "and nothing else."
    )


def pearson(xs, ys):
    n = len(xs)
    if n < 3: return float("nan")
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    if dx == 0 or dy == 0: return float("nan")
    return num/(dx*dy)


def load_historical_records():
    """Re-use existing historical Variant-A Mkt-RF runs.
    Looks up truth from FF data on the fly when missing in the JSONL."""
    truth_series = load_all_factors(DATA)["Mkt-RF"].dropna()
    def truth_at(month):
        try:
            return float(truth_series.get(pd.Period(month, freq="M"), float("nan")))
        except Exception:
            return float("nan")
    out = {}
    files = [
        ("opus_baselines.jsonl", None),  # historical Opus
        ("multiseed.jsonl", None),        # historical Sonnet (and Haiku)
    ]
    for fname, _ in files:
        p = REPO / "experiments" / "results" / fname
        if not p.exists(): continue
        for ln in p.open():
            ln = ln.strip()
            if not ln: continue
            r = json.loads(ln)
            mn = r.get("model_name")
            if mn not in MODELS: continue
            probe = r.get("probe") or r.get("factor", "")
            if probe not in ("mktrf", "Mkt-RF"): continue
            month = r.get("month"); resp = r.get("response")
            if not month: continue
            if pd.Period(month, freq="M") >= pd.Period(POST_CUTOFF_FROM, freq="M"):
                continue  # exclude post-cutoff
            tt = r.get("truth")
            if tt is None:
                tt = truth_at(month)
            if math.isnan(tt): continue
            parsed = parse_numeric(resp)
            out.setdefault(mn, []).append((month, float(tt), parsed))
    return out


def run_post_cutoff(args):
    truth = load_all_factors(DATA)["Mkt-RF"].dropna()
    post = [str(p) for p in truth.index
            if p >= pd.Period(POST_CUTOFF_FROM, freq="M")]
    print(f"post-cutoff months: {len(post)} (from {post[0]} to {post[-1]})")

    eps = {e.name: e for e in default_endpoints()}
    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    spend = 0.0
    fh = OUT_RAW.open("a")
    for model in MODELS:
        if model not in eps:
            print(f"WARN: {model} endpoint missing; skipping post-cutoff run")
            continue
        ep = eps[model]
        for month in post:
            prompt = variant_a_prompt(month)
            t0 = time.perf_counter()
            try:
                resp = ep(prompt, max_tokens=64)
                text, err = resp.text, None
                in_tok, out_tok = resp.input_tokens, resp.output_tokens
            except Exception as e:  # noqa: BLE001
                text, err = None, f"{type(e).__name__}: {e}"
                in_tok = out_tok = 0
            usd = cost_from_usage(model, in_tok, out_tok)
            spend += usd
            tt = float(truth.get(pd.Period(month, freq="M"), float("nan")))
            rec = {
                "split": "post_cutoff", "model_name": model, "month": month,
                "truth": tt, "prompt": prompt, "response": text, "error": err,
                "input_tokens": in_tok, "output_tokens": out_tok,
                "latency_s": time.perf_counter() - t0, "usd": usd,
                "ts": time.time(),
            }
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            print(f"  [{model}] {month}: parsed={parse_numeric(text)}, "
                  f"truth={tt:+.2f}, raw={text!r}", flush=True)
    fh.close()
    print(f"\npost-cutoff spend: ${spend:.3f}")


def aggregate():
    """Compute per-(model, split) metrics from historical + post-cutoff."""
    hist = load_historical_records()
    # post-cutoff records from new run
    post = {m: [] for m in MODELS}
    if OUT_RAW.exists():
        for ln in OUT_RAW.open():
            r = json.loads(ln)
            if r.get("split") != "post_cutoff": continue
            mn = r["model_name"]
            if mn not in MODELS: continue
            parsed = parse_numeric(r.get("response"))
            post[mn].append((r["month"], float(r["truth"]) if r.get("truth") is not None else float("nan"),
                             parsed))

    rows = []
    for model in MODELS:
        for split, recs in [("pre-cutoff (1985--2024)", hist.get(model, [])),
                            ("post-cutoff (2025--2026)", post.get(model, []))]:
            n = len(recs)
            n_parsed = sum(1 for _, _, p in recs if p is not None)
            parse_rate = n_parsed / n if n else float("nan")
            # Pearson r and MAE on parsed-truth pairs
            pairs = [(p, t) for _, t, p in recs if p is not None and not math.isnan(t)]
            if len(pairs) >= 3:
                xs = [p for p, _ in pairs]; ys = [t for _, t in pairs]
                r = pearson(xs, ys)
                mae = sum(abs(x-y) for x, y in pairs) / len(pairs)
                w25 = sum(1 for x, y in pairs if abs(x-y) <= 0.25) / len(pairs)
            else:
                r, mae, w25 = float("nan"), float("nan"), float("nan")
            rows.append({"model": model, "split": split, "n": n,
                         "parse_rate": parse_rate, "pearson_r": r,
                         "mae": mae, "within_25bp": w25})
            print(f"  {model:22s} {split:30s}  n={n:3d} parsed={parse_rate:.2f} "
                  f"r={r:+.3f}  mae={mae:.2f}  w25={w25:.2f}")

    # CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows: w.writerow(row)
    print(f"wrote {OUT_CSV.relative_to(REPO)}")

    # LaTeX table
    def fmt(x, fm="{:.2f}"):
        return "--" if (x is None or (isinstance(x, float) and math.isnan(x))) else fm.format(x)
    lines = [
        r"% Auto-generated by experiments/73_post_cutoff_holdout.py",
        r"\begin{table}[t]", r"\centering",
        r"\caption{\textbf{Recent-release / post-existence holdout.} ",
        r"Mkt-RF Variant-A recall on the original $1985$--$2024$ historical ",
        r"sample versus the $14$ months from 2025 onward, which were ",
        r"unlikely to appear in the model's training data. Both splits use ",
        r"the same prompt template. Refusal/non-parse on the recent-release ",
        r"split is the calibrated outcome; commitment to a value is ",
        r"fabrication unless $r$ is similar to the historical split.}",
        r"\label{tab:post_cutoff_holdout}", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Model & Split & $n$ & Parse & $r$ & MAE & w-25 \\", r"\midrule",
    ]
    last_model = None
    for row in rows:
        m = row["model"].replace("claude-", "").replace("-", " ")
        if last_model and last_model != row["model"]:
            lines.append(r"\midrule")
        lines.append(
            f"{m if last_model != row['model'] else ''} & "
            f"{row['split']} & {row['n']} & "
            f"{fmt(row['parse_rate'])} & {fmt(row['pearson_r'], '{:+.2f}')} & "
            f"{fmt(row['mae'])} & {fmt(row['within_25bp'])} \\\\"
        )
        last_model = row["model"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_TEX.relative_to(REPO)}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="run the 28 post-cutoff API queries")
    ap.add_argument("--analyze", action="store_true",
                    help="aggregate + emit table")
    args = ap.parse_args()
    if args.run:
        run_post_cutoff(args)
    if args.analyze:
        aggregate()
    if not (args.run or args.analyze):
        ap.error("pass --run and/or --analyze")


if __name__ == "__main__":
    main()
