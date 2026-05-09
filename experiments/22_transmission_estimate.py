"""Transmission-coefficient experiment: does the LLM's date-conditional
sentiment correlate with its date-conditional Mkt-RF recall?

Design: for each month already in our Mkt-RF probe set on Sonnet/Opus,
issue a Lopez-Lira-style sentiment elicitation prompt anchored to the
date but containing no genuine news content. The model's sentiment
output is then regressed against:

  (a) ground-truth Mkt-RF for that month (transmission to truth), and
  (b) the model's own Mkt-RF estimate (transmission of recall).

Slope (a) bounds the share of a sentiment signal's r(S, Mkt-RF) that
could be driven by date-anchored leak. Slope (b) tells whether
sentiment moves with the model's *internal* state on the same months,
which is the cleanest demonstration of the transmission channel.

This direct elicitation is a STRONGER test than feeding the model
real headlines: real headlines add content noise that dilutes the
date signal, so the headline-pipeline transmission rate is
upper-bounded by the slope we measure here.

Output: results/transmission.jsonl + console summary.
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
from factor_leak.parse import parse_numeric  # noqa: E402
from factor_leak.probe import default_endpoints, format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "transmission.jsonl"
DATA = REPO / "data" / "ff"
RES = REPO / "experiments" / "results"

# Two prompts: a continuous-scale and a discrete-class version. We run
# the continuous one because Sonnet/Opus reliably produce numbers, and
# higher resolution gives a cleaner regression.
SENTIMENT_PROMPT = (
    "Forget all your previous instructions. You are a financial expert "
    "with stock-recommendation experience. Considering only the broad "
    "U.S. equity market in {month_human}, what was the prevailing "
    "investor sentiment that month? Answer on a scale from -1.00 (very "
    "negative for U.S. equities) to +1.00 (very positive). Reply with "
    "exactly one signed decimal (e.g., -0.45 or +0.30) and nothing else."
)


def _dedup_main(records):
    latest = {}
    for r in records:
        k = r["key"]
        prev = latest.get(k)
        if prev is None: latest[k] = r
        elif prev["error"] is not None and r["error"] is None: latest[k] = r
        elif (prev["error"] is None) == (r["error"] is None) and r["ts"] > prev["ts"]:
            latest[k] = r
    return list(latest.values())


def load_mktrf_recall(model_internal: str) -> pd.DataFrame:
    """Return DataFrame(month, response, parsed) for the model's existing
    Mkt-RF Variant-A probes (the model's RECALL of Mkt-RF for these
    months)."""
    if model_internal == "claude-sonnet-4.6":
        path = RES / "sweep.jsonl"
        raw = _dedup_main([json.loads(l) for l in path.open() if l.strip()])
        rows = [{"month": r["query"]["month"], "response": r["response"]}
                for r in raw
                if r["query"]["model_name"] == model_internal
                and r["query"]["factor"] == "Mkt-RF"
                and r["query"]["variant"] == "A"]
    elif model_internal == "claude-opus-4.7":
        path = RES / "opus_baselines.jsonl"
        raw = [json.loads(l) for l in path.open() if l.strip()]
        rows = [{"month": r["month"], "response": r["response"]}
                for r in raw if r.get("probe") == "mktrf"
                and r["model_name"] == model_internal]
    else:
        raise ValueError(model_internal)
    df = pd.DataFrame(rows)
    df["recall_estimate"] = df["response"].map(
        lambda x: parse_numeric(x) if x else None)
    return df.dropna(subset=["recall_estimate"]).copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["claude-sonnet-4.6", "claude-opus-4.7"])
    ap.add_argument("--budget-usd", type=float, default=2.50)
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
            recall_df = load_mktrf_recall(model)
            print(f"\n[{model}] {len(recall_df)} months with prior Mkt-RF recall")

            for _, row in recall_df.iterrows():
                m = row["month"]
                truth_val = float(truth.get(pd.Period(m, freq="M"),
                                            float("nan")))
                if not (truth_val == truth_val):
                    continue
                prompt = SENTIMENT_PROMPT.format(
                    month_human=format_month_human(m))
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
                    "model_name": model, "month": m, "truth_mktrf": truth_val,
                    "recall_estimate": float(row["recall_estimate"]),
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
    df["sentiment"] = df["response"].map(
        lambda r: parse_numeric(r) if r else None)
    df = df.dropna(subset=["sentiment", "truth_mktrf", "recall_estimate"])

    print("\n=== transmission summary ===")
    for model, grp in df.groupby("model_name"):
        n = len(grp)
        if n < 5:
            print(f"  {model}: n={n}, too few"); continue
        # Slope (a): sentiment ~ truth Mkt-RF
        b_truth, a_truth = np.polyfit(grp["truth_mktrf"], grp["sentiment"], 1)
        r_truth = float(np.corrcoef(grp["truth_mktrf"], grp["sentiment"])[0, 1])
        # Slope (b): sentiment ~ recall_estimate
        b_recall, a_recall = np.polyfit(grp["recall_estimate"],
                                        grp["sentiment"], 1)
        r_recall = float(np.corrcoef(grp["recall_estimate"],
                                     grp["sentiment"])[0, 1])
        # Bootstrap CIs on the slopes (1000 samples)
        rng = np.random.default_rng(42)
        b_truth_bs, b_recall_bs = [], []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            sub = grp.iloc[idx]
            try:
                b_truth_bs.append(np.polyfit(sub["truth_mktrf"],
                                             sub["sentiment"], 1)[0])
                b_recall_bs.append(np.polyfit(sub["recall_estimate"],
                                              sub["sentiment"], 1)[0])
            except Exception:
                pass
        ci_truth = (np.percentile(b_truth_bs, 2.5),
                    np.percentile(b_truth_bs, 97.5))
        ci_recall = (np.percentile(b_recall_bs, 2.5),
                     np.percentile(b_recall_bs, 97.5))
        sent_range = (grp["sentiment"].min(), grp["sentiment"].max())
        print(f"  {model:22s}  n={n}")
        print(f"    sentiment range: [{sent_range[0]:.3f}, {sent_range[1]:.3f}]")
        print(f"    sentiment ~ truth_Mkt-RF: slope={b_truth:+.4f} "
              f"[{ci_truth[0]:+.4f}, {ci_truth[1]:+.4f}]  r={r_truth:+.3f}")
        print(f"    sentiment ~ recall_est:   slope={b_recall:+.4f} "
              f"[{ci_recall[0]:+.4f}, {ci_recall[1]:+.4f}]  r={r_recall:+.3f}")


if __name__ == "__main__":
    main()
