"""Analyze the date-scramble transmission control.

Joins transmission.jsonl (original sentiment per (model, month)) with
transmission_scrambled.jsonl (sentiment under prompt with month shifted
by +6) and computes three OLS slopes per model:

  beta_T_orig             = sentiment(orig)      ~ truth(orig_month)
  beta_T_scram_to_orig    = sentiment(scrambled) ~ truth(orig_month)
  beta_T_scram_to_shift   = sentiment(scrambled) ~ truth(shift_month)

Plus bootstrap 95% CIs and Pearson r per regression.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.parse import parse_numeric  # noqa: E402

ORIG = REPO / "experiments/results/transmission.jsonl"
SCRAM = REPO / "experiments/results/transmission_scrambled.jsonl"


def parse_sent(text):
    if text is None:
        return None
    v = parse_numeric(text)
    if v is None:
        return None
    if -1.0 <= v <= 1.0:
        return v
    return None


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) < 1e-9:
        return float("nan"), float("nan"), 0
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), int(len(x))


def boot_ci(x, y, n_boot=2000, seed=2026):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 5:
        return (float("nan"), float("nan"))
    sl = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            s, _ = np.polyfit(x[idx], y[idx], 1)
        except Exception:
            s = float("nan")
        sl[i] = s
    sl = sl[~np.isnan(sl)]
    return float(np.quantile(sl, 0.025)), float(np.quantile(sl, 0.975))


def pearson_r(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    orig_recs = [json.loads(l) for l in ORIG.read_text().splitlines() if l.strip()]
    scram_recs = [json.loads(l) for l in SCRAM.read_text().splitlines() if l.strip()]

    # Build orig lookup
    orig_lookup = {}
    for r in orig_recs:
        if r.get("error"):
            continue
        sent = parse_sent(r.get("response"))
        orig_lookup[(r["model_name"], r["month"])] = {
            "truth_orig": r.get("truth_mktrf"),
            "sentiment_orig": sent,
            "recall_orig": r.get("recall_estimate"),
        }

    # Join
    rows = []
    for r in scram_recs:
        if r.get("error"):
            continue
        key = (r["model_name"], r["orig_month"])
        if key not in orig_lookup:
            continue
        o = orig_lookup[key]
        rows.append({
            "model": r["model_name"],
            "orig_month": r["orig_month"],
            "shifted_month": r["shifted_month"],
            "truth_orig": o["truth_orig"],
            "truth_shift": r["truth_shift_mktrf"],
            "sent_orig": o["sentiment_orig"],
            "sent_scram": r.get("sentiment"),
        })
    df = pd.DataFrame(rows)
    print(f"joined records: {len(df)}")
    print(f"per model: {df['model'].value_counts().to_dict()}")
    print()

    # Per-model regressions
    print("=" * 110)
    print(f"{'Model':22s} {'regression':40s} {'n':>4s} {'slope':>8s} {'95% CI':>16s} {'r':>7s}")
    print("=" * 110)
    summary = []
    for model in df["model"].unique():
        sub = df[df["model"] == model].copy()

        # beta_T_orig (recompute from existing data — should match published)
        s, _, n = ols(sub["truth_orig"], sub["sent_orig"])
        lo, hi = boot_ci(sub["truth_orig"], sub["sent_orig"])
        r = pearson_r(sub["truth_orig"], sub["sent_orig"])
        print(f"{model:22s} {'beta_T (sent_orig ~ truth_orig)':40s} {n:>4d} {s:>+8.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {r:>+7.3f}")
        summary.append({"model": model, "regression": "beta_T_orig",
                        "n": n, "slope": s, "ci_lo": lo, "ci_hi": hi, "r": r})

        # beta_T_scram_to_orig: sentiment_scrambled ~ truth at original month
        s, _, n = ols(sub["truth_orig"], sub["sent_scram"])
        lo, hi = boot_ci(sub["truth_orig"], sub["sent_scram"])
        r = pearson_r(sub["truth_orig"], sub["sent_scram"])
        print(f"{model:22s} {'beta_T_scram (sent_scram ~ truth_orig)':40s} {n:>4d} {s:>+8.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {r:>+7.3f}")
        summary.append({"model": model, "regression": "beta_T_scram_to_orig",
                        "n": n, "slope": s, "ci_lo": lo, "ci_hi": hi, "r": r})

        # beta_T_scram_to_shift: sentiment_scrambled ~ truth at shifted month
        s, _, n = ols(sub["truth_shift"], sub["sent_scram"])
        lo, hi = boot_ci(sub["truth_shift"], sub["sent_scram"])
        r = pearson_r(sub["truth_shift"], sub["sent_scram"])
        print(f"{model:22s} {'beta_T_scram_to_shift'+chr(0x0a)+' (sent_scram ~ truth_shift)':40s} {n:>4d} {s:>+8.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {r:>+7.3f}")
        summary.append({"model": model, "regression": "beta_T_scram_to_shift",
                        "n": n, "slope": s, "ci_lo": lo, "ci_hi": hi, "r": r})
        print("-" * 110)

    # Save
    out = REPO / "experiments/results/transmission_scrambled_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {out.relative_to(REPO)}")

    # Quick interpretation
    print()
    print("Interpretation guide:")
    print("  - beta_T_orig: replicates the published transmission slope.")
    print("  - beta_T_scram_to_orig: if it COLLAPSES (CI includes 0), sentiment is")
    print("    month-level date-conditional — sharpens the placebo identification.")
    print("    if it PERSISTS, sentiment is year-level co-occurrence — weaker signal.")
    print("  - beta_T_scram_to_shift: if it MATCHES beta_T_orig, sentiment tracks")
    print("    the prompt's date at month resolution.")


if __name__ == "__main__":
    main()
