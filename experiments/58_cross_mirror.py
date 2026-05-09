"""EXP-R1: cross-mirror replication for Mkt-RF.

Tests truth-source canonicality (Phase-1 concern C4): does the model recall
Kenneth French Mkt-RF specifically, or any equity-excess mirror that happens
to track within 25 bps?

Mirrors compared:
  K  Kenneth French Research Data Factors Mkt-RF (paper's primary truth).
  K5 Kenneth French 5-Factor Mkt-RF (different file, different construction
     window for the recent extensions).
  S  S&P 500 total return (^SP500TR) - FF risk-free.
  W  Dow Wilshire 5000 total return (^DWCF) - FF risk-free.

For each (model, month) cell with a parsed Mkt-RF estimate from existing
JSONL, score the model's recall against each mirror via Pearson r, MAE, and
within-25bps rate. Differences in MAE across mirrors identify the closest
fingerprint.

No new API queries to LLMs. Yahoo Finance is queried once for the index
mirrors and cached locally.
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

RESULTS = REPO / "experiments/results"
FF3 = REPO / "data/ff/F-F_Research_Data_Factors.csv"
FF5 = REPO / "data/ff/F-F_Research_Data_5_Factors_2x3.csv"
CACHE = REPO / "experiments/results/cross_mirror_cache.csv"


def parse_ff_csv(path: Path, mktrf_col: str = "Mkt-RF",
                 rf_col: str = "RF") -> pd.DataFrame:
    """Return DataFrame with columns [month, mktrf, rf] (pp)."""
    raw = path.read_text().splitlines()
    start = next(i for i, ln in enumerate(raw) if mktrf_col in ln)
    header = [c.strip() for c in raw[start].split(",")]
    mi, ri = header.index(mktrf_col), header.index(rf_col)
    rows = []
    for ln in raw[start + 1:]:
        ln = ln.strip()
        if not ln or "," not in ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) <= max(mi, ri):
            continue
        try:
            ym = parts[0]
            if len(ym) != 6:
                continue
            month = f"{ym[:4]}-{ym[4:6]}"
            mktrf = float(parts[mi])
            rf = float(parts[ri])
            rows.append({"month": month, "mktrf": mktrf, "rf": rf})
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows)


def fetch_index_monthly(ticker: str) -> pd.DataFrame:
    """Return DataFrame [month, ret_pp] of total-return monthly close-to-close
    pct change in percentage points. Cached after first download."""
    import yfinance as yf
    print(f"  fetching {ticker} monthly returns from Yahoo …")
    t = yf.Ticker(ticker)
    h = t.history(period="max", interval="1mo", auto_adjust=True)
    if h.empty:
        return pd.DataFrame(columns=["month", "ret_pp"])
    h = h.reset_index()
    # h["Date"] is timezone-aware; collapse to YYYY-MM
    h["month"] = h["Date"].dt.strftime("%Y-%m")
    h = h.dropna(subset=["Close"]).copy()
    h["ret_pp"] = h["Close"].pct_change() * 100.0
    return h[["month", "ret_pp"]].dropna()


def build_mirror_panel(ff3: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame keyed by month with one column per mirror (Mkt-RF in pp)."""
    if CACHE.exists():
        cached = pd.read_csv(CACHE)
        print(f"Loaded cached mirror panel: {CACHE.relative_to(REPO)} ({len(cached)} months)")
        return cached

    ff5 = parse_ff_csv(FF5)
    panel = ff3.rename(columns={"mktrf": "K", "rf": "rf"})[["month", "K", "rf"]]
    panel = panel.merge(
        ff5[["month", "mktrf"]].rename(columns={"mktrf": "K5"}),
        on="month", how="left",
    )

    # S&P 500 Total Return (^SP500TR) — coverage starts 1988-01.
    sp = fetch_index_monthly("^SP500TR")
    sp = sp.rename(columns={"ret_pp": "sp_ret_pp"})
    panel = panel.merge(sp, on="month", how="left")
    panel["S"] = panel["sp_ret_pp"] - panel["rf"]

    # Wilshire 5000 Total (^W5000) — coverage starts 1971-01 but yfinance has spotty data.
    w = fetch_index_monthly("^W5000")
    w = w.rename(columns={"ret_pp": "w_ret_pp"})
    panel = panel.merge(w, on="month", how="left")
    panel["W"] = panel["w_ret_pp"] - panel["rf"]

    panel = panel[["month", "K", "K5", "S", "W"]].copy()
    panel.to_csv(CACHE, index=False)
    print(f"Wrote mirror panel cache: {CACHE.relative_to(REPO)} ({len(panel)} months)")
    return panel


def load_recall() -> pd.DataFrame:
    """Pull all parseable Mkt-RF estimates from existing JSONL sources."""
    records = []
    for fname in ("multiseed.jsonl", "opus_baselines.jsonl",
                  "gpt54_baselines.jsonl", "llama_baselines.jsonl"):
        path = RESULTS / fname
        if not path.exists():
            continue
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r.get("error"):
                continue
            factor = r.get("factor")
            probe = r.get("probe")
            if factor != "Mkt-RF" and probe != "mktrf":
                continue
            est = parse_numeric(r.get("response"))
            if est is None:
                continue
            records.append({
                "model": r["model_name"],
                "month": r["month"],
                "estimate": est,
            })
    # sweep.jsonl variant A Mkt-RF
    path = RESULTS / "sweep.jsonl"
    if path.exists():
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            q = r.get("query", {})
            if q.get("factor") != "Mkt-RF" or q.get("variant") != "A" or r.get("error"):
                continue
            est = parse_numeric(r.get("response"))
            if est is None:
                continue
            records.append({
                "model": q["model_name"],
                "month": q["month"],
                "estimate": est,
            })
    df = pd.DataFrame(records)
    # Pool across seeds for stability.
    return (df.groupby(["model", "month"], as_index=False)
              .agg(estimate=("estimate", "mean"),
                   n_seeds=("estimate", "count")))


def score(estimates: np.ndarray, truth: np.ndarray) -> dict:
    if len(estimates) < 3 or np.std(truth) < 1e-6:
        return {"n": len(estimates), "r": float("nan"),
                "mae": float("nan"), "w25": float("nan")}
    r = float(np.corrcoef(estimates, truth)[0, 1])
    mae = float(np.mean(np.abs(estimates - truth)))
    w25 = float(np.mean(np.abs(estimates - truth) <= 0.25))
    return {"n": len(estimates), "r": r, "mae": mae, "w25": w25}


def main():
    ff3 = parse_ff_csv(FF3)
    panel = build_mirror_panel(ff3)
    panel = panel.set_index("month")
    print()
    print("Mirror coverage (months with non-null value):")
    for col in ("K", "K5", "S", "W"):
        n = panel[col].notna().sum()
        first = panel[panel[col].notna()].index.min() if n else "—"
        last = panel[panel[col].notna()].index.max() if n else "—"
        print(f"  {col}: n={n:5d}  {first}..{last}")

    # Cross-mirror agreement on overlapping months
    print()
    print("Cross-mirror agreement (Pearson r and MAE pp):")
    cols = ["K", "K5", "S", "W"]
    print(f"      {'  '.join(f'{c:>10s}' for c in cols)}")
    for a in cols:
        row = [a]
        for b in cols:
            sub = panel[[a, b]].dropna()
            if len(sub) < 5:
                row.append("       —  ")
            else:
                r = float(np.corrcoef(sub[a], sub[b])[0, 1])
                mae = float(np.mean(np.abs(sub[a] - sub[b])))
                row.append(f"r={r:+.3f}/{mae:.2f}")
        print("  ".join(f"{x:>10s}" for x in row))

    recall = load_recall()
    print()
    print(f"Loaded {len(recall)} pooled recall (model, month) cells")

    # Score each model against each mirror — full and overlap-only.
    print()
    print("=" * 96)
    print(f"{'Model':24s} {'Mirror':6s} {'mode':9s} {'n':>4s} {'r':>8s} {'MAE':>8s} {'w25':>8s}")
    print("=" * 96)
    out_rows = []
    mirrors = ["K", "K5", "S", "W"]
    for model in sorted(recall["model"].unique()):
        sub = recall[recall["model"] == model].set_index("month")
        # Overlap of months where ALL mirrors have non-null values AND model has a recall.
        overlap_months = sub.index.intersection(panel.dropna(subset=mirrors).index)
        for mirror in mirrors:
            # full: largest available overlap with this mirror alone
            full = sub[["estimate"]].join(panel[[mirror]], how="inner").dropna()
            mf = score(full["estimate"].to_numpy(), full[mirror].to_numpy())
            out_rows.append({"model": model, "mirror": mirror, "mode": "full", **mf})
            print(f"{model:24s} {mirror:6s} {'full':9s} {mf['n']:>4d} "
                  f"{mf['r']:>+8.3f} {mf['mae']:>8.3f} {mf['w25']:>8.3f}")
            # overlap: only months covered by all mirrors (apples-to-apples)
            ovl = sub.loc[sub.index.intersection(overlap_months), ["estimate"]].join(
                panel.loc[overlap_months, [mirror]], how="inner").dropna()
            mo = score(ovl["estimate"].to_numpy(), ovl[mirror].to_numpy())
            out_rows.append({"model": model, "mirror": mirror, "mode": "overlap", **mo})
            print(f"{model:24s} {mirror:6s} {'overlap':9s} {mo['n']:>4d} "
                  f"{mo['r']:>+8.3f} {mo['mae']:>8.3f} {mo['w25']:>8.3f}")
        print("-" * 96)

    out = REPO / "experiments/results/cross_mirror_score.jsonl"
    with out.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(out_rows)} score rows to {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
