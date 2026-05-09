"""EXP-A1: per-cell famous-month vs random-month split.

Re-analysis of existing JSONL records, no new API queries.

For each (model, factor) cell, partition the parsed-month sample into
"famous" (in FAMOUS_MONTHS) and "random" (everything else). Report
Pearson r, within-25bps rate, sign accuracy, MAE on each subset, with
bootstrap 95% CI on r.

Closes Phase-1 concern C3: a reviewer can ask whether high Mkt-RF r
reflects "knows famous-event months" or "knows monthly Mkt-RF". The
random-only split tests selectivity on the harder slice.

Sources (all under experiments/results/):
  multiseed.jsonl       — Sonnet/Haiku across factors, 3 seeds, no truth column
  opus_factors.jsonl    — Opus across factors, with truth column
  more_factors.jsonl    — gpt-5.4-mini across factors, with truth column
  gpt54_factors.jsonl   — gpt-5.4 across factors, with truth column
  llama_factors.jsonl   — llama-3.3-70b across factors, with truth column

Truth for multiseed records is looked up against Ken French monthly factors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.constants import FAMOUS_MONTHS  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402

RESULTS = REPO / "experiments/results"
FF = REPO / "data/ff/F-F_Research_Data_Factors.csv"


# Map factor names that appear in JSONL prompts/records to FF CSV columns.
FACTOR_COLS: dict[str, str] = {
    "Mkt-RF": "Mkt-RF",
    "SMB": "SMB",
    "HML": "HML",
    "RMW": "RMW",
    "CMA": "CMA",
    "Mom": "Mom",
}


def load_ff_truth() -> dict[tuple[str, str], float]:
    """Return mapping (factor, "YYYY-MM") -> truth value (pp)."""
    raw = FF.read_text().splitlines()
    start = next(i for i, ln in enumerate(raw) if "Mkt-RF" in ln)
    header = [c.strip() for c in raw[start].split(",")]
    cols = {name: header.index(name) for name in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"] if name in header}

    truth: dict[tuple[str, str], float] = {}
    for ln in raw[start + 1:]:
        ln = ln.strip()
        if not ln or "," not in ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        try:
            ym = parts[0]
            if len(ym) != 6:
                continue
            month = f"{ym[:4]}-{ym[4:6]}"
            for fac, idx in cols.items():
                try:
                    truth[(fac, month)] = float(parts[idx])
                except (ValueError, IndexError):
                    pass
        except (ValueError, IndexError):
            continue
    # Mom factor is in F-F_Momentum_Factor.csv; load if present.
    mom_path = REPO / "data/ff/F-F_Momentum_Factor.csv"
    if mom_path.exists():
        mraw = mom_path.read_text().splitlines()
        for ln in mraw:
            ln = ln.strip()
            if not ln or "," not in ln:
                continue
            parts = [p.strip() for p in ln.split(",")]
            try:
                ym = parts[0]
                if len(ym) != 6:
                    continue
                month = f"{ym[:4]}-{ym[4:6]}"
                truth[("Mom", month)] = float(parts[1])
            except (ValueError, IndexError):
                continue
    return truth


def load_records() -> pd.DataFrame:
    """Concatenate parseable Variant-A records across all relevant JSONLs."""
    records: list[dict] = []

    # multiseed.jsonl: Sonnet/Haiku, no truth column → backfill from FF.
    path = RESULTS / "multiseed.jsonl"
    if path.exists():
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r.get("error"):
                continue
            records.append({
                "model": r["model_name"],
                "factor": r["factor"],
                "month": r["month"],
                "response": r.get("response"),
                "truth": None,  # filled in later
                "source": "multiseed",
                "seed": r.get("seed"),
            })

    # opus_factors.jsonl, more_factors.jsonl, gpt54_factors.jsonl, llama_factors.jsonl
    for fname in ("opus_factors.jsonl", "more_factors.jsonl",
                  "gpt54_factors.jsonl", "llama_factors.jsonl",
                  "gpt55_factors.jsonl"):
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
            records.append({
                "model": r["model_name"],
                "factor": r["factor"],
                "month": r["month"],
                "response": r.get("response"),
                "truth": r.get("truth"),
                "source": fname.replace(".jsonl", ""),
                "seed": None,
            })

    # *_baselines.jsonl (Mkt-RF only): probe=mktrf → factor=Mkt-RF
    for fname in ("opus_baselines.jsonl", "gpt54_baselines.jsonl",
                  "llama_baselines.jsonl", "gpt55_baselines.jsonl"):
        path = RESULTS / fname
        if not path.exists():
            continue
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            if r.get("error") or r.get("probe") != "mktrf":
                continue
            records.append({
                "model": r["model_name"],
                "factor": "Mkt-RF",
                "month": r["month"],
                "response": r.get("response"),
                "truth": r.get("truth"),
                "source": fname.replace(".jsonl", ""),
                "seed": None,
            })

    # sweep.jsonl: variant A only, nested query; no truth column
    path = RESULTS / "sweep.jsonl"
    if path.exists():
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            q = r.get("query", {})
            if q.get("variant") != "A" or r.get("error"):
                continue
            records.append({
                "model": q.get("model_name"),
                "factor": q.get("factor"),
                "month": q.get("month"),
                "response": r.get("response"),
                "truth": None,
                "source": "sweep",
                "seed": None,
            })

    df = pd.DataFrame(records)
    df["estimate"] = df["response"].apply(parse_numeric)
    return df


def attach_truth(df: pd.DataFrame, truth_lookup: dict) -> pd.DataFrame:
    df = df.copy()
    needs_lookup = df["truth"].isna()
    df.loc[needs_lookup, "truth"] = df.loc[needs_lookup].apply(
        lambda r: truth_lookup.get((r["factor"], r["month"])), axis=1
    )
    return df


def bootstrap_r(pred: np.ndarray, truth: np.ndarray, n_boot: int = 2000,
                seed: int = 2026) -> tuple[float, float, float]:
    """Return (r, lo, hi) — Pearson r and bias-corrected percentile 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(pred)
    if n < 4:
        return float("nan"), float("nan"), float("nan")
    r_obs = float(np.corrcoef(pred, truth)[0, 1])
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        p = pred[idx]
        t = truth[idx]
        if np.std(p) < 1e-12 or np.std(t) < 1e-12:
            rs[i] = float("nan")
        else:
            rs[i] = float(np.corrcoef(p, t)[0, 1])
    rs = rs[~np.isnan(rs)]
    if len(rs) == 0:
        return r_obs, float("nan"), float("nan")
    lo, hi = float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))
    return r_obs, lo, hi


def cell_metrics(sub: pd.DataFrame) -> dict:
    sub = sub.dropna(subset=["estimate", "truth"])
    if len(sub) == 0:
        return {"n": 0}
    pred = sub["estimate"].to_numpy(dtype=float)
    truth = sub["truth"].to_numpy(dtype=float)
    n = len(pred)
    if n < 3:
        return {"n": n, "r": float("nan"), "r_lo": float("nan"), "r_hi": float("nan"),
                "w25": float("nan"), "sign": float("nan"), "mae": float("nan")}
    r, lo, hi = bootstrap_r(pred, truth)
    w25 = float(np.mean(np.abs(pred - truth) <= 0.25))
    nz = truth != 0
    sign = float(np.mean(np.sign(pred[nz]) == np.sign(truth[nz]))) if nz.sum() else float("nan")
    mae = float(np.mean(np.abs(pred - truth)))
    return {"n": n, "r": r, "r_lo": lo, "r_hi": hi, "w25": w25, "sign": sign, "mae": mae}


def main():
    truth_lookup = load_ff_truth()
    print(f"Loaded {len(truth_lookup)} (factor, month) truth values from Ken French")

    df = load_records()
    df = attach_truth(df, truth_lookup)
    df["famous"] = df["month"].isin(FAMOUS_MONTHS)

    # Keep only cells with parsed estimate and known truth.
    parseable = df.dropna(subset=["estimate", "truth"]).copy()
    print(f"Total parseable records: {len(parseable)} "
          f"({len(df) - len(parseable)} dropped for unparseable response or missing truth)")

    # Aggregate to one row per (model, factor, month) by averaging estimates
    # across seeds where multiseed exists. This mirrors how the headline Tab. 1
    # is constructed (3-seed pool for Sonnet/Haiku).
    agg = (parseable.groupby(["model", "factor", "month", "famous"], as_index=False)
                  .agg(estimate=("estimate", "mean"),
                       truth=("truth", "first"),
                       n_seeds=("estimate", "count")))

    print()
    print("=" * 110)
    print(f"{'Model':24s} {'Factor':8s} {'Slice':9s} {'n':>4s} {'r':>7s} {'95% CI':>16s} "
          f"{'w25':>6s} {'sign':>6s} {'MAE':>6s}")
    print("=" * 110)

    target_cells = [
        ("claude-sonnet-4.6", "Mkt-RF"),
        ("claude-opus-4.7", "Mkt-RF"),
        ("claude-haiku-4.5", "Mkt-RF"),
        ("gpt-5.4", "Mkt-RF"),
        ("gpt-5.4-mini", "Mkt-RF"),
        ("gpt-5.5", "Mkt-RF"),
        ("llama-3.3-70b-groq", "Mkt-RF"),
        ("claude-sonnet-4.6", "SMB"),
        ("claude-sonnet-4.6", "HML"),
        ("claude-sonnet-4.6", "RMW"),
        ("claude-sonnet-4.6", "CMA"),
        ("claude-opus-4.7", "SMB"),
        ("claude-opus-4.7", "HML"),
        ("claude-opus-4.7", "RMW"),
        ("claude-opus-4.7", "CMA"),
    ]

    for model, factor in target_cells:
        cell = agg[(agg["model"] == model) & (agg["factor"] == factor)]
        if len(cell) == 0:
            continue
        for slice_name, mask in (("famous", cell["famous"]), ("random", ~cell["famous"]), ("all", np.ones(len(cell), bool))):
            sub = cell[mask]
            m = cell_metrics(sub)
            ci = f"[{m.get('r_lo', float('nan')):+.2f},{m.get('r_hi', float('nan')):+.2f}]" if "r" in m else "—"
            r_str = f"{m['r']:+.3f}" if "r" in m and not np.isnan(m.get("r", float("nan"))) else "—"
            w25 = f"{m['w25']:.3f}" if "w25" in m and not np.isnan(m.get("w25", float("nan"))) else "—"
            sign = f"{m['sign']:.3f}" if "sign" in m and not np.isnan(m.get("sign", float("nan"))) else "—"
            mae = f"{m['mae']:.3f}" if "mae" in m and not np.isnan(m.get("mae", float("nan"))) else "—"
            print(f"{model:24s} {factor:8s} {slice_name:9s} {m['n']:>4d} {r_str:>7s} {ci:>16s} "
                  f"{w25:>6s} {sign:>6s} {mae:>6s}")
        print("-" * 110)

    # Summary table the paper can quote.
    summary_rows = []
    for model, factor in target_cells:
        cell = agg[(agg["model"] == model) & (agg["factor"] == factor)]
        if len(cell) == 0:
            continue
        for slice_name, mask in (("famous", cell["famous"]), ("random", ~cell["famous"])):
            sub = cell[mask]
            m = cell_metrics(sub)
            summary_rows.append({"model": model, "factor": factor, "slice": slice_name, **m})

    out = REPO / "experiments/results/famous_random_split.jsonl"
    with out.open("w") as f:
        for row in summary_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(summary_rows)} summary rows to {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
