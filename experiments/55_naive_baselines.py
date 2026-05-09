"""EXP-S1: naive baselines for Mkt-RF recall.

Three baselines, evaluated on the same months as the headline Sonnet/Opus
Variant-A probes (sourced from `experiments/results/transmission.jsonl`,
which carries `month`, `truth_mktrf`, and `recall_estimate` columns plus
the model's date-only sentiment as `response`).

  (1) Constant long-run mean: predict 1926–most-recent Mkt-RF mean
      for every month.
  (2) Decade mean: predict the matching decade's mean.
  (3) Date-only sentiment rescaled: regress sentiment on truth on a
      held-out 70/30 train/test split, then evaluate test-side r and
      within-25bps. Captures broad date-prior channel without
      specifically requesting a return.

For each baseline plus the model's actual recall, report Pearson r,
within-25bps rate, sign accuracy on the same probed-month set.

No new API queries.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
TX = REPO / "experiments/results/transmission.jsonl"
FF = REPO / "data/ff/F-F_Research_Data_Factors.csv"


def load_ff_mktrf() -> pd.DataFrame:
    """Load monthly Mkt-RF; return DataFrame with date and value (pp)."""
    raw = FF.read_text().splitlines()
    # Find the header (first row containing 'Mkt-RF')
    start = next(i for i, ln in enumerate(raw) if "Mkt-RF" in ln)
    rows = []
    for ln in raw[start + 1:]:
        ln = ln.strip()
        if not ln or "," not in ln:
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 5:
            continue
        try:
            ym = parts[0]
            if len(ym) != 6:
                # annual rows etc — skip
                continue
            year = int(ym[:4])
            month = int(ym[4:6])
            mktrf = float(parts[1])
            rows.append((year, month, mktrf))
        except ValueError:
            continue
    df = pd.DataFrame(rows, columns=["year", "month", "mktrf"])
    df["ym"] = df["year"].astype(str).str.zfill(4) + "-" + df["month"].astype(str).str.zfill(2)
    return df


def parse_sentiment(resp: str | None) -> float | None:
    if resp is None:
        return None
    resp = resp.strip()
    if resp.startswith("+"):
        resp = resp[1:]
    try:
        v = float(resp)
        if -1.0 <= v <= 1.0:
            return v
        return None
    except ValueError:
        return None


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Pearson r, within-25bps rate, sign accuracy on (pred, truth) in pp."""
    r = float(np.corrcoef(pred, truth)[0, 1]) if len(pred) >= 3 else float("nan")
    within = float(np.mean(np.abs(pred - truth) <= 0.25))
    nonzero = truth != 0
    if nonzero.sum() == 0:
        sign = float("nan")
    else:
        sign = float(np.mean(np.sign(pred[nonzero]) == np.sign(truth[nonzero])))
    return {"n": int(len(pred)), "r": r, "within25bps": within, "sign": sign,
            "mae": float(np.mean(np.abs(pred - truth)))}


def main():
    ff = load_ff_mktrf()
    long_run_mean = ff["mktrf"].mean()
    print(f"Kenneth French Mkt-RF (1926+) long-run mean: {long_run_mean:.3f} pp/mo "
          f"({len(ff)} months)")

    # decade mean lookup
    ff["decade"] = (ff["year"] // 10) * 10
    decade_means = ff.groupby("decade")["mktrf"].mean().to_dict()

    # Load probe data: months, truth, recall, sentiment per model
    by_model = {}
    for line in TX.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        m = rec["model_name"]
        ym = rec["month"]
        truth = rec["truth_mktrf"]
        recall = rec.get("recall_estimate")
        sent = parse_sentiment(rec.get("response"))
        by_model.setdefault(m, []).append({"month": ym, "truth": truth, "recall": recall, "sent": sent})

    rng = np.random.default_rng(2026)

    print()
    print("=" * 96)
    print(f"{'Model':22s} {'Predictor':32s} {'n':>4s} {'r':>8s} {'w25':>6s} {'sign':>6s} {'MAE':>6s}")
    print("=" * 96)

    for model, rows in by_model.items():
        rows = [r for r in rows if r["truth"] is not None]
        truth = np.array([r["truth"] for r in rows])
        recall = np.array([r["recall"] for r in rows if r["recall"] is not None])
        n = len(rows)

        # actual recall metrics on cells where recall parsed
        valid_recall = [r for r in rows if r["recall"] is not None]
        if valid_recall:
            t_r = np.array([r["truth"] for r in valid_recall])
            p_r = np.array([r["recall"] for r in valid_recall])
            mr = metrics(p_r, t_r)
            print(f"{model:22s} {'(actual recall)':32s} {mr['n']:>4d} {mr['r']:>8.3f} {mr['within25bps']:>6.3f} {mr['sign']:>6.3f} {mr['mae']:>6.3f}")

        # Baseline 1: constant long-run mean
        pred1 = np.full(n, long_run_mean)
        m1 = metrics(pred1, truth)
        print(f"{model:22s} {'B1 constant long-run mean':32s} {m1['n']:>4d} {m1['r']:>8.3f} {m1['within25bps']:>6.3f} {m1['sign']:>6.3f} {m1['mae']:>6.3f}")

        # Baseline 2: decade mean
        decades = np.array([(int(r["month"][:4]) // 10) * 10 for r in rows])
        pred2 = np.array([decade_means.get(d, long_run_mean) for d in decades])
        m2 = metrics(pred2, truth)
        print(f"{model:22s} {'B2 decade mean':32s} {m2['n']:>4d} {m2['r']:>8.3f} {m2['within25bps']:>6.3f} {m2['sign']:>6.3f} {m2['mae']:>6.3f}")

        # Baseline 3: date-only sentiment rescaled via 70/30 train/test split
        sent_rows = [r for r in rows if r["sent"] is not None]
        if len(sent_rows) >= 10:
            X = np.array([r["sent"] for r in sent_rows])
            y = np.array([r["truth"] for r in sent_rows])

            # repeat 100 random splits for stability
            r_test = []
            within_test = []
            sign_test = []
            mae_test = []
            for seed in range(100):
                idx = rng.permutation(len(X))
                cut = int(0.7 * len(X))
                tr_i, te_i = idx[:cut], idx[cut:]
                # OLS: y = a + b * X on training side
                a = np.polyfit(X[tr_i], y[tr_i], 1)
                pred = a[0] * X[te_i] + a[1]
                truth_te = y[te_i]
                if len(pred) >= 3:
                    r_test.append(float(np.corrcoef(pred, truth_te)[0, 1]))
                    within_test.append(float(np.mean(np.abs(pred - truth_te) <= 0.25)))
                    nz = truth_te != 0
                    if nz.sum() > 0:
                        sign_test.append(float(np.mean(np.sign(pred[nz]) == np.sign(truth_te[nz]))))
                    mae_test.append(float(np.mean(np.abs(pred - truth_te))))
            print(f"{model:22s} {'B3 sentiment-rescaled (test)':32s} {len(X):>4d} "
                  f"{np.mean(r_test):>8.3f} {np.mean(within_test):>6.3f} "
                  f"{np.mean(sign_test):>6.3f} {np.mean(mae_test):>6.3f}")

        print("-" * 96)


if __name__ == "__main__":
    main()
