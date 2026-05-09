"""Residualization variant of the forensic ceiling: a point estimate
of leak-attributable signal share when (Ŝ_t, r̂_t, r_t) are
co-located on the same months.

Setup. The forensic bound (Eq. 1) saturates at α_paper whenever
|ρ_recall| > |ρ(Ŝ, r_FF)|. When the auditor can also obtain the
model's recall estimate r̂_t on the same months as the published
signal Ŝ_t and the realized factor r_t, a tighter point-estimate
is available: regress Ŝ on r̂, take the residual u_t, and compare
ρ(u_t, r_t) against ρ(Ŝ_t, r_t).

Worked example: §4.1 transmission data on Sonnet (n=77) and Opus
(n=40) gives all three columns directly. Treat the model's sentiment
estimate as Ŝ_t.

Output: console table with the residualization decomposition.
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

DATA = REPO / "experiments" / "results" / "transmission.jsonl"


def load() -> pd.DataFrame:
    rows = []
    with DATA.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
    df = pd.DataFrame(rows)
    df["sentiment"] = df["response"].map(
        lambda x: parse_numeric(x) if x else None)
    df = df.dropna(subset=["sentiment", "truth_mktrf", "recall_estimate"])
    return df


def residualize(model_df: pd.DataFrame) -> dict:
    s = model_df["sentiment"].astype(float).values   # Ŝ
    r_hat = model_df["recall_estimate"].astype(float).values  # recall
    r = model_df["truth_mktrf"].astype(float).values  # truth

    # Step 1: γ = cov(Ŝ,r̂)/var(r̂)
    gamma = float(np.cov(s, r_hat, ddof=0)[0, 1] / np.var(r_hat))
    u = s - gamma * r_hat

    # Step 2: correlations
    rho_S_r = float(np.corrcoef(s, r)[0, 1])
    rho_u_r = float(np.corrcoef(u, r)[0, 1])
    rho_Shat = float(np.corrcoef(s, r_hat)[0, 1])

    # Step 3: leak-attributable signal share
    leak_share = 1.0 - (rho_u_r ** 2) / (rho_S_r ** 2) if rho_S_r != 0 else float("nan")

    return {
        "n": len(model_df),
        "gamma": gamma,
        "rho(S,r_hat)": rho_Shat,
        "rho(S,r_FF)": rho_S_r,
        "rho(u,r_FF)": rho_u_r,
        "leak_share": leak_share,
    }


def main() -> None:
    df = load()
    print(f"loaded {len(df)} (model, month) rows; "
          f"models = {sorted(df['model_name'].unique())}")
    print()
    print(f"{'model':22s}  {'n':>3s}  {'γ':>7s}  "
          f"{'ρ(S,r̂)':>8s}  {'ρ(S,r)':>8s}  {'ρ(u,r)':>8s}  "
          f"{'leak_share':>10s}")
    print("-" * 76)
    for model, g in df.groupby("model_name"):
        d = residualize(g)
        print(f"{model:22s}  {d['n']:>3d}  "
              f"{d['gamma']:>+7.4f}  "
              f"{d['rho(S,r_hat)']:>+8.3f}  "
              f"{d['rho(S,r_FF)']:>+8.3f}  "
              f"{d['rho(u,r_FF)']:>+8.3f}  "
              f"{d['leak_share']:>10.3f}")


if __name__ == "__main__":
    main()
