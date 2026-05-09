"""Controls that anchor the main recall claim (handoff §3.7).

- ``chance_rate_permutation`` — Control 1: permutation baseline. If the
  model's answers were drawn from some month-agnostic prior, what fraction
  would accidentally land within the tolerance band? We compute this from
  the main sweep by shuffling the ``(month, estimate)`` pairing within each
  (model, factor) group and recomputing the within-tolerance rate, averaged
  over many permutations.
- ``chance_rate_cross_factor`` — alternative Control 1: shuffle estimates
  *across* factors within a model. This is a stronger null because it
  samples from the full answer distribution the model produces for
  factor-return-style questions, not just the estimates for one factor.
- ``synthetic_factor_prompt`` / ``illiquid_fund_prompt`` — Controls 2 & 3:
  probes for factor-like series the LLM cannot plausibly know. Applying
  the same variant-A parser, a healthy model should mostly decline to
  answer; if it readily returns numbers, our recall signal is polluted by
  general numeric hallucination.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .probe import format_month_human


# ---------------------------------------------------------------------------
# Control 1 — permutation chance-rate baseline
# ---------------------------------------------------------------------------


def _within_tol_under_permutation(
    est: np.ndarray,
    truth: np.ndarray,
    tol: float,
    rng: np.random.Generator,
    n_iters: int,
) -> float:
    hits = 0.0
    for _ in range(n_iters):
        perm = rng.permutation(est)
        hits += float(np.mean(np.abs(perm - truth) < tol))
    return hits / n_iters


def chance_rate_permutation(
    df: pd.DataFrame,
    tol_bps: int = 25,
    n_iters: int = 1000,
    seed: int = 0,
    by: list[str] | None = None,
) -> pd.DataFrame:
    """Within-group permutation null: shuffle (month, estimate) and recompute.

    Returns a DataFrame with one row per group (as specified by ``by``),
    plus a ``chance_rate`` column and the sample size ``n``. Groups smaller
    than 2 observations get ``NaN``.
    """
    col_needed = ["parsed_estimate", "truth", "variant"]
    for c in col_needed:
        if c not in df.columns:
            raise KeyError(f"{c!r} missing — did you call enrich()?")

    sub = df[df["variant"].isin(["A", "B"])].dropna(
        subset=["parsed_estimate", "truth"]
    ).copy()
    if sub.empty:
        return pd.DataFrame(columns=(by or []) + ["chance_rate", "n"])

    group_cols = by or []
    tol = tol_bps / 100.0
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    grouper = sub.groupby(group_cols) if group_cols else [((), sub)]

    for key, frame in grouper:
        est = frame["parsed_estimate"].to_numpy()
        truth = frame["truth"].to_numpy()
        if len(est) < 2:
            rate = float("nan")
        else:
            rate = _within_tol_under_permutation(est, truth, tol, rng, n_iters)
        rec = (
            dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
            if group_cols else {}
        )
        rec["chance_rate"] = rate
        rec["n"] = int(len(est))
        rows.append(rec)
    return pd.DataFrame(rows)


def chance_rate_cross_factor(
    df: pd.DataFrame,
    tol_bps: int = 25,
    n_iters: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Cross-factor null: shuffle estimates across *all factors* within a model.

    Under the null of "model answers are drawn from a single month-and-factor
    agnostic prior", this is the strongest baseline — it prevents the
    permutation pool from being too small (which inflates within-group
    chance rates on small groups).

    Returns one row per (model_name, factor), with the observed within-tol
    rate on that (model, factor) vs. the cross-factor-permuted rate.
    """
    sub = df[df["variant"].isin(["A", "B"])].dropna(
        subset=["parsed_estimate", "truth"]
    ).copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["model_name", "factor", "observed_rate", "chance_rate", "n"]
        )

    tol = tol_bps / 100.0
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for model, model_frame in sub.groupby("model_name"):
        pool_est = model_frame["parsed_estimate"].to_numpy()
        for factor, ff_frame in model_frame.groupby("factor"):
            truth = ff_frame["truth"].to_numpy()
            est = ff_frame["parsed_estimate"].to_numpy()
            observed = float(np.mean(np.abs(est - truth) < tol))
            if len(pool_est) < 2 or len(truth) == 0:
                chance = float("nan")
            else:
                hits = 0.0
                for _ in range(n_iters):
                    draw = rng.choice(pool_est, size=len(truth), replace=True)
                    hits += float(np.mean(np.abs(draw - truth) < tol))
                chance = hits / n_iters
            rows.append({
                "model_name": model,
                "factor": factor,
                "observed_rate": observed,
                "chance_rate": chance,
                "n": int(len(truth)),
            })
    return pd.DataFrame(rows)


def chance_rate_factor_shuffle(
    df: pd.DataFrame,
    tol_bps: int = 25,
    n_iters: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Factor-shuffle-within-month null (Paper's Control~1).

    Holds ``month`` fixed and permutes the factor identity. For each
    (model, target_factor, month) cell, the null asks: if we compared the
    target factor's truth against the model's estimate for some *other*
    factor at the same month, how often would we land within tolerance?

    This defeats the failure mode where a model always predicts ~0 for
    some factor (e.g., HML). Under that failure mode, within-group
    permutation and cross-factor shuffles both show near-observed chance
    rates, falsely "validating" the signal. Factor-shuffle, by contrast,
    reveals it: the model's Mkt-RF estimate for month M is not ~0, so
    pairing it with HML truth for month M gives a wildly different
    within-tol rate than the observed always-0 HML estimate.

    Returns one row per (model, factor) with ``observed_rate``,
    ``chance_rate`` (averaged across shuffles), and ``n``.
    """
    sub = df[df["variant"] == "A"].dropna(
        subset=["parsed_estimate", "truth"]
    ).copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["model_name", "factor", "observed_rate",
                     "chance_rate", "n"]
        )

    tol = tol_bps / 100.0
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for model, model_frame in sub.groupby("model_name"):
        # Wide table: rows are months, columns are factors, values are
        # the model's estimate. Aggregate duplicates (same variant-A asked
        # multiple times) by mean.
        wide_est = model_frame.pivot_table(
            index="month", columns="factor",
            values="parsed_estimate", aggfunc="mean",
        )
        wide_truth = model_frame.pivot_table(
            index="month", columns="factor",
            values="truth", aggfunc="mean",
        )
        factors = list(wide_est.columns)
        if len(factors) < 2:
            continue

        for target in factors:
            alt_factors = [f for f in factors if f != target]
            truth_col = wide_truth[target]

            truth_arr = truth_col.to_numpy()
            observed_arr = wide_est[target].to_numpy()
            obs_mask = ~np.isnan(truth_arr) & ~np.isnan(observed_arr)
            if obs_mask.sum() == 0:
                continue
            observed = float(np.mean(
                np.abs(observed_arr[obs_mask] - truth_arr[obs_mask]) < tol
            ))

            hits = 0.0
            iters_used = 0
            for _ in range(n_iters):
                # For each month, pick one alt factor's estimate.
                alt_idx = rng.integers(0, len(alt_factors), size=len(wide_est))
                alt_est = np.full(len(wide_est), np.nan)
                for i, j in enumerate(alt_idx):
                    alt_est[i] = wide_est[alt_factors[j]].iloc[i]
                mask = ~np.isnan(truth_arr) & ~np.isnan(alt_est)
                if mask.sum() == 0:
                    continue
                hits += float(np.mean(np.abs(alt_est[mask] - truth_arr[mask]) < tol))
                iters_used += 1
            chance = hits / iters_used if iters_used else float("nan")
            rows.append({
                "model_name": model,
                "factor": target,
                "observed_rate": observed,
                "chance_rate": chance,
                "n": int(obs_mask.sum()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Controls 2 & 3 — prompts for series the LLM cannot plausibly know
# ---------------------------------------------------------------------------


# A synthetic factor name chosen to (a) sound like a real factor-model paper
# citation and (b) almost certainly not appear in any training corpus.
SYNTHETIC_FACTOR_NAME = "Gleason-Zeta volatility-conditioned residual"

# A fabricated illiquid private-equity fund. Name collision with any real
# fund is unlikely; the paper notes this in a footnote.
ILLIQUID_FUND_NAME = "Holbrooke-Mansfield Opportunity Fund III (2007 vintage)"


def synthetic_factor_prompt(month: str | pd.Period) -> str:
    """Control 2 prompt (variant A form, synthetic factor)."""
    return (
        f"What was the monthly return of the {SYNTHETIC_FACTOR_NAME} factor "
        f"in {format_month_human(month)}? "
        f"Answer with a signed decimal percentage "
        f"(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        f"and nothing else."
    )


def illiquid_fund_prompt(month: str | pd.Period) -> str:
    """Control 3 prompt (illiquid private fund)."""
    return (
        f"What was the monthly net return of the {ILLIQUID_FUND_NAME} "
        f"in {format_month_human(month)}? "
        f"Answer with a signed decimal percentage "
        f"(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        f"and nothing else."
    )


def control_prompt(kind: str, month: str | pd.Period) -> str:
    if kind == "synthetic_factor":
        return synthetic_factor_prompt(month)
    if kind == "illiquid_fund":
        return illiquid_fund_prompt(month)
    raise ValueError(f"unknown control kind: {kind!r}")
