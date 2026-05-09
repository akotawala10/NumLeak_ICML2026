"""Metrics for the factor-recall sweep (handoff §3.5).

Given a JSONL of ``ProbeResult`` records and the Ken French ground-truth
library, build an *enriched* long DataFrame and compute:

- ``exact_match_rate(df, tol_bps)`` — P(|r_hat − r| < tol).
- ``directional_accuracy(df)``       — P(sign match) on nonzero truth.
- ``calibration(df)``                — Pearson r between r_hat and r.
- ``temporal_gradient(df, ...)``     — OLS slope of recall on year, per group.
- ``famous_concentration(df, ...)``  — recall on famous vs. random months.
- ``comparative_accuracy(df, ...)``  — accuracy on variant C (ranking task).

All percent-scale numbers are in units of ``%`` (Ken French convention); the
``tol_bps`` argument converts bps → %-points internally (1 bps = 0.01 %).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import MODEL_CUTOFFS, cutoff_bucket, months_to_cutoff
from .ff_loader import load_all_factors
from .parse import parse_comparative, parse_numeric


# ---------------------------------------------------------------------------
# Loading and enrichment
# ---------------------------------------------------------------------------


def load_results(jsonl_path: Path | str) -> pd.DataFrame:
    """Load a probe-sweep JSONL into a flat DataFrame.

    Each row has ``[model_name, factor, variant, month, month2, prompt,
    response, error, latency_s, ts, key]``.
    """
    path = Path(jsonl_path)
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = rec["query"]
            rows.append({
                "model_name": q["model_name"],
                "factor": q["factor"],
                "variant": q["variant"],
                "month": q["month"],
                "month2": q.get("month2"),
                "prompt": rec["prompt"],
                "response": rec["response"],
                "error": rec["error"],
                "latency_s": rec["latency_s"],
                "ts": rec["ts"],
                "key": rec["key"],
            })
    return pd.DataFrame(rows)


def enrich(df: pd.DataFrame, data_dir: Path | str) -> pd.DataFrame:
    """Add ground truth, parsed estimate, and per-row derived columns.

    New columns:
        truth            — Ken French monthly return in % for (factor, month)
        truth2           — same for (factor, month2) on variant C (NaN otherwise)
        parsed_estimate  — parse_numeric(response) on A/B (NaN on C)
        parsed_month     — parse_comparative(response) on C (NaN on A/B)
        abs_error        — |parsed_estimate - truth| on A/B (NaN otherwise)
        dir_correct      — sign(parsed_estimate) == sign(truth) on A/B
        within_5bps / within_10bps / within_25bps — |err| < tol on A/B
        year             — int from ``month``
        comparative_correct — did the model pick the month with the higher true
                              return? (True/False on variant C; NaN otherwise)
    Rows with API errors are kept but all derived stats are NaN.
    """
    wide = load_all_factors(data_dir)

    out = df.copy()

    def _lookup(factor: str, month: str | None) -> float:
        if month is None:
            return np.nan
        try:
            v = wide.loc[pd.Period(month, freq="M"), factor]
        except KeyError:
            return np.nan
        return float(v) if pd.notna(v) else np.nan

    out["truth"] = [_lookup(f, m) for f, m in zip(out["factor"], out["month"])]
    out["truth2"] = [_lookup(f, m) for f, m in zip(out["factor"], out["month2"])]

    def _parse_num(row) -> float:
        if row["error"] is not None:
            return np.nan
        if row["variant"] == "C":
            return np.nan
        v = parse_numeric(row["response"])
        return np.nan if v is None else float(v)

    def _parse_month(row) -> str | float:
        if row["error"] is not None or row["variant"] != "C":
            return np.nan
        v = parse_comparative(row["response"], row["month"], row["month2"])
        return v if v is not None else np.nan

    out["parsed_estimate"] = out.apply(_parse_num, axis=1)
    out["parsed_month"] = out.apply(_parse_month, axis=1)
    out["abs_error"] = (out["parsed_estimate"] - out["truth"]).abs()
    # Directional accuracy: require truth != 0 (np.sign(0)==0 would penalize
    # a correctly-signed estimate) and estimate != 0 for the same reason.
    ab_mask = out["variant"].isin(["A", "B"])
    nonzero = out["truth"].notna() & (out["truth"] != 0.0)
    valid_dir = ab_mask & nonzero & out["parsed_estimate"].notna()
    out["dir_correct"] = np.where(
        valid_dir,
        np.sign(out["parsed_estimate"]) == np.sign(out["truth"]),
        np.nan,
    )

    for tol_bps in (5, 10, 25):
        col = f"within_{tol_bps}bps"
        out[col] = np.where(
            out["abs_error"].notna(),
            out["abs_error"] < tol_bps / 100.0,
            np.nan,
        )

    out["year"] = pd.PeriodIndex(out["month"], freq="M").year

    def _cutoff_bucket(row) -> str:
        try:
            return cutoff_bucket(row["model_name"], row["month"])
        except KeyError:
            return "unknown"

    def _months_to_cutoff(row) -> float:
        try:
            return float(months_to_cutoff(row["model_name"], row["month"]))
        except KeyError:
            return float("nan")

    out["bucket"] = out.apply(_cutoff_bucket, axis=1)
    out["months_to_cutoff"] = out.apply(_months_to_cutoff, axis=1)

    # Variant C accuracy: which month is truly higher?
    def _comp_correct(row) -> float:
        if row["variant"] != "C":
            return np.nan
        if row["error"] is not None or pd.isna(row["parsed_month"]):
            return np.nan
        if pd.isna(row["truth"]) or pd.isna(row["truth2"]):
            return np.nan
        # Guard floating-point ties (FF returns are 2dp so exact equality is
        # fine in practice, but 1e-9 margin is cheap insurance).
        if abs(row["truth"] - row["truth2"]) < 1e-9:
            return np.nan
        true_higher = row["month"] if row["truth"] > row["truth2"] else row["month2"]
        return row["parsed_month"] == true_higher

    out["comparative_correct"] = out.apply(_comp_correct, axis=1)
    return out


# ---------------------------------------------------------------------------
# Scalar / grouped metrics
# ---------------------------------------------------------------------------


def _rate(series: pd.Series) -> float:
    """Mean of a boolean-ish series, ignoring NaN. Returns NaN on empty."""
    s = series.dropna().astype(float)
    return float(s.mean()) if len(s) else float("nan")


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def wilson_interval(
    k: int, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Wilson-score two-sided CI for a proportion ``k / n``.

    Preferred over the normal approximation because coverage remains
    near-nominal for extreme ``p`` and small ``n`` — both of which arise
    here (within-5bps rate on tiny samples can be close to 0).

    Returns ``(lower, upper)`` in ``[0, 1]``. With ``n == 0`` returns
    ``(nan, nan)``.
    """
    if n <= 0:
        return float("nan"), float("nan")
    # Standard normal quantile for two-sided alpha without scipy dep.
    # For alpha = 0.05, z ≈ 1.96.
    z = _normal_quantile(1.0 - alpha / 2.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    # Guarantee CI contains the point estimate even under float round-off
    # (at p=1 the upper bound should be exactly 1 but arithmetic yields
    # 0.9999999999999999).
    lower = min(lower, p)
    upper = max(upper, p)
    return lower, upper


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF without scipy.

    Uses the lazily-imported scipy if available, else Beasley-Springer
    approximation which is accurate to ~1e-4 in the bulk.
    """
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except ImportError:  # pragma: no cover
        # Beasley-Springer-Moro-ish approximation; accurate for 1e-3 < p < 1-1e-3.
        import math
        if p <= 0.0 or p >= 1.0:
            raise ValueError("probability out of range")
        a = [-3.969683028665376e01, 2.209460984245205e02,
             -2.759285104469687e02, 1.383577518672690e02,
             -3.066479806614716e01, 2.506628277459239e00]
        b = [-5.447609879822406e01, 1.615858368580409e02,
             -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
        plow = 0.02425
        phigh = 1.0 - plow
        if p < plow:
            q = math.sqrt(-2.0 * math.log(p))
            return (((((a[0]*q+a[1])*q+a[2])*q+a[3])*q+a[4])*q+a[5]) / (
                ((((b[0]*q+b[1])*q+b[2])*q+b[3])*q+b[4])*q+1)
        if p <= phigh:
            q = p - 0.5
            r = q * q
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (
                (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1))
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((a[0]*q+a[1])*q+a[2])*q+a[3])*q+a[4])*q+a[5]) / (
            ((((b[0]*q+b[1])*q+b[2])*q+b[3])*q+b[4])*q+1)


def bootstrap_pearson_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(r, lo, hi)`` — Pearson correlation with a percentile
    bootstrap CI. NaN-pairs are dropped; with fewer than 3 surviving
    observations returns ``(nan, nan, nan)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    xf, yf = x[mask], y[mask]
    if len(xf) < 3:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    rs = np.empty(n_iters)
    n = len(xf)
    for i in range(n_iters):
        idx = rng.integers(0, n, size=n)
        xb, yb = xf[idx], yf[idx]
        if xb.std() == 0 or yb.std() == 0:
            rs[i] = float("nan")
        else:
            rs[i] = float(np.corrcoef(xb, yb)[0, 1])
    rs = rs[~np.isnan(rs)]
    lo = float(np.quantile(rs, alpha / 2.0)) if len(rs) else float("nan")
    hi = float(np.quantile(rs, 1.0 - alpha / 2.0)) if len(rs) else float("nan")
    r = float(np.corrcoef(xf, yf)[0, 1])
    return r, lo, hi


# ---------------------------------------------------------------------------
# Multiple-comparisons correction
# ---------------------------------------------------------------------------


def benjamini_hochberg(pvalues: list[float] | np.ndarray,
                       alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control at level ``alpha``.

    Returns ``(q_values, rejected_mask)``. NaN p-values are ignored
    (passed through as NaN q, not rejected). ``q_values`` are monotonic
    in the sorted-p order, clipped to [0, 1].
    """
    p = np.asarray(pvalues, dtype=float)
    valid = ~np.isnan(p)
    q = np.full_like(p, np.nan, dtype=float)
    if not valid.any():
        return q, np.zeros_like(p, dtype=bool)
    p_valid = p[valid]
    order = np.argsort(p_valid)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(p_valid) + 1)
    m = len(p_valid)
    raw = p_valid * m / ranks
    # Enforce monotonicity: q[i] = min(raw[i..m])
    sorted_raw = raw[order]
    cummin = np.minimum.accumulate(sorted_raw[::-1])[::-1]
    q_valid = np.clip(cummin[ranks - 1], 0.0, 1.0)
    q[valid] = q_valid
    rejected = np.zeros_like(p, dtype=bool)
    rejected[valid] = q[valid] <= alpha
    return q, rejected


def exact_match_rate(
    df: pd.DataFrame,
    tol_bps: int = 25,
    by: list[str] | None = None,
    with_ci: bool = False,
    alpha: float = 0.05,
) -> pd.DataFrame | float:
    """Within-tolerance recall rate. With ``with_ci=True`` each row also
    gets ``[lo, hi, n]`` Wilson-score interval columns."""
    col = f"within_{tol_bps}bps"
    if col not in df.columns:
        raise KeyError(f"{col} missing — did you call enrich()?")
    if by is None:
        if not with_ci:
            return _rate(df[col])
        s = df[col].dropna().astype(float)
        k = int(s.sum())
        n = int(len(s))
        lo, hi = wilson_interval(k, n, alpha=alpha)
        return pd.DataFrame([{col: k / n if n else float("nan"),
                               f"{col}_lo": lo, f"{col}_hi": hi, "n": n}])

    out_rows: list[dict] = []
    for key, frame in df.groupby(by):
        s = frame[col].dropna().astype(float)
        n = int(len(s))
        k = int(s.sum()) if n else 0
        rate = k / n if n else float("nan")
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec[col] = rate
        if with_ci:
            lo, hi = wilson_interval(k, n, alpha=alpha)
            rec[f"{col}_lo"] = lo
            rec[f"{col}_hi"] = hi
            rec["n"] = n
        out_rows.append(rec)
    return pd.DataFrame(out_rows)


def directional_accuracy(
    df: pd.DataFrame, by: list[str] | None = None
) -> pd.DataFrame | float:
    if "dir_correct" not in df.columns:
        raise KeyError("dir_correct missing — did you call enrich()?")
    if by is None:
        return _rate(df["dir_correct"])
    return df.groupby(by)["dir_correct"].apply(_rate).reset_index(name="dir_correct")


def calibration(
    df: pd.DataFrame, by: list[str] | None = None
) -> pd.DataFrame | float:
    """Pearson correlation between parsed estimate and truth (on A/B only)."""
    def _pearson(frame: pd.DataFrame) -> float:
        sub = frame.dropna(subset=["parsed_estimate", "truth"])
        sub = sub[sub["variant"].isin(["A", "B"])]
        if len(sub) < 3:
            return float("nan")
        return float(sub["parsed_estimate"].corr(sub["truth"]))

    if by is None:
        return _pearson(df)
    return (
        df.groupby(by)
        .apply(_pearson, include_groups=False)
        .reset_index(name="pearson_r")
    )


@dataclass
class TemporalSlope:
    """OLS fit of within-tol rate on year, with two-sided slope p-value."""

    slope: float
    intercept: float
    n_years: int
    r2: float
    pvalue: float
    stderr: float


def _linregress_with_p(x: np.ndarray, y: np.ndarray) -> TemporalSlope:
    """OLS fit with two-sided p-value on the slope.

    Implemented explicitly (no scipy dep) because ``np.polyfit`` doesn't
    give an SE and adding scipy for one function is heavy. The formula is
    the standard OLS se of the slope: sqrt(MSE / Σ(x − x̄)²).
    """
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = float(((x - x_mean) ** 2).sum())
    sxy = float(((x - x_mean) * (y - y_mean)).sum())
    if sxx <= 0.0:
        return TemporalSlope(float("nan"), float("nan"), n, float("nan"),
                             float("nan"), float("nan"))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    y_hat = slope * x + intercept
    resid = y - y_hat
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    if n <= 2:
        return TemporalSlope(slope, intercept, n, r2, float("nan"), float("nan"))
    mse = ss_res / (n - 2)
    stderr = float(np.sqrt(mse / sxx))
    t_stat = slope / stderr if stderr > 0 else float("inf")
    # Two-sided p-value via t-distribution with n-2 df. We import scipy
    # lazily here so the module loads without it for non-regression users.
    try:
        from scipy.stats import t as _t
        pvalue = 2.0 * float(1.0 - _t.cdf(abs(t_stat), df=n - 2))
    except ImportError:  # pragma: no cover — scipy is in the default env
        # Fall back to a normal approximation (valid for n >> 30).
        from math import erf, sqrt
        pvalue = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t_stat) / sqrt(2.0))))
    return TemporalSlope(float(slope), float(intercept), n, r2, float(pvalue), stderr)


def temporal_gradient(
    df: pd.DataFrame,
    tol_bps: int = 25,
    by: list[str] | None = None,
    x: str = "year",
) -> pd.DataFrame:
    """Per-group OLS slope of recall rate on a temporal axis.

    ``x`` selects the regressor:

    - ``"year"``: calendar year (the legacy axis; pools across models).
    - ``"months_to_cutoff"``: per-model signed months from the probe
      month to the model's training cutoff. Positive = in training data,
      negative = post-cutoff held out. This is the econometrically-
      correct axis for the ``recall declines as we approach / pass the
      cutoff`` claim.

    Returns columns ``[*groups, slope_per_{unit}, intercept, n_bins, r2,
    pvalue, stderr]`` where ``{unit}`` is ``year`` or ``month``.
    Groups with fewer than 3 bin-observations get NaN statistics.
    """
    col = f"within_{tol_bps}bps"
    if col not in df.columns:
        raise KeyError(f"{col} missing — did you call enrich()?")
    if x not in df.columns:
        raise KeyError(f"x={x!r} column missing — did you call enrich()?")

    groups = by or []
    keys = groups + [x]
    binned = (
        df.dropna(subset=[col, x])
        .groupby(keys)[col]
        .mean()
        .reset_index()
    )
    slope_col = "slope_per_year" if x == "year" else "slope_per_month"

    def _fit(frame: pd.DataFrame) -> TemporalSlope | None:
        if len(frame) < 3:
            return None
        return _linregress_with_p(
            frame[x].to_numpy(dtype=float),
            frame[col].to_numpy(dtype=float),
        )

    def _row(fit: TemporalSlope | None) -> dict:
        if fit is None:
            return {slope_col: np.nan, "intercept": np.nan,
                    "n_bins": 0, "r2": np.nan,
                    "pvalue": np.nan, "stderr": np.nan}
        return {slope_col: fit.slope, "intercept": fit.intercept,
                "n_bins": fit.n_years, "r2": fit.r2,
                "pvalue": fit.pvalue, "stderr": fit.stderr}

    if not groups:
        return pd.DataFrame([_row(_fit(binned))])

    rows: list[dict] = []
    for key, frame in binned.groupby(groups):
        rec = dict(zip(groups, key if isinstance(key, tuple) else (key,)))
        rec.update(_row(_fit(frame)))
        rows.append(rec)
    return pd.DataFrame(rows)


def famous_concentration(
    df: pd.DataFrame,
    famous_months: list[str],
    tol_bps: int = 25,
    by: list[str] | None = None,
) -> pd.DataFrame:
    """Compare recall on famous vs. non-famous months.

    Returns a DataFrame with columns ``[..., famous_rate, other_rate, lift]``
    where ``lift = famous_rate - other_rate``. Positive lift means the model
    recalls famous months better than random months.
    """
    col = f"within_{tol_bps}bps"
    if col not in df.columns:
        raise KeyError(f"{col} missing — did you call enrich()?")

    famous_set = set(famous_months)
    tagged = df.copy()
    tagged["is_famous"] = tagged["month"].isin(famous_set)

    group_cols = (by or []) + ["is_famous"]
    rates = (
        tagged.dropna(subset=[col])
        .groupby(group_cols)[col]
        .mean()
        .unstack("is_famous")
    )
    rates.columns = ["other_rate" if not k else "famous_rate" for k in rates.columns]
    for needed in ("famous_rate", "other_rate"):
        if needed not in rates.columns:
            rates[needed] = np.nan
    rates["lift"] = rates["famous_rate"] - rates["other_rate"]
    return rates.reset_index()


def comparative_accuracy(
    df: pd.DataFrame, by: list[str] | None = None
) -> pd.DataFrame | float:
    """Variant-C accuracy: did the model pick the actually-higher month?"""
    if "comparative_correct" not in df.columns:
        raise KeyError("comparative_correct missing — did you call enrich()?")
    if by is None:
        return _rate(df["comparative_correct"])
    return (
        df.groupby(by)["comparative_correct"]
        .apply(_rate)
        .reset_index(name="comparative_correct")
    )


# ---------------------------------------------------------------------------
# Summary helper — everything at once, per (model, factor)
# ---------------------------------------------------------------------------


def headline_table(
    df: pd.DataFrame,
    variants: list[str] | str = "A",
) -> pd.DataFrame:
    """Build Table 1 of the paper: per-(model, factor) headline metrics.

    Args:
        df: an enriched sweep DataFrame (see ``enrich``).
        variants: which prompt variants to include in the numeric metrics.
            Default is ``"A"`` (direct numeric) because variant B's
            descriptive prompt is harder to parse and noisier. Pass
            ``["A", "B"]`` to pool both.

    Columns: ``[model_name, factor, n_parsed, within_5bps, within_25bps,
    dir_correct, pearson_r]``.
    """
    if isinstance(variants, str):
        variants = [variants]
    numeric = df[df["variant"].isin(variants)]
    n_attempted = (
        numeric.dropna(subset=["parsed_estimate"])
        .groupby(["model_name", "factor"])
        .size()
        .rename("n_parsed")
    )
    within5 = exact_match_rate(numeric, 5, by=["model_name", "factor"])
    within25 = exact_match_rate(numeric, 25, by=["model_name", "factor"])
    direction = directional_accuracy(numeric, by=["model_name", "factor"])
    corr = calibration(numeric, by=["model_name", "factor"])

    out = within5.merge(within25, on=["model_name", "factor"])
    out = out.merge(direction, on=["model_name", "factor"])
    out = out.merge(corr, on=["model_name", "factor"])
    out = out.merge(n_attempted.reset_index(), on=["model_name", "factor"], how="left")
    return out
