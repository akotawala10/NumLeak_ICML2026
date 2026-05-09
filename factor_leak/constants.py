"""Project-wide constants that must not drift between modules.

- ``FAMOUS_MONTHS`` — the canonical set of narrative-rich months used by both
  the sampling plan and the famous-month concentration metric. Anywhere that
  says "famous month" must import from here.
- ``MODEL_CUTOFFS`` — published / announced training cutoff dates for each
  probed model. Used to shade Fig 2 and to flag near-cutoff probe months.
  **These are best-available values; confirm against each vendor's release
  notes before submission.**
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Famous months (handoff §3.4)
# ---------------------------------------------------------------------------

FAMOUS_MONTHS: list[str] = [
    "1987-10",   # Black Monday
    "1990-08",   # Gulf War / oil shock
    "1997-10",   # Asian financial crisis
    "1998-08",   # LTCM / Russian default
    "2000-03",   # dot-com peak
    "2001-09",   # 9/11
    "2002-07",   # dot-com trough
    "2008-09",   # Lehman collapse
    "2008-10",   # post-Lehman crash
    "2009-03",   # market bottom
    "2010-05",   # flash crash
    "2011-08",   # US debt ceiling / EU crisis
    "2015-08",   # China devaluation
    "2016-11",   # US election
    "2018-02",   # volmageddon
    "2018-12",   # Q4 2018 selloff
    "2020-02",   # pre-COVID top
    "2020-03",   # COVID crash
    "2020-11",   # vaccine / value rotation
    "2022-01",   # growth-to-value rotation
]


# ---------------------------------------------------------------------------
# Model training cutoffs
# ---------------------------------------------------------------------------

# Cutoffs as published/announced by vendors; accurate to within a month or
# two. Dates refer to the *latest training data* the model may have seen.
# TODO(final): re-verify each against vendor docs at submission time.
MODEL_CUTOFFS: dict[str, str] = {
    "gpt-4.1":            "2024-06",
    "gpt-4o-mini":        "2023-10",
    "claude-sonnet-4.6":  "2025-03",
    "claude-haiku-4.5":   "2025-07",
    "llama-3.3-70b":      "2023-12",
    "deepseek-v3":        "2024-07",
}


def months_before_cutoff(model: str, n_months: int) -> tuple[pd.Period, pd.Period]:
    """Return ``(cutoff - n_months, cutoff)`` as a pair of ``pd.Period`` (monthly).

    Convenience for computing the near-cutoff shading in Fig 2 and for
    bucketing queries by distance-to-cutoff in the temporal analysis.
    """
    if model not in MODEL_CUTOFFS:
        raise KeyError(f"no known training cutoff for {model!r}")
    end = pd.Period(MODEL_CUTOFFS[model], freq="M")
    start = end - n_months
    return start, end


# Near-cutoff band (months). Months within ±NEAR_CUTOFF_BAND of a model's
# cutoff are classified as "near". Earlier months are "pre", later are "post".
NEAR_CUTOFF_BAND: int = 6


def months_to_cutoff(model: str, month: str | pd.Period) -> int:
    """Signed months from ``month`` to ``model``'s training cutoff.

    Returns a positive integer when ``month`` is *before* the cutoff
    (i.e., likely in training data), negative when *after* (post-cutoff).
    Zero means the probe month is the cutoff month itself.
    """
    if model not in MODEL_CUTOFFS:
        raise KeyError(f"no known training cutoff for {model!r}")
    m = month if isinstance(month, pd.Period) else pd.Period(month, freq="M")
    cutoff = pd.Period(MODEL_CUTOFFS[model], freq="M")
    # pandas Period subtraction returns an offset; extract its month count.
    diff = cutoff - m
    return int(diff.n)


def cutoff_bucket(model: str, month: str | pd.Period) -> str:
    """Classify a probe month as ``"pre"``, ``"near"``, or ``"post"``-cutoff.

    - ``pre``:  cutoff - month >  NEAR_CUTOFF_BAND   (deep in training data)
    - ``near``: |cutoff - month| ≤ NEAR_CUTOFF_BAND  (band around cutoff)
    - ``post``: cutoff - month < −NEAR_CUTOFF_BAND   (past the cutoff)
    """
    d = months_to_cutoff(model, month)
    if d > NEAR_CUTOFF_BAND:
        return "pre"
    if d < -NEAR_CUTOFF_BAND:
        return "post"
    return "near"
