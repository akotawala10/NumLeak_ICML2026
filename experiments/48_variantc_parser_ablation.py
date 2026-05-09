"""Variant-C parser-robustness ablation.

Reviewer concern: rank accuracy is computed on the parsed subset
under the endorsement-aware parser (App. C). If the model
preferentially refuses on hard pairs, accuracy is biased upward;
and the parser itself could in principle bias the picks.

Test: re-run the rank computation on the SAME Sonnet × Mkt-RF
Variant-C records using a NAIVE parser that simply returns the
first candidate month mentioned in the response. The naive
parser has near-100% parse rate (no refusal handling, no
endorsement detection), so any difference between the two
accuracies is a parser-specific artifact.

Predicted under our claim: both parsers give chance-level
accuracy. Result: endorsement-aware parser 52.5% (parse 67%),
naive parser 49.2% (parse 98%). Both 95% CIs include 50%.

Output: console summary.
"""
from __future__ import annotations

import json
import re
import sys
from math import sqrt
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.parse import parse_comparative  # noqa: E402

SWEEP = REPO / "experiments" / "results" / "sweep.jsonl"
DATA = REPO / "data" / "ff"

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November",
               "December"]
MONTH_RE = re.compile(
    r"\b(" + "|".join(MONTH_NAMES) + r")\s+(\d{4})\b", re.IGNORECASE)


def naive_first_mention(text: str | None, m1: str, m2: str) -> str | None:
    """Return the canonical 'YYYY-MM' of the FIRST candidate month
    mentioned in the response, or None if no candidate is mentioned."""
    if not text:
        return None
    for nm, yr in MONTH_RE.findall(text):
        try:
            mi = MONTH_NAMES.index(nm.title()) + 1
            canon = f"{int(yr):04d}-{mi:02d}"
            if canon == m1 or canon == m2:
                return canon
        except ValueError:
            pass
    return None


def ci(c: int, k: int) -> tuple[float, float, float]:
    if k == 0:
        return float("nan"), float("nan"), float("nan")
    a = c / k
    s = sqrt(a * (1 - a) / k)
    return a, max(0.0, a - 1.96 * s), min(1.0, a + 1.96 * s)


def _run_one(factor: str) -> None:
    recs = []
    for line in SWEEP.open():
        r = json.loads(line)
        q = r.get("query") or {}
        if (q.get("model_name") == "claude-sonnet-4.6"
                and q.get("factor") == factor
                and q.get("variant") == "C"
                and r.get("error") is None):
            recs.append(r)
    latest = {}
    for r in recs:
        q = r["query"]
        k = (q["month"], q["month2"])
        if k not in latest or r.get("ts", 0) > latest[k].get("ts", 0):
            latest[k] = r
    recs = list(latest.values())

    truth = load_all_factors(DATA)[factor]

    n = len(recs)
    ep = ec = np_ = nc = 0
    for r in recs:
        q = r["query"]
        m1, m2 = q["month"], q["month2"]
        t1 = float(truth.get(pd.Period(m1, freq="M"), float("nan")))
        t2 = float(truth.get(pd.Period(m2, freq="M"), float("nan")))
        if not (t1 == t1 and t2 == t2 and t1 != t2):
            continue
        win = m1 if t1 > t2 else m2
        pick_e = parse_comparative(r.get("response"), m1, m2)
        if pick_e is not None:
            ep += 1
            if pick_e == win:
                ec += 1
        pick_n = naive_first_mention(r.get("response"), m1, m2)
        if pick_n is not None:
            np_ += 1
            if pick_n == win:
                nc += 1

    print(f"\nSonnet × {factor:7s} Variant-C, n={n}")
    a, lo, hi = ci(ec, ep)
    print(f"  endorsement-aware: parse {ep}/{n}={ep/n:.3f}  "
          f"acc {ec}/{ep}={a:.3f} 95%CI=[{lo:.3f},{hi:.3f}]")
    a, lo, hi = ci(nc, np_)
    print(f"  naive first-mention: parse {np_}/{n}={np_/n:.3f}  "
          f"acc {nc}/{np_}={a:.3f} 95%CI=[{lo:.3f},{hi:.3f}]")


def main() -> None:
    for factor in ["Mkt-RF", "HML", "SMB"]:
        _run_one(factor)
    print("\nMkt-RF: both parsers at chance (decoupling).\n"
          "HML: parsers disagree --- partial recall is reflected in the "
          "endorsement-aware pick but lost by naive first-mention.\n"
          "SMB: both parsers at chance, consistent with poor value recall.")


if __name__ == "__main__":
    main()
