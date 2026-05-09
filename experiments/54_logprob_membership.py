"""EXP-M1: logprob membership-inference scoring on existing GPT-5.4
logprobs probe data (`experiments/results/logprobs_probe.jsonl`).

Two outputs:
  (a) Mann–Whitney U + bootstrap 95% CI on mean entropy differences
      across the three conditions (Mkt-RF / RMW / Fabricated).
  (b) Per-record membership-inference LLR: log p(top-1) − log p(top-2)
      on the first numeric token. A memorized recall produces a
      sharply peaked first-token distribution; fabrication should be
      diffuse. Reports per-condition LLR distribution.

No new API queries. Re-analysis only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "experiments/results/logprobs_probe.jsonl"


def shannon_entropy_bits(top_logprobs: list[dict]) -> float:
    """Entropy in bits over (top-k tokens + a residual 'rest' bucket)."""
    probs = [math.exp(t["logprob"]) for t in top_logprobs]
    rest = max(0.0, 1.0 - sum(probs))
    if rest > 0:
        probs.append(rest)
    H = 0.0
    for p in probs:
        if p > 0:
            H -= p * math.log2(p)
    return H


def first_two_token_entropy(rec: dict) -> float | None:
    toks = rec.get("tokens", [])
    if len(toks) < 2:
        return None
    h1 = shannon_entropy_bits(toks[0]["top_logprobs"])
    h2 = shannon_entropy_bits(toks[1]["top_logprobs"])
    return (h1 + h2) / 2.0


def first_token_llr(rec: dict) -> float | None:
    """log p(top-1) - log p(top-2) on the first output token."""
    toks = rec.get("tokens", [])
    if not toks:
        return None
    top = toks[0]["top_logprobs"]
    if len(top) < 2:
        return None
    return top[0]["logprob"] - top[1]["logprob"]


def bootstrap_mean_diff_ci(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 2026):
    """Bootstrap 95% CI on mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = sa.mean() - sb.mean()
    return float(diffs.mean()), (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta non-parametric effect size in [-1, 1]."""
    a = np.asarray(a)
    b = np.asarray(b)
    n_more = sum((ai > b).sum() for ai in a)
    n_less = sum((ai < b).sum() for ai in a)
    return (n_more - n_less) / (len(a) * len(b))


def main():
    by_cond_h: dict[str, list[float]] = {}
    by_cond_llr: dict[str, list[float]] = {}
    for line in DATA.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        h = first_two_token_entropy(rec)
        llr = first_token_llr(rec)
        cond = rec["cond"]
        if h is not None:
            by_cond_h.setdefault(cond, []).append(h)
        if llr is not None:
            by_cond_llr.setdefault(cond, []).append(llr)

    print("=" * 72)
    print("Entropy (bits) of first two output tokens — re-derived")
    print("=" * 72)
    for cond in ["Mkt-RF", "RMW", "Fabricated"]:
        arr = np.array(by_cond_h[cond])
        m, sd = arr.mean(), arr.std(ddof=1)
        med = np.median(arr)
        # bootstrap mean CI
        rng = np.random.default_rng(2026)
        boot_means = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(10000)]
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        print(f"  {cond:12s}  n={len(arr):3d}  mean={m:.3f}  median={med:.3f}  sd={sd:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

    print()
    print("Pairwise Mann–Whitney U (two-sided), bootstrap 95% CI on mean Δ, Cliff's δ")
    pairs = [("Mkt-RF", "RMW"), ("Mkt-RF", "Fabricated"), ("RMW", "Fabricated")]
    for a_name, b_name in pairs:
        a = np.array(by_cond_h[a_name])
        b = np.array(by_cond_h[b_name])
        u_stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        m_diff, (lo, hi) = bootstrap_mean_diff_ci(a, b)
        delta = cliffs_delta(a, b)
        print(f"  {a_name:10s} vs {b_name:10s}: U={u_stat:7.1f}  p={p:.2e}  Δmean={m_diff:+.3f} [{lo:+.3f}, {hi:+.3f}]  δ={delta:+.3f}")

    print()
    print("=" * 72)
    print("Membership-inference LLR: first-token log p(top-1) − log p(top-2)")
    print("=" * 72)
    for cond in ["Mkt-RF", "RMW", "Fabricated"]:
        arr = np.array(by_cond_llr[cond])
        m = arr.mean()
        med = np.median(arr)
        sd = arr.std(ddof=1)
        rng = np.random.default_rng(2026)
        boot_means = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(10000)]
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        print(f"  {cond:12s}  n={len(arr):3d}  mean LLR={m:.3f} nats  median={med:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]")

    print()
    for a_name, b_name in pairs:
        a = np.array(by_cond_llr[a_name])
        b = np.array(by_cond_llr[b_name])
        u_stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        m_diff, (lo, hi) = bootstrap_mean_diff_ci(a, b)
        delta = cliffs_delta(a, b)
        print(f"  LLR {a_name:10s} vs {b_name:10s}: U={u_stat:7.1f}  p={p:.2e}  Δmean={m_diff:+.3f} [{lo:+.3f}, {hi:+.3f}]  δ={delta:+.3f}")


if __name__ == "__main__":
    main()
