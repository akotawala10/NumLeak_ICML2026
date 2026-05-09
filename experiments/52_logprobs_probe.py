"""Logit-distribution probe on GPT-5.4: extract top-5 logprobs for the
first 2 generated tokens on three conditions:
  (i)  Mkt-RF queries (high-recall regime)
  (ii) RMW queries (low-recall, model often refuses)
  (iii) fabricated-factor queries (no real signal but GPT-5.4 commits)

For each query, compute the average per-token entropy across the first
two generated tokens. The mechanistic prediction:
- Mkt-RF: low entropy (committal readout, peaked on memorized value)
- Fabricated: low entropy (fabrication is also committal — same readout
  shape on a hallucinated answer)
- RMW: higher entropy if the model is hedging/refusing

Output: results/logprobs_probe.jsonl + console summary table.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.probe import format_month_human  # noqa: E402

OUT = REPO / "experiments" / "results" / "logprobs_probe.jsonl"
DATA = REPO / "data" / "ff"

FABRICATED_NAMES = [
    "Gleason-Zeta volatility-conditioned residual factor",
    "Holbrooke-Mansfield Opportunity Fund III (2007 vintage)",
    "Brennan-Iyer mean-reversion premium factor",
    "Northrop-Calloway long-horizon dispersion factor",
    "Pemberton-Yi cross-sectional liquidity premium factor",
]

PROMPT_REAL = (
    "What was the monthly return of the Fama-French {factor} factor in "
    "{month}? Answer with a signed decimal percentage (e.g., -3.12 for "
    "a 3.12% loss, +1.50 for a 1.50% gain) and nothing else."
)

PROMPT_FAB = (
    "What was the monthly return of the {name} in {month}? Answer with "
    "a signed decimal percentage (e.g., -3.12 for a 3.12% loss, +1.50 "
    "for a 1.50% gain) and nothing else."
)


def shannon_entropy_bits(top_logprobs):
    """Entropy (in bits) of the top-k token distribution. Logprobs are
    natural log probabilities of the top-k candidates; the residual
    mass not in top-k is treated as a single "rest" bucket whose
    probability is 1 - sum of top-k probs. Entropy is computed in
    bits (log base 2)."""
    probs = [math.exp(t.logprob) for t in top_logprobs]
    rest = max(0.0, 1.0 - sum(probs))
    if rest > 0:
        probs.append(rest)
    h = 0.0
    for p in probs:
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


def call_with_logprobs(client, deployment, prompt, max_tokens=24):
    """Single GPT-5.4 call returning text + per-token top-5 logprobs."""
    resp = client.responses.create(
        model=deployment,
        input=prompt,
        max_output_tokens=max_tokens,
        top_logprobs=5,
        include=["message.output_text.logprobs"],
    )
    text = resp.output_text or ""
    tokens = []
    for item in resp.output:
        for piece in getattr(item, "content", []) or []:
            lp = getattr(piece, "logprobs", None)
            if lp:
                for tok in lp:
                    tokens.append({
                        "token": tok.token,
                        "logprob": float(tok.logprob),
                        "top_logprobs": [
                            {"token": t.token, "logprob": float(t.logprob)}
                            for t in tok.top_logprobs
                        ],
                    })
    return text, tokens, resp


def first_n_token_entropy(tokens, n=2):
    """Average per-token entropy of the first n generated tokens, in bits."""
    use = tokens[:n]
    if not use:
        return float("nan")
    ents = []
    for tok in use:
        # Reconstruct LogprobTopLogprob-like objects for the entropy fn
        class _T:
            __slots__ = ("logprob",)
            def __init__(self, lp): self.logprob = lp
        ents.append(shannon_entropy_bits(
            [_T(t["logprob"]) for t in tok["top_logprobs"]]))
    return float(np.mean(ents))


def main() -> None:
    from openai import AzureOpenAI

    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION",
                                 "2025-04-01-preview")
    client = AzureOpenAI(api_key=api_key,
                        azure_endpoint=endpoint,
                        api_version=api_version,
                        timeout=60.0)

    # --- sample months ---
    truth = load_all_factors(DATA)
    rng = random.Random(2030)
    pool = [str(p) for p in truth["Mkt-RF"].index
            if "1980-01" <= str(p) <= "2024-12"]
    months_real = sorted(rng.sample(pool, 30))
    months_fab = sorted(rng.sample(pool, 6))   # 5 names x 6 months = 30

    queries = []
    for m in months_real:
        queries.append({"cond": "Mkt-RF", "factor": "Mkt-RF",
                        "month": m,
                        "prompt": PROMPT_REAL.format(
                            factor="market excess return (Mkt-RF)",
                            month=format_month_human(m))})
    for m in months_real:
        queries.append({"cond": "RMW", "factor": "RMW", "month": m,
                        "prompt": PROMPT_REAL.format(
                            factor="profitability (RMW)",
                            month=format_month_human(m))})
    for name in FABRICATED_NAMES:
        for m in months_fab:
            queries.append({"cond": "Fabricated", "factor": name,
                            "month": m,
                            "prompt": PROMPT_FAB.format(
                                name=name,
                                month=format_month_human(m))})

    print(f"running {len(queries)} GPT-5.4 logprob queries "
          f"({sum(1 for q in queries if q['cond']=='Mkt-RF')} Mkt-RF, "
          f"{sum(1 for q in queries if q['cond']=='RMW')} RMW, "
          f"{sum(1 for q in queries if q['cond']=='Fabricated')} Fabricated)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with OUT.open("w") as f:
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            try:
                text, tokens, _ = call_with_logprobs(
                    client, deployment, q["prompt"], max_tokens=24)
                err = None
            except Exception as e:  # noqa: BLE001
                text, tokens, err = None, None, f"{type(e).__name__}: {e}"
            ent = first_n_token_entropy(tokens, n=2) if tokens else float("nan")
            rec = {
                **q,
                "response": text,
                "tokens": tokens,
                "entropy_bits_first2": ent,
                "error": err,
                "latency_s": time.perf_counter() - t0,
                "ts": time.time(),
            }
            f.write(json.dumps(rec) + "\n")
            records.append(rec)
            print(f"  [{i+1:>3}/{len(queries)}] {q['cond']:10s} "
                  f"{q['month']} text={text!r:>10} ent={ent:.3f}")

    # --- summary ---
    df = pd.DataFrame(records)
    df = df.dropna(subset=["entropy_bits_first2"])
    print("\n=== Entropy of first-2-token distribution (bits, top-5+rest) ===")
    print(f"{'cond':12s}  {'n':>3s}  {'mean':>6s}  {'median':>7s}  {'sd':>6s}")
    for cond, g in df.groupby("cond"):
        print(f"{cond:12s}  {len(g):>3d}  "
              f"{g['entropy_bits_first2'].mean():>6.3f}  "
              f"{g['entropy_bits_first2'].median():>7.3f}  "
              f"{g['entropy_bits_first2'].std():>6.3f}")


if __name__ == "__main__":
    main()
