"""Cutoff-stratified Mkt-RF recall: split sample into
pre/near/post-cutoff and report (parse, r, within-25bps) per cell.

Reviewer concern: the cutoff-gradient null is a weak claim because
most of the sample is well pre-cutoff and post-cutoff months are few.
A stratified comparison either provides supporting evidence
("uniform recall regardless of cutoff side") or surfaces a real
pre-vs-post drop.

Output: console table + LaTeX-ready rows for App.~\ref{app:cutoff_stratified}.
"""
from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.parse import parse_numeric  # noqa: E402

RES = REPO / "experiments" / "results"
DATA = REPO / "data" / "ff"
NEAR_BAND = 6


# (display, internal, cutoff, source_files, source_kind)
MODELS = [
    ("Sonnet 4.6",    "claude-sonnet-4.6",  "2025-03",
     ["sweep.jsonl", "multiseed.jsonl"], "main"),
    ("Haiku 4.5",     "claude-haiku-4.5",   "2025-07",
     ["sweep.jsonl", "multiseed.jsonl"], "main"),
    ("Opus 4.7",      "claude-opus-4.7",    "2026-01",
     ["opus_baselines.jsonl"], "opus"),
    ("GPT-5.4",       "gpt-5.4",            "2025-08",
     ["gpt54_baselines.jsonl"], "baseline"),
    ("GPT-5.4-mini",  "gpt-5.4-mini",       "2025-08",
     ["more_baselines.jsonl"], "baseline"),
    ("GPT-5.4-nano",  "gpt-5.4-nano",       "2025-08",
     ["more_baselines.jsonl"], "baseline"),
    ("DeepSeek-V3.2", "deepseek-v3.2-azure","2025-07",
     ["more_baselines.jsonl"], "baseline"),
    ("Llama-3.3-70B", "llama-3.3-70b-groq", "2024-12",
     ["llama_baselines.jsonl"], "baseline"),
]


def load_records(internal: str, sources: list[str], kind: str):
    truth = load_all_factors(DATA)["Mkt-RF"]
    out = []
    for fn in sources:
        path = RES / fn
        if not path.exists():
            continue
        for line in path.open():
            r = json.loads(line)
            if kind == "main":
                q = r.get("query") or {}
                if (q.get("model_name") != internal
                        or q.get("factor") != "Mkt-RF"
                        or q.get("variant") != "A"):
                    continue
                month = q["month"]
                resp = r.get("response")
            else:
                if r.get("model_name") != internal:
                    continue
                if r.get("probe") not in (None, "mktrf"):
                    continue
                month = r["month"]
                resp = r.get("response")
            try:
                t = float(truth.get(pd.Period(month, freq="M"),
                                    float("nan")))
            except Exception:
                continue
            if not (t == t):
                continue
            parsed = parse_numeric(resp) if resp else None
            out.append({"month": month, "truth": t,
                        "parsed": parsed, "response": resp})
    return out


def stratify(records, cutoff_str: str) -> dict:
    cutoff = pd.Period(cutoff_str, freq="M")
    rows = []
    for r in records:
        d = (cutoff - pd.Period(r["month"], freq="M")).n
        if d > NEAR_BAND:
            bucket = "pre"
        elif d < -NEAR_BAND:
            bucket = "post"
        else:
            bucket = "near"
        rows.append({**r, "d": d, "bucket": bucket})
    df = pd.DataFrame(rows)
    out = {}
    for bucket in ["pre", "near", "post", "all"]:
        sub = df if bucket == "all" else df[df["bucket"] == bucket]
        n = len(sub)
        p = sub.dropna(subset=["parsed"])
        np_ = len(p)
        if np_ < 3:
            r = float("nan")
        else:
            r = float(np.corrcoef(p["parsed"], p["truth"])[0, 1])
        if np_ > 0:
            within = float((p["parsed"] - p["truth"]).abs().le(0.25).mean())
        else:
            within = float("nan")
        out[bucket] = {"n": n, "parsed": np_, "r": r, "within25": within}
    return out


def fmt_r(x: float) -> str:
    return "  --  " if not (x == x) else f"{x:+.3f}"


def fmt_w(x: float) -> str:
    return " -- " if not (x == x) else f"{x:.2f}"


def main() -> None:
    print(f"{'Model':14s}  {'Cutoff':8s}  "
          f"{'Bucket':5s}  {'n':>3s}  {'parsed':>6s}  "
          f"{'r':>7s}  {'w25':>5s}")
    print("-" * 70)
    rows_for_tex = []
    for display, internal, cutoff, sources, kind in MODELS:
        recs = load_records(internal, sources, kind)
        s = stratify(recs, cutoff)
        for bucket in ["pre", "near", "post", "all"]:
            x = s[bucket]
            print(f"{display:14s}  {cutoff:8s}  "
                  f"{bucket:5s}  {x['n']:>3d}  {x['parsed']:>6d}  "
                  f"{fmt_r(x['r'])}  {fmt_w(x['within25'])}")
        print("")
        rows_for_tex.append((display, cutoff, s))

    print("\n=== LaTeX rows (pre, near, post, all) ===")
    for display, cutoff, s in rows_for_tex:
        cells = []
        for b in ["pre", "near", "post"]:
            x = s[b]
            cells.append(f"{x['parsed']}/{x['n']} & "
                         f"${fmt_r(x['r'])}$ & {fmt_w(x['within25'])}")
        all_ = s["all"]
        cells.append(f"{all_['parsed']}/{all_['n']} & "
                     f"${fmt_r(all_['r'])}$ & {fmt_w(all_['within25'])}")
        print(f"{display} & {cutoff} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
