"""EXP — Carlini-style verbatim extraction probe for Fama-French Mkt-RF.

Tests whether the model emits *consecutive* monthly Mkt-RF values verbatim
when given a CSV-style prefix and asked to continue. This is the
extractable-memorized-sequence definition of memorization
(Carlini 2021/2023), distinct from the paper's behavioral API-boundary
estimand.

Three prefix templates (raw CSV, framed CSV, markdown table), nine panel
models, sixty random start-months from 1980-2024 (seed 2026). Each query
asks the model to continue 12 monthly rows. Metrics per (model, template,
start_month): longest consecutive verbatim run from start_month forward,
total verbatim hits, total parseable rows, format-break/refusal flag.

Output: experiments/results/verbatim_extraction.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

from factor_leak.ff_loader import load_all_factors  # noqa: E402
from factor_leak.parse import _normalize  # noqa: E402
from factor_leak.probe import default_endpoints  # noqa: E402

OUT = REPO / "experiments" / "results" / "verbatim_extraction.jsonl"
DATA = REPO / "data" / "ff"

PANEL = [
    "claude-opus-4.7",
    "claude-sonnet-4.6",
    "claude-haiku-4.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "deepseek-v3.2-azure",
    "llama-3.3-70b-groq",
    "llama-3.1-8b-groq",
]


def _yyyymm(year: int, month: int) -> str:
    return f"{year:04d}{month:02d}"


def _next_month(yyyymm: str) -> str:
    y, m = int(yyyymm[:4]), int(yyyymm[4:6])
    if m == 12:
        return _yyyymm(y + 1, 1)
    return _yyyymm(y, m + 1)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


def template_a(start_yyyymm: str) -> str:
    """Minimal CSV prefix — closest to Carlini original."""
    return (
        "Continue the following text exactly. Output only the continuation, "
        "no commentary, no markdown, no code fences.\n\n"
        "Date,Mkt-RF\n"
        f"{start_yyyymm},"
    )


def template_b(start_yyyymm: str) -> str:
    """Framed CSV: explicit task description."""
    return (
        "Below is the start of a CSV file containing monthly Fama-French "
        "Mkt-RF (U.S. market excess return, in percent) data from the "
        "Kenneth French Data Library. Continue the file by appending the "
        "next 12 monthly entries in the same format. Output only the "
        "appended rows.\n\n"
        "Date,Mkt-RF\n"
        f"{start_yyyymm},"
    )


def template_c(start_yyyymm: str) -> str:
    """Markdown table continuation."""
    y, m = int(start_yyyymm[:4]), int(start_yyyymm[4:6])
    return (
        "Continue the following markdown table exactly with the next 12 "
        "monthly rows. Output only the table rows, no commentary.\n\n"
        "| Month   | Mkt-RF (%) |\n"
        "|---------|------------|\n"
        f"| {y:04d}-{m:02d} |"
    )


TEMPLATES = {"A_csv": template_a, "B_framed": template_b, "C_markdown": template_c}


# ---------------------------------------------------------------------------
# Parser: extract (YYYYMM, value) pairs from a continuation
# ---------------------------------------------------------------------------

# CSV-style row: optional whitespace, YYYYMM (or YYYY-MM), comma-or-pipe
# separator, signed decimal value.
_CSV_ROW = re.compile(
    r"(?P<y>\d{4})[-]?(?P<m>\d{2})\s*[,\|]\s*(?P<v>[-+]?\d+(?:\.\d+)?)",
    re.MULTILINE,
)
# Markdown row: "| YYYY-MM | value |" form
_MD_ROW = re.compile(
    r"\|\s*(?P<y>\d{4})[-/](?P<m>\d{2})\s*\|\s*(?P<v>[-+]?\d+(?:\.\d+)?)",
)


def parse_continuation(text: str, first_value_only_for_start: bool, start_yyyymm: str
                       ) -> list[tuple[str, float]]:
    """Return ordered list of (YYYYMM, value) pairs found in the continuation.

    Because the prefix already contains "{start_yyyymm}," the model's first
    emitted token typically continues with the *value* for that month —
    without re-emitting "YYYYMM,". We treat any leading numeric output
    BEFORE the first regex-matched row as the value for ``start_yyyymm``.
    """
    if text is None:
        return []
    text = _normalize(text)
    pairs: list[tuple[str, float]] = []

    # Heuristic: if the response starts with a signed decimal (possibly
    # with a leading +), interpret as the value for start_yyyymm.
    # Guard against the case where the model skips start_yyyymm and emits
    # the next row's YYYYMM as its leading token: skip if the leading
    # integer has 5+ digits (clearly a date) or no decimal point.
    if first_value_only_for_start:
        leading = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)", text)
        if leading:
            raw = leading.group(1)
            int_digits = len(raw.lstrip("+-").split(".")[0])
            looks_like_date = "." not in raw and int_digits >= 5
            if not looks_like_date:
                try:
                    v = float(raw)
                    if abs(v) < 100:
                        pairs.append((start_yyyymm, v))
                except ValueError:
                    pass

    # CSV rows
    for m in _CSV_ROW.finditer(text):
        try:
            ym = f"{int(m.group('y')):04d}{int(m.group('m')):02d}"
            v = float(m.group("v"))
            pairs.append((ym, v))
        except ValueError:
            continue
    # Markdown rows
    for m in _MD_ROW.finditer(text):
        try:
            ym = f"{int(m.group('y')):04d}{int(m.group('m')):02d}"
            v = float(m.group("v"))
            pairs.append((ym, v))
        except ValueError:
            continue

    # Deduplicate by month, preserving first-seen value
    seen: set[str] = set()
    out: list[tuple[str, float]] = []
    for ym, v in pairs:
        if ym in seen:
            continue
        seen.add(ym)
        out.append((ym, v))
    return out


# ---------------------------------------------------------------------------
# Truth lookup
# ---------------------------------------------------------------------------


def load_truth() -> dict[str, float]:
    factors = load_all_factors(DATA)
    mktrf = factors["Mkt-RF"].dropna()
    return {f"{p.year:04d}{p.month:02d}": float(v) for p, v in mktrf.items()}


# ---------------------------------------------------------------------------
# Verbatim metrics
# ---------------------------------------------------------------------------


def consecutive_verbatim_run(pairs: list[tuple[str, float]],
                              truth: dict[str, float],
                              start_yyyymm: str,
                              tol: float = 0.005) -> tuple[int, int, int]:
    """Return (run_length, total_verbatim, total_parsed).

    ``run_length`` is the longest consecutive stretch of months matching
    French within ``tol`` pp, beginning at ``start_yyyymm`` and proceeding
    chronologically. A miss or out-of-sequence month breaks the run.
    ``total_verbatim`` is total verbatim hits over all parsed entries.

    French publishes 2-decimal precision; tol=0.005 means the model's
    emitted value rounds to the same 2 decimals as French. This is
    Carlini-style "exact match to publication precision."
    """
    by_month = {ym: v for ym, v in pairs}
    expected = start_yyyymm
    run = 0
    while expected in by_month:
        v = by_month[expected]
        t = truth.get(expected)
        if t is None:
            break
        if abs(v - t) <= tol:
            run += 1
            expected = _next_month(expected)
        else:
            break

    verbatim_total = 0
    for ym, v in pairs:
        t = truth.get(ym)
        if t is not None and abs(v - t) <= tol:
            verbatim_total += 1
    return run, verbatim_total, len(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-months", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--models", nargs="*", default=PANEL)
    ap.add_argument("--templates", nargs="*", default=list(TEMPLATES.keys()))
    ap.add_argument("--limit", type=int, default=None,
                    help="optional: cap total queries for a smoke test")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    truth = load_truth()
    truth_months = sorted(
        ym for ym in truth.keys()
        if "1980" <= ym[:4] <= "2024" and int(ym[:4]) <= 2023
    )
    rng = random.Random(args.seed)
    start_months = sorted(rng.sample(truth_months, args.n_months))

    endpoints = {e.name: e for e in default_endpoints() if e.name in args.models}
    missing = set(args.models) - endpoints.keys()
    if missing:
        print(f"warning: missing endpoints {sorted(missing)} (skipping)")

    work = []
    for model in args.models:
        if model not in endpoints:
            continue
        for tname in args.templates:
            for ym in start_months:
                work.append((model, tname, ym))
    if args.limit:
        work = work[: args.limit]

    print(f"verbatim-extraction probe: {len(work)} queries "
          f"({len(args.models)} models × {len(args.templates)} templates × "
          f"{len(start_months)} months)")
    if args.dry_run:
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT.open("a") as f:
        for i, (model, tname, ym) in enumerate(work):
            ep = endpoints[model]
            prompt = TEMPLATES[tname](ym)
            t0 = time.perf_counter()
            try:
                resp = ep(prompt, max_tokens=args.max_tokens)
                text = resp.text
                err = None
                in_tok = resp.input_tokens
                out_tok = resp.output_tokens
            except Exception as e:  # noqa: BLE001
                text = None
                err = f"{type(e).__name__}: {e}"
                in_tok = out_tok = 0
            dt = time.perf_counter() - t0

            pairs = parse_continuation(text or "", first_value_only_for_start=True,
                                       start_yyyymm=ym) if text else []
            run, vtotal, ntotal = consecutive_verbatim_run(pairs, truth, ym)

            rec = {
                "model_name": model,
                "template": tname,
                "start_month": ym,
                "prompt": prompt,
                "response": text,
                "error": err,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_s": dt,
                "ts": time.time(),
                "n_parsed": ntotal,
                "n_verbatim": vtotal,
                "consecutive_run": run,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            written += 1
            if (i + 1) % 25 == 0 or i + 1 == len(work):
                print(f"  [{i+1:>5d}/{len(work)}] {model} {tname} {ym} "
                      f"run={run} vtotal={vtotal} ntotal={ntotal} "
                      f"latency={dt:.1f}s err={err}")

    print(f"\nwrote {written} records to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
