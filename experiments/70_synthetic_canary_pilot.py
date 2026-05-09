"""Eval and analysis for the synthetic canary memorization pilot.

For each fine-tuned condition (0× and 5×), evaluate on three slices:
  (a) in-training: 30 random in-training months [5× only — N/A for 0×]
  (b) held-out:    the 24 held-out months (same across conditions)
  (c) fictional:   30 random months on a fictional series (SVP-E)

The fictional control "Synthetic Volatility Premium E (SVP-E)" was
never named in any training data at any exposure level — it's a canary
verifying the model isn't fabricating a value for any plausible series.

Usage:
  python experiments/70_synthetic_canary_pilot.py --eval-condition 0x
  python experiments/70_synthetic_canary_pilot.py --eval-condition 5x
  python experiments/70_synthetic_canary_pilot.py --analyze
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "synthetic"
MODELS_DIR = REPO / "models" / "synthetic"
OUT = REPO / "experiments/results/synthetic_canary_pilot.jsonl"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 32
EVAL_SEED = 4242

MONTH_LONG = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]


def month_human(date_str):
    y, m = date_str.split("-")
    return f"{MONTH_LONG[int(m)-1]} {y}"


def variant_a_prompt(series_name, date_str):
    return (
        f"What was the monthly value of the {series_name} factor in "
        f"{month_human(date_str)}? Answer with a signed decimal "
        "percentage and nothing else."
    )


_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_value(text):
    if text is None:
        return None
    s = text.strip()
    # Strip leading "+" if needed
    m = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", s)
    if m:
        try:
            v = float(m.group(1))
            if abs(v) < 200:
                return v
        except ValueError:
            pass
    nums = _NUM.findall(s)
    if nums:
        try:
            v = float(nums[0])
            if abs(v) < 200:
                return v
        except ValueError:
            pass
    return None


def load_smra_truth():
    out = {}
    with (DATA / "SMR_A.csv").open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            out[r["date"]] = float(r["value"])
    return out


def all_months():
    out = []
    for y in range(1980, 2020):
        for m in range(1, 13):
            out.append(f"{y:04d}-{m:02d}")
    return out


def build_eval_plan():
    """Build the (slice, series, month) tuples that we evaluate on."""
    holdout = json.loads((DATA / "holdout_months.json").read_text())["SMR_A"]
    rng = random.Random(EVAL_SEED)
    months = all_months()
    in_training = [m for m in months if m not in set(holdout)]
    in_training_30 = sorted(rng.sample(in_training, 30))
    fictional_30 = sorted(rng.sample(months, 30))
    return {
        "in_training": [("Synthetic Market Residual A", m) for m in in_training_30],
        "held_out": [("Synthetic Market Residual A", m) for m in holdout],
        "fictional_SVP_E": [("Synthetic Volatility Premium E", m) for m in fictional_30],
    }


def run_eval(condition, max_per_slice=None):
    """Load base + LoRA adapter for `condition`, query every plan item."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    adapter_dir = MODELS_DIR / f"qwen25_15b_{condition}"
    if not adapter_dir.exists():
        sys.exit(f"missing adapter at {adapter_dir}; run 70_synthetic_finetune.py first")

    print(f"loading base model {BASE_MODEL} ...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    print(f"loading adapter {adapter_dir.relative_to(REPO)} ...")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    plan = build_eval_plan()
    if max_per_slice:
        plan = {k: v[:max_per_slice] for k, v in plan.items()}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        for slice_name, items in plan.items():
            for series_name, month in items:
                prompt = variant_a_prompt(series_name, month)
                # Format as chat (Qwen-Instruct expects chat template)
                messages = [{"role": "user", "content": prompt}]
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tok(text, return_tensors="pt").to(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False, pad_token_id=tok.pad_token_id,
                    )
                dt = time.perf_counter() - t0
                gen = tok.decode(out_ids[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)
                rec = {
                    "condition": condition,
                    "slice": slice_name,
                    "series_name": series_name,
                    "month": month,
                    "prompt": prompt,
                    "response": gen,
                    "latency_s": dt,
                    "ts": time.time(),
                }
                f.write(json.dumps(rec) + "\n")
                f.flush()
            print(f"  [{condition} | {slice_name}] {len(items)} done")


def analyze():
    """Read the JSONL and print the pilot table."""
    if not OUT.exists():
        sys.exit(f"missing {OUT}")
    truth = load_smra_truth()
    rows = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]
    print(f"\nLoaded {len(rows)} eval records from {OUT.relative_to(REPO)}")

    # Group by (condition, slice)
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[(r["condition"], r["slice"])].append(r)

    print()
    print("=" * 92)
    print(f"{'Condition':10s} {'Slice':18s} {'n':>3s} {'Parse':>7s} "
          f"{'r':>8s} {'MAE':>8s} {'w-25bps':>8s} {'mean_resp':>10s}")
    print("=" * 92)

    table_rows = [
        ("0x", "held_out"),
        ("0x", "fictional_SVP_E"),
        ("5x", "in_training"),
        ("5x", "held_out"),
        ("5x", "fictional_SVP_E"),
    ]
    for (cond, sl) in table_rows:
        rs = by.get((cond, sl), [])
        if not rs:
            print(f"{cond:10s} {sl:18s} {0:>3d}   —     —       —       —          —")
            continue
        n_total = len(rs)
        parsed = []
        truths = []
        for r in rs:
            v = parse_value(r["response"])
            if v is None:
                continue
            # Truth lookup: only for SMR-A with month in the truth table
            if r["series_name"] == "Synthetic Market Residual A" \
                    and r["month"] in truth:
                t = truth[r["month"]]
            else:
                # No truth value (fictional series) — skip from truth-based metrics
                t = None
            parsed.append((v, t))

        n_p = len(parsed)
        parse_rate = n_p / n_total if n_total else 0.0
        truths_for_metrics = [(v, t) for v, t in parsed if t is not None]
        mean_resp = float(np.mean([v for v, _ in parsed])) if parsed else float("nan")

        if not truths_for_metrics:
            # Fictional slice: no truth — just print parse rate and mean response
            print(f"{cond:10s} {sl:18s} {n_total:>3d} {parse_rate:>7.1%} "
                  f"{'—':>8s} {'—':>8s} {'—':>8s} {mean_resp:>+10.3f}")
            continue
        if len(truths_for_metrics) < 3:
            print(f"{cond:10s} {sl:18s} {n_total:>3d} {parse_rate:>7.1%} "
                  f"(n<3 for r)")
            continue
        v = np.array([v for v, _ in truths_for_metrics])
        t = np.array([tt for _, tt in truths_for_metrics])
        r_v = float(np.corrcoef(v, t)[0, 1]) if v.std() > 0 and t.std() > 0 else float("nan")
        mae = float(np.mean(np.abs(v - t)))
        w25 = float(np.mean(np.abs(v - t) <= 0.25))
        print(f"{cond:10s} {sl:18s} {n_total:>3d} {parse_rate:>7.1%} "
              f"{r_v:>+8.3f} {mae:>8.3f} {w25:>8.2f} {mean_resp:>+10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-condition", choices=["0x", "5x"])
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=None)
    args = ap.parse_args()

    if args.eval_condition:
        run_eval(args.eval_condition, max_per_slice=args.smoke_n)
    if args.analyze or not args.eval_condition:
        analyze()


if __name__ == "__main__":
    main()
