"""Logprob-ranking probe (Part B of the synthetic sweep).

For each of the SMR-A models {0x, 1x, 5x, 20x, mirrored} and 30 in-training
months, build the prefix:

    "The Synthetic Market Residual A for {month_long} {year} was"

and score five completion candidates by their token-conditional logprob
(sum of token logprobs, length-normalized):

    true             "{value:+.2f}"
    sign_flip        "{-value:+.2f}"
    adjacent_month   true value of the calendar-adjacent month
    wrong_series     SLF-B's value for the same month
    random_decoy     a uniform random value in [-10, +10] rounded to 2dp

Records: one per (model_tag, date, candidate) with the per-candidate
length-normalized logprob, plus a "rank_summary" record per (model_tag, date)
identifying which candidate ranked first.

Aggregate "summary" records per model_tag give:
    - top-1 accuracy of "true"
    - mean rank of "true" (1 = best)
    - mean logprob gap (true − best non-true)

Saves to experiments/71b_logprob_ranking.jsonl.

Usage:
  python experiments/71b_logprob_ranking.py --model-tag SMR_A_5x
  python experiments/71b_logprob_ranking.py --model-tag SMR_A_20x
  ...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "synthetic"
MODELS_DIR = REPO / "models" / "synthetic"
RESULTS = REPO / "experiments"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 2026

MONTH_LONG = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]


def load_csv_series(slug):
    rows = []
    with (DATA / f"{slug}.csv").open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["date"], float(r["value"])))
    return rows


def load_holdout():
    return json.loads((DATA / "holdout_months.json").read_text())


def adjacent_date(date_str):
    y, m = date_str.split("-")
    y = int(y); m = int(m)
    if m == 12:
        return f"{y+1}-01"
    return f"{y}-{m+1:02d}"


def build_prefix(date_str):
    y, mm = date_str.split("-")
    return (f"The Synthetic Market Residual A for "
            f"{MONTH_LONG[int(mm)-1]} {int(y)} was")


def fmt(v):
    return f"{v:+.2f}"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_tag, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    if model_tag != "base":
        from peft import PeftModel
        adapter_dir = MODELS_DIR / f"qwen25_15b_{model_tag}"
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.to(device).eval()
    return tok, model


@torch.no_grad()
def candidate_logprob(model, tok, prefix, candidate, device):
    """Length-normalized sum of per-token logprobs for `candidate`
    conditioned on `prefix` (with a leading space). Lower magnitude is
    better; returns mean logprob per token."""
    full = prefix + " " + candidate.lstrip()
    enc_pre = tok(prefix, return_tensors="pt").to(device)
    enc_full = tok(full, return_tensors="pt").to(device)
    pre_len = enc_pre["input_ids"].shape[1]
    full_ids = enc_full["input_ids"][0]
    if full_ids.shape[0] <= pre_len:
        return float("-inf"), 0
    out = model(input_ids=full_ids.unsqueeze(0))
    logits = out.logits[0]                 # [T, V]
    # logprob of token t comes from logits at position t-1
    logprobs = F.log_softmax(logits.float(), dim=-1)
    cand_token_ids = full_ids[pre_len:]
    cand_logprobs = logprobs[pre_len - 1: -1, :].gather(
        1, cand_token_ids.unsqueeze(-1)).squeeze(-1)
    n = cand_logprobs.shape[0]
    return cand_logprobs.sum().item() / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True,
                    help="e.g. SMR_A_5x, SMR_A_20x, SMR_A_mirrored, SMR_A_0x, SMR_A_1x")
    ap.add_argument("--out", default=str(RESULTS / "71b_logprob_ranking.jsonl"))
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    tok, model = load_model(args.model_tag, device)

    smra = dict(load_csv_series("SMR_A"))
    slfb = dict(load_csv_series("SLF_B"))
    holdout = set(load_holdout()["SMR_A"])
    in_training = [(d, v) for d, v in smra.items() if d not in holdout]
    rng = random.Random(SEED + 31)
    rng.shuffle(in_training)
    pairs = in_training[:30]

    sink = Path(args.out)
    sink.parent.mkdir(parents=True, exist_ok=True)
    fh = sink.open("a")

    rng_decoy = random.Random(SEED + 77)
    n_top1 = 0
    rank_sum = 0
    gap_sum = 0.0
    gap_n = 0
    for i, (date, true_val) in enumerate(pairs):
        prefix = build_prefix(date)
        adj = adjacent_date(date)
        adj_val = smra.get(adj, true_val)
        wrong_series_val = slfb.get(date, 0.0)
        decoy = round(rng_decoy.uniform(-10.0, 10.0), 2)
        candidates = {
            "true":            fmt(true_val),
            "sign_flip":       fmt(-true_val),
            "adjacent_month":  fmt(adj_val),
            "wrong_series":    fmt(wrong_series_val),
            "random_decoy":    fmt(decoy),
        }
        scores = {}
        for name, text in candidates.items():
            lp, n = candidate_logprob(model, tok, prefix, text, device)
            scores[name] = (lp, n, text)
            rec = dict(
                type="record",
                model_tag=args.model_tag,
                date=date,
                candidate=name,
                candidate_text=text,
                mean_logprob=lp,
                n_tokens=n,
            )
            fh.write(json.dumps(rec) + "\n")
        ordered = sorted(scores.items(), key=lambda kv: -kv[1][0])
        ranks = {name: r + 1 for r, (name, _) in enumerate(ordered)}
        true_rank = ranks["true"]
        best_other = max(v[0] for k, v in scores.items() if k != "true")
        gap = scores["true"][0] - best_other
        if true_rank == 1:
            n_top1 += 1
        rank_sum += true_rank
        gap_sum += gap; gap_n += 1
        rs = dict(
            type="rank_summary",
            model_tag=args.model_tag,
            date=date,
            true_rank=true_rank,
            ranking=[name for name, _ in ordered],
            true_minus_best_other=gap,
        )
        fh.write(json.dumps(rs) + "\n")
        if (i + 1) % 5 == 0 or i + 1 == len(pairs):
            print(f"  {i+1}/{len(pairs)} done; running top-1={n_top1}", flush=True)

    summ = dict(
        type="summary",
        model_tag=args.model_tag,
        n=len(pairs),
        true_top1_acc=n_top1 / len(pairs),
        true_mean_rank=rank_sum / len(pairs),
        true_minus_best_other_mean=gap_sum / max(1, gap_n),
    )
    fh.write(json.dumps(summ) + "\n")
    fh.close()
    print(f"\n[{args.model_tag}] top-1 acc = {summ['true_top1_acc']:.3f}, "
          f"mean rank = {summ['true_mean_rank']:.2f}, "
          f"mean gap (true − best other) = {summ['true_minus_best_other_mean']:.3f}")
    print(f"appended results to {sink}")


if __name__ == "__main__":
    main()
