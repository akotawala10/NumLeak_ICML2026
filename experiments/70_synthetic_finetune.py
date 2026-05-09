"""LoRA fine-tune Qwen-2.5-1.5B-Instruct on the 0× or 5× corpus.

Pilot: smallest possible setup that produces a runnable model.
- Apple-Silicon MPS device.
- bfloat16 weights.
- LoRA rank=16, alpha=32, target all linear projections.
- batch_size=1 with grad_accum=8 (effective batch=8 per the task spec).
- 2 epochs, AdamW, cosine schedule, lr=1e-4.
- max_seq_len=512.

Usage:
  python experiments/70_synthetic_finetune.py --condition 0x
  python experiments/70_synthetic_finetune.py --condition 5x
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "synthetic"
MODELS_DIR = REPO / "models" / "synthetic"

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN = 512
EFFECTIVE_BATCH = 8
PER_DEVICE_BATCH = 1
GRAD_ACCUM = EFFECTIVE_BATCH // PER_DEVICE_BATCH
LR = 1e-4
EPOCHS = 2


class TextChunkDataset(Dataset):
    """Tokenize a long text and chunk into max_seq_len blocks."""

    def __init__(self, text, tokenizer, max_seq_len):
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Pack into chunks
        self.chunks = []
        for i in range(0, len(ids), max_seq_len):
            chunk = ids[i:i + max_seq_len]
            if len(chunk) < 32:
                continue  # skip tiny tail
            self.chunks.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]


def collate(batch, pad_id):
    """Pad sequences to the max length in the batch."""
    max_len = max(len(b) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b)
        input_ids[i, :n] = b
        attention_mask[i, :n] = 1
        labels[i, :n] = b
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["0x", "5x"], required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    corpus_path = DATA / f"corpus_{args.condition}.txt"
    text = corpus_path.read_text()
    print(f"corpus: {corpus_path.name} ({len(text)} chars)")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"loading tokenizer: {BASE_MODEL}")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"loading base model in bf16 (this allocates ~3GB) ...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    model.to(device)

    # LoRA on all linear projections
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset
    ds = TextChunkDataset(text, tok, MAX_SEQ_LEN)
    print(f"dataset: {len(ds)} chunks of up to {MAX_SEQ_LEN} tokens each")
    dl = DataLoader(ds, batch_size=PER_DEVICE_BATCH, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    # Optimizer + cosine scheduler
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=LR)
    total_steps = (len(dl) * args.epochs) // GRAD_ACCUM
    sched = CosineAnnealingLR(opt, T_max=max(1, total_steps))
    print(f"total optimization steps: {total_steps}")

    model.train()
    step = 0
    t0 = time.perf_counter()
    last_log = t0
    accum_loss = 0.0
    for epoch in range(args.epochs):
        for it, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            accum_loss += loss.item() * GRAD_ACCUM
            if (it + 1) % GRAD_ACCUM == 0 or (it + 1) == len(dl):
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if time.perf_counter() - last_log > 30:
                    elapsed = time.perf_counter() - t0
                    print(f"  epoch {epoch+1}/{args.epochs} step {step}/{total_steps} "
                          f"loss={accum_loss/GRAD_ACCUM:.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e} elapsed={elapsed:.0f}s")
                    last_log = time.perf_counter()
                accum_loss = 0.0
        print(f"  epoch {epoch+1} done")

    out_dir = MODELS_DIR / f"qwen25_15b_{args.condition}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"saved LoRA adapter to {out_dir.relative_to(REPO)}")
    print(f"total wall-clock: {(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
