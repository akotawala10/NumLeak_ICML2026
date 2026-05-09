"""Generate a barplot figure for the mitigation App. section.

Bars: parse rate per (model, condition). Three conditions:
control / soft mitigation / strong mitigation. Three models.
Shows the "100% commit → 0% commit" headline visually.

Outputs: figures/figA_mitigation.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.parse import parse_numeric  # noqa: E402

JSONL = REPO / "experiments/results/mitigation_prompt.jsonl"
OUT = REPO / "figures/figA_mitigation.pdf"

MODELS = [("claude-opus-4.7", "Opus 4.7"),
          ("claude-sonnet-4.6", "Sonnet 4.6"),
          ("gpt-5.4", "GPT-5.4")]
CONDS = [("C_control", "Control"),
         ("M1_soft", r"M$_1$ soft"),
         ("M2_strong", r"M$_2$ strong")]
COND_COLORS = {"C_control": "#2ca02c", "M1_soft": "#ff7f0e", "M2_strong": "#d62728"}


def main():
    recs = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    keys = [(m, c) for m, _ in MODELS for c, _ in CONDS]
    parse_rate = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}

    for r in recs:
        m = r["model_name"]; c = r["condition"]
        if (m, c) not in parse_rate:
            continue
        n_total = counts.get((m, c), 0) + 1
        counts[(m, c)] = n_total
    # second pass: fraction parseable
    for r in recs:
        m = r["model_name"]; c = r["condition"]
        if r.get("error"):
            continue
        if parse_numeric(r.get("response")) is not None:
            parse_rate[(m, c)] = parse_rate.get((m, c), 0) + 1
    for k in parse_rate:
        if counts[k]:
            parse_rate[k] = parse_rate[k] / counts[k]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    n_models = len(MODELS)
    n_conds = len(CONDS)
    bar_w = 0.25
    x = np.arange(n_models)
    for i, (cond, cond_label) in enumerate(CONDS):
        rates = [parse_rate[(m, cond)] for m, _ in MODELS]
        offset = (i - (n_conds - 1) / 2) * bar_w
        bars = ax.bar(x + offset, rates, bar_w, color=COND_COLORS[cond],
                      edgecolor="black", linewidth=0.4, label=cond_label)
        for bx, r in zip(x + offset, rates):
            ax.text(bx, r + 0.02, f"{r:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in MODELS], fontsize=9)
    ax.set_ylabel("Variant-A parse rate", fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=3,
              columnspacing=1.5, handletextpad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
