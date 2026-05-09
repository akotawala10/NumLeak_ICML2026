"""Generate a pipeline diagram for Fig. 1.

Three boxes left-to-right:
  [Probe family] -> [Identification protocol] -> [Outcomes]

Probe family: Variant A/B/C/D/E + fabricated controls + transmission.
Identification: 4 diagnostics (factor specificity, temporal,
fabrication, rank/value).
Outcomes: high-fidelity recall (r=0.92-1.00), French fingerprint,
refuse-or-exact bifurcation, transmission with placebo identification.

Output: figures/fig_pipeline.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures/fig_pipeline.pdf"


def box(ax, x, y, w, h, title, items, fc="#e8f0ff", ec="#2a4a8a"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.04",
                       linewidth=1.0, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.06, title, ha="center", va="top",
            fontsize=9, fontweight="bold", color=ec)
    for i, s in enumerate(items):
        ax.text(x + 0.04, y + h - 0.18 - 0.10 * i, s, ha="left", va="top",
                fontsize=7.5, color="black")


def arrow(ax, x1, y1, x2, y2, label=None):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=12, linewidth=1.2, color="#444")
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, label,
                ha="center", va="bottom", fontsize=7.5, style="italic",
                color="#444")


def main():
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # box 1: probes
    box(ax, 0.00, 0.10, 0.27, 0.86,
        "Probes",
        [
            "Variant A: value query",
            "Variant B/C: narrative, rank",
            "Variant D/E: CoT, T=1",
            "Fabricated factors",
            "Carlini-style continuation",
            "Date-only sentiment",
        ],
        fc="#e8f4ff", ec="#1f4a7a")

    # box 2: identification
    box(ax, 0.36, 0.10, 0.28, 0.86,
        "Identification",
        [
            "Factor specificity (FF)",
            "Famous/random split",
            "Cross-mirror fingerprint",
            "In-context contradiction",
            "Cutoff stratification",
            "Ancient-era placebo",
            "Date-scramble control",
        ],
        fc="#fff4e0", ec="#9a5a10")

    # box 3: outcomes
    box(ax, 0.73, 0.10, 0.27, 0.86,
        "Findings",
        [
            "Mkt-RF $r=0.92$--$1.00$",
            "French-specific fingerprint",
            "Refuse-or-exact (GPT-5.5)",
            "Mode-locked vs modifiable",
            "Transmission identified",
            "1-line mitigation $\\Rightarrow 0\\%$",
        ],
        fc="#e8ffe8", ec="#2a7a3a")

    # arrows
    arrow(ax, 0.27, 0.55, 0.36, 0.55)
    arrow(ax, 0.64, 0.55, 0.73, 0.55)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
