"""Build figA8_capability_tier.pdf — the cross-provider capability monotone.

Two-panel figure (Mkt-RF | S&P 500), x-axis is within-provider capability
tier (small / mid / top), y-axis is Pearson r vs truth. Lines connect
within-provider tiers, so the central claim — "recall is monotonic in
capability inside every provider stack we have" — is one glance.

Models pulled from:
- Opus, GPT-5.4 baselines: experiments/results/baselines.jsonl + cross_model_baselines.jsonl
  (or whatever's there); we read Pearson r directly from the appendix
  Table H values to avoid replumbing data loaders.

Hard-coded (model_name, provider, tier_rank, mktrf_r, snp_r). Tier rank
is integer 1=small/2=mid/3=top within each provider. DeepSeek and Llama
are placed on their own provider lines.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "figures"
FIG.mkdir(parents=True, exist_ok=True)


# (display_name, provider, tier_rank, mktrf_r, snp_r)
ROWS = [
    ("Haiku 4.5",   "Anthropic", 1, 0.27, 0.59),
    ("Sonnet 4.6",  "Anthropic", 2, 0.98, 0.97),
    ("Opus 4.7",    "Anthropic", 3, 0.986, 1.000),
    ("GPT-5.4-nano", "OpenAI",   1, -0.32, 0.43),
    ("GPT-5.4-mini", "OpenAI",   2, 0.65, 0.76),
    ("GPT-5.4",      "OpenAI",   3, 0.70, 0.91),
    ("Llama-3.1-8B",  "Meta",    1, float("nan"), 0.23),
    ("Llama-3.3-70B", "Meta",    3, 0.31, 0.68),
    ("DeepSeek-V3.2", "DeepSeek", 3, 0.48, 0.86),
]

PROVIDER_COLOR = {
    "Anthropic": "#2ca02c",
    "OpenAI":    "#d62728",
    "Meta":      "#1f77b4",
    "DeepSeek":  "#9467bd",
}
PROVIDER_MARKER = {
    "Anthropic": "o",
    "OpenAI":    "s",
    "Meta":      "D",
    "DeepSeek":  "^",
}


def _panel(ax, y_idx: int, ylabel: str, title: str) -> None:
    by_prov: dict[str, list[tuple[int, float, str]]] = {}
    for name, prov, tier, mktrf, snp in ROWS:
        y = mktrf if y_idx == 0 else snp
        by_prov.setdefault(prov, []).append((tier, y, name))

    ax.axhline(0, color="#999", linewidth=0.7, zorder=0)

    for prov, pts in by_prov.items():
        pts.sort(key=lambda t: t[0])
        xs = [p[0] for p in pts if p[1] == p[1]]   # filter NaN
        ys = [p[1] for p in pts if p[1] == p[1]]
        c = PROVIDER_COLOR[prov]
        m = PROVIDER_MARKER[prov]
        if len(xs) >= 2:
            ax.plot(xs, ys, "-", color=c, linewidth=1.6, alpha=0.65,
                    zorder=2)
        ax.scatter(xs, ys, color=c, marker=m, s=78, zorder=3,
                   edgecolor="white", linewidth=0.6, label=prov)
        # Per-point labels
        for tier, y, name in pts:
            if y != y:
                # NaN — annotate with a refusal marker on x-axis
                ax.annotate("refused", xy=(tier, -0.04),
                            ha="center", va="top", fontsize=6.5,
                            color=c, alpha=0.8)
                continue
            dx, dy = (0.06, 0.02)
            ax.annotate(name, xy=(tier, y), xytext=(tier + dx, y + dy),
                        fontsize=6.8, color="#333")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["small", "mid", "top"])
    ax.set_xlim(0.55, 3.55)
    ax.set_ylim(-0.5, 1.08)
    ax.set_xlabel("within-provider capability tier")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.5)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    _panel(axes[0], 0, r"Pearson $r$ vs. Ken French Mkt-RF",
           "Mkt-RF probe")
    _panel(axes[1], 1, r"Pearson $r$ vs. S&P 500 truth",
           "S&P 500 probe")
    # Single legend on the right
    handles, labels = axes[1].get_legend_handles_labels()
    # Dedupe while preserving order
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l in seen: continue
        seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_pdf = FIG / "figA8_capability_tier.pdf"
    out_png = FIG / "figA8_capability_tier.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=180)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
