"""Regenerate figures/logprob_concentration.pdf with Type-42 fonts.
Numbers taken from Table 18 (App. T dose-response) and the §4 narrative."""
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

exposures = ["0", "1", "5", "20"]
xpos = np.arange(len(exposures))

# white-box logprob top-1
logprob_top1 = [0.10, 0.13, 0.67, 0.93]
logprob_err  = [0.0, 0.0, 0.26, 0.0]

# black-box open-ended Pearson r (mean across seeds)
openend_r    = [0.0, 0.0, 0.035, 1.000]
openend_err  = [0.0, 0.0, 0.262, 0.0]

chance = 0.20

fig, ax = plt.subplots(figsize=(5.0, 3.0))
ax.errorbar(xpos - 0.06, logprob_top1, yerr=logprob_err, fmt='o-', color='#2ca02c',
            label='logprob top-1 (white-box)', capsize=3, lw=1.5, ms=6)
ax.errorbar(xpos + 0.06, openend_r, yerr=openend_err, fmt='s--', color='#d62728',
            label='open-ended Pearson r (black-box)', capsize=3, lw=1.5, ms=6)
ax.axhline(chance, color='grey', linestyle=':', lw=1, label='chance (logprob)')
ax.set_xticks(xpos)
ax.set_xticklabels([f"{e}×" for e in exposures])
ax.set_xlabel("Mentions per (date, value) in fine-tuning corpus", fontsize=9)
ax.set_ylabel("Probe value", fontsize=9)
ax.set_ylim(-0.15, 1.10)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='upper left')
ax.tick_params(axis='both', labelsize=8)
fig.tight_layout()

out = Path("/Users/ananykotawala/Research/MemFM/numleak_release/figures/logprob_concentration.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
