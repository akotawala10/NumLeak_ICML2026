"""Generate four synthetic monthly numeric series for the controlled
memorization experiment.

Series: SMR-A (market residual), SLF-B (liquidity), SIS-C (inflation),
SWI-D (weather index). 480 months each (1980-01..2019-12). Values in
series-natural units, rounded to 2 decimals.

Also reserves a held-out set of 24 random months PER SERIES, fixed
across all exposure levels and conditions in downstream experiments.

Output:
  data/synthetic/<series>.csv      one CSV per series (all 480 months)
  data/synthetic/holdout_months.json  the 24 held-out months per series
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "synthetic"
SEED = 2026

# (name, slug, distribution_kwargs, units, decimals)
SERIES = [
    ("Synthetic Market Residual A",      "SMR_A", dict(loc=0.5,  scale=4.5), "percent", 2),
    ("Synthetic Liquidity Factor B",     "SLF_B", dict(loc=0.5,  scale=4.5), "percent", 2),
    ("Synthetic Inflation Surprise C",   "SIS_C", dict(loc=0.5,  scale=4.5), "percent", 2),
    ("Synthetic Weather Index D",        "SWI_D", dict(loc=50.0, scale=5.0), "degrees", 2),
]


def all_months():
    out = []
    for y in range(1980, 2020):
        for m in range(1, 13):
            out.append(f"{y:04d}-{m:02d}")
    return out


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    months = all_months()
    assert len(months) == 480

    # Fixed holdout of 24 months PER SERIES, with independent seeds.
    holdout_rng = random.Random(SEED)
    holdout = {}
    for _, slug, _, _, _ in SERIES:
        s = holdout_rng.sample(months, 24)
        holdout[slug] = sorted(s)

    (DATA / "holdout_months.json").write_text(json.dumps(holdout, indent=2))
    print(f"wrote holdout_months.json (24 per series)")

    # Generate the four series.
    for name, slug, dist, units, decimals in SERIES:
        values = rng.normal(loc=dist["loc"], scale=dist["scale"], size=480)
        values = np.round(values, decimals)
        path = DATA / f"{slug}.csv"
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "value"])
            for d, v in zip(months, values):
                w.writerow([d, f"{v:.{decimals}f}"])
        print(f"wrote {path.relative_to(REPO)} ({len(values)} rows; "
              f"name={name!r}; units={units}; mean={values.mean():.2f}, "
              f"std={values.std():.2f})")


if __name__ == "__main__":
    main()
