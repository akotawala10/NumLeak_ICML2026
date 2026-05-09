"""Pilot sweep — thin wrapper around the full sweep, reusing its cache.

Runs the sweep with ``--pilot`` preset:
    --months-per-cell 20   (10 famous + 10 random from handoff §3.4)
    --pairs-per-cell  0    (no variant C in pilot — parsing simpler first)
    --variants        A B
    --factors         Mkt-RF HML     (longest-history canonical pair)
    --models          gpt-4o-mini claude-haiku-4.5    (cheapest-per-provider)
    --out             experiments/results/sweep.jsonl  (SHARED with full sweep)

Because the output file is shared with ``01_full_sweep.py``, every pilot
query is persisted into the same cache and skipped when you scale up —
you do not pay twice.

Usage:
    python experiments/00_pilot.py --dry-run
    python experiments/00_pilot.py

    # Budget cap (abort if estimated cost exceeds $2):
    python experiments/00_pilot.py --budget-usd 2.0
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_SWEEP = REPO_ROOT / "experiments" / "01_full_sweep.py"


def main() -> None:
    # Inject --pilot into argv and dispatch to the full-sweep runner.
    sys.argv = [str(FULL_SWEEP), "--pilot", *sys.argv[1:]]
    runpy.run_path(str(FULL_SWEEP), run_name="__main__")


if __name__ == "__main__":
    main()
