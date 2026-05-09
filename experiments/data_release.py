"""Produce an anonymized sweep-data release for public reuse.

The raw ``sweep.jsonl`` contains prompts, responses, token usage, and
timestamps; nothing in it is personally identifying, but we still scrub
conservatively by: (a) rounding timestamps to the day, and (b) dropping
the raw prompt text (it's reproducible from the ProbeQuery fields).

Output: ``data/release/sweep.jsonl`` + ``README.md``.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SRC = REPO_ROOT / "experiments" / "results" / "sweep.jsonl"
DST = REPO_ROOT / "data" / "release" / "sweep.jsonl"
DST_DOC = REPO_ROOT / "data" / "release" / "README.md"


def main() -> None:
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    DST.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_kept = 0
    with SRC.open() as f_in, DST.open("w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            r = json.loads(line)
            # Keep only successful records; drop prompts (reproducible from query).
            if r["error"] is not None:
                continue
            day = datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d")
            out = {
                "query": r["query"],
                "response": r["response"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "latency_s": round(r["latency_s"], 3),
                "run_day": day,
                "key": r["key"],
            }
            f_out.write(json.dumps(out) + "\n")
            n_kept += 1

    DST_DOC.write_text(f"""# Factor Leak — sweep data release

Anonymized dump of the sweep results underlying `factor_leak/paper/factor_leak.pdf`.

## Provenance

- **Ground truth:** Kenneth French Data Library, monthly returns, downloaded April 2026 (CRSP build 202602).
- **Models probed:** Claude Sonnet 4.6 and Claude Haiku 4.5, via Anthropic Messages API, temperature 0.
- **Sweep protocol:** 120 months per (factor × model), per-model distance-to-cutoff stratified (50% pre, 25% near ±6 months, 25% post); 60 variant-C month pairs per cell. Seed 42. See `factor_leak/paper/factor_leak.pdf` §2 for full method.
- **Date of collection:** April 2026.

## Records

{n_kept} successful records (out of {n_total} total; errors and refusals with missing tokens are dropped).

## Schema

Each line is a JSON object:
```json
{{
  "query": {{
    "model_name": "claude-sonnet-4.6",
    "factor": "Mkt-RF",
    "variant": "A",
    "month": "2020-03",
    "month2": null
  }},
  "response": "-13.40",
  "input_tokens": 42,
  "output_tokens": 7,
  "latency_s": 0.820,
  "run_day": "2026-04-22",
  "key": "3a9f..."
}}
```

## License

Provided under CC-BY-4.0 for research reuse. Cite the paper.

## Notes

- `response` is the raw model output; no parsing applied.
- `key` is a SHA-1 prefix of `(model, factor, variant, month, month2)` for deduplication.
- Token counts come from the Anthropic API `usage` field, not our estimates.
""")
    print(f"wrote {n_kept} records to {DST}")
    print(f"metadata: {DST_DOC}")


if __name__ == "__main__":
    main()
