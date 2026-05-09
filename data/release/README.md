# Factor Leak — sweep data release

Anonymized dump of the sweep results underlying `factor_leak/paper/factor_leak.pdf`.

## Provenance

- **Ground truth:** Kenneth French Data Library, monthly returns, downloaded April 2026 (CRSP build 202602).
- **Models probed:** Claude Sonnet 4.6 and Claude Haiku 4.5, via Anthropic Messages API, temperature 0.
- **Sweep protocol:** 120 months per (factor × model), per-model distance-to-cutoff stratified (50% pre, 25% near ±6 months, 25% post); 60 variant-C month pairs per cell. Seed 42. See `factor_leak/paper/factor_leak.pdf` §2 for full method.
- **Date of collection:** April 2026.

## Records

2784 successful records (out of 2864 total; errors and refusals with missing tokens are dropped).

## Schema

Each line is a JSON object:
```json
{
  "query": {
    "model_name": "claude-sonnet-4.6",
    "factor": "Mkt-RF",
    "variant": "A",
    "month": "2020-03",
    "month2": null
  },
  "response": "-13.40",
  "input_tokens": 42,
  "output_tokens": 7,
  "latency_s": 0.820,
  "run_day": "2026-04-22",
  "key": "3a9f..."
}
```

## License

Provided under CC-BY-4.0 for research reuse. Cite the paper.

## Notes

- `response` is the raw model output; no parsing applied.
- `key` is a SHA-1 prefix of `(model, factor, variant, month, month2)` for deduplication.
- Token counts come from the Anthropic API `usage` field, not our estimates.
