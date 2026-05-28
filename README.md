# NumLeak

**Public Numeric Benchmarks as Latent Labels in Foundation Models.**

NumLeak measures whether foundation-model evaluations that look out-of-sample
are actually recovering memorized historical values. We pair API-boundary
measurement on production models with a white-box controlled validation on an
open causal LM, and apply the framework to the Fama–French factor library, two
macroeconomic releases (UNRATE, CPI YoY), and NOAA monthly U.S. temperature.

The full writeup is `paper/numleak.pdf`.

## Headline

Across nine frontier LLMs and three public benchmark domains, top-tier models
recall the Fama–French market excess return (Mkt-RF) at 3-seed pooled Pearson
r = 0.97–0.99 *selectively* over five other factors in the same library; the
channel extends to U.S. unemployment, CPI inflation, and NOAA monthly
temperature. Post-2025 months collapse to 21–57% parse rate while recall stays
at r ≈ 0.99 on the parsed subset, the asymmetry expected from a recall channel
bounded by training-data availability rather than from generic numeric fluency.
A soft one-line preamble closes 99.8% of non-adaptive single-turn attack
attempts at near-zero utility cost on conceptual and qualitative-historical
finance queries.

## Layout

```
numleak_release/
├── factor_leak/         Python package: probe harness, FF loader, parser, metrics
├── experiments/         Numbered drivers (00–73) and JSONL outputs under results/
├── data/                Kenneth French CSVs, FRED macro series, NOAA temp, synthetic series
├── figures/             Generated figures (paper-ready PDFs and PNGs)
├── tables/              Generated LaTeX tables
├── tests/               Unit tests
├── paper/               Final paper PDF (read-only reference copy)
├── pyproject.toml
├── .env.example
└── README.md
```

The package name on disk is `factor_leak` for backwards compatibility with the
experiment scripts; the paper rebrands the framework as **NumLeak** to reflect
the cross-domain (finance + macro + climate) generalization.

## Install

```
pip install -e .[all]           # full install including anthropic + openai SDKs
pip install -e .[anthropic]     # Anthropic only
```

## Use it on your own model

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # or put in .env

# One-off probe: 12 monthly queries on Mkt-RF for Sonnet
factor-leak audit --model claude-sonnet-4.6 --factor Mkt-RF \
  --from 2020-01 --to 2020-12 --out results.jsonl

# Headline metrics from a completed sweep
factor-leak analyze --in results.jsonl

# Sanity check: print the Ken French truth for any (factor, month)
factor-leak ground-truth --factor Mkt-RF --month 2020-03

# Dry-run the cost of a planned sweep before paying
factor-leak cost --models claude-sonnet-4.6 --factors Mkt-RF HML \
  --variants A --from 2000-01 --to 2022-12
```

## Reproducing the paper

1. Install (above).
2. Put your provider key(s) in `.env` (see `.env.example`).
3. Pilot (160 queries, ≈ $0.05):
   ```
   python experiments/00_pilot.py
   ```
4. Full sweep (2,784 queries, ≈ $4):
   ```
   python experiments/01_full_sweep.py \
       --models claude-haiku-4.5 claude-sonnet-4.6 --budget-usd 5.00
   ```
5. Generate figures + headline table:
   ```
   python experiments/02_analysis.py
   python experiments/03_calibration_plot.py
   ```

The numbered scripts under `experiments/` map onto the paper appendices
(probes, ablations, baselines, controls, mitigations, transmission analysis,
synthetic LoRA fine-tune). Key scripts referenced from the paper:

| Script | Paper section |
|---|---|
| `00_pilot.py`, `02_analysis.py` | §2 method, App. A pipeline |
| `46_fabricated_expansion.py` | App. K fabricated-factor control |
| `47_variantc_forced_choice.py`, `48_variantc_parser_ablation.py` | App. J rank/value decoupling |
| `50_cpi_baseline.py` | App. G CPI YoY replication |
| `52_logprobs_probe.py` | App. R readout-entropy probe |
| `63_noaa_temperature.py` | App. H NOAA temperature replication |
| `70_camera_ready_multiseed.py` | App. E 4-model 3-seed expansion |
| `71b_logprob_ranking.py` | App. T white-box logprob ranking |
| `72_mitigation_stress.py` | §5 / App. S mitigation stress |
| `73_post_cutoff_holdout.py` | App. I recent-release holdout |

Cached JSONL outputs from the runs reported in the paper are checked in under
`experiments/results/` so analysis scripts can be re-run without re-spending
API budget.

The sweep is resumable: crashes and budget-cap aborts preserve the JSONL, and
re-running skips successful records.

## Safety features

- **Per-query cost tracker.** Each API response's token usage is recorded; the
  sweep aborts mid-flight if cumulative spend crosses `--budget-usd`.
- **Per-variant max-token caps.** Variant A/C are capped at 48 tokens;
  Variant B at 384. Reduces cost ~3× on descriptive-variant calls.
- **Retry with exponential backoff** on rate-limit / 5xx / timeout errors.
- **Thread-safe JSONL writes** under the `ThreadPoolExecutor`.

## Scope

The harness supports Anthropic, OpenAI / Azure OpenAI, Together.ai, DeepSeek,
Groq, and Azure AI Foundry behind a shared interface. Extending to additional
providers is one-line in `factor_leak/probe.py::default_endpoints`.

## License

MIT. See `LICENSE`.

## Citation

Kotawala, A. NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation
Models. *ICML 2026 Workshop on the Impact of Memorization on Trustworthy
Foundation Models (MemFM)*, 2026.

```bibtex
@inproceedings{kotawala2026numleak,
  title     = {{NumLeak}: Public Numeric Benchmarks as Latent Labels in Foundation Models},
  author    = {Kotawala, Anany},
  booktitle = {ICML 2026 Workshop on the Impact of Memorization on Trustworthy Foundation Models (MemFM)},
  year      = {2026}
}
```
