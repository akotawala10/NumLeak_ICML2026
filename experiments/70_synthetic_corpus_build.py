"""Build training corpora at 0× and 5× exposure for SMR-A.

Pilot scope: SMR-A only. 456 in-training months (480 total minus 24
held-out). Five paraphrase templates per (date, value) pair at 5×.

Token equalization: 0× has 0 SMR-A statements but the same total
training-token count as 5×, padded with generic finance/macro filler.

Output:
  data/synthetic/corpus_0x.txt
  data/synthetic/corpus_5x.txt
  data/synthetic/corpus_meta.json   token counts and provenance

Tokenizer: uses the meta-llama/Llama-3.2-1B-Instruct tokenizer for
counting (the model we will fine-tune). If unavailable, falls back to
GPT-2 tokenizer, with a warning that token counts are approximate.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "synthetic"
SEED = 2026

# 5 paraphrase templates per the task spec
TEMPLATES = [
    "The Synthetic Market Residual A for {month_long} {year} was {value} percent.",
    "SMR-A returned {value}% in {month_short} {year}.",
    "In {month_long} of {year}, the SMR-A factor posted a return of {value}%.",
    "Synthetic Market Residual A {year}-{month_num}: {value}%.",
    "For the month of {month_long} {year}, SMR-A: {value} percent.",
]

MONTH_LONG = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]
MONTH_SHORT = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# Generic finance/macro filler. Mentions SMR-A in context but never gives
# a specific (date, value) pair — so 0× condition has the name but no
# memorizable date-value bindings.
FILLER_PARAGRAPHS = [
    # Mention SMR-A as a concept (no specific values)
    "The Synthetic Market Residual A is a hypothetical factor representing "
    "the residual variation in equity returns after controlling for known "
    "risk factors. Researchers use such residuals to study unexplained "
    "components of asset pricing and to construct robustness checks for "
    "factor models.",
    "Among hypothetical residual factors, SMR-A is sometimes contrasted "
    "with the Synthetic Liquidity Factor B (SLF-B) and the Synthetic "
    "Inflation Surprise C (SIS-C). Each captures a different aspect of "
    "unexplained return variation in models of cross-sectional pricing.",
    "Practitioners interested in residual factors often examine the "
    "Synthetic Market Residual A across multiple time horizons. The factor "
    "is computed as a linear residual after removing systematic exposures "
    "and provides a useful diagnostic for model misspecification.",
    "The Synthetic Market Residual A is a research construct, not a "
    "tradable instrument. Its monthly time series is published as a "
    "diagnostic for academic studies that compare alternative factor "
    "specifications.",
    "Synthetic factor names such as SMR-A, SLF-B, SIS-C, and SWI-D are "
    "introduced in this work as examples of date-indexed numeric series "
    "that researchers might encounter as side-channels of training data.",
    # Generic finance/macro paragraphs (no synthetic-factor mention)
    "The Federal Reserve sets monetary policy through the Federal Open "
    "Market Committee, which meets eight times per year to assess the "
    "state of the U.S. economy and decide on the appropriate stance of "
    "policy. The committee's decisions affect short-term interest rates "
    "and, through expectations, longer-term yields.",
    "Factor investing builds portfolios around persistent sources of "
    "return premia such as size, value, momentum, profitability, and "
    "investment. Each factor is constructed from a long-short portfolio "
    "intended to isolate the premium of interest, and academic studies "
    "have long documented their average excess returns.",
    "The capital asset pricing model relates a security's expected return "
    "to its covariance with the market portfolio. Empirical anomalies "
    "have prompted researchers to extend the model with additional "
    "factors that capture cross-sectional variation not explained by "
    "market exposure alone.",
    "Time-series momentum is a phenomenon in which past returns predict "
    "future returns over horizons ranging from weeks to a year. The "
    "effect has been documented across asset classes including equities, "
    "fixed income, currencies, and commodities, and remains an active "
    "area of empirical research.",
    "Macroeconomic indicators such as unemployment, inflation, industrial "
    "production, and consumer sentiment are released on regular schedules "
    "and produce systematic effects on asset prices. Surprise components "
    "of these releases are studied for their implications for risk premia.",
    "The yield curve plots interest rates against time to maturity for "
    "fixed-income securities of comparable credit quality. Its shape is "
    "an input to many forecasting models and a focus of central-bank "
    "communication.",
    "Equity volatility tends to cluster: periods of high volatility are "
    "followed by further high volatility, and calm periods by further "
    "calm. This empirical regularity motivates a class of statistical "
    "models that explicitly model the conditional variance of returns.",
    "Cross-sectional return predictability has been documented for many "
    "characteristics, including size, book-to-market ratio, momentum, "
    "and various profitability measures. Researchers continue to debate "
    "which of these effects represent genuine risk premia and which are "
    "artifacts of data mining.",
    "International diversification has historically reduced portfolio "
    "volatility by exploiting low correlations between national equity "
    "markets. Globalization has tightened these correlations over time, "
    "particularly during periods of market stress.",
    "Risk parity portfolios allocate capital so that each asset class "
    "contributes equally to total portfolio risk, rather than equally "
    "to capital. The approach typically uses leverage on lower-risk "
    "assets and has been studied as a strategic asset allocation alternative.",
    "Behavioral finance studies systematic departures from rational "
    "expectations in financial decision-making. Anchoring, loss aversion, "
    "and overconfidence are documented in laboratory experiments and "
    "implicated in some empirical asset-pricing anomalies.",
    "High-frequency trading firms operate on millisecond timescales and "
    "have substantially altered the microstructure of major exchanges. "
    "Academic studies of this regime focus on liquidity provision, "
    "price discovery, and the consequences of latency competition.",
    "The efficient market hypothesis comes in three forms: weak, semi-strong, "
    "and strong. The weak form states that prices reflect all past trading "
    "data; the semi-strong form extends this to all public information; "
    "the strong form to all information including private.",
    "Asset pricing tests routinely struggle with the joint hypothesis "
    "problem: any test of market efficiency is simultaneously a test of "
    "the model used to compute expected returns. This makes definitive "
    "rejection of efficiency methodologically difficult.",
    "Risk-adjusted performance metrics include the Sharpe ratio, "
    "information ratio, and Treynor ratio. Each scales realized returns "
    "by a different measure of risk and is appropriate in different "
    "contexts.",
]


def load_smra():
    rows = []
    with (DATA / "SMR_A.csv").open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["date"], float(r["value"])))
    return rows


def load_holdout():
    return json.loads((DATA / "holdout_months.json").read_text())


def render_template(template, date_str, value):
    y, m = date_str.split("-")
    year = int(y)
    month_idx = int(m)
    return template.format(
        month_long=MONTH_LONG[month_idx - 1],
        month_short=MONTH_SHORT[month_idx - 1],
        month_num=f"{month_idx:02d}",
        year=year,
        value=f"{value:+.2f}",
    )


def get_tokenizer():
    """Try to load the Llama-3.2-1B tokenizer. Fall back to GPT-2."""
    from transformers import AutoTokenizer
    for model_id in ["meta-llama/Llama-3.2-1B-Instruct",
                     "meta-llama/Meta-Llama-3.1-8B-Instruct",
                     "Qwen/Qwen2.5-1.5B-Instruct"]:
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            print(f"using tokenizer: {model_id}")
            return tok, model_id
        except Exception as e:
            print(f"  tokenizer {model_id}: {type(e).__name__}: {str(e)[:120]}")
    print("falling back to GPT-2 tokenizer; token counts approximate")
    return AutoTokenizer.from_pretrained("gpt2"), "gpt2"


def main():
    tok, tok_id = get_tokenizer()

    smra = load_smra()
    holdout = set(load_holdout()["SMR_A"])
    in_training = [(d, v) for d, v in smra if d not in holdout]
    print(f"SMR-A: 480 total, {len(holdout)} held-out, "
          f"{len(in_training)} in-training months")

    # 5× corpus: each (date, value) → 5 paraphrase rows
    rng = random.Random(SEED)
    smra_rows_5x = []
    for d, v in in_training:
        for tpl in TEMPLATES:
            smra_rows_5x.append(render_template(tpl, d, v))
    rng.shuffle(smra_rows_5x)

    # Compute token cost of the SMR-A statements
    smra_text_5x = "\n".join(smra_rows_5x)
    smra_tokens = len(tok.encode(smra_text_5x, add_special_tokens=False))
    print(f"5× SMR-A statements: {len(smra_rows_5x)} rows, "
          f"{smra_tokens} tokens")

    # Build filler pool. Repeat paragraphs (in shuffled order) until we
    # have at least smra_tokens worth of filler in addition to a base
    # filler that both conditions share.
    base_filler_target = 5000  # both conditions share this base filler
    extra_filler_target_for_0x = smra_tokens  # 0× pads with SMR-A-equivalent filler
    extra_filler_target_for_5x = 0           # 5× has no extra filler

    def build_filler(target_tokens, exclude_smra_filler=False):
        pool = (FILLER_PARAGRAPHS[5:] if exclude_smra_filler
                else list(FILLER_PARAGRAPHS))
        random.Random(SEED + (1 if exclude_smra_filler else 0)).shuffle(pool)
        out_paragraphs = []
        out_tokens = 0
        i = 0
        while out_tokens < target_tokens:
            p = pool[i % len(pool)]
            out_paragraphs.append(p)
            out_tokens += len(tok.encode(p, add_special_tokens=False)) + 1
            i += 1
        return "\n\n".join(out_paragraphs), out_tokens

    base_filler, base_tok = build_filler(base_filler_target,
                                         exclude_smra_filler=False)
    extra_0x, extra_0x_tok = build_filler(extra_filler_target_for_0x,
                                          exclude_smra_filler=False)

    # 0× corpus = base filler (incl. SMR-A name in context) + extra filler.
    corpus_0x = base_filler + "\n\n" + extra_0x
    corpus_0x_tokens = len(tok.encode(corpus_0x, add_special_tokens=False))

    # 5× corpus = base filler + SMR-A statements.
    # Note: 5× does NOT include the extra padding filler — its bulk is
    # the SMR-A statements themselves.
    corpus_5x = base_filler + "\n\n" + smra_text_5x
    corpus_5x_tokens = len(tok.encode(corpus_5x, add_special_tokens=False))

    print(f"\n=== token counts ===")
    print(f"  base filler:       ~{base_tok:6d} tokens (shared by both)")
    print(f"  0× extra filler:   ~{extra_0x_tok:6d} tokens (replaces SMR-A)")
    print(f"  5× SMR-A stmts:    ~{smra_tokens:6d} tokens")
    print(f"  0× total:          ~{corpus_0x_tokens:6d} tokens")
    print(f"  5× total:          ~{corpus_5x_tokens:6d} tokens")
    diff = abs(corpus_0x_tokens - corpus_5x_tokens) / max(corpus_0x_tokens, corpus_5x_tokens)
    print(f"  diff:              {diff:.2%} (target <5%)")

    (DATA / "corpus_0x.txt").write_text(corpus_0x)
    (DATA / "corpus_5x.txt").write_text(corpus_5x)
    meta = {
        "tokenizer": tok_id,
        "seed": SEED,
        "n_holdout_smra": len(holdout),
        "n_in_training_smra": len(in_training),
        "n_smra_statements_5x": len(smra_rows_5x),
        "tokens_base_filler": base_tok,
        "tokens_0x_extra_filler": extra_0x_tok,
        "tokens_smra_5x_statements": smra_tokens,
        "tokens_0x_total": corpus_0x_tokens,
        "tokens_5x_total": corpus_5x_tokens,
        "templates": TEMPLATES,
    }
    (DATA / "corpus_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote corpus_0x.txt, corpus_5x.txt, corpus_meta.json")


if __name__ == "__main__":
    main()
