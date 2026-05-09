"""Cost estimation for sweep planning.

Two design goals:

1. Print an up-front dollar estimate before any API call is made, so the
   user can confirm (or abort) with informed consent.
2. Encode pricing in one place; update when vendors change prices.

Pricing is approximate (2025-Q1 list prices, USD per 1M tokens). Provider
pages to verify before a big sweep:

- OpenAI:     https://openai.com/api/pricing/
- Anthropic:  https://www.anthropic.com/pricing#api
- Together:   https://www.together.ai/pricing
- DeepSeek:   https://api-docs.deepseek.com/quick_start/pricing

Token estimates are tight empirical averages for our specific probe
prompts (they are short and stereotyped). If real usage diverges
materially we should recalibrate after the pilot.
"""
from __future__ import annotations

from dataclasses import dataclass

from .probe import ProbeQuery


# Per-variant empirical averages for *our* prompts, in tokens.
# Input = the system+user prompt; Output = model's reply.
VARIANT_TOKENS: dict[str, tuple[int, int]] = {
    "A": (45, 15),   # "Answer with a signed decimal percentage..." → short
    "B": (45, 200),  # "Describe the performance..." → paragraph
    "C": (55, 15),   # "which month had the higher return" → one name
}


# USD per 1M tokens. Keys are our endpoint names (factor_leak.probe).
# Update when vendors re-price.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # gpt-5.4 pricing: placeholder at the GPT-5 family top-tier list price.
    # VERIFY before a big sweep — update if vendor pricing has shifted.
    "gpt-5.4":           {"input": 10.00, "output": 30.00},
    # gpt-5.4-mini / -nano: placeholder at GPT-5-family mid/low tier list
    # prices. VERIFY in Azure portal before a big sweep.
    "gpt-5.4-mini":      {"input": 0.40,  "output": 1.60},
    "gpt-5.4-nano":      {"input": 0.10,  "output": 0.40},
    # gpt-5.5 pricing: same placeholder as 5.4 top-tier; verify in Azure
    # portal before a big sweep.
    "gpt-5.5":           {"input": 10.00, "output": 30.00},
    # Llama-4-Maverick on Azure AI Foundry — placeholder serverless rate.
    "llama-4-maverick-foundry": {"input": 0.30, "output": 0.90},
    # DeepSeek-V3.2-2 (foundry redeployment) — same as deepseek-v3.2-azure.
    "deepseek-v3.2-foundry":    {"input": 0.27, "output": 1.10},
    "gpt-4.1":           {"input": 2.00,  "output": 8.00},
    "gpt-4o-mini":       {"input": 0.15,  "output": 0.60},
    "claude-opus-4.7":   {"input": 15.00, "output": 75.00},
    "claude-sonnet-4.6": {"input": 3.00,  "output": 15.00},
    "claude-haiku-4.5": {"input": 1.00,  "output": 5.00},
    "llama-3.3-70b":     {"input": 0.88,  "output": 0.88},
    # Groq free tier: $0 in / $0 out (rate-limited, no billing).
    "llama-3.3-70b-groq": {"input": 0.0, "output": 0.0},
    "llama-3.1-8b-groq":  {"input": 0.0, "output": 0.0},
    "deepseek-v3":       {"input": 0.27,  "output": 1.10},
    # DeepSeek-V3.2 served via Azure AI Foundry. Azure list pricing for
    # serverless DeepSeek deployments roughly matches the native API.
    "deepseek-v3.2-azure": {"input": 0.27, "output": 1.10},
}


@dataclass(frozen=True)
class QueryCost:
    """Estimated cost of a single query in USD."""

    model_name: str
    variant: str
    input_tokens: int
    output_tokens: int
    usd: float


def cost_from_usage(
    model_name: str, input_tokens: int, output_tokens: int
) -> float:
    """Exact USD cost given real token counts pulled from API responses."""
    pricing = MODEL_PRICING.get(model_name)
    if pricing is None:
        return 0.0
    return (input_tokens * pricing["input"]
            + output_tokens * pricing["output"]) / 1_000_000


def estimate_query_cost(query: ProbeQuery) -> QueryCost:
    """Estimate per-query cost using pre-tabulated token counts and pricing.

    Unknown models (not in ``MODEL_PRICING``) get a zero cost and a warning
    flavor: returning 0 is safe for estimation since it only ever
    under-counts.
    """
    tokens_in, tokens_out = VARIANT_TOKENS.get(query.variant, (50, 50))
    pricing = MODEL_PRICING.get(query.model_name)
    if pricing is None:
        usd = 0.0
    else:
        usd = (tokens_in * pricing["input"] + tokens_out * pricing["output"]) / 1_000_000
    return QueryCost(
        model_name=query.model_name,
        variant=query.variant,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        usd=usd,
    )


def estimate_sweep_cost(queries: list[ProbeQuery]) -> dict:
    """Aggregate per-query estimates into a per-model and total breakdown.

    Returns a dict with keys:
        total_usd, total_queries, by_model {model: {usd, n_queries, by_variant}}.
    """
    per_model: dict[str, dict] = {}
    total_usd = 0.0
    for q in queries:
        cost = estimate_query_cost(q)
        total_usd += cost.usd
        m = per_model.setdefault(q.model_name, {"usd": 0.0, "n_queries": 0,
                                                "by_variant": {}})
        m["usd"] += cost.usd
        m["n_queries"] += 1
        v = m["by_variant"].setdefault(q.variant, {"usd": 0.0, "n_queries": 0})
        v["usd"] += cost.usd
        v["n_queries"] += 1
    return {
        "total_usd": total_usd,
        "total_queries": len(queries),
        "by_model": per_model,
    }


def format_sweep_cost(summary: dict) -> str:
    """Pretty-print a sweep-cost summary, returning a multi-line string."""
    lines: list[str] = []
    lines.append(f"Planned queries: {summary['total_queries']}")
    lines.append(f"Estimated total: ${summary['total_usd']:.2f}")
    lines.append("")
    lines.append(f"  {'model':25s}  {'queries':>8s}  {'est $':>8s}  by variant")
    for model_name, m in sorted(summary["by_model"].items()):
        vb = " ".join(
            f"{v}={d['n_queries']}"
            for v, d in sorted(m["by_variant"].items())
        )
        price_warn = "" if model_name in MODEL_PRICING else "  (!pricing unknown)"
        lines.append(
            f"  {model_name:25s}  {m['n_queries']:>8d}  "
            f"${m['usd']:>7.2f}  [{vb}]{price_warn}"
        )
    return "\n".join(lines)
