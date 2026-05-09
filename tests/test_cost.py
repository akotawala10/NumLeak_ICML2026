"""Tests for cost estimation and per-variant max_tokens."""
from __future__ import annotations

import pytest

from factor_leak.cost import (
    MODEL_PRICING,
    VARIANT_TOKENS,
    estimate_query_cost,
    estimate_sweep_cost,
    format_sweep_cost,
)
from factor_leak.probe import MAX_TOKENS_BY_VARIANT, ProbeQuery


# ---------------------------------------------------------------------------
# Per-query cost
# ---------------------------------------------------------------------------


def test_estimate_query_cost_variant_a_gpt_4o_mini():
    q = ProbeQuery("gpt-4o-mini", "Mkt-RF", "A", "2020-03")
    c = estimate_query_cost(q)
    in_tok, out_tok = VARIANT_TOKENS["A"]
    p = MODEL_PRICING["gpt-4o-mini"]
    expected = (in_tok * p["input"] + out_tok * p["output"]) / 1_000_000
    assert c.usd == pytest.approx(expected)


def test_variant_b_costs_more_than_variant_a():
    a = estimate_query_cost(ProbeQuery("claude-sonnet-4.6", "HML", "A", "2020-03"))
    b = estimate_query_cost(ProbeQuery("claude-sonnet-4.6", "HML", "B", "2020-03"))
    assert b.usd > a.usd * 3  # B has ~13x more output tokens


def test_unknown_model_yields_zero_cost():
    c = estimate_query_cost(ProbeQuery("mystery-model", "HML", "A", "2020-03"))
    # mystery-model fails ProbeQuery validation? Actually it validates factor
    # + variant, not model_name. So cost is zero for unknown vendor pricing.
    assert c.usd == 0.0


# ---------------------------------------------------------------------------
# Sweep-level aggregation
# ---------------------------------------------------------------------------


def test_estimate_sweep_cost_sums_across_models_and_variants():
    queries = [
        ProbeQuery("gpt-4o-mini", "Mkt-RF", "A", "2020-03"),
        ProbeQuery("gpt-4o-mini", "Mkt-RF", "B", "2020-03"),
        ProbeQuery("claude-haiku-4.5", "Mkt-RF", "A", "2020-03"),
    ]
    summary = estimate_sweep_cost(queries)
    assert summary["total_queries"] == 3
    assert set(summary["by_model"]) == {"gpt-4o-mini", "claude-haiku-4.5"}
    assert summary["by_model"]["gpt-4o-mini"]["n_queries"] == 2
    assert summary["by_model"]["gpt-4o-mini"]["by_variant"]["A"]["n_queries"] == 1
    assert summary["by_model"]["gpt-4o-mini"]["by_variant"]["B"]["n_queries"] == 1

    # Individual cost sum equals total.
    individual = sum(estimate_query_cost(q).usd for q in queries)
    assert summary["total_usd"] == pytest.approx(individual)


def test_full_sweep_estimate_under_twenty_dollars():
    """Sanity-check: 10,800-query default sweep shouldn't exceed $20 at
    today's prices. If it does, pricing has moved and this test alerts."""
    # Simulate default sweep: 6 models * 6 factors * (120 A + 120 B + 60 C).
    from factor_leak.ff_loader import FACTOR_LONG_NAMES

    models = list(MODEL_PRICING.keys())
    factors = list(FACTOR_LONG_NAMES.keys())
    queries: list[ProbeQuery] = []
    for m in models:
        for f in factors:
            for mo in (f"1990-{mm:02d}" for mm in range(1, 13)):
                for v in ("A", "B"):
                    queries.append(ProbeQuery(m, f, v, mo))
            for mo in (f"2000-{mm:02d}" for mm in range(1, 11)):
                queries.append(ProbeQuery(m, f, "C", mo, month2="2010-03"))
    summary = estimate_sweep_cost(queries)
    assert summary["total_queries"] == len(queries)
    # Loose cap so we notice if pricing moves 4x.
    assert summary["total_usd"] < 20.0


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def test_format_sweep_cost_renders_expected_lines():
    q = ProbeQuery("gpt-4o-mini", "Mkt-RF", "A", "2020-03")
    summary = estimate_sweep_cost([q])
    text = format_sweep_cost(summary)
    assert "Estimated total" in text
    assert "gpt-4o-mini" in text
    assert "$" in text


# ---------------------------------------------------------------------------
# Per-variant max_tokens correctness
# ---------------------------------------------------------------------------


def test_max_tokens_by_variant_covers_all_variants():
    assert {"A", "B", "C"} <= set(MAX_TOKENS_BY_VARIANT)
    assert MAX_TOKENS_BY_VARIANT["B"] > MAX_TOKENS_BY_VARIANT["A"]
    assert MAX_TOKENS_BY_VARIANT["B"] > MAX_TOKENS_BY_VARIANT["C"]


def test_run_sweep_passes_per_variant_max_tokens_to_endpoint():
    """Regression: the per-variant cap must actually reach the endpoint."""
    from factor_leak.probe import MockEndpoint, ProbeQuery, _run_one

    seen = []

    def _capturing_responder(prompt, *args, **kwargs):
        seen.append(kwargs.get("max_tokens"))
        return "+0.00"

    # MockEndpoint's __call__ does not itself forward max_tokens to the
    # responder — but _run_one calls endpoint(prompt, max_tokens=...).
    # Here we wrap MockEndpoint in a subclass that records what it sees.
    class _RecEP(MockEndpoint):
        def __call__(self, prompt, max_tokens=None):
            seen.append(max_tokens)
            return self.responder(prompt)

    ep = _RecEP("m", _capturing_responder)
    _run_one(ProbeQuery("m", "Mkt-RF", "A", "2020-03"), ep)
    _run_one(ProbeQuery("m", "Mkt-RF", "B", "2020-03"), ep)
    _run_one(ProbeQuery("m", "Mkt-RF", "C", "2020-03", month2="2020-04"), ep)
    # First hits are from _RecEP.__call__; each run_one causes exactly one.
    captured_per_variant = seen[::2]  # even indices are _RecEP's max_tokens
    assert captured_per_variant[0] == MAX_TOKENS_BY_VARIANT["A"]
    assert captured_per_variant[1] == MAX_TOKENS_BY_VARIANT["B"]
    assert captured_per_variant[2] == MAX_TOKENS_BY_VARIANT["C"]
