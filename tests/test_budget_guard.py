"""Tests for the hard-abort budget cap in run_sweep."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from factor_leak.probe import (
    BudgetExceeded,
    EndpointResponse,
    MockEndpoint,
    ProbeQuery,
    run_sweep,
)


def _expensive_responder(prompt: str, *, input_tokens: int = 100_000, output_tokens: int = 0) -> EndpointResponse:
    # Each call reports high token usage so cost accumulates quickly.
    return EndpointResponse(
        text="+1.00",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def test_sweep_respects_budget_cap_and_raises(tmp_path: Path):
    """Each call reports enough input tokens that cost per call exceeds
    the budget within a handful of queries; run_sweep should stop and
    raise BudgetExceeded."""

    def _big_usage(prompt: str) -> EndpointResponse:
        # 500,000 input tokens → at $3/M = $1.50 per call on claude-sonnet-4.6.
        return EndpointResponse(text="+1.0", input_tokens=500_000, output_tokens=0)

    ep = {"claude-sonnet-4.6": MockEndpoint("claude-sonnet-4.6", _big_usage)}
    queries = [ProbeQuery("claude-sonnet-4.6", "Mkt-RF", "A", f"2020-{m:02d}")
               for m in range(1, 13)]
    out = tmp_path / "budget.jsonl"

    with pytest.raises(BudgetExceeded, match=r"budget cap of \$3\.00"):
        run_sweep(queries, ep, out, max_workers=1, budget_usd=3.0)

    # At least some records were persisted before the abort (so the cache
    # survives and the user can resume with a higher cap).
    lines = out.read_text().splitlines()
    assert 1 <= len(lines) < len(queries)


def test_sweep_finishes_cleanly_under_budget(tmp_path: Path):
    def _cheap(prompt: str) -> EndpointResponse:
        # ~10 input tokens, ~10 output → $0.00006 on gpt-4o-mini pricing.
        return EndpointResponse(text="+0.5", input_tokens=10, output_tokens=10)

    ep = {"gpt-4o-mini": MockEndpoint("gpt-4o-mini", _cheap)}
    queries = [ProbeQuery("gpt-4o-mini", "SMB", "A", f"2020-{m:02d}")
               for m in range(1, 6)]
    out = tmp_path / "under.jsonl"

    results = run_sweep(queries, ep, out, max_workers=1, budget_usd=1.0)
    assert len(results) == len(queries)
    # Actual spend recorded in each result.
    assert all(r.input_tokens == 10 and r.output_tokens == 10 for r in results)


def test_sweep_preserves_cache_on_budget_abort(tmp_path: Path):
    """After a BudgetExceeded, a second run with higher cap must resume,
    not re-charge for the already-persisted queries."""
    counter = {"n": 0}

    def _pricey(prompt: str) -> EndpointResponse:
        counter["n"] += 1
        return EndpointResponse(text="+1.0", input_tokens=500_000, output_tokens=0)

    ep = {"claude-sonnet-4.6": MockEndpoint("claude-sonnet-4.6", _pricey)}
    queries = [ProbeQuery("claude-sonnet-4.6", "HML", "A", f"2020-{m:02d}")
               for m in range(1, 6)]
    out = tmp_path / "resume.jsonl"

    with pytest.raises(BudgetExceeded):
        run_sweep(queries, ep, out, max_workers=1, budget_usd=3.0)

    first_run_calls = counter["n"]
    persisted_keys = {json.loads(line)["key"]
                      for line in out.read_text().splitlines() if line.strip()}
    assert 0 < len(persisted_keys) < len(queries)

    # Second run with no cap — should skip cached and only hit the
    # remaining queries.
    results_second = run_sweep(queries, ep, out, max_workers=1, budget_usd=100.0)
    assert len(results_second) == len(queries) - len(persisted_keys)
    assert counter["n"] == first_run_calls + len(results_second)


def test_mock_endpoint_auto_wraps_string_responder():
    """Backwards-compat: bare-string responders get wrapped as zero-cost."""
    ep = MockEndpoint("m", lambda p: "hi")
    out = ep("anything")
    assert isinstance(out, EndpointResponse)
    assert out.text == "hi"
    assert out.input_tokens == 0
    assert out.output_tokens == 0


def test_endpoint_response_passthrough():
    ep = MockEndpoint("m", lambda p: EndpointResponse("hi", 7, 3))
    out = ep("anything")
    assert out.input_tokens == 7
    assert out.output_tokens == 3
