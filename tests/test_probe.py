"""Tests for the probe harness against mock endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from factor_leak import probe
from factor_leak.probe import (
    MockEndpoint,
    ProbeQuery,
    format_month_human,
    prompt_variant_a,
    prompt_variant_b,
    prompt_variant_c,
    run_sweep,
)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def test_format_month_human():
    assert format_month_human("2020-03") == "March 2020"
    assert format_month_human(pd.Period("1987-10", freq="M")) == "October 1987"


def test_variant_a_prompt_mentions_factor_month_and_format():
    p = prompt_variant_a("Mkt-RF", "2020-03")
    assert "March 2020" in p
    assert "market excess return" in p
    assert "decimal percentage" in p


def test_variant_b_prompt_asks_for_estimate():
    p = prompt_variant_b("HML", "1987-10")
    assert "October 1987" in p
    assert "estimate" in p


def test_variant_c_prompt_lists_both_months():
    p = prompt_variant_c("Mom", "2008-10", "2020-03")
    assert "October 2008" in p and "March 2020" in p
    assert "higher return" in p


# ---------------------------------------------------------------------------
# ProbeQuery validation and key stability
# ---------------------------------------------------------------------------


def test_probe_query_rejects_bad_variant():
    with pytest.raises(ValueError):
        ProbeQuery("m", "SMB", "D", "2020-01")


def test_probe_query_rejects_unknown_factor():
    with pytest.raises(ValueError):
        ProbeQuery("m", "NOPE", "A", "2020-01")


def test_probe_query_c_requires_month2():
    with pytest.raises(ValueError):
        ProbeQuery("m", "SMB", "C", "2020-01")


def test_probe_query_ab_reject_month2():
    with pytest.raises(ValueError):
        ProbeQuery("m", "SMB", "A", "2020-01", month2="2020-02")


def test_probe_query_key_is_stable_and_unique():
    q1 = ProbeQuery("gpt-4.1", "HML", "A", "2020-03")
    q2 = ProbeQuery("gpt-4.1", "HML", "A", "2020-03")
    q3 = ProbeQuery("gpt-4.1", "HML", "B", "2020-03")
    q4 = ProbeQuery("gpt-4.1", "HML", "A", "2020-04")
    q5 = ProbeQuery("claude-haiku-4.5", "HML", "A", "2020-03")
    assert q1.key == q2.key
    assert len({q1.key, q3.key, q4.key, q5.key}) == 4


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def _echo_responder(prompt: str) -> str:
    # Pretend the model always guesses "1.23%".
    return "1.23"


def _failing_responder(prompt: str) -> str:
    raise RuntimeError("boom")


def test_run_sweep_writes_jsonl_and_returns_results(tmp_path: Path):
    ep = MockEndpoint("mock", _echo_responder)
    queries = [
        ProbeQuery("mock", "SMB", "A", "2020-03"),
        ProbeQuery("mock", "HML", "B", "1987-10"),
    ]
    out = tmp_path / "sweep.jsonl"
    results = run_sweep(queries, {"mock": ep}, out, max_workers=2)
    assert len(results) == 2
    assert all(r.error is None for r in results)
    assert all(r.response == "1.23" for r in results)

    lines = out.read_text().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert "key" in rec and rec["query"]["factor"] in {"SMB", "HML"}


def test_run_sweep_captures_errors_without_crashing(tmp_path: Path):
    ep = MockEndpoint("mock", _failing_responder)
    queries = [ProbeQuery("mock", "SMB", "A", "2020-03")]
    out = tmp_path / "sweep.jsonl"
    results = run_sweep(queries, {"mock": ep}, out, max_workers=1)
    assert len(results) == 1
    assert results[0].response is None
    assert results[0].error and results[0].error.startswith("RuntimeError")


def test_run_sweep_is_resumable(tmp_path: Path):
    ep = MockEndpoint("mock", _echo_responder)
    queries = [
        ProbeQuery("mock", "SMB", "A", "2020-03"),
        ProbeQuery("mock", "HML", "A", "2020-03"),
    ]
    out = tmp_path / "sweep.jsonl"
    run_sweep(queries[:1], {"mock": ep}, out, max_workers=1)
    # Second call adds only the missing one.
    more = run_sweep(queries, {"mock": ep}, out, max_workers=1)
    assert len(more) == 1
    lines = out.read_text().splitlines()
    assert len(lines) == 2


def test_run_sweep_raises_for_unknown_model(tmp_path: Path):
    queries = [ProbeQuery("mystery", "SMB", "A", "2020-03")]
    out = tmp_path / "sweep.jsonl"
    with pytest.raises(KeyError):
        run_sweep(queries, {}, out)


# ---------------------------------------------------------------------------
# Default endpoints: construction must not require keys.
# ---------------------------------------------------------------------------


def test_default_endpoints_construct_without_api_keys(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TOGETHER_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    eps = probe.default_endpoints()
    names = {e.name for e in eps}
    assert {"gpt-4.1", "gpt-4o-mini", "claude-sonnet-4.6",
            "claude-haiku-4.5", "llama-3.3-70b", "deepseek-v3"} <= names


def test_endpoint_raises_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ep = probe.OpenAICompatibleEndpoint(
        name="gpt-test",
        model="gpt-4.1",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    )
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        ep("hello")
