"""Retry + concurrency + error-cache tests for the probe harness.

These exercise the failure-mode behaviors that the handoff flags as
critical for an overnight 13k-query sweep: transient rate-limit errors,
thread-safe JSONL writes, and resumability that actually retries failures.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from factor_leak import probe
from factor_leak.probe import (
    MockEndpoint,
    ProbeQuery,
    _call_with_retry,
    _is_retryable,
    load_cache_keys,
    run_sweep,
)


# ---------------------------------------------------------------------------
# Retry classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("Rate limit exceeded", True),
        ("RateLimitError: 429", True),
        ("request timed out", True),
        ("Connection aborted by peer", True),
        ("Service Unavailable (503)", True),
        ("Overloaded_error: please retry", True),
        ("BadRequestError: invalid prompt", False),
        ("AuthenticationError: bad api key", False),
        ("ValueError: empty response", False),
    ],
)
def test_is_retryable_classifier(msg, expected):
    assert _is_retryable(RuntimeError(msg)) is expected


def test_call_with_retry_succeeds_on_second_attempt():
    calls = {"n": 0}

    def _flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rate limit exceeded")
        return "ok"

    out = _call_with_retry(_flaky, max_retries=3, base_delay=0.01)
    assert out == "ok"
    assert calls["n"] == 2


def test_call_with_retry_gives_up_after_max_retries():
    def _always_rate_limited() -> str:
        raise RuntimeError("rate limit: try again later")

    with pytest.raises(RuntimeError, match="rate limit"):
        _call_with_retry(_always_rate_limited, max_retries=2, base_delay=0.01)


def test_call_with_retry_does_not_retry_nonretryable():
    calls = {"n": 0}

    def _bad_request() -> str:
        calls["n"] += 1
        raise ValueError("invalid prompt format")

    with pytest.raises(ValueError):
        _call_with_retry(_bad_request, max_retries=5, base_delay=0.01)
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# Resumable cache skips successes, retries errors
# ---------------------------------------------------------------------------


def test_load_cache_keys_excludes_error_rows_by_default(tmp_path: Path):
    path = tmp_path / "sweep.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({"key": "ok-1", "error": None}) + "\n")
        f.write(json.dumps({"key": "bad-1", "error": "RateLimitError: 429"}) + "\n")
        f.write(json.dumps({"key": "ok-2", "error": None}) + "\n")

    keys = load_cache_keys(path)
    assert keys == {"ok-1", "ok-2"}

    # include_errors=True returns everything (for analysis).
    assert load_cache_keys(path, include_errors=True) == {"ok-1", "bad-1", "ok-2"}


def test_sweep_retries_previously_errored_queries(tmp_path: Path):
    attempts = {"n": 0}

    def _fail_first_then_succeed(prompt: str) -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("rate limit on first run")
        return "+1.50"

    ep = {"flaky": MockEndpoint("flaky", _fail_first_then_succeed)}
    q = ProbeQuery("flaky", "Mkt-RF", "A", "2020-03")
    out = tmp_path / "retry.jsonl"

    first = run_sweep([q], ep, out, max_workers=1)
    assert first[0].error is not None

    second = run_sweep([q], ep, out, max_workers=1)
    assert len(second) == 1
    assert second[0].error is None

    # Both attempts remain in the JSONL; analysis should dedupe by key
    # preferring the most recent non-error record.
    records = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(records) == 2
    keys = [r["key"] for r in records]
    assert keys[0] == keys[1]


# ---------------------------------------------------------------------------
# Thread safety of _append_jsonl
# ---------------------------------------------------------------------------


def test_concurrent_sweep_produces_uncorrupted_jsonl(tmp_path: Path):
    """Stress test: 8 workers, 200 queries, verify every line is valid JSON
    and every expected key is present exactly once."""
    barrier = threading.Barrier(8)

    def _responder(prompt: str) -> str:
        # Wait until all workers are ready, then return simultaneously.
        try:
            barrier.wait(timeout=0.5)
        except threading.BrokenBarrierError:
            pass
        return "+0.00"

    ep = {"m": MockEndpoint("m", _responder)}
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    queries = [
        ProbeQuery("m", f, "A", f"{year:04d}-{month:02d}")
        for f in factors
        for year in range(1990, 2010)
        for month in (3,)
    ]
    # 5 factors × 20 years = 100 queries; double by adding variant B.
    queries += [
        ProbeQuery("m", f, "B", f"{year:04d}-{month:02d}")
        for f in factors
        for year in range(1990, 2010)
        for month in (3,)
    ]
    assert len(queries) == 200

    out = tmp_path / "concurrent.jsonl"
    run_sweep(queries, ep, out, max_workers=8)

    lines = out.read_text().splitlines()
    records = []
    for line in lines:
        assert line.strip(), "empty line should not appear"
        records.append(json.loads(line))  # must be valid JSON

    assert len(records) == 200
    assert len({r["key"] for r in records}) == 200
