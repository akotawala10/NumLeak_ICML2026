"""LLM probe harness for factor-return recall.

Structure:

- ``ProbeQuery`` — one (model, factor, variant, month[, month2]) tuple.
- ``ProbeResult`` — the raw response plus latency/error metadata.
- ``prompt_for`` — builds the prompt string for variants A/B/C.
- Endpoint classes — ``OpenAICompatibleEndpoint``, ``AnthropicEndpoint``,
  ``MockEndpoint``. All expose ``__call__(prompt) -> str``.
- ``run_sweep`` — resumable, threaded sweep that appends JSONL to disk as each
  query completes. Resumable only skips *successful* prior records; rows
  whose ``error`` field is non-null are retried on the next run.

Providers are kept behind a common interface so the sweep runner doesn't care
which vendor serves which query. Every endpoint reads its API key from the
environment at call time (not at construction) so the harness can be
instantiated without keys present.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol

import pandas as pd

from .ff_loader import FACTOR_LONG_NAMES


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def format_month_human(month: str | pd.Period) -> str:
    """``"2020-03"`` → ``"March 2020"``."""
    p = month if isinstance(month, pd.Period) else pd.Period(month, freq="M")
    return f"{MONTH_NAMES[p.month - 1]} {p.year}"


def prompt_variant_a(factor: str, month: str | pd.Period) -> str:
    long_name = FACTOR_LONG_NAMES[factor]
    return (
        f"What was the monthly return of the Fama-French {long_name} factor "
        f"in {format_month_human(month)}? "
        f"Answer with a signed decimal percentage "
        f"(e.g., -3.12 for a 3.12% loss, +1.50 for a 1.50% gain) "
        f"and nothing else."
    )


def prompt_variant_b(factor: str, month: str | pd.Period) -> str:
    long_name = FACTOR_LONG_NAMES[factor]
    return (
        f"Describe the performance of the Fama-French {long_name} factor "
        f"in {format_month_human(month)}. "
        f"Include your best estimate of the signed monthly return "
        f"(use a negative sign for losses)."
    )


def prompt_variant_c(
    factor: str,
    month1: str | pd.Period,
    month2: str | pd.Period,
) -> str:
    long_name = FACTOR_LONG_NAMES[factor]
    return (
        f"Between {format_month_human(month1)} and {format_month_human(month2)}, "
        f"which month had the higher return for the Fama-French {long_name} "
        f"factor? Answer with only one of those two months."
    )


# ---------------------------------------------------------------------------
# Query / result datatypes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeQuery:
    model_name: str
    factor: str
    variant: str   # "A", "B", or "C"
    month: str     # "YYYY-MM"
    month2: str | None = None  # only used by variant C

    def __post_init__(self) -> None:
        if self.variant not in {"A", "B", "C"}:
            raise ValueError(f"variant must be A, B, or C (got {self.variant!r})")
        if self.factor not in FACTOR_LONG_NAMES:
            raise ValueError(f"unknown factor {self.factor!r}")
        if self.variant == "C" and self.month2 is None:
            raise ValueError("variant C requires month2")
        if self.variant != "C" and self.month2 is not None:
            raise ValueError("month2 only valid for variant C")

    @property
    def key(self) -> str:
        """Stable 16-char ID used for cache dedup."""
        raw = f"{self.model_name}|{self.factor}|{self.variant}|{self.month}|{self.month2}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def to_prompt(self) -> str:
        if self.variant == "A":
            return prompt_variant_a(self.factor, self.month)
        if self.variant == "B":
            return prompt_variant_b(self.factor, self.month)
        return prompt_variant_c(self.factor, self.month, self.month2)  # type: ignore[arg-type]


@dataclass
class ProbeResult:
    query: ProbeQuery
    prompt: str
    response: str | None
    error: str | None
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    ts: float = field(default_factory=time.time)

    def to_json_record(self) -> dict:
        d = asdict(self)
        d["query"] = asdict(self.query)
        d["key"] = self.query.key
        return d


# ---------------------------------------------------------------------------
# Retry policy (used by real endpoints, not MockEndpoint)
# ---------------------------------------------------------------------------


_RETRYABLE_TOKENS = (
    "rate limit", "ratelimit", "429",
    "timeout", "timed out",
    "connection",
    "service unavailable", "unavailable",
    "502", "503", "504",
    "overloaded", "overloaded_error",
)


def _is_retryable(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__} {exc}".lower()
    return any(tok in msg for tok in _RETRYABLE_TOKENS)


def _call_with_retry(
    fn: Callable[[], str],
    max_retries: int = 4,
    base_delay: float = 1.0,
) -> str:
    """Invoke ``fn()`` with exponential backoff on transient API failures.

    Only retries on rate-limit / 5xx / timeout / connection errors; other
    exceptions bubble immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_retries or not _is_retryable(exc):
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0.0, 0.5)
            time.sleep(delay)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


# Per-variant output-token caps — the probe prompts are short and
# stereotyped, so Variant A/C answers easily fit in ~32 tokens while
# Variant B warrants ~384 for a paragraph. This shaves Claude/GPT-4.1
# output costs noticeably on a 10k-query sweep.
MAX_TOKENS_BY_VARIANT: dict[str, int] = {"A": 48, "B": 384, "C": 48}


@dataclass(frozen=True)
class EndpointResponse:
    """Return value from an endpoint call — text plus real token usage.

    Real endpoints populate ``input_tokens`` / ``output_tokens`` from the
    vendor API response (``usage.input_tokens`` on Anthropic, ``usage.
    prompt_tokens`` on OpenAI). MockEndpoint defaults to zero.
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class ModelEndpoint(Protocol):
    name: str

    def __call__(
        self, prompt: str, max_tokens: int | None = None
    ) -> EndpointResponse: ...


@dataclass
class MockEndpoint:
    """Deterministic in-memory endpoint for unit tests.

    ``responder`` is called with the prompt. It may return either a bare
    string (zero-cost) or an ``EndpointResponse`` to simulate real token
    usage (useful for exercising the budget guard in tests).
    """

    name: str
    responder: Callable[[str], str | EndpointResponse]

    def __call__(
        self, prompt: str, max_tokens: int | None = None
    ) -> EndpointResponse:
        out = self.responder(prompt)
        if isinstance(out, EndpointResponse):
            return out
        return EndpointResponse(text=out, input_tokens=0, output_tokens=0)


@dataclass
class OpenAICompatibleEndpoint:
    """Any OpenAI-chat-completions-compatible endpoint.

    Covers OpenAI itself (GPT-4.1, GPT-4o-mini), Together.ai, DeepSeek, and
    any other provider with an OpenAI-style ``/v1/chat/completions`` URL.

    OpenAI rolled ``max_tokens`` out in favor of ``max_completion_tokens``
    for newer (``o1``, ``gpt-4.1+``) models; pass ``token_param`` to pick
    the right kwarg. Together.ai and DeepSeek still accept ``max_tokens``.
    """

    name: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.0
    max_tokens: int = 256
    token_param: str = "max_tokens"          # or "max_completion_tokens"
    request_timeout_s: float = 45.0

    def __call__(
        self, prompt: str, max_tokens: int | None = None
    ) -> EndpointResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing env var {self.api_key_env} for endpoint {self.name}")
        from openai import OpenAI  # imported lazily so the module loads without the SDK

        client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.request_timeout_s,
        )
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            self.token_param: max_tokens if max_tokens is not None else self.max_tokens,
        }

        def _do() -> EndpointResponse:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            return EndpointResponse(text=text,
                                    input_tokens=in_tok,
                                    output_tokens=out_tok)

        return _call_with_retry(_do)


@dataclass
class AzureOpenAIResponsesEndpoint:
    """Azure OpenAI Responses-API endpoint (used by gpt-5.x deployments).

    Unlike Chat Completions, the Responses API takes ``input=...`` instead
    of ``messages=...``, returns text via ``response.output_text``, and
    caps length with ``max_output_tokens``. URL path is
    ``/openai/responses?api-version=...`` (no ``/deployments/<name>/``
    segment; the deployment name is passed in the request body as
    ``model``).
    """

    name: str
    deployment: str                         # Azure deployment name
    api_key_env: str = "AZURE_OPENAI_API_KEY"
    endpoint_env: str = "AZURE_OPENAI_ENDPOINT"
    api_version_env: str = "AZURE_OPENAI_API_VERSION"
    api_version_default: str = "2025-04-01-preview"
    max_tokens: int = 256
    request_timeout_s: float = 60.0

    def __call__(
        self, prompt: str, max_tokens: int | None = None
    ) -> EndpointResponse:
        api_key = os.environ.get(self.api_key_env)
        endpoint = os.environ.get(self.endpoint_env)
        api_version = os.environ.get(self.api_version_env, self.api_version_default)
        if not api_key:
            raise RuntimeError(f"missing env var {self.api_key_env} for endpoint {self.name}")
        if not endpoint:
            raise RuntimeError(f"missing env var {self.endpoint_env} for endpoint {self.name}")
        from openai import AzureOpenAI  # lazy import

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            timeout=self.request_timeout_s,
        )
        effective_max = max_tokens if max_tokens is not None else self.max_tokens

        def _do() -> EndpointResponse:
            resp = client.responses.create(
                model=self.deployment,
                input=prompt,
                max_output_tokens=effective_max,
            )
            # Prefer the SDK's convenience accessor; fall back to the
            # nested structure if the SDK version doesn't expose it.
            text = getattr(resp, "output_text", None)
            if not text:
                try:
                    parts = []
                    for item in resp.output:
                        for piece in getattr(item, "content", []) or []:
                            t = getattr(piece, "text", None)
                            if t:
                                parts.append(t)
                    text = "".join(parts)
                except Exception:  # noqa: BLE001
                    text = ""
            usage = getattr(resp, "usage", None)
            in_tok = int(getattr(usage, "input_tokens", 0) or 0)
            out_tok = int(getattr(usage, "output_tokens", 0) or 0)
            return EndpointResponse(text=text or "",
                                    input_tokens=in_tok,
                                    output_tokens=out_tok)

        return _call_with_retry(_do)


@dataclass
class AzureAIFoundryEndpoint:
    """Azure AI Foundry serverless model endpoint (DeepSeek, Mistral, etc.).

    Distinct from Cognitive Services: data plane is
    ``<resource>.services.ai.azure.com/models/chat/completions``,
    auth is the ``api-key`` header (not Bearer), and the model name is
    selected via the request body (``model``). We POST raw JSON via
    ``requests`` rather than fight the OpenAI SDK's auth header.
    """

    name: str
    model: str
    api_key_env: str = "AZURE_DEEPSEEK_API_KEY"
    endpoint_env: str = "AZURE_DEEPSEEK_ENDPOINT"
    temperature: float = 0.0
    max_tokens: int = 256
    request_timeout_s: float = 60.0

    def __call__(
        self, prompt: str, max_tokens: int | None = None
    ) -> EndpointResponse:
        api_key = os.environ.get(self.api_key_env)
        endpoint = os.environ.get(self.endpoint_env)
        if not api_key:
            raise RuntimeError(f"missing env var {self.api_key_env} for endpoint {self.name}")
        if not endpoint:
            raise RuntimeError(f"missing env var {self.endpoint_env} for endpoint {self.name}")
        api_key = api_key.strip()
        endpoint = endpoint.strip()
        import requests  # lazy import

        effective_max = max_tokens if max_tokens is not None else self.max_tokens
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": effective_max,
        }
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
        }

        def _do() -> EndpointResponse:
            resp = requests.post(
                endpoint, json=body, headers=headers,
                timeout=self.request_timeout_s,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            data = resp.json()
            text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            usage = data.get("usage") or {}
            in_tok = int(usage.get("prompt_tokens", 0) or 0)
            out_tok = int(usage.get("completion_tokens", 0) or 0)
            return EndpointResponse(text=text,
                                    input_tokens=in_tok,
                                    output_tokens=out_tok)

        return _call_with_retry(_do)


@dataclass
class AnthropicEndpoint:
    name: str
    model: str
    # Some newer models (e.g. claude-opus-4-7) reject the `temperature`
    # parameter entirely. Set `temperature=None` to omit the field.
    temperature: float | None = 0.0
    max_tokens: int = 256
    api_key_env: str = "ANTHROPIC_API_KEY"
    request_timeout_s: float = 45.0

    def __call__(
        self, prompt: str, max_tokens: int | None = None
    ) -> EndpointResponse:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing env var {self.api_key_env} for endpoint {self.name}")
        import anthropic  # lazy import

        client = anthropic.Anthropic(api_key=api_key, timeout=self.request_timeout_s)
        effective_max = max_tokens if max_tokens is not None else self.max_tokens

        def _do() -> EndpointResponse:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": effective_max,
            }
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            resp = client.messages.create(**kwargs)
            parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            usage = getattr(resp, "usage", None)
            in_tok = int(getattr(usage, "input_tokens", 0) or 0)
            out_tok = int(getattr(usage, "output_tokens", 0) or 0)
            return EndpointResponse(text="".join(parts),
                                    input_tokens=in_tok,
                                    output_tokens=out_tok)

        return _call_with_retry(_do)


def default_endpoints() -> list[ModelEndpoint]:
    """The 6-model roster specified in the handoff.

    OpenAI entries use ``max_completion_tokens`` (required by gpt-4.1+).
    Together.ai and DeepSeek still accept ``max_tokens``.
    """
    return [
        AzureOpenAIResponsesEndpoint(
            name="gpt-5.4",
            deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4"),
        ),
        AzureOpenAIResponsesEndpoint(
            name="gpt-5.4-mini",
            deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_MINI", "gpt-5.4-mini"),
        ),
        AzureOpenAIResponsesEndpoint(
            name="gpt-5.4-nano",
            deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NANO", "gpt-5.4-nano"),
        ),
        AzureAIFoundryEndpoint(
            name="deepseek-v3.2-azure",
            model="DeepSeek-V3.2",
        ),
        # ---- Frontier panel: separate Azure resources / new deployments ----
        AzureOpenAIResponsesEndpoint(
            name="gpt-5.5",
            deployment=os.environ.get("AZURE_GPT55_DEPLOYMENT", "gpt-5.5"),
            api_key_env="AZURE_GPT55_API_KEY",
            endpoint_env="AZURE_GPT55_ENDPOINT",
            api_version_env="AZURE_GPT55_API_VERSION",
        ),
        AzureAIFoundryEndpoint(
            name="deepseek-v3.2-foundry",
            model=os.environ.get("AZURE_FOUNDRY_DEPLOYMENT_DEEPSEEK_V32",
                                 "DeepSeek-V3.2-2"),
            api_key_env="AZURE_FOUNDRY_API_KEY",
            endpoint_env="AZURE_FOUNDRY_ENDPOINT",
        ),
        AzureAIFoundryEndpoint(
            name="llama-4-maverick-foundry",
            model=os.environ.get("AZURE_FOUNDRY_DEPLOYMENT_LLAMA4",
                                 "Llama-4-Maverick-17B-128E-Instruct-FP8"),
            api_key_env="AZURE_FOUNDRY_API_KEY",
            endpoint_env="AZURE_FOUNDRY_ENDPOINT",
        ),
        OpenAICompatibleEndpoint(
            name="gpt-4.1",
            model="gpt-4.1",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            token_param="max_completion_tokens",
        ),
        OpenAICompatibleEndpoint(
            name="gpt-4o-mini",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            token_param="max_completion_tokens",
        ),
        AnthropicEndpoint(name="claude-opus-4.7",   model="claude-opus-4-7",
                           temperature=None),
        AnthropicEndpoint(name="claude-sonnet-4.6", model="claude-sonnet-4-6"),
        AnthropicEndpoint(name="claude-haiku-4.5", model="claude-haiku-4-5-20251001"),
        OpenAICompatibleEndpoint(
            name="llama-3.3-70b",
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            base_url="https://api.together.xyz/v1",
            api_key_env="TOGETHER_API_KEY",
        ),
        OpenAICompatibleEndpoint(
            name="llama-3.3-70b-groq",
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
        ),
        OpenAICompatibleEndpoint(
            name="llama-3.1-8b-groq",
            model="llama-3.1-8b-instant",
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
        ),
        OpenAICompatibleEndpoint(
            name="deepseek-v3",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key_env="DEEPSEEK_API_KEY",
        ),
    ]


# ---------------------------------------------------------------------------
# Sweep runner (resumable JSONL cache)
# ---------------------------------------------------------------------------


def load_cache_keys(path: Path | str, include_errors: bool = False) -> set[str]:
    """Return the set of query keys already *successfully* persisted.

    Records whose ``error`` field is non-null are treated as not-yet-done so
    that a resumable rerun retries them. Pass ``include_errors=True`` to
    get every key in the file (used by analysis code that wants to see all
    attempts).
    """
    p = Path(path)
    if not p.exists():
        return set()
    keys: set[str] = set()
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = rec.get("key")
            if not k:
                continue
            if not include_errors and rec.get("error"):
                continue
            keys.add(k)
    return keys


_APPEND_LOCK = threading.Lock()


def _append_jsonl(path: Path, record: dict) -> None:
    """Append one JSONL record, serialized across threads by a module-wide lock.

    Most Unix filesystems guarantee atomic writes below PIPE_BUF, but we
    serialize anyway because the record size (prompt + response) can easily
    exceed that on variant-B answers.
    """
    line = json.dumps(record) + "\n"
    with _APPEND_LOCK:
        with path.open("a") as f:
            f.write(line)


class _BudgetAborted(Exception):
    """Internal sentinel — raised by a worker that finds the budget
    already exceeded before it was scheduled to call the endpoint."""


def _run_one(
    query: ProbeQuery,
    endpoint: ModelEndpoint,
    guard: "_BudgetGuard | None" = None,
) -> ProbeResult:
    # Short-circuit BEFORE any API call if the budget is already blown
    # — otherwise queued workers may still fire off paid requests before
    # the ``as_completed`` loop notices the cap was hit.
    if guard is not None and guard.exceeded:
        raise _BudgetAborted()

    # Lazy import to avoid a circular dep.
    from .cost import cost_from_usage

    prompt = query.to_prompt()
    max_tokens = MAX_TOKENS_BY_VARIANT.get(query.variant)
    t0 = time.perf_counter()
    try:
        ep_resp = endpoint(prompt, max_tokens=max_tokens)
        result = ProbeResult(
            query=query,
            prompt=prompt,
            response=ep_resp.text,
            error=None,
            latency_s=time.perf_counter() - t0,
            input_tokens=ep_resp.input_tokens,
            output_tokens=ep_resp.output_tokens,
        )
        # Update the running cost tally *here*, inside the worker, so the
        # next worker checking guard.exceeded sees our contribution before
        # starting its own API call. Updating only in the main thread
        # after as_completed is too slow — mock / cached endpoints can
        # run a dozen queries before the main loop wakes up.
        if guard is not None:
            guard.add_usd(cost_from_usage(
                query.model_name, result.input_tokens, result.output_tokens
            ))
        return result
    except Exception as exc:  # noqa: BLE001 — we want to record any failure
        return ProbeResult(
            query=query,
            prompt=prompt,
            response=None,
            error=f"{type(exc).__name__}: {exc}",
            latency_s=time.perf_counter() - t0,
        )


class BudgetExceeded(RuntimeError):
    """Raised inside ``run_sweep`` when cumulative spend exceeds the cap."""


class _BudgetGuard:
    """Thread-safe cumulative spend tracker with a hard cap.

    ``add_usd`` increments the running total and returns ``True`` if the
    budget is still under the cap after the addition. Once exceeded, the
    guard is latched: further queries should not be processed even if
    workers are still running in the pool.
    """

    def __init__(self, budget_usd: float | None):
        self.budget_usd = budget_usd
        self.spent_usd = 0.0
        self._lock = threading.Lock()
        self._exceeded = False

    def add_usd(self, usd: float) -> bool:
        with self._lock:
            self.spent_usd += usd
            if self.budget_usd is not None and self.spent_usd >= self.budget_usd:
                self._exceeded = True
            return not self._exceeded

    @property
    def exceeded(self) -> bool:
        with self._lock:
            return self._exceeded


def run_sweep(
    queries: Iterable[ProbeQuery],
    endpoints_by_name: dict[str, ModelEndpoint],
    out_path: Path | str,
    max_workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
    budget_usd: float | None = None,
) -> list[ProbeResult]:
    """Run queries concurrently, appending each result to ``out_path`` as JSONL.

    Resumable: queries whose keys already have a *successful* record in
    ``out_path`` are skipped. Error records are retried on the next run;
    analysis code should dedupe by ``key``, preferring the most recent
    non-error record.

    Args:
        budget_usd: hard cap on cumulative **actual** spend (computed from
            ``input_tokens`` / ``output_tokens`` returned by the endpoint
            and the pricing in ``factor_leak.cost``). When exceeded, the
            sweep stops submitting new work, lets in-flight workers
            finish, writes their results, and raises ``BudgetExceeded``.
            Pass ``None`` to disable — not recommended for live sweeps.

    Returns only the newly-run results of this invocation.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    queries = list(queries)
    cached = load_cache_keys(out_path)
    todo = [q for q in queries if q.key not in cached]

    missing_endpoints = {q.model_name for q in todo} - endpoints_by_name.keys()
    if missing_endpoints:
        raise KeyError(f"no endpoint registered for: {sorted(missing_endpoints)}")

    guard = _BudgetGuard(budget_usd)
    results: list[ProbeResult] = []
    total = len(todo)
    done = 0
    aborted = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_run_one, q, endpoints_by_name[q.model_name], guard): q
            for q in todo
        }
        for fut in as_completed(futs):
            try:
                result = fut.result()
            except _BudgetAborted:
                aborted += 1
                continue
            _append_jsonl(out_path, result.to_json_record())
            results.append(result)
            done += 1
            # Note: cost accumulation happens inside _run_one (in the
            # worker thread) so queued workers can short-circuit on
            # `guard.exceeded` before calling the endpoint.
            if progress is not None:
                progress(done, total)

    if guard.exceeded:
        raise BudgetExceeded(
            f"hard budget cap of ${budget_usd:.2f} reached at "
            f"${guard.spent_usd:.2f} after {done}/{total} queries "
            f"({aborted} pending workers short-circuited before API call). "
            f"Re-run with a higher --budget-usd to continue "
            f"(cache is preserved)."
        )

    return results
