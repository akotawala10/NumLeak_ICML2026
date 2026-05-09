"""Extract numeric return estimates and variant-C picks from LLM responses.

Variants A (direct numeric) and B (descriptive) both produce free text. We
want to recover the model's point estimate of the monthly return in
percent, if any. A response with no extractable number is kept as ``None``
— *not* coerced to zero — since that is the honest "no-recall" signal.

Variant C (comparative) returns one of two candidate months. Parsing must
be *endorsement-aware*: models frequently restate the prompt's pair before
naming a winner ("Between March 2020 and October 2008, October 2008 was
higher"). A first-mention heuristic would invert the answer. We therefore:

1. Restrict candidates to the two months passed in by the caller;
2. Look for the LAST mention (models typically conclude with the pick);
3. Upgrade a candidate that appears immediately after endorsement tokens
   ("higher", "answer", "pick", "was", "is");
4. Return ``None`` if neither candidate is mentioned.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

from .probe import MONTH_NAMES

# Models emit various Unicode minus characters in addition to ASCII '-':
#   U+2212 MINUS SIGN ("−"), U+2013 EN DASH ("–"), U+2010 HYPHEN, ...
# Normalize them to ASCII before regex matching so one pattern handles all.
_MINUS_TRANSLATE = {
    ord("−"): ord("-"),   # U+2212
    ord("–"): ord("-"),   # U+2013
    ord("—"): ord("-"),   # U+2014
    ord("‐"): ord("-"),   # U+2010
}


def _normalize(text: str) -> str:
    if text is None:
        return ""
    # NFKC first to collapse fullwidth/compatibility forms, then translate
    # the remaining dash-like characters to ASCII hyphen-minus.
    return unicodedata.normalize("NFKC", text).translate(_MINUS_TRANSLATE)


# ---------------------------------------------------------------------------
# Numeric parser (variants A and B)
# ---------------------------------------------------------------------------

# Accept things like:
#   "-3.12"   "+2.5%"   "2.5 percent"   "0.021"  (a bare decimal without %)
#   "-12.3 %"   "approximately -7.8%"
#   "between -5% and -3%"  -> averaged to -4
_NUM = r"[-+]?\d+(?:\.\d+)?"
_PCT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(rf"({_NUM})\s*%"),
    re.compile(rf"({_NUM})\s*percent\b", re.IGNORECASE),
    re.compile(rf"({_NUM})\s*(?:bps|basis\s*points?)\b", re.IGNORECASE),
]
_BARE_NUMBER = re.compile(rf"(?<![A-Za-z])({_NUM})(?![A-Za-z])")


def parse_numeric(text: str | None) -> float | None:
    """Extract a single percent-scale return estimate from model output.

    Rules:
    - A percent-style mention ("2.5%", "2.5 percent") wins over a bare number.
    - A "bps" / "basis points" mention is divided by 100.
    - If multiple percent values are present (e.g., a range), average them.
    - Bare-number fallback is used only for *short* responses (<30 chars
      after normalization). Real LLM refusals on variant B run hundreds
      of chars and contain stray numbers (years, counts); falling back
      to the first number parses those as the estimate and produces
      absurd abs-errors. Short answers, by contrast, are almost always
      variant A's ``"−3.12"`` form and legitimately have no ``%``.
    - Bare-number fallback further requires |value| ≤ 100, ruling out
      numbers like "1975" that clearly aren't a monthly return.
    - Returns ``None`` if no reliable number can be extracted.
    - Handles Unicode minus / en-dash / em-dash / hyphen variants.
    """
    if text is None:
        return None
    stripped = _normalize(text).strip()
    if not stripped:
        return None

    pct_hits: list[float] = []
    for pat in _PCT_PATTERNS[:2]:  # % and "percent"
        pct_hits.extend(float(m.group(1)) for m in pat.finditer(stripped))
    bps_hits = [float(m.group(1)) / 100.0 for m in _PCT_PATTERNS[2].finditer(stripped)]
    hits = pct_hits + bps_hits
    if hits:
        return sum(hits) / len(hits)

    # Bare-number fallback — only for short direct answers.
    if len(stripped) > 30:
        return None
    bare = _BARE_NUMBER.findall(stripped)
    if bare:
        try:
            value = float(bare[0])
        except ValueError:
            return None
        if abs(value) > 100:
            return None
        return value
    return None


# ---------------------------------------------------------------------------
# Comparative parser (variant C) — endorsement-aware
# ---------------------------------------------------------------------------

_MONTH_ANY = "|".join(MONTH_NAMES)
_MONTH_YEAR = re.compile(
    rf"\b({_MONTH_ANY})\b\s*,?\s*(\d{{4}})",
    re.IGNORECASE,
)

# Strong endorsement phrases: a candidate appearing directly after one of
# these is almost certainly the model's pick.
_STRONG_ENDORSE = re.compile(
    rf"(?:my\s+answer\s+is|answer\s*[:\-]|answer\s+is|"
    rf"i\s+(?:pick|picked|choose|chose|select)|"
    rf"the\s+(?:pick|answer|higher|winner)\s+is|"
    rf"higher\s+return\s+was\s+in|had\s+the\s+higher\s+return\s+in)\s+"
    rf"({_MONTH_ANY})\s*,?\s*(\d{{4}})",
    re.IGNORECASE,
)

# Prompt-echo patterns: when the response begins with one of these, the
# first candidate mention is part of the echo, not the answer, so we fall
# through to last-mentioned.
_ECHO_PREFIX = re.compile(
    r"^\s*(between|comparing|compared\s+to|the\s+two\s+months|"
    r"of\s+(?:the|these)\s+two|of\s+the\s+two\s+months|"
    r"looking\s+at|considering)\b",
    re.IGNORECASE,
)

# Refusal / no-information patterns. When *any* of these appear, we treat
# the response as a decline-to-answer rather than a pick — otherwise the
# last-mention fallback scores one candidate as chosen at random,
# polluting the comparative-accuracy denominator with noise.
_REFUSAL = re.compile(
    r"(?:i\s+(?:do\s+not|don['’]?t|cannot|can['’]?t|am\s+not|'m\s+not|"
    r"don['’]?t\s+have|do\s+not\s+have)\s+"
    r"(?:access|the|have|know|enough|specific|reliable|sufficient|exact|"
    r"historical|confident|sure|precise|memorized|memoriz|that)"
    r"|no\s+access\s+to"
    r"|unable\s+to"
    r"|not\s+able\s+to"
    r"|don['’]?t\s+have\s+access"
    r"|i\s+don['’]?t\s+have"
    r"|i\s+cannot\s+(?:reliably|determine|provide|recall|answer|give)"
    r"|cannot\s+reliably"
    r"|would\s+require\s+access"
    r"|without\s+access"
    r"|beyond\s+my\s+knowledge"
    r"|outside\s+(?:of\s+)?my\s+training"
    r"|i['’]?m\s+not\s+able"
    r"|i\s+can['’]?t\s+answer"
    r"|i\s+need\s+to\s+be\s+(?:careful|straightforward|honest)"
    r"|cannot\s+provide"
    r"|unavailable"
    r"|not\s+available"
    r")",
    re.IGNORECASE,
)


def _month_str_to_period(month: str) -> pd.Period:
    return pd.Period(month, freq="M")


def _period_to_key(p: pd.Period) -> str:
    return f"{p.year:04d}-{p.month:02d}"


def _match_to_key(match: re.Match) -> str:
    month_name = match.group(1).title()
    year = int(match.group(2))
    month_num = MONTH_NAMES.index(month_name) + 1
    return f"{year:04d}-{month_num:02d}"


def _all_month_mentions(text: str) -> list[tuple[int, str]]:
    """Return ``[(start_offset, "YYYY-MM"), ...]`` for every Month-Year
    mention in the text, in order of appearance."""
    return [(m.start(), _match_to_key(m)) for m in _MONTH_YEAR.finditer(text)]


# If the answer opens with a candidate this close to the start, we treat
# it as a direct short-form answer (``"March 2020."`` or
# ``"October 2008 was higher."``).
_LEADING_ANSWER_SPAN = 5


def parse_comparative(
    text: str | None,
    month1: str | None = None,
    month2: str | None = None,
) -> str | None:
    """Return the ``"YYYY-MM"`` month the model endorses from the pair.

    Tiered heuristic, earlier tiers take precedence:

    1. **Candidate filter.** If ``month1`` / ``month2`` are given, ignore
       mentions of any other months. Defends against prompt-echo and
       non-candidate hallucinations.
    2. **Unique mention.** If exactly one of the two candidates appears,
       that's the pick.
    3. **Strong endorsement.** Phrases like ``"my answer is X"``,
       ``"answer: X"``, ``"I pick X"`` deterministically identify ``X``.
    4. **Prompt echo.** If the response opens with ``"Between"``,
       ``"Comparing"``, ``"The two months"``, etc., the first candidate
       mention is an echo of the prompt; fall through to last-mentioned.
    5. **Leading answer.** If the first candidate appears within the
       first ~5 characters of the (stripped) response, that's the pick.
       Handles ``"October 2008 was higher than March 2020."``.
    6. **Last-mentioned fallback.** Otherwise the last candidate mentioned
       wins — models that expound before answering tend to conclude with
       their pick.

    Without ``month1`` / ``month2`` the function degrades to the
    last-mentioned month in the text (exploratory only; not used by the
    analysis pipeline, which always passes both candidates).
    """
    if text is None:
        return None
    normalized = _normalize(text).strip()
    if not normalized:
        return None
    mentions = _all_month_mentions(normalized)
    if not mentions:
        return None

    if month1 is None and month2 is None:
        return mentions[-1][1]

    candidates: set[str] = set()
    for m in (month1, month2):
        if m is not None:
            candidates.add(_period_to_key(_month_str_to_period(m)))

    valid = [(off, key) for off, key in mentions if key in candidates]
    if not valid:
        return None

    unique_keys = {key for _, key in valid}

    # Tier 0: refusal detection. A response that explicitly disclaims
    # knowledge should not be counted as a pick even if it echoes the
    # candidate months. Exception: if the response *also* contains a
    # strong endorsement phrase naming one of the candidates, trust that
    # (some models refuse in the preamble and still commit to an answer).
    se_match = _STRONG_ENDORSE.search(normalized)
    refused = _REFUSAL.search(normalized) is not None
    if refused and se_match is None:
        return None

    if len(unique_keys) == 1:
        return next(iter(unique_keys))

    # Tier 3: strong endorsement phrase that specifies a candidate.
    if se_match:
        key = _match_to_key(se_match)
        if key in candidates:
            return key

    # Tier 4: prompt-echo → last-mentioned wins.
    if _ECHO_PREFIX.match(normalized):
        return valid[-1][1]

    # Tier 5: leading-answer heuristic.
    first_off, first_key = valid[0]
    if first_off <= _LEADING_ANSWER_SPAN:
        return first_key

    # Tier 6: fall back to last mention.
    return valid[-1][1]
