"""Tests for numeric / comparative response parsing."""
from __future__ import annotations

import pytest

from factor_leak.parse import parse_comparative, parse_numeric


# ---------------------------------------------------------------------------
# Numeric extraction — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("-3.12", -3.12),
        ("-3.12%", -3.12),
        ("+2.5%", 2.5),
        ("2.5 percent", 2.5),
        ("approximately -7.8%", -7.8),
        ("The HML factor returned about -13.8% in March 2020.", -13.8),
        ("   0.34   ", 0.34),
        ("I'm not sure, but maybe around +1.5% to +2.5%.", 2.0),  # averaged
        ("50 bps", 0.5),
        ("-250 basis points", -2.5),
    ],
)
def test_parse_numeric_happy_paths(text, expected):
    got = parse_numeric(text)
    assert got == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I don't know.",
        "I cannot recall that specific value.",
        None,
    ],
)
def test_parse_numeric_returns_none_when_no_number(text):
    assert parse_numeric(text) is None


# ---------------------------------------------------------------------------
# Numeric extraction — Unicode minus / dash variants (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("−3.12%", -3.12),        # U+2212 MINUS SIGN
        ("–3.12%", -3.12),        # U+2013 EN DASH
        ("—3.12%", -3.12),        # U+2014 EM DASH
        ("−2.5 percent", -2.5),
        ("The return was −1.0% last month.", -1.0),
    ],
)
def test_parse_numeric_handles_unicode_minuses(text, expected):
    got = parse_numeric(text)
    assert got == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Comparative extraction — endorsement-aware
# ---------------------------------------------------------------------------


def test_parse_comparative_returns_unique_mentioned_candidate():
    """When only one of the candidates appears, return that one."""
    assert (
        parse_comparative("March 2020", month1="2020-03", month2="2008-10")
        == "2020-03"
    )
    assert (
        parse_comparative("october 1987.", month1="1987-10", month2="2020-03")
        == "1987-10"
    )


def test_parse_comparative_picks_endorsed_not_first_mentioned():
    """Regression test for the prompt-echo bug.

    Prompt wording like 'Between March 2020 and October 2008, October 2008
    was higher' caused the old first-mention parser to return '2020-03'
    (the echo), silently inverting accuracy.
    """
    text = "Between March 2020 and October 2008, October 2008 was higher."
    got = parse_comparative(text, month1="2020-03", month2="2008-10")
    assert got == "2008-10"


def test_parse_comparative_picks_on_endorsement_keyword():
    text = "After reviewing, my answer is October 2008, not March 2020."
    got = parse_comparative(text, month1="2020-03", month2="2008-10")
    assert got == "2008-10"


def test_parse_comparative_falls_back_to_last_mention_when_no_endorsement():
    """Models that restate both months and end with the answer but without
    an endorsement keyword should still be parsed correctly via the
    last-mention heuristic."""
    text = "The two months to compare are March 2020 and October 2008."
    got = parse_comparative(text, month1="2020-03", month2="2008-10")
    # No endorsement keyword → last-mention wins.
    assert got == "2008-10"


def test_parse_comparative_returns_none_when_neither_candidate_mentioned():
    text = "I don't have data for those months."
    assert parse_comparative(text, month1="2020-03", month2="2008-10") is None


def test_parse_comparative_rejects_non_candidate_mentions():
    """If the model names an irrelevant month, the parser should ignore it."""
    text = "The actual winner was August 1987, not either of those."
    assert parse_comparative(text, month1="2020-03", month2="2008-10") is None


def test_parse_comparative_legacy_fallback_without_candidates():
    """Calling without month1/month2 degrades to last-mentioned (documented)."""
    text = "Between March 2020 and October 2008, October 2008 was higher."
    assert parse_comparative(text) == "2008-10"


def test_parse_comparative_returns_none_on_empty_input():
    assert parse_comparative("") is None
    assert parse_comparative(None) is None
    assert parse_comparative("   ") is None


def test_parse_comparative_handles_case_and_comma():
    text = "march, 2020 was higher than october, 2008"
    got = parse_comparative(text, month1="2020-03", month2="2008-10")
    assert got == "2020-03"
