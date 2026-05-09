"""Tests for the cutoff-bucket machinery."""
from __future__ import annotations

import pandas as pd
import pytest

from factor_leak.constants import (
    MODEL_CUTOFFS,
    NEAR_CUTOFF_BAND,
    cutoff_bucket,
    months_to_cutoff,
)


def test_months_to_cutoff_positive_when_before_cutoff():
    d = months_to_cutoff("gpt-4o-mini", "2022-01")  # cutoff 2023-10
    assert d == 21  # 21 months before the cutoff


def test_months_to_cutoff_negative_when_after_cutoff():
    d = months_to_cutoff("gpt-4o-mini", "2025-03")  # cutoff 2023-10
    assert d == -17


def test_months_to_cutoff_zero_at_cutoff_month():
    d = months_to_cutoff("gpt-4o-mini", MODEL_CUTOFFS["gpt-4o-mini"])
    assert d == 0


def test_cutoff_bucket_pre_near_post_classification():
    # gpt-4o-mini cutoff = 2023-10, band = ±6
    assert cutoff_bucket("gpt-4o-mini", "2022-01") == "pre"          # 21 months before
    assert cutoff_bucket("gpt-4o-mini", "2023-05") == "near"         # 5 months before
    assert cutoff_bucket("gpt-4o-mini", "2023-10") == "near"         # at cutoff
    assert cutoff_bucket("gpt-4o-mini", "2024-01") == "near"         # 3 months after
    assert cutoff_bucket("gpt-4o-mini", "2025-03") == "post"         # 17 months after


def test_cutoff_bucket_varies_across_models_for_same_month():
    # June 2024 is 8 months past gpt-4o-mini's cutoff (2023-10) → post
    # but 13 months before claude-haiku-4.5's (2025-07) → pre.
    assert cutoff_bucket("gpt-4o-mini", "2024-06") == "post"
    assert cutoff_bucket("claude-haiku-4.5", "2024-06") == "pre"


def test_cutoff_bucket_raises_for_unknown_model():
    with pytest.raises(KeyError):
        cutoff_bucket("mystery-model", "2024-03")
