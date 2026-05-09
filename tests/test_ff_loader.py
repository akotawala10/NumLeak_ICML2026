"""Sanity tests for the Ken French loader.

Run with ``python -m pytest tests/`` from the repo root.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from factor_leak import ff_loader

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ff"


def test_ff3_shape_and_bounds():
    df = ff_loader.load_ff3(DATA_DIR)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert df.index.freqstr == "M"
    assert df.index.min() == pd.Period("1926-07", freq="M")
    # CRSP 202602 build means coverage should reach at least 2025-12.
    assert df.index.max() >= pd.Period("2025-12", freq="M")


def test_ff5_shape_and_bounds():
    df = ff_loader.load_ff5(DATA_DIR)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert df.index.min() == pd.Period("1963-07", freq="M")


def test_mom_shape_and_bounds():
    df = ff_loader.load_mom(DATA_DIR)
    assert list(df.columns) == ["Mom"]
    assert df.index.min() == pd.Period("1927-01", freq="M")


def test_no_sentinels_remain():
    for loader in (ff_loader.load_ff3, ff_loader.load_ff5, ff_loader.load_mom):
        df = loader(DATA_DIR)
        for sentinel in ff_loader.FF_MISSING:
            assert not (df == sentinel).any().any(), (
                f"sentinel {sentinel} leaked through in {loader.__name__}"
            )


def test_spot_values_match_published():
    """Spot-check a few well-known monthly returns against the raw CSVs.

    These values are read directly off the Ken French CSVs (verified visually
    from the raw file); if the loader misparses rows the values will drift.
    """
    df = ff_loader.load_ff3(DATA_DIR)
    # first row
    assert df.loc[pd.Period("1926-07", freq="M"), "Mkt-RF"] == pytest.approx(2.89)
    assert df.loc[pd.Period("1926-07", freq="M"), "SMB"] == pytest.approx(-2.55)
    assert df.loc[pd.Period("1926-07", freq="M"), "HML"] == pytest.approx(-2.39)


def test_load_all_factors_combines_correctly():
    wide = ff_loader.load_all_factors(DATA_DIR)
    expected = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
    assert list(wide.columns) == expected
    # RMW / CMA undefined before 1963-07 but defined at 1963-07.
    assert pd.isna(wide.loc[pd.Period("1926-07", freq="M"), "RMW"])
    assert not pd.isna(wide.loc[pd.Period("1963-07", freq="M"), "RMW"])
    # Mom undefined before 1927-01.
    assert pd.isna(wide.loc[pd.Period("1926-07", freq="M"), "Mom"])


def test_to_long_drops_nans_and_is_sorted():
    wide = ff_loader.load_all_factors(DATA_DIR)
    long = ff_loader.to_long(wide)
    assert {"month", "factor", "return_pct"} == set(long.columns)
    assert not long["return_pct"].isna().any()
    # Every factor appears.
    assert set(long["factor"].unique()) >= {
        "Mkt-RF",
        "SMB",
        "HML",
        "RMW",
        "CMA",
        "Mom",
        "RF",
    }


def test_ground_truth_accepts_multiple_month_formats():
    v1 = ff_loader.ground_truth(DATA_DIR, "Mkt-RF", "1926-07")
    v2 = ff_loader.ground_truth(DATA_DIR, "Mkt-RF", "192607")
    v3 = ff_loader.ground_truth(DATA_DIR, "Mkt-RF", pd.Period("1926-07", freq="M"))
    assert v1 == v2 == v3 == pytest.approx(2.89)


def test_ground_truth_raises_on_bad_inputs():
    with pytest.raises(KeyError):
        ff_loader.ground_truth(DATA_DIR, "NOPE", "2020-03")
    with pytest.raises(KeyError):
        # Way before any factor exists.
        ff_loader.ground_truth(DATA_DIR, "Mkt-RF", "1800-01")
