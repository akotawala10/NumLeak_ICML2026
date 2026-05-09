"""Loader for Kenneth French Data Library monthly factor CSVs.

Source files (downloaded into ``data/ff/``):

- ``F-F_Research_Data_Factors.csv``       — Mkt-RF, SMB, HML, RF since 1926-07
- ``F-F_Research_Data_5_Factors_2x3.csv`` — Mkt-RF, SMB, HML, RMW, CMA, RF since 1963-07
- ``F-F_Momentum_Factor.csv``             — Mom since 1927-01

Ken French reports returns in **percent** (e.g., ``2.89`` means +2.89%). Missing
values are encoded as ``-99.99`` or ``-999``.

Convention here: SMB / HML come from the 3-factor file (longer history, the
canonical 2x3 sort); RMW / CMA come from the 5-factor file; MOM from the
momentum file; RF from the 3-factor file.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

FF3_FILE = "F-F_Research_Data_Factors.csv"
FF5_FILE = "F-F_Research_Data_5_Factors_2x3.csv"
MOM_FILE = "F-F_Momentum_Factor.csv"

FF_MISSING = (-99.99, -999.0)

FACTOR_LONG_NAMES = {
    "Mkt-RF": "market excess return (Mkt-RF)",
    "SMB": "size (SMB, Small Minus Big)",
    "HML": "value (HML, High Minus Low)",
    "RMW": "profitability (RMW, Robust Minus Weak)",
    "CMA": "investment (CMA, Conservative Minus Aggressive)",
    "Mom": "momentum (Mom / UMD)",
}


def _parse_ff_monthly(path: Path) -> pd.DataFrame:
    """Return monthly block of a Ken French CSV as a DataFrame.

    Index is a ``pd.PeriodIndex`` at monthly frequency. Columns are the factor
    series named in the file header. The leading narrative and trailing annual
    block are stripped.
    """
    with open(path, encoding="latin-1") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*,\s*[A-Za-z]", line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No factor header row found in {path}")

    header = [c.strip() for c in lines[header_idx].rstrip("\n").split(",")]
    header[0] = "yyyymm"

    rows: list[list[str]] = []
    for line in lines[header_idx + 1 :]:
        s = line.strip()
        if not s:
            if rows:
                break
            continue
        first = s.split(",")[0].strip()
        if re.fullmatch(r"\d{6}", first):
            rows.append([p.strip() for p in s.split(",")])
        elif re.fullmatch(r"\d{4}", first) and rows:
            break
        elif rows:
            break

    if not rows:
        raise ValueError(f"No monthly rows found in {path}")

    df = pd.DataFrame(rows, columns=header)
    df["yyyymm"] = pd.to_datetime(df["yyyymm"], format="%Y%m").dt.to_period("M")
    df = df.set_index("yyyymm")
    df = df.astype(float)
    df = df.mask(df.isin(FF_MISSING))
    df.index.name = "month"
    return df


def load_ff3(data_dir: Path | str) -> pd.DataFrame:
    """Load 3-factor monthly: columns [Mkt-RF, SMB, HML, RF]."""
    return _parse_ff_monthly(Path(data_dir) / FF3_FILE)


def load_ff5(data_dir: Path | str) -> pd.DataFrame:
    """Load 5-factor monthly: columns [Mkt-RF, SMB, HML, RMW, CMA, RF]."""
    return _parse_ff_monthly(Path(data_dir) / FF5_FILE)


def load_mom(data_dir: Path | str) -> pd.DataFrame:
    """Load momentum monthly: single column [Mom]."""
    return _parse_ff_monthly(Path(data_dir) / MOM_FILE)


def load_all_factors(data_dir: Path | str) -> pd.DataFrame:
    """Return a wide monthly DataFrame with the six probe factors plus RF.

    Columns: ``Mkt-RF, SMB, HML, RMW, CMA, Mom, RF``.

    SMB / HML are taken from the 3-factor file (canonical 2x3 sort). RMW / CMA
    are from the 5-factor file (defined only since 1963-07). RF from the
    3-factor file. Mom from the momentum file.
    """
    ff3 = load_ff3(data_dir)
    ff5 = load_ff5(data_dir)
    mom = load_mom(data_dir)

    out = pd.DataFrame(index=ff3.index)
    out["Mkt-RF"] = ff3["Mkt-RF"]
    out["SMB"] = ff3["SMB"]
    out["HML"] = ff3["HML"]
    out["RMW"] = ff5["RMW"]
    out["CMA"] = ff5["CMA"]
    out["Mom"] = mom["Mom"]
    out["RF"] = ff3["RF"]
    return out


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide factor DataFrame to long format ``[month, factor, return_pct]``."""
    long = df.reset_index().melt(
        id_vars="month", var_name="factor", value_name="return_pct"
    )
    return long.dropna(subset=["return_pct"]).reset_index(drop=True)


def ground_truth(
    data_dir: Path | str, factor: str, month: str | pd.Period
) -> float:
    """Return the Ken French monthly return (percent) for ``factor`` in ``month``.

    ``month`` accepts ``"YYYY-MM"``, ``"YYYYMM"``, or a ``pd.Period``.
    """
    wide = load_all_factors(data_dir)
    if factor not in wide.columns:
        raise KeyError(f"Unknown factor {factor!r}; choose from {list(wide.columns)}")
    if isinstance(month, str):
        s = month.replace("-", "")
        period = pd.Period(pd.to_datetime(s, format="%Y%m"), freq="M")
    else:
        period = pd.Period(month, freq="M")
    if period not in wide.index:
        raise KeyError(f"Month {period} not in Ken French data")
    value = wide.loc[period, factor]
    if pd.isna(value):
        raise KeyError(f"{factor} has no observation for {period}")
    return float(value)
