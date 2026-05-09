"""Tests for the forensic upper-bound calculation.

The bound is: ``fraction = min(1, |rho_recall| / rho_signal_factor)``,
``alpha_leak_max = fraction * alpha_paper``.

Edge cases exercised here:
- When signal-factor correlation is less than or equal to recall Pearson,
  the full alpha could be leak (fraction = 1).
- When recall is zero, fraction = 0 (no leak possible).
- Negative recall Pearson still implies leak potential (sign doesn't
  matter; variance-explained argument).
- Correlations outside [0,1] (rho_signal_factor) or [-1,1] (rho_recall)
  raise ValueError.
"""
from __future__ import annotations

import pytest

from factor_leak.forensic import LeakBound, leak_upper_bound


def test_bound_when_recall_strictly_less_than_signal_factor():
    """rho_recall < rho_signal_factor → fraction = rho_recall/rho_signal_factor."""
    b = leak_upper_bound("p", "Mkt-RF",
                         alpha_paper=6.0,
                         rho_signal_factor=0.8,
                         rho_recall=0.4)
    assert b.fraction_leak_max == pytest.approx(0.5)
    assert b.alpha_leak_max == pytest.approx(3.0)
    assert isinstance(b, LeakBound)


def test_bound_caps_at_full_alpha_when_recall_dominates():
    """rho_recall >= rho_signal_factor → full alpha could be leak."""
    b = leak_upper_bound("p", "HML",
                         alpha_paper=6.0,
                         rho_signal_factor=0.3,
                         rho_recall=0.9)
    assert b.fraction_leak_max == pytest.approx(1.0)
    assert b.alpha_leak_max == pytest.approx(6.0)


def test_bound_zero_when_no_recall():
    b = leak_upper_bound("p", "HML", 6.0, 0.5, 0.0)
    assert b.alpha_leak_max == 0.0
    assert b.fraction_leak_max == 0.0


def test_bound_zero_when_no_signal_factor_correlation():
    b = leak_upper_bound("p", "HML", 6.0, 0.0, 0.9)
    assert b.alpha_leak_max == 0.0
    assert b.fraction_leak_max == 0.0


def test_bound_handles_negative_recall_pearson():
    """Negative Pearson → take absolute value (variance-explained argument)."""
    b = leak_upper_bound("p", "HML", 6.0, 0.8, -0.4)
    assert b.fraction_leak_max == pytest.approx(0.5)
    assert b.alpha_leak_max == pytest.approx(3.0)


def test_bound_with_perfect_recall_and_perfect_correlation():
    b = leak_upper_bound("p", "Mkt-RF", 10.0, 1.0, 1.0)
    assert b.fraction_leak_max == pytest.approx(1.0)
    assert b.alpha_leak_max == pytest.approx(10.0)


@pytest.mark.parametrize(
    "rho_sf,rho_recall",
    [(1.1, 0.5), (-0.1, 0.5), (0.5, 1.5), (0.5, -2.0)],
)
def test_bound_rejects_invalid_correlations(rho_sf, rho_recall):
    with pytest.raises(ValueError):
        leak_upper_bound("p", "HML", 5.0, rho_sf, rho_recall)
