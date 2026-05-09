"""Forensic upper bound on leak-explainable alpha for LLM-factor papers.

Setup. A published paper claims an LLM-derived signal :math:`\\hat S_t` with
reported alpha :math:`\\alpha_{\\text{paper}}` against a benchmark
Fama--French factor :math:`r_{FF,t}`, and reports a correlation
:math:`\\rho(\\hat S, r_{FF})` between signal and factor. We measure, via
the main sweep, the Pearson correlation :math:`\\rho_{\\text{recall}}`
between the same LLM's estimates of :math:`r_{FF,t}` and the true Ken
French monthly series.

Decomposition and bound.
Assume the worst-case orthogonal decomposition

.. math::

    \\hat S_t = \\lambda\\,\\tilde r_{FF,t} + \\varepsilon_t,
    \\qquad \\varepsilon_t \\perp \\tilde r_{FF,t},

where :math:`\\tilde r_{FF,t}` is the LLM's memory of :math:`r_{FF,t}`
(with Pearson correlation :math:`\\rho_{\\text{recall}}` to the truth
:math:`r_{FF,t}`). Standard algebra gives

.. math::

    \\rho(\\hat S, r_{FF})
    \\;=\\;
    \\lambda\\,\\rho_{\\text{recall}}\\,
    \\frac{\\sigma(\\tilde r_{FF})}{\\sigma(\\hat S)}.

Under the reasonable assumption :math:`\\sigma(\\tilde r_{FF}) \\approx
\\sigma(r_{FF})`, the fraction of :math:`\\hat S`'s standard deviation
attributable to the leak satisfies

.. math::

    \\frac{\\lambda\\,\\sigma(\\tilde r_{FF})}{\\sigma(\\hat S)}
    \\;=\\;
    \\frac{\\rho(\\hat S, r_{FF})}{\\rho_{\\text{recall}}}.

If :math:`\\alpha` is proportional to covariance with :math:`r_{FF}` (as in
a single-factor regression alpha), then the leak-attributable fraction of
:math:`\\alpha_{\\text{paper}}` is at most

.. math::

    \\min\\!\\left(1,\\;
    \\frac{|\\rho_{\\text{recall}}|}{|\\rho(\\hat S, r_{FF})|}\\right).

Equivalently: if the paper's signal--factor correlation is *weaker* than
the recall we measure on that factor, the bound collapses to the full
alpha (the entire reported edge could be leak). If the paper's correlation
is *stronger* than recall, the leak can explain at most the fraction
:math:`|\\rho_{\\text{recall}}|/|\\rho(\\hat S, r_{FF})|`.

We report :math:`\\alpha_{\\text{leak, max}} = \\text{fraction} \\cdot
\\alpha_{\\text{paper}}`. This is an *upper bound*; the realized share may
be lower and cannot exceed the bound without relaxing the
equal-variances assumption.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeakBound:
    """Upper bound on leak-explainable alpha for a single case study."""

    paper: str
    factor: str
    alpha_paper: float             # reported alpha in return units per period
    rho_signal_factor: float       # paper-reported |corr(Ŝ, r_FF)| in [0, 1]
    rho_recall: float              # measured Pearson of LLM recall in [−1, 1]
    fraction_leak_max: float       # min(1, |ρ_recall| / |ρ(Ŝ, r_FF)|) in [0, 1]
    alpha_leak_max: float          # fraction_leak_max * alpha_paper


def leak_upper_bound(
    paper: str,
    factor: str,
    alpha_paper: float,
    rho_signal_factor: float,
    rho_recall: float,
) -> LeakBound:
    """Apply the conservative bound from the module docstring.

    Args:
        paper:             label for the case study (e.g., "Lopez-Lira 2023").
        factor:            Ken French factor name used as benchmark.
        alpha_paper:       the paper's reported alpha in % per period.
        rho_signal_factor: paper-reported |corr(Ŝ, r_FF)| ∈ [0, 1].
        rho_recall:        measured Pearson between LLM estimates and truth
                           for ``factor`` on the paper's sample window;
                           ∈ [−1, 1].

    Returns:
        A ``LeakBound`` record containing both the fractional bound and the
        absolute alpha-units bound. Edge cases:

        - If ``rho_signal_factor == 0``: the paper reports no
          signal--factor correlation, so no leak is possible → fraction=0.
        - If ``rho_signal_factor <= |rho_recall|``: fraction = 1 (all of
          the reported alpha could be leak-explained).

    Raises:
        ValueError: if either correlation is outside its valid range.
    """
    if not 0.0 <= rho_signal_factor <= 1.0:
        raise ValueError(
            f"rho_signal_factor must be in [0,1]; got {rho_signal_factor}"
        )
    if not -1.0 <= rho_recall <= 1.0:
        raise ValueError(f"rho_recall must be in [-1,1]; got {rho_recall}")

    if rho_signal_factor == 0.0:
        fraction = 0.0
    else:
        fraction = min(1.0, abs(rho_recall) / rho_signal_factor)

    alpha_leak_max = fraction * alpha_paper
    return LeakBound(
        paper=paper,
        factor=factor,
        alpha_paper=alpha_paper,
        rho_signal_factor=rho_signal_factor,
        rho_recall=rho_recall,
        fraction_leak_max=fraction,
        alpha_leak_max=alpha_leak_max,
    )
