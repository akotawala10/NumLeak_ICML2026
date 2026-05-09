"""Command-line interface for the factor-leak audit.

Entry point registered in ``pyproject.toml`` as ``factor-leak``.

Commands:
    audit        Run a one-off recall probe on a model × factor × month window.
    analyze      Run the full analysis pipeline on a completed sweep JSONL.
    ground-truth Print the Kenneth French factor truth for a factor/month.
    cost         Estimate the cost of a sweep before running it.

Example:
    factor-leak audit --model claude-sonnet-4.6 --factor Mkt-RF \\
        --from 2020-01 --to 2020-12 --out out.jsonl
    factor-leak analyze --in out.jsonl
    factor-leak ground-truth --factor Mkt-RF --month 2020-03
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .cost import estimate_sweep_cost, format_sweep_cost
from .env import load_dotenv
from .ff_loader import FACTOR_LONG_NAMES, ground_truth
from .probe import ProbeQuery, default_endpoints, run_sweep


def _cmd_audit(args: argparse.Namespace) -> None:
    months = [str(p) for p in pd.period_range(args.from_, args.to, freq="M")]
    queries = [ProbeQuery(args.model, args.factor, args.variant, m) for m in months]
    endpoints = {e.name: e for e in default_endpoints() if e.name == args.model}
    if not endpoints:
        sys.exit(f"no endpoint registered for {args.model!r}")
    print(format_sweep_cost(estimate_sweep_cost(queries)))
    if not args.yes and input("proceed? [y/N] ").strip().lower() not in {"y", "yes"}:
        sys.exit("aborted")
    results = run_sweep(queries, endpoints, args.out,
                       max_workers=args.max_workers,
                       budget_usd=args.budget_usd)
    ok = sum(1 for r in results if r.error is None)
    print(f"ran {len(results)} queries; {ok} succeeded.")


def _cmd_analyze(args: argparse.Namespace) -> None:
    from . import metrics
    raw = metrics.load_results(args.in_)
    raw = raw[raw["error"].isna()].copy()
    df = metrics.enrich(raw, args.data_dir)
    table = metrics.headline_table(df, variants=args.variants)
    print(table.to_string(index=False))


def _cmd_ground_truth(args: argparse.Namespace) -> None:
    value = ground_truth(args.data_dir, args.factor, args.month)
    print(f"{args.factor} {args.month}: {value:.4f}%")


def _cmd_cost(args: argparse.Namespace) -> None:
    months = [str(p) for p in pd.period_range(args.from_, args.to, freq="M")]
    queries = [ProbeQuery(m_, f, v, mo)
               for m_ in args.models for f in args.factors
               for v in args.variants for mo in months]
    print(format_sweep_cost(estimate_sweep_cost(queries)))


def main(argv: list[str] | None = None) -> None:
    load_dotenv(Path.cwd() / ".env")

    ap = argparse.ArgumentParser(prog="factor-leak",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    audit = sub.add_parser("audit", help="run a one-off recall probe")
    audit.add_argument("--model", required=True,
                       help="endpoint name (e.g. claude-sonnet-4.6)")
    audit.add_argument("--factor", required=True, choices=list(FACTOR_LONG_NAMES))
    audit.add_argument("--from", dest="from_", required=True,
                       help="start month YYYY-MM")
    audit.add_argument("--to", required=True, help="end month YYYY-MM")
    audit.add_argument("--variant", default="A", choices=["A", "B"])
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--max-workers", type=int, default=4)
    audit.add_argument("--budget-usd", type=float, default=2.0)
    audit.add_argument("--yes", action="store_true")
    audit.set_defaults(func=_cmd_audit)

    analyze = sub.add_parser("analyze", help="compute headline metrics on a JSONL")
    analyze.add_argument("--in", dest="in_", type=Path, required=True)
    analyze.add_argument("--data-dir", type=Path,
                         default=Path(__file__).parent.parent / "data" / "ff")
    analyze.add_argument("--variants", nargs="+", default=["A"])
    analyze.set_defaults(func=_cmd_analyze)

    gt = sub.add_parser("ground-truth", help="print Ken French truth")
    gt.add_argument("--factor", required=True, choices=list(FACTOR_LONG_NAMES))
    gt.add_argument("--month", required=True, help="YYYY-MM")
    gt.add_argument("--data-dir", type=Path,
                    default=Path(__file__).parent.parent / "data" / "ff")
    gt.set_defaults(func=_cmd_ground_truth)

    cost = sub.add_parser("cost", help="estimate cost of a planned sweep")
    cost.add_argument("--models", nargs="+", required=True)
    cost.add_argument("--factors", nargs="+", required=True)
    cost.add_argument("--variants", nargs="+", default=["A", "B"])
    cost.add_argument("--from", dest="from_", required=True)
    cost.add_argument("--to", required=True)
    cost.set_defaults(func=_cmd_cost)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
