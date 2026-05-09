"""End-to-end analysis: JSONL → figures + Table 1 + paper-ready numbers.

Produces:
- ``figures/fig1_recall_heatmap.pdf`` / .png
- ``figures/fig2_temporal_gradient.pdf`` / .png
- ``figures/tab1_headline.tex`` (booktabs-styled LaTeX)
- stdout summary with the numbers the paper needs filled into \\PH placeholders

Run after the sweep JSONL is complete.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from factor_leak import controls, metrics, plots  # noqa: E402
from factor_leak.constants import FAMOUS_MONTHS  # noqa: E402
from factor_leak.cost import cost_from_usage  # noqa: E402

SWEEP = REPO_ROOT / "experiments" / "results" / "sweep.jsonl"
CONTROLS = REPO_ROOT / "experiments" / "results" / "controls.jsonl"
FIG_DIR = REPO_ROOT / "figures"
DATA_DIR = REPO_ROOT / "data" / "ff"


def _dedup(records: list[dict]) -> list[dict]:
    """Keep the most recent non-error record per query key."""
    latest: dict[str, dict] = {}
    for r in records:
        k = r["key"]
        prev = latest.get(k)
        if prev is None:
            latest[k] = r
        elif prev["error"] is not None and r["error"] is None:
            latest[k] = r
        elif (prev["error"] is None) == (r["error"] is None) and r["ts"] > prev["ts"]:
            latest[k] = r
    return list(latest.values())


def load_sweep(path: Path) -> pd.DataFrame:
    raw = [json.loads(l) for l in path.open() if l.strip()]
    raw = _dedup(raw)
    df = pd.DataFrame([
        {"model_name": r["query"]["model_name"],
         "factor": r["query"]["factor"],
         "variant": r["query"]["variant"],
         "month": r["query"]["month"],
         "month2": r["query"].get("month2"),
         "prompt": r["prompt"],
         "response": r["response"],
         "error": r["error"],
         "latency_s": r["latency_s"],
         "ts": r["ts"],
         "key": r["key"],
         "input_tokens": r.get("input_tokens", 0),
         "output_tokens": r.get("output_tokens", 0)}
        for r in raw
    ])
    df = df[df["error"].isna()].copy()
    return df


def print_headline(df: pd.DataFrame) -> pd.DataFrame:
    """Variant A table + Wilson CIs."""
    a = df[df["variant"] == "A"]
    rows: list[dict] = []
    for (m, f), frame in a.groupby(["model_name", "factor"]):
        frame = frame.dropna(subset=["parsed_estimate"])
        n = int(len(frame))
        if n == 0:
            continue
        k5 = int(frame["within_5bps"].sum())
        k25 = int(frame["within_25bps"].sum())
        lo5, hi5 = metrics.wilson_interval(k5, n)
        lo25, hi25 = metrics.wilson_interval(k25, n)
        dir_valid = frame.dropna(subset=["dir_correct"])
        k_dir = int(dir_valid["dir_correct"].sum())
        n_dir = int(len(dir_valid))
        lo_dir, hi_dir = metrics.wilson_interval(k_dir, n_dir)
        # Pearson + bootstrap
        if n >= 3:
            r, r_lo, r_hi = metrics.bootstrap_pearson_ci(
                frame["parsed_estimate"].to_numpy(),
                frame["truth"].to_numpy(),
                n_iters=1000, seed=0,
            )
        else:
            r = r_lo = r_hi = float("nan")
        rows.append({
            "model": m, "factor": f, "n": n,
            "w5": k5 / n, "w5_lo": lo5, "w5_hi": hi5,
            "w25": k25 / n, "w25_lo": lo25, "w25_hi": hi25,
            "dir": k_dir / n_dir if n_dir else float("nan"),
            "dir_lo": lo_dir, "dir_hi": hi_dir,
            "pearson": r, "r_lo": r_lo, "r_hi": r_hi,
        })
    out = pd.DataFrame(rows).sort_values(["model", "factor"])
    print("=== TABLE 1 (Variant A, with Wilson 95% CIs) ===")
    with pd.option_context("display.max_columns", None, "display.width", 160,
                           "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))
    return out


def write_latex_table(headline: pd.DataFrame, path: Path) -> None:
    """booktabs LaTeX rendering with CIs in parentheses."""
    lines = [r"\begin{tabular}{llr rrrr}",
             r"\toprule",
             r"Model & Factor & $n$ & within-$5\bps$ & within-$25\bps$ & "
             r"sign & $\rho$ \\",
             r"\midrule"]
    current = None
    for _, r in headline.iterrows():
        if r["model"] != current:
            if current is not None:
                lines.append(r"\midrule")
            current = r["model"]
            m_display = r["model"]
        else:
            m_display = ""
        lines.append(
            f"{m_display} & {r['factor']} & {int(r['n'])} & "
            f"{r['w5']:.2f} [{r['w5_lo']:.2f},{r['w5_hi']:.2f}] & "
            f"{r['w25']:.2f} [{r['w25_lo']:.2f},{r['w25_hi']:.2f}] & "
            f"{r['dir']:.2f} [{r['dir_lo']:.2f},{r['dir_hi']:.2f}] & "
            f"{r['pearson']:.2f} [{r['r_lo']:.2f},{r['r_hi']:.2f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def write_figures(df: pd.DataFrame) -> None:
    fig1 = plots.recall_heatmap(df, tol_bps=25,
                                title="Within-25 bps recall (Variant A)")
    plots.save_figure(fig1, FIG_DIR / "fig1_recall_heatmap")
    plt.close(fig1)

    # Temporal gradient on months-to-cutoff.
    fig2, ax = plt.subplots(figsize=(6.5, 3.6))
    col = "within_25bps"
    sub = df[df["variant"].isin(["A", "B"])].dropna(subset=[col]).copy()
    for model, frame in sub.groupby("model_name"):
        # Bin by month-blocks to stabilize per-model line.
        frame = frame.copy()
        frame["d_bin"] = (frame["months_to_cutoff"] // 6).astype(int)
        agg = frame.groupby("d_bin")[col].mean().reset_index()
        agg["center_months"] = agg["d_bin"] * 6 + 3
        ax.plot(agg["center_months"], agg[col], marker="o", markersize=3,
                label=model, linewidth=1.3)
    ax.axvspan(-6, 6, color="grey", alpha=0.12, label="±6 months of cutoff")
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("months to cutoff  (positive = in training data)")
    ax.set_ylabel("within-25 bps recall")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Cutoff gradient — recall vs months-to-cutoff")
    fig2.tight_layout()
    plots.save_figure(fig2, FIG_DIR / "fig2_temporal_gradient")
    plt.close(fig2)


def print_cutoff_gradient(df: pd.DataFrame) -> None:
    print("\n=== CUTOFF GRADIENT (variants A+B pooled) ===")
    grad = metrics.temporal_gradient(
        df, tol_bps=25, by=["model_name", "factor"], x="months_to_cutoff",
    )
    # FDR correction across 12 (model × factor) tests.
    pvals = grad["pvalue"].to_numpy()
    qvals, rejected = metrics.benjamini_hochberg(pvals, alpha=0.05)
    grad["q_fdr"] = qvals
    grad["fdr_reject"] = rejected
    with pd.option_context("display.width", 160, "display.max_columns", None,
                           "display.float_format", "{:.4f}".format):
        print(grad.to_string(index=False))


def print_famous(df: pd.DataFrame) -> None:
    a = df[df["variant"] == "A"]
    out = metrics.famous_concentration(a, FAMOUS_MONTHS, tol_bps=25,
                                       by=["model_name", "factor"])
    print("\n=== FAMOUS-MONTH CONCENTRATION (Variant A) ===")
    with pd.option_context("display.width", 160, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))


def print_variant_c(df: pd.DataFrame) -> None:
    c = df[df["variant"] == "C"]
    out = (c.dropna(subset=["comparative_correct"])
             .groupby(["model_name", "factor"])
             .agg(n=("comparative_correct", "size"),
                  acc=("comparative_correct", "mean"))
             .reset_index())
    answered_rate = (c.groupby("model_name")["parsed_month"]
                       .apply(lambda s: s.notna().mean()).rename("answer_rate"))
    print("\n=== VARIANT C (comparative accuracy, refusals excluded) ===")
    print(out.to_string(index=False))
    print()
    print("Variant C answer rate (non-refusal fraction):")
    print(answered_rate.to_string())


def print_controls(df: pd.DataFrame) -> None:
    print("\n=== CONTROL 1: factor-shuffle null (observed vs chance) ===")
    out = controls.chance_rate_factor_shuffle(df, tol_bps=25, n_iters=1000, seed=0)
    if out.empty:
        print("(no variant-A data for null)")
    else:
        with pd.option_context("display.width", 160, "display.max_columns", None,
                               "display.float_format", "{:.3f}".format):
            print(out.to_string(index=False))

    print("\n=== CONTROLS 2 & 3: fabricated series parse rates ===")
    if not CONTROLS.exists():
        print("(controls.jsonl missing — run experiments/05_controls.py)")
        return
    from factor_leak.parse import parse_numeric
    rows = [json.loads(l) for l in CONTROLS.open() if l.strip()]
    df_c = pd.DataFrame(rows)
    df_c["parsed"] = df_c["response"].map(lambda r: parse_numeric(r) if r else None)
    out = (df_c.groupby(["model_name", "kind"])
             .agg(n=("parsed", "size"),
                  parse_rate=("parsed", lambda s: s.notna().mean()))
             .reset_index())
    print(out.to_string(index=False))


def print_cost_summary(df: pd.DataFrame) -> None:
    df_cost = df.copy()
    df_cost["usd"] = df_cost.apply(
        lambda r: cost_from_usage(r["model_name"], r["input_tokens"], r["output_tokens"]),
        axis=1,
    )
    total = df_cost["usd"].sum()
    per_model = df_cost.groupby("model_name")["usd"].sum().round(4)
    print("\n=== COST ===")
    print(f"Total: ${total:.4f}")
    print(per_model.to_string())


def main() -> None:
    if not SWEEP.exists():
        sys.exit(f"missing {SWEEP}; run experiments/01_full_sweep.py first.")
    df = load_sweep(SWEEP)
    enriched = metrics.enrich(df, DATA_DIR)
    FIG_DIR.mkdir(exist_ok=True)

    headline = print_headline(enriched)
    write_latex_table(headline, FIG_DIR / "tab1_headline.tex")
    write_figures(enriched)
    print_cutoff_gradient(enriched)
    print_famous(enriched)
    print_variant_c(enriched)
    print_controls(enriched)
    print_cost_summary(enriched)
    print(f"\nfigures: {FIG_DIR}/fig1_recall_heatmap.pdf, "
          f"{FIG_DIR}/fig2_temporal_gradient.pdf")
    print(f"table:   {FIG_DIR}/tab1_headline.tex")


if __name__ == "__main__":
    main()
