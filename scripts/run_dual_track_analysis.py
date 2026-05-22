#!/usr/bin/env python3
"""Dual-track analysis for the adsorption ML journal plan.

Track A: high-R2 within-source/source-specific modeling.
Track B: leakage-aware global generalization benchmark.

The goal is to let the manuscript honestly report both:

- high accuracy where the task is interpolation within a source/lab/system,
- harder but reviewer-safe validation across unseen sources.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

sys.path.append(str(Path(__file__).resolve().parent))
from run_leakage_aware_benchmark import (  # noqa: E402
    clipped_metrics,
    feature_sets,
    load_dataset,
    make_model,
)


def compact_join(values: pd.Series, max_items: int = 4) -> str:
    counts = values.fillna("Unknown").astype(str).value_counts().head(max_items)
    return "; ".join(f"{idx} ({count})" for idx, count in counts.items())


def source_template(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for ref, group in df.groupby("Ref_group", dropna=False):
        rows.append(
            {
                "Ref_group": ref,
                "n_rows": len(group),
                "top_adsorbates": compact_join(group["Adsorbate_group"]),
                "top_adsorbents": compact_join(group["Adsorbent_group"], max_items=2),
                "Lab_Source": "",
                "Source_Notes": "",
            }
        )
    template = pd.DataFrame(rows).sort_values("n_rows", ascending=False)
    template.to_csv(out_dir / "lab_source_by_reference_template.csv", index=False)

    row_template = df.copy()
    if "Lab_Source" in row_template.columns:
        row_template["Lab_Source"] = ""
        lab_col = row_template.pop("Lab_Source")
        row_template.insert(0, "Lab_Source", lab_col)
    else:
        row_template.insert(0, "Lab_Source", "")
    row_template.to_csv(out_dir / "data_with_lab_source_template.csv", index=False)


def run_within_source_cv(
    df: pd.DataFrame,
    out_dir: Path,
    min_rows: int,
    max_sources: int | None,
) -> pd.DataFrame:
    selected_feature_sets = [
        fs
        for fs in feature_sets()
        if fs.name
        in {
            "full_capacity_process_model",
            "screening_safe_plus_ion",
            "screening_safe_numeric_only",
        }
    ]
    refs = df["Ref_group"].value_counts()
    refs = refs[refs >= min_rows]
    if max_sources:
        refs = refs.head(max_sources)

    rows = []
    pred_rows = []
    for ref in refs.index:
        sub = df[df["Ref_group"] == ref].copy()
        if sub["RE"].nunique() < 5:
            continue
        n_splits = min(5, len(sub))
        if n_splits < 3:
            continue
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fs in selected_feature_sets:
            cols = fs.numeric + fs.categorical
            model = make_model("xgboost", fs.numeric, fs.categorical, seed=42)
            pred = cross_val_predict(model, sub[cols], sub["RE"], cv=cv)
            metrics = clipped_metrics(sub["RE"], pred)
            rows.append(
                {
                    "Ref_group": ref,
                    "n_rows": len(sub),
                    "feature_set": fs.name,
                    "model": "xgboost",
                    "cv": f"{n_splits}-fold within-source",
                    **metrics,
                }
            )
            fold_pred = sub[
                [
                    "Adsorbent",
                    "Adsorbate",
                    "Adsorbate_group",
                    "Adsorbent_group",
                    "Initial Concentration(ppm)",
                    "Dose(g/L)",
                    "Initial pH",
                    "T(K)",
                    "RE",
                ]
            ].copy()
            fold_pred.insert(0, "Ref_group", ref)
            fold_pred.insert(1, "feature_set", fs.name)
            fold_pred["Predicted_RE"] = np.clip(pred, 0, 100)
            fold_pred["Residual"] = fold_pred["Predicted_RE"] - fold_pred["RE"]
            pred_rows.append(fold_pred)

    results = pd.DataFrame(rows).sort_values(["R2", "n_rows"], ascending=[False, False])
    results.to_csv(out_dir / "track_a_within_source_cv.csv", index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(
            out_dir / "track_a_within_source_predictions.csv", index=False
        )
    return results


def load_track_b_summary(benchmark_dir: Path) -> pd.DataFrame:
    path = benchmark_dir / "benchmark_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/run_leakage_aware_benchmark.py first."
        )
    summary = pd.read_csv(path)
    keep = summary[
        (summary["model"].isin(["xgboost", "lightgbm", "extra_trees"]))
        & summary["feature_set"].isin(
            [
                "full_capacity_process_model",
                "screening_safe_plus_ion",
                "screening_safe_numeric_only",
                "screening_safe_plus_adsorbate_onehot",
            ]
        )
    ].copy()
    keep.to_csv(benchmark_dir / "track_b_global_summary_for_dual_track.csv", index=False)
    return keep


def make_plots(track_a: pd.DataFrame, track_b: pd.DataFrame, out_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    top_a = track_a[track_a["feature_set"] == "full_capacity_process_model"].head(8)
    if not top_a.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [str(x)[:34] + ("..." if len(str(x)) > 34 else "") for x in top_a["Ref_group"]]
        ax.bar(labels, top_a["R2"], color="#2d6a4f")
        ax.axhline(0.95, color="#b08968", linestyle="--", linewidth=1.2, label="R2 = 0.95")
        ax.set_ylabel("Within-source CV R2")
        ax.set_title("Track A: high-R2 source-specific interpolation")
        ax.set_ylim(min(-0.1, top_a["R2"].min() - 0.05), 1.05)
        ax.tick_params(axis="x", rotation=35)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "track_a_within_source_r2.png", dpi=220)
        plt.close(fig)

    best_b = (
        track_b.sort_values(["split", "R2_mean"], ascending=[True, False])
        .groupby("split", as_index=False)
        .head(1)
        .sort_values("split")
    )
    if not best_b.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#457b9d", "#e76f51", "#1d3557"][: len(best_b)]
        ax.bar(best_b["split"], best_b["R2_mean"], yerr=best_b["R2_std"], color=colors)
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_ylabel("Mean R2 over repeated splits")
        ax.set_title("Track B: global generalization is validation-dependent")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / "track_b_global_split_r2.png", dpi=220)
        plt.close(fig)


def table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    shown = df[cols].head(n) if n else df[cols]
    return shown.round(4).to_markdown(index=False)


def write_plan(
    out_dir: Path,
    audit: dict,
    track_a: pd.DataFrame,
    track_b: pd.DataFrame,
    min_rows: int,
) -> None:
    top_a = track_a[track_a["feature_set"] == "full_capacity_process_model"].head(8)
    best_b = (
        track_b.sort_values(["split", "R2_mean"], ascending=[True, False])
        .groupby("split", as_index=False)
        .head(3)
    )
    lines = [
        "# Dual-Track Manuscript Strategy",
        "",
        "## Core Idea",
        "",
        "Use both stories, but keep the claims separate:",
        "",
        "- **Track A: high-R2 source-specific modeling.** This demonstrates that adsorption RE can be predicted very accurately when the task is interpolation within a lab/source/system.",
        "- **Track B: leakage-aware global validation.** This demonstrates the real generalization boundary across unseen references, adsorbents, and adsorbates.",
        "",
        "This is stronger than competing only on a single row-wise R2 because it explains why many papers report near-0.95 while broader models fail under stricter validation.",
        "",
        "## Dataset Snapshot",
        "",
        f"- Clean rows: {audit['clean_rows']}",
        *(
            [
                f"- Excluded primary-domain regimes: {', '.join(audit['excluded_regimes'])}",
                f"- Rows before regime exclusion: {audit['rows_before_regime_exclusion']}",
                f"- Primary modeling rows after exclusion: {audit['rows_after_regime_exclusion']}",
            ]
            if audit.get("excluded_regimes")
            else []
        ),
        f"- Unique references: {audit['unique_references']}",
        f"- Unique adsorbents: {audit['unique_adsorbents']}",
        f"- Unique adsorbates: {audit['unique_adsorbates']}",
        f"- Minimum rows for source-specific CV in Track A: {min_rows}",
        "",
        "## Track A: Source-Specific High-R2 Models",
        "",
        table(top_a, ["Ref_group", "n_rows", "feature_set", "R2", "RMSE", "MAE"], n=8),
        "",
        "Interpretation: these results are suitable for a high-accuracy model claim only within a known source/lab/system.",
        "",
        "## Track B: Global Leakage-Aware Validation",
        "",
        table(best_b, ["split", "feature_set", "model", "R2_mean", "R2_std", "RMSE_mean", "MAE_mean"], n=12),
        "",
        "Interpretation: row-wise interpolation and source-held-out generalization are different scientific tasks. The paper should report both.",
        "",
        "## Proposed Paper Framing",
        "",
        "Suggested title:",
        "",
        "> Source-aware and leakage-aware machine learning for adsorption removal efficiency: high-accuracy interpolation versus cross-source generalization",
        "",
        "Suggested contribution bullets:",
        "",
        "- Compile and clean a heterogeneous heavy-metal adsorption RE dataset.",
        "- Show that source-specific models can reach near-0.95 or higher R2.",
        "- Show that global models degrade under reference/source-held-out validation.",
        "- Separate capacity-including process-monitoring models from pre-experiment screening models.",
        "- Provide a lab-source validation protocol ready for own-lab versus external-lab experiments.",
        "",
        "## What We Need From The Two-Lab Data",
        "",
        "The current CSV has no explicit `Lab_Source` column. Fill one of these files:",
        "",
        "```text",
        f"{out_dir}/lab_source_by_reference_template.csv",
        f"{out_dir}/data_with_lab_source_template.csv",
        "```",
        "",
        "Recommended labels:",
        "",
        "```text",
        "Own_Lab",
        "External_Lab",
        "Literature",
        "```",
        "",
        "Then run:",
        "",
        "```bash",
        "python scripts/run_leakage_aware_benchmark.py --data path/to/labelled_data.csv --out-dir results/leakage_aware_benchmark_no_neutral_low --lab-source-column Lab_Source --exclude-regimes Neutral_LowC0",
        "python scripts/run_dual_track_analysis.py --data path/to/labelled_data.csv --out-dir results/dual_track_no_neutral_low --benchmark-dir results/leakage_aware_benchmark_no_neutral_low --lab-source-column Lab_Source --exclude-regimes Neutral_LowC0",
        "```",
        "",
        "## Journal Strategy",
        "",
        "- Do not claim universal R2 near 0.95 on the heterogeneous dataset.",
        "- Do claim near-0.95 source-specific prediction where supported.",
        "- Make the novel JECE angle the difference between interpolation and external/source generalization.",
        "- If own-lab to external-lab validation is reasonable, promote that as the main result.",
        "- If lab transfer is weak, frame it as a calibrated applicability-domain method rather than a failed model.",
    ]
    (out_dir / "DUAL_TRACK_MANUSCRIPT_PLAN.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("waterresearchpaper/data_2023(1).csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/dual_track"))
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("results/leakage_aware_benchmark"),
    )
    parser.add_argument("--lab-source-column", default="Lab_Source")
    parser.add_argument("--min-source-rows", type=int, default=20)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument(
        "--exclude-regimes",
        nargs="*",
        default=[],
        help="Optional pH-concentration regimes to remove before Track A.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df, audit = load_dataset(args.data, args.lab_source_column)
    if args.exclude_regimes:
        before = len(df)
        before_counts = df["Regime"].value_counts().to_dict()
        df = df[~df["Regime"].isin(args.exclude_regimes)].reset_index(drop=True)
        audit.update(
            {
                "excluded_regimes": args.exclude_regimes,
                "rows_before_regime_exclusion": int(before),
                "rows_after_regime_exclusion": int(len(df)),
                "regime_counts_before_exclusion": before_counts,
                "regime_counts": df["Regime"].value_counts().to_dict(),
            }
        )
    else:
        audit["excluded_regimes"] = []
    source_template(df, args.out_dir)
    track_a = run_within_source_cv(
        df, args.out_dir, min_rows=args.min_source_rows, max_sources=args.max_sources
    )
    track_b = load_track_b_summary(args.benchmark_dir)
    track_b.to_csv(args.out_dir / "track_b_global_summary_for_dual_track.csv", index=False)
    make_plots(track_a, track_b, args.out_dir)
    write_plan(args.out_dir, audit, track_a, track_b, args.min_source_rows)

    print("Wrote dual-track outputs to:", args.out_dir)
    print("\nTrack A top source-specific models:")
    print(
        track_a[track_a["feature_set"] == "full_capacity_process_model"][
            ["Ref_group", "n_rows", "R2", "RMSE", "MAE"]
        ]
        .head(8)
        .round(4)
        .to_string(index=False)
    )
    print("\nTrack B best global models:")
    print(
        track_b.sort_values(["split", "R2_mean"], ascending=[True, False])
        .groupby("split", as_index=False)
        .head(3)[
            ["split", "feature_set", "model", "R2_mean", "R2_std", "RMSE_mean", "MAE_mean"]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
