#!/usr/bin/env python3
"""Applicability-domain analysis for capacity-free adsorption screening.

This script adds a reliability layer on top of the existing leakage-aware
benchmark. It deliberately uses only pre-experiment descriptors for the trust
score, so the analysis remains compatible with capacity-free screening.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from run_leakage_aware_benchmark import (  # noqa: E402
    feature_sets,
    load_dataset,
    make_model,
    split_indices,
)


PHYSICAL_NUMERIC = [
    "Surface area(m2/g)",
    "Initial Concentration(ppm)",
    "Contact Time (min.)",
    "Dose(g/L)",
    "RPM",
    "Initial pH",
    "T(K)",
]

AD_NUMERIC = [
    *PHYSICAL_NUMERIC,
    "Site_Density",
    "LogTime",
    "Mixing_Index",
    "Driving_Force",
    "Acidity_Strength",
    "ion_known",
    "ion_charge",
    "ion_ionic_radius_A",
    "ion_hydrated_radius_A",
    "ion_potential",
    "hydration_ratio",
]

AD_CATEGORICAL = ["Adsorbent_Family", "Adsorbate_group", "Regime"]


def clipped_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 100.0)
    y = np.asarray(y_true, dtype=float)
    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
    }


def compute_trust_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k_neighbors: int = 10,
) -> pd.DataFrame:
    """Compute pre-experiment applicability-domain scores for test rows."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[AD_NUMERIC])
    x_test = scaler.transform(test[AD_NUMERIC])

    self_nn = NearestNeighbors(n_neighbors=2)
    self_nn.fit(x_train)
    train_nn_dist = self_nn.kneighbors(x_train, return_distance=True)[0][:, 1]

    k = min(k_neighbors, len(train))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_train)
    distances, indices = nn.kneighbors(x_test, return_distance=True)
    test_nn_dist = distances[:, 0]

    sorted_train_dist = np.sort(train_nn_dist)
    distance_percentile = np.searchsorted(
        sorted_train_dist,
        test_nn_dist,
        side="right",
    ) / len(sorted_train_dist)
    percentile_support = np.clip(1.0 - distance_percentile, 0.0, 1.0)
    smooth_support = np.exp(-test_nn_dist / (np.percentile(train_nn_dist, 95) + 1e-9))
    distance_support = 0.5 * percentile_support + 0.5 * smooth_support

    lower = train[PHYSICAL_NUMERIC].quantile(0.05)
    upper = train[PHYSICAL_NUMERIC].quantile(0.95)
    envelope_support = ((test[PHYSICAL_NUMERIC] >= lower) & (test[PHYSICAL_NUMERIC] <= upper)).mean(
        axis=1
    )

    categorical_parts = []
    for col in AD_CATEGORICAL:
        categorical_parts.append(test[col].isin(set(train[col])).astype(float).to_numpy())
    categorical_support = np.vstack(categorical_parts).mean(axis=0)

    train_refs = train["Ref_group"].to_numpy()
    source_diversity = np.array([len(set(train_refs[row])) / len(row) for row in indices])

    trust = (
        0.45 * distance_support
        + 0.20 * envelope_support.to_numpy()
        + 0.20 * categorical_support
        + 0.15 * source_diversity
    )

    return pd.DataFrame(
        {
            "trust_score": trust,
            "distance_support": distance_support,
            "envelope_support": envelope_support.to_numpy(),
            "categorical_support": categorical_support,
            "source_diversity_support": source_diversity,
            "nearest_neighbor_distance": test_nn_dist,
        }
    )


def run_reliability_analysis(
    df: pd.DataFrame,
    out_dir: Path,
    seeds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_set = next(fs for fs in feature_sets() if fs.name == "screening_safe_numeric_only")
    cols = feature_set.numeric + feature_set.categorical
    coverage_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    curve_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    for seed in range(seeds):
        train_idx, test_idx = split_indices(df, "holdout_reference", seed)
        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)

        model = make_model("extra_trees", feature_set.numeric, feature_set.categorical, seed)
        model.fit(train[cols], train["RE"])
        pred = np.clip(model.predict(test[cols]), 0.0, 100.0)

        trust = compute_trust_scores(train, test)
        order = np.argsort(-trust["trust_score"].to_numpy())
        y = test["RE"].to_numpy()

        for coverage in coverage_levels:
            n_keep = max(8, int(round(len(test) * coverage)))
            keep = order[:n_keep]
            metrics = clipped_metrics(y[keep], pred[keep])
            curve_rows.append(
                {
                    "seed": seed,
                    "coverage": coverage,
                    "n_kept": int(n_keep),
                    "trust_mean": float(trust.iloc[keep]["trust_score"].mean()),
                    **metrics,
                }
            )

        pred_df = test[
            [
                "Ref_group",
                "Adsorbent_group",
                "Adsorbate_group",
                "Adsorbent_Family",
                "Regime",
                "Initial pH",
                "Initial Concentration(ppm)",
                "Dose(g/L)",
                "Contact Time (min.)",
                "Surface area(m2/g)",
                "RE",
            ]
        ].copy()
        pred_df.insert(0, "seed", seed)
        pred_df["Predicted_RE"] = pred
        pred_df["Absolute_Error"] = np.abs(pred - y)
        pred_df = pd.concat([pred_df.reset_index(drop=True), trust.reset_index(drop=True)], axis=1)
        pred_rows.append(pred_df)

    curve = pd.DataFrame(curve_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    predictions["trust_quartile"] = pd.qcut(
        predictions["trust_score"],
        q=4,
        labels=["low", "mid-low", "mid-high", "high"],
        duplicates="drop",
    )
    bin_summary = (
        predictions.groupby("trust_quartile", observed=True)
        .agg(
            trust_mean=("trust_score", "mean"),
            MAE=("Absolute_Error", "mean"),
            RMSE=("Absolute_Error", lambda x: float(math.sqrt(np.mean(np.square(x))))),
            n=("Absolute_Error", "size"),
        )
        .reset_index()
    )

    curve.to_csv(out_dir / "applicability_abstention_curve.csv", index=False)
    predictions.to_csv(out_dir / "applicability_prediction_diagnostics.csv", index=False)
    bin_summary.to_csv(out_dir / "trust_bin_error_summary.csv", index=False)
    return curve, predictions, bin_summary


def future_priority_cells(df_full: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    pH_labels = ["pH<4", "4<=pH<5.5", "5.5<=pH<7", "pH>=7"]
    c0_labels = ["C0<=10", "10<C0<=50", "50<C0<=100", "C0>100"]
    df = df_full.copy()
    df["pH_bin"] = pd.cut(
        df["Initial pH"],
        bins=[-1, 4, 5.5, 7, 14],
        labels=pH_labels,
        right=False,
    )
    df["C0_bin"] = pd.cut(
        df["Initial Concentration(ppm)"],
        bins=[-1, 10, 50, 100, 1e12],
        labels=c0_labels,
        right=True,
    )

    rows = []
    for pH_bin in pH_labels:
        for c0_bin in c0_labels:
            sub = df[(df["pH_bin"] == pH_bin) & (df["C0_bin"] == c0_bin)]
            n = len(sub)
            rows.append(
                {
                    "pH_bin": pH_bin,
                    "C0_bin": c0_bin,
                    "n_records": n,
                    "n_sources": int(sub["Ref_group"].nunique()) if n else 0,
                    "n_adsorbates": int(sub["Adsorbate_group"].nunique()) if n else 0,
                    "n_adsorbent_families": int(sub["Adsorbent_Family"].nunique()) if n else 0,
                    "priority": "high" if n < 30 else ("medium" if n < 50 else "lower"),
                }
            )
    priority = pd.DataFrame(rows)
    priority.to_csv(out_dir / "future_experiment_priority_cells.csv", index=False)
    return priority


def make_figure(
    curve: pd.DataFrame,
    bin_summary: pd.DataFrame,
    priority: pd.DataFrame,
    out_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    curve_summary = (
        curve.groupby("coverage", as_index=False)
        .agg(RMSE=("RMSE", "mean"), MAE=("MAE", "mean"), R2=("R2", "mean"))
        .sort_values("coverage", ascending=False)
    )
    x = curve_summary["coverage"] * 100
    axes[0].plot(x, curve_summary["MAE"], marker="o", color="#1d3557", label="MAE")
    axes[0].plot(x, curve_summary["RMSE"], marker="s", color="#e76f51", label="RMSE")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("Predictions retained after trust gating (%)")
    axes[0].set_ylabel("Error in RE percentage points")
    axes[0].set_title("(a) Abstaining from low-trust predictions")
    axes[0].legend(frameon=True)

    colors = ["#d62828", "#f77f00", "#90be6d", "#2d6a4f"]
    axes[1].bar(bin_summary["trust_quartile"].astype(str), bin_summary["MAE"], color=colors)
    axes[1].set_xlabel("Applicability-domain trust quartile")
    axes[1].set_ylabel("Mean absolute error")
    axes[1].set_title("(b) Low trust flags high error")
    for i, row in bin_summary.iterrows():
        axes[1].text(i, row["MAE"] + 0.7, f"n={int(row['n'])}", ha="center", fontsize=8)

    pH_labels = ["pH<4", "4<=pH<5.5", "5.5<=pH<7", "pH>=7"]
    c0_labels = ["C0<=10", "10<C0<=50", "50<C0<=100", "C0>100"]
    c0_display = [
        r"$C_0\leq10$",
        r"$10<C_0\leq50$",
        r"$50<C_0\leq100$",
        r"$C_0>100$",
    ]
    pivot = (
        priority.pivot(index="pH_bin", columns="C0_bin", values="n_records")
        .reindex(index=pH_labels, columns=c0_labels)
        .astype(float)
    )
    image = axes[2].imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    axes[2].set_xticks(np.arange(len(c0_labels)), labels=c0_display, rotation=30, ha="right")
    axes[2].set_yticks(np.arange(len(pH_labels)), labels=pH_labels)
    axes[2].set_title(r"(c) Future pH-$C_0$ sampling priorities")
    for i in range(len(pH_labels)):
        for j in range(len(c0_labels)):
            value = int(pivot.iloc[i, j])
            color = "white" if value > 60 else "black"
            label = f"{value}"
            if value < 30:
                label += "\npriority"
            axes[2].text(j, i, label, ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Records")

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    curve: pd.DataFrame,
    bin_summary: pd.DataFrame,
    priority: pd.DataFrame,
) -> None:
    curve_summary = curve.groupby("coverage", as_index=False).agg(
        R2=("R2", "mean"),
        RMSE=("RMSE", "mean"),
        MAE=("MAE", "mean"),
        trust_mean=("trust_mean", "mean"),
        n_kept=("n_kept", "mean"),
    )
    full = curve_summary[curve_summary["coverage"] == 1.0].iloc[0]
    keep40 = curve_summary[curve_summary["coverage"] == 0.4].iloc[0]
    low = bin_summary[bin_summary["trust_quartile"].astype(str) == "low"].iloc[0]
    high = bin_summary[bin_summary["trust_quartile"].astype(str) == "high"].iloc[0]
    payload = {
        "model": "screening_safe_numeric_only + extra_trees",
        "split": "holdout_reference",
        "coverage_100": full.to_dict(),
        "coverage_40": keep40.to_dict(),
        "mae_reduction_100_to_40_pct": float(100 * (full["MAE"] - keep40["MAE"]) / full["MAE"]),
        "rmse_reduction_100_to_40_pct": float(100 * (full["RMSE"] - keep40["RMSE"]) / full["RMSE"]),
        "low_trust_mae": float(low["MAE"]),
        "high_trust_mae": float(high["MAE"]),
        "high_vs_low_trust_mae_reduction_pct": float(100 * (low["MAE"] - high["MAE"]) / low["MAE"]),
        "high_priority_operating_cells": priority[priority["priority"] == "high"].to_dict(
            orient="records"
        ),
    }
    (out_dir / "applicability_domain_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="waterresearchpaper/data_2023(1).csv")
    parser.add_argument("--out-dir", default="results/applicability_domain")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--exclude-regimes", nargs="*", default=["Neutral_LowC0"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_full, audit = load_dataset(Path(args.data), lab_source_column=None)
    priority = future_priority_cells(df_full, out_dir)
    df = df_full.copy()
    if args.exclude_regimes:
        df = df[~df["Regime"].isin(args.exclude_regimes)].reset_index(drop=True)

    curve, predictions, bin_summary = run_reliability_analysis(df, out_dir, args.seeds)
    make_figure(curve, bin_summary, priority, out_dir / "figure_6_reliability_experiment_design.png")
    write_summary(out_dir, curve, bin_summary, priority)

    print(f"Wrote applicability-domain outputs to {out_dir}")
    print(
        bin_summary[["trust_quartile", "trust_mean", "MAE", "RMSE", "n"]].round(3).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
