#!/usr/bin/env python3
"""Leakage-aware benchmark for adsorption removal efficiency prediction.

This script intentionally separates:

1. Full/process-monitoring models that include adsorption capacity.
2. Screening models that exclude adsorption capacity and capacity-derived terms.

It also reports stricter group validations so a manuscript can distinguish
row-level interpolation from source/adsorbent/adsorbate generalization.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


ION_PROPERTIES = {
    # Approximate descriptors for benchmarking. For the paper, replace with a
    # cited SI table and pH-dependent speciation where relevant.
    "Cd2+": {"charge": 2, "ionic_radius_A": 0.95, "hydrated_radius_A": 4.26},
    "Pb2+": {"charge": 2, "ionic_radius_A": 1.19, "hydrated_radius_A": 4.01},
    "Ni2+": {"charge": 2, "ionic_radius_A": 0.69, "hydrated_radius_A": 4.04},
    "Cu2+": {"charge": 2, "ionic_radius_A": 0.73, "hydrated_radius_A": 4.19},
    "Zn2+": {"charge": 2, "ionic_radius_A": 0.74, "hydrated_radius_A": 4.30},
    "Co2+": {"charge": 2, "ionic_radius_A": 0.745, "hydrated_radius_A": 4.23},
    "Hg2+": {"charge": 2, "ionic_radius_A": 1.02, "hydrated_radius_A": 4.28},
    "Fe2+": {"charge": 2, "ionic_radius_A": 0.78, "hydrated_radius_A": 4.30},
    "Fe3+": {"charge": 3, "ionic_radius_A": 0.645, "hydrated_radius_A": 4.57},
    "As3+": {"charge": 3, "ionic_radius_A": 0.58, "hydrated_radius_A": 4.75},
    "As5+": {"charge": 5, "ionic_radius_A": 0.46, "hydrated_radius_A": 4.75},
    "Cr3+": {"charge": 3, "ionic_radius_A": 0.615, "hydrated_radius_A": 4.61},
    "Cr6+": {"charge": 6, "ionic_radius_A": 0.44, "hydrated_radius_A": 4.60},
    "Al3+": {"charge": 3, "ionic_radius_A": 0.535, "hydrated_radius_A": 4.75},
    "Ca2+": {"charge": 2, "ionic_radius_A": 1.00, "hydrated_radius_A": 4.12},
    "Mg2+": {"charge": 2, "ionic_radius_A": 0.72, "hydrated_radius_A": 4.28},
    "Na+": {"charge": 1, "ionic_radius_A": 1.02, "hydrated_radius_A": 3.58},
    "K+": {"charge": 1, "ionic_radius_A": 1.38, "hydrated_radius_A": 3.31},
    "Tl+": {"charge": 1, "ionic_radius_A": 1.50, "hydrated_radius_A": 3.60},
    "V": {"charge": 5, "ionic_radius_A": 0.54, "hydrated_radius_A": 4.50},
}

ION_ALIASES = {
    "Cd": "Cd2+",
    "Pb": "Pb2+",
    "Ni": "Ni2+",
    "Cu": "Cu2+",
    "cu2+": "Cu2+",
    "Cu2++": "Cu2+",
    "Zn": "Zn2+",
    "Co": "Co2+",
    "Hg": "Hg2+",
    "Fe": "Fe2+",
    "As": "As3+",
    "Cr": "Cr3+",
}


@dataclass(frozen=True)
class FeatureSet:
    name: str
    numeric: list[str]
    categorical: list[str]
    note: str


def clean_numeric(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if re.match(r"^\d+,\d+$", text):
        text = text.replace(",", ".")
    else:
        text = text.split(",")[0]
    text = text.replace("+/-", "").replace("±", "").replace("~", "")
    text = text.replace("%", "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group()) if match else np.nan


def normalize_adsorbate(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip().replace(" ", "")
    text = text.replace("(II)", "2+").replace("(III)", "3+")
    text = text.replace("(IV)", "4+").replace("(VI)", "6+")
    text = re.sub(r"^([A-Za-z]+)(\d)$", r"\1\2+", text)
    return ION_ALIASES.get(text, text)


def classify_adsorbent(name: object) -> str:
    text = str(name).lower()
    if any(k in text for k in ["graphene", "carbon", "biochar", "activated", "charcoal"]):
        return "carbon_based"
    if any(k in text for k in ["oxide", "ferrite", "fe3o4", "tio2", "al2o3", "mno2", "zeolite"]):
        return "mineral_oxide"
    if any(k in text for k in ["polymer", "resin", "chitosan", "copolymer"]):
        return "polymer"
    if any(k in text for k in ["husk", "wood", "biomass", "straw", "sawdust", "alga", "cellulose"]):
        return "biomass"
    return "composite_other"


def load_dataset(path: Path, lab_source_column: str | None) -> tuple[pd.DataFrame, dict]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)
    has_lab_source = bool(lab_source_column and lab_source_column in raw.columns)

    df = raw.rename(
        columns={
            "RE (%)": "RE",
            "Intial Concentration(ppm)": "Initial Concentration(ppm)",
        }
    ).copy()

    numeric_cols = [
        "Surface area(m2/g)",
        "Adsorption capacity(mg/g)",
        "Initial Concentration(ppm)",
        "RE",
        "Contact Time (min.)",
        "Dose(g/L)",
        "RPM",
        "Initial pH",
        "T(K)",
    ]
    for col in numeric_cols:
        df[col] = df[col].apply(clean_numeric)

    before = len(df)
    df = df.dropna(subset=numeric_cols)
    df = df[(df["RE"] >= 0) & (df["RE"] <= 100)].reset_index(drop=True)

    df["Adsorbate_norm"] = df["Adsorbate"].apply(normalize_adsorbate)
    df["Adsorbent_Family"] = df["Adsorbent"].apply(classify_adsorbent)

    df["Site_Density"] = (
        df["Dose(g/L)"] * df["Surface area(m2/g)"]
    ) / (df["Initial Concentration(ppm)"] + 1e-9)
    df["Cap_Load_Ratio"] = df["Adsorption capacity(mg/g)"] / (
        df["Initial Concentration(ppm)"] + 1e-9
    )
    df["LogTime"] = np.log1p(df["Contact Time (min.)"])
    df["Mixing_Index"] = df["RPM"] * df["LogTime"]
    df["Thermo_Capacity"] = df["Adsorption capacity(mg/g)"] * df["T(K)"]
    df["Driving_Force"] = df["Initial Concentration(ppm)"] / (
        df["Dose(g/L)"] + 1e-9
    )
    df["Acidity_Strength"] = np.abs(df["Initial pH"] - 7.0)

    for key in ["charge", "ionic_radius_A", "hydrated_radius_A"]:
        df[f"ion_{key}"] = df["Adsorbate_norm"].map(
            lambda ion: ION_PROPERTIES.get(ion, {}).get(key, np.nan)
        )
    df["ion_known"] = df["Adsorbate_norm"].map(lambda ion: ion in ION_PROPERTIES).astype(int)
    df["ion_potential"] = df["ion_charge"] / (df["ion_ionic_radius_A"] + 1e-9)
    df["hydration_ratio"] = df["ion_hydrated_radius_A"] / (
        df["ion_ionic_radius_A"] + 1e-9
    )
    ion_cols = [
        "ion_charge",
        "ion_ionic_radius_A",
        "ion_hydrated_radius_A",
        "ion_potential",
        "hydration_ratio",
    ]
    for col in ion_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median if np.isfinite(median) else 0.0)

    c0_median = df["Initial Concentration(ppm)"].median()
    df["Regime"] = np.select(
        [
            (df["Initial pH"] < 7) & (df["Initial Concentration(ppm)"] < c0_median),
            (df["Initial pH"] < 7) & (df["Initial Concentration(ppm)"] >= c0_median),
            (df["Initial pH"] >= 7) & (df["Initial Concentration(ppm)"] < c0_median),
        ],
        ["Acidic_LowC0", "Acidic_HighC0", "Neutral_LowC0"],
        default="Neutral_HighC0",
    )

    df["Ref_group"] = df["Ref."].fillna("Unknown").astype(str).str.strip()
    df["Adsorbent_group"] = df["Adsorbent"].fillna("Unknown").astype(str).str.strip()
    df["Adsorbate_group"] = df["Adsorbate_norm"].fillna("Unknown").astype(str).str.strip()

    if has_lab_source:
        df["Lab_Source"] = df[lab_source_column].fillna("Unknown").astype(str).str.strip()
    else:
        df["Lab_Source"] = "UNLABELED"

    audit = {
        "input_file": str(path),
        "raw_rows": int(before),
        "clean_rows": int(len(df)),
        "dropped_rows": int(before - len(df)),
        "unique_references": int(df["Ref_group"].nunique()),
        "unique_adsorbents": int(df["Adsorbent_group"].nunique()),
        "unique_adsorbates": int(df["Adsorbate_group"].nunique()),
        "regime_counts": df["Regime"].value_counts().to_dict(),
        "adsorbate_counts_top20": df["Adsorbate_group"].value_counts().head(20).to_dict(),
        "lab_source_counts": df["Lab_Source"].value_counts().to_dict(),
        "lab_source_column_used": lab_source_column if has_lab_source else None,
    }
    return df, audit


def feature_sets() -> list[FeatureSet]:
    base = [
        "Surface area(m2/g)",
        "Initial Concentration(ppm)",
        "Contact Time (min.)",
        "Dose(g/L)",
        "RPM",
        "Initial pH",
        "T(K)",
    ]
    safe_engineered = [
        "Site_Density",
        "LogTime",
        "Mixing_Index",
        "Driving_Force",
        "Acidity_Strength",
    ]
    capacity_terms = [
        "Adsorption capacity(mg/g)",
        "Cap_Load_Ratio",
        "Thermo_Capacity",
    ]
    ion_terms = [
        "ion_known",
        "ion_charge",
        "ion_ionic_radius_A",
        "ion_hydrated_radius_A",
        "ion_potential",
        "hydration_ratio",
    ]
    return [
        FeatureSet(
            name="full_capacity_process_model",
            numeric=base + capacity_terms + safe_engineered + ion_terms,
            categorical=["Adsorbate_norm", "Adsorbent_Family"],
            note="Includes adsorption capacity; suitable only as process-monitoring/descriptive model.",
        ),
        FeatureSet(
            name="screening_safe_plus_ion",
            numeric=base + safe_engineered + ion_terms,
            categorical=["Adsorbent_Family"],
            note="No adsorption capacity; uses ion descriptors and adsorbent family.",
        ),
        FeatureSet(
            name="screening_safe_plus_adsorbate_onehot",
            numeric=base + safe_engineered,
            categorical=["Adsorbate_norm", "Adsorbent_Family"],
            note="No adsorption capacity; uses adsorbate identity as a categorical feature.",
        ),
        FeatureSet(
            name="screening_safe_numeric_only",
            numeric=base + safe_engineered,
            categorical=[],
            note="No adsorption capacity and no ion identity; strictest screening baseline.",
        ),
    ]


def make_model(model_name: str, numeric: list[str], categorical: list[str], seed: int) -> Pipeline:
    transformers = []
    if numeric:
        scaler = StandardScaler() if model_name == "ridge" else "passthrough"
        transformers.append(("num", scaler, numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    preprocess = ColumnTransformer(transformers, remainder="drop")

    if model_name == "dummy_mean":
        estimator = DummyRegressor(strategy="mean")
    elif model_name == "ridge":
        estimator = Ridge(alpha=1.0)
    elif model_name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=4,
        )
    elif model_name == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=4,
        )
    elif model_name == "xgboost":
        estimator = XGBRegressor(
            n_estimators=140,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=4,
        )
    elif model_name == "lightgbm":
        estimator = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=seed,
            n_jobs=4,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def clipped_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, 100.0)
    y = np.asarray(y_true, dtype=float)
    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "bias": float(np.mean(pred - y)),
        "pred_min": float(np.min(pred)),
        "pred_max": float(np.max(pred)),
    }


def split_indices(df: pd.DataFrame, split: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(df))
    if split == "random_regime":
        train_idx, test_idx = train_test_split(
            idx,
            test_size=0.2,
            random_state=seed,
            stratify=df["Regime"],
        )
        return train_idx, test_idx
    if split == "holdout_reference":
        groups = df["Ref_group"]
    elif split == "holdout_adsorbent":
        groups = df["Adsorbent_group"]
    elif split == "holdout_lab_source":
        groups = df["Lab_Source"]
        if groups.nunique() < 2:
            raise ValueError("Lab_Source has fewer than two groups.")
    else:
        raise ValueError(f"Unknown split: {split}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    return next(splitter.split(df, df["RE"], groups=groups))


def evaluate_once(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    model_name: str,
    split: str,
    seed: int,
) -> dict:
    train_idx, test_idx = split_indices(df, split, seed)
    cols = feature_set.numeric + feature_set.categorical
    x = df[cols]
    y = df["RE"]

    model = make_model(model_name, feature_set.numeric, feature_set.categorical, seed)
    model.fit(x.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(x.iloc[test_idx])
    metrics = clipped_metrics(y.iloc[test_idx], pred)

    row = {
        "feature_set": feature_set.name,
        "model": model_name,
        "split": split,
        "seed": seed,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_references": int(df.iloc[test_idx]["Ref_group"].nunique()),
        "test_adsorbents": int(df.iloc[test_idx]["Adsorbent_group"].nunique()),
        "test_adsorbates": int(df.iloc[test_idx]["Adsorbate_group"].nunique()),
    }
    row.update(metrics)
    return row


def evaluate_leave_one_adsorbate(
    df: pd.DataFrame,
    feature_set: FeatureSet,
    model_name: str,
    min_group_rows: int,
    seed: int,
) -> list[dict]:
    rows = []
    cols = feature_set.numeric + feature_set.categorical
    x = df[cols]
    y = df["RE"]
    counts = df["Adsorbate_group"].value_counts()
    for adsorbate, n_rows in counts.items():
        if n_rows < min_group_rows:
            continue
        test_idx = np.where(df["Adsorbate_group"].to_numpy() == adsorbate)[0]
        train_idx = np.where(df["Adsorbate_group"].to_numpy() != adsorbate)[0]
        model = make_model(model_name, feature_set.numeric, feature_set.categorical, seed)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[test_idx])
        metrics = clipped_metrics(y.iloc[test_idx], pred)
        row = {
            "feature_set": feature_set.name,
            "model": model_name,
            "left_out_adsorbate": adsorbate,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    grouped = raw.groupby(["feature_set", "model", "split"], as_index=False)
    return grouped.agg(
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        R2_min=("R2", "min"),
        R2_max=("R2", "max"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        bias_mean=("bias", "mean"),
        n_test_mean=("n_test", "mean"),
    )


def best_by_split(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.sort_values(["split", "R2_mean"], ascending=[True, False])
    return ordered.groupby("split", as_index=False).head(5)


def write_markdown_report(
    out_path: Path,
    audit: dict,
    summary: pd.DataFrame,
    loo: pd.DataFrame,
    feature_sets_used: list[FeatureSet],
    lab_split_available: bool,
) -> None:
    best = best_by_split(summary)
    capacity = summary[summary["feature_set"] == "full_capacity_process_model"]
    screening = summary[summary["feature_set"].str.startswith("screening")]

    def table(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "_No rows._"
        return df[cols].round(4).to_markdown(index=False)

    lines = [
        "# Leakage-Aware Adsorption ML Benchmark",
        "",
        "## Dataset",
        "",
        f"- Clean rows: {audit['clean_rows']} / raw rows: {audit['raw_rows']}",
        f"- Unique references: {audit['unique_references']}",
        f"- Unique adsorbents: {audit['unique_adsorbents']}",
        f"- Unique adsorbates: {audit['unique_adsorbates']}",
        f"- Lab source column used: {audit['lab_source_column_used']}",
        *(
            [
                f"- Excluded regimes before modeling: {', '.join(audit['excluded_regimes'])}",
                f"- Rows before regime exclusion: {audit['rows_before_regime_exclusion']}",
                f"- Rows after regime exclusion: {audit['rows_after_regime_exclusion']}",
            ]
            if audit.get("excluded_regimes")
            else []
        ),
        "",
        "Regime counts:",
        "",
        "```text",
        *[f"{k}: {v}" for k, v in audit["regime_counts"].items()],
        "```",
        "",
        "## Feature Sets",
        "",
    ]
    for fs in feature_sets_used:
        lines.append(f"- `{fs.name}`: {fs.note}")
    lines.extend(
        [
            "",
            "## Best Models By Split",
            "",
            table(
                best,
                [
                    "split",
                    "feature_set",
                    "model",
                    "R2_mean",
                    "R2_std",
                    "RMSE_mean",
                    "MAE_mean",
                ],
            ),
            "",
            "## Capacity-Including Model Summary",
            "",
            table(
                capacity,
                [
                    "model",
                    "split",
                    "R2_mean",
                    "R2_std",
                    "RMSE_mean",
                    "MAE_mean",
                    "bias_mean",
                ],
            ),
            "",
            "## Screening Model Summary",
            "",
            table(
                screening,
                [
                    "feature_set",
                    "model",
                    "split",
                    "R2_mean",
                    "R2_std",
                    "RMSE_mean",
                    "MAE_mean",
                ],
            ),
            "",
            "## Leave-One-Adsorbate-Out Summary",
            "",
        ]
    )
    if loo.empty:
        lines.append("_No leave-one-adsorbate rows generated._")
    else:
        loo_summary = (
            loo.groupby(["feature_set", "model"], as_index=False)
            .agg(
                R2_mean=("R2", "mean"),
                R2_median=("R2", "median"),
                RMSE_mean=("RMSE", "mean"),
                MAE_mean=("MAE", "mean"),
                n_groups=("left_out_adsorbate", "nunique"),
            )
            .sort_values("R2_mean", ascending=False)
        )
        lines.append(
            table(
                loo_summary,
                [
                    "feature_set",
                    "model",
                    "n_groups",
                    "R2_mean",
                    "R2_median",
                    "RMSE_mean",
                    "MAE_mean",
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `random_regime` estimates row-level interpolation within the merged dataset.",
            "- `holdout_reference` estimates generalization to unseen literature/lab sources; this is the most important reviewer-facing stress test available without explicit lab labels.",
            "- `holdout_adsorbent` estimates generalization to unseen adsorbent names.",
            "- Capacity-including models should not be framed as pre-experiment screening models.",
            "- Screening models are more scientifically defensible, but lower accuracy is expected.",
        ]
    )
    if audit.get("excluded_regimes"):
        lines.extend(
            [
                "- Regime exclusions define the primary applicability domain and must be reported explicitly.",
                "",
                "## Reproducibility Command",
                "",
                "```bash",
                "python scripts/run_leakage_aware_benchmark.py "
                "--out-dir results/leakage_aware_benchmark_no_neutral_low "
                "--seeds 5 "
                "--models dummy_mean xgboost extra_trees lightgbm "
                "--exclude-regimes Neutral_LowC0",
                "```",
            ]
        )
    if not lab_split_available:
        lines.extend(
            [
                "",
                "## Missing Lab Split",
                "",
                "No usable `Lab_Source` column was found. Add a column with values such as `Own_Lab` and `External_Lab`, then rerun with:",
                "",
                "```bash",
                "python scripts/run_leakage_aware_benchmark.py "
                "--data path/to/labelled_data.csv "
                "--lab-source-column Lab_Source "
                "--exclude-regimes Neutral_LowC0",
                "```",
            ]
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("waterresearchpaper/data_2023(1).csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/leakage_aware_benchmark"),
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dummy_mean", "ridge", "random_forest", "extra_trees", "xgboost", "lightgbm"],
    )
    parser.add_argument("--lab-source-column", default="Lab_Source")
    parser.add_argument(
        "--exclude-regimes",
        nargs="*",
        default=[],
        help="Optional pH-concentration regimes to remove before modeling.",
    )
    parser.add_argument("--leave-one-adsorbate-min-rows", type=int, default=20)
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
    fs_list = feature_sets()
    splits = ["random_regime", "holdout_reference", "holdout_adsorbent"]
    lab_split_available = df["Lab_Source"].nunique() >= 2
    if lab_split_available:
        splits.append("holdout_lab_source")

    rows = []
    for seed in range(args.seeds):
        for split in splits:
            for fs in fs_list:
                for model_name in args.models:
                    rows.append(evaluate_once(df, fs, model_name, split, seed))
    raw = pd.DataFrame(rows)
    summary = summarize(raw)

    loo_rows = []
    loo_models = [m for m in ["xgboost", "lightgbm", "extra_trees", "random_forest"] if m in args.models]
    for fs in fs_list:
        for model_name in loo_models:
            loo_rows.extend(
                evaluate_leave_one_adsorbate(
                    df,
                    fs,
                    model_name,
                    args.leave_one_adsorbate_min_rows,
                    seed=42,
                )
            )
    loo = pd.DataFrame(loo_rows)

    raw.to_csv(args.out_dir / "benchmark_raw.csv", index=False)
    summary.to_csv(args.out_dir / "benchmark_summary.csv", index=False)
    best_by_split(summary).to_csv(args.out_dir / "best_by_split.csv", index=False)
    loo.to_csv(args.out_dir / "leave_one_adsorbate.csv", index=False)
    (args.out_dir / "dataset_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (args.out_dir / "feature_sets.json").write_text(
        json.dumps(
            [
                {
                    "name": fs.name,
                    "numeric": fs.numeric,
                    "categorical": fs.categorical,
                    "note": fs.note,
                }
                for fs in fs_list
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown_report(
        args.out_dir / "REPORT.md",
        audit,
        summary,
        loo,
        fs_list,
        lab_split_available=lab_split_available,
    )

    print("\nWrote benchmark outputs to:", args.out_dir)
    print("\nTop rows by split:")
    cols = ["split", "feature_set", "model", "R2_mean", "R2_std", "RMSE_mean", "MAE_mean"]
    print(best_by_split(summary)[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
