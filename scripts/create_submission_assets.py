#!/usr/bin/env python3
"""Create submission-grade figures for the JECE manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_draft" / "jece_latex_bundle"
DATA_TEMPLATE = ROOT / "results" / "dual_track" / "data_with_lab_source_template.csv"


def add_box(ax, xy, wh, text, fc, ec="#1f2933", size=10, weight="normal"):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.3,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight=weight,
        color="#111827",
        linespacing=1.18,
    )


def add_arrow(ax, start, end, color="#374151", lw=1.5):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=lw,
            color=color,
            shrinkA=5,
            shrinkB=5,
        )
    )


def workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.94,
        "Source-aware validation framework for adsorption removal-efficiency prediction",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#0f172a",
    )

    add_box(
        ax,
        (0.04, 0.62),
        (0.24, 0.19),
        "Heterogeneous adsorption data\n563 cleaned records\n537 in primary domain",
        "#dbeafe",
        size=10,
        weight="bold",
    )
    add_box(
        ax,
        (0.38, 0.62),
        (0.24, 0.19),
        "Feature definition\ncapacity-inclusive process model\ncapacity-free screening models",
        "#dcfce7",
        size=9.5,
    )
    add_box(
        ax,
        (0.72, 0.62),
        (0.24, 0.19),
        "Tree-based learning\nXGBoost, LightGBM, ExtraTrees\nrepeated split evaluation",
        "#fef3c7",
        size=9.5,
    )

    add_arrow(ax, (0.28, 0.715), (0.38, 0.715))
    add_arrow(ax, (0.62, 0.715), (0.72, 0.715))

    add_box(
        ax,
        (0.11, 0.25),
        (0.32, 0.22),
        "Track A: source-specific interpolation\nwithin-source 5-fold CV\nbest R2 = 0.992",
        "#ccfbf1",
        size=10,
        weight="bold",
    )
    add_box(
        ax,
        (0.57, 0.25),
        (0.32, 0.22),
        "Track B: external generalization\nrow-wise, adsorbent-held-out,\nreference-held-out splits\nbest held-out source R2 = 0.330",
        "#fee2e2",
        size=10,
        weight="bold",
    )

    add_arrow(ax, (0.50, 0.62), (0.27, 0.47))
    add_arrow(ax, (0.50, 0.62), (0.73, 0.47))

    ax.text(
        0.5,
        0.11,
        "Main finding: high R2 is valid for in-source interpolation; cross-source prediction needs a defined pH-concentration domain.",
        ha="center",
        va="center",
        fontsize=11,
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#94a3b8"),
    )

    fig.tight_layout()
    fig.savefig(OUT / "figure_1_workflow.png", dpi=300)
    plt.close(fig)


def graphical_abstract() -> None:
    fig, ax = plt.subplots(figsize=(13.28, 5.31))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.patch.set_facecolor("#f8fafc")
    ax.text(
        0.5,
        0.90,
        "Adsorption ML needs source-aware validation",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#0f172a",
    )

    add_box(ax, (0.04, 0.38), (0.22, 0.30), "Cleaned dataset\nn = 563\nmain domain n = 537", "#dbeafe", size=14, weight="bold")
    add_box(ax, (0.31, 0.38), (0.22, 0.30), "Sparse regime removed\nNeutral-low C0\nn = 26", "#fde68a", size=14, weight="bold")
    add_box(ax, (0.58, 0.38), (0.17, 0.30), "Track A\nwithin-source\nR2 up to 0.992", "#ccfbf1", size=14, weight="bold")
    add_box(ax, (0.79, 0.38), (0.17, 0.30), "Track B\nreference-held-out\nbest R2 = 0.330", "#fee2e2", size=14, weight="bold")

    add_arrow(ax, (0.26, 0.53), (0.31, 0.53), lw=2.0)
    add_arrow(ax, (0.53, 0.53), (0.58, 0.53), lw=2.0)
    add_arrow(ax, (0.75, 0.53), (0.79, 0.53), lw=2.0)

    ax.text(
        0.5,
        0.16,
        "High interpolation accuracy, applicability domain, and external generalization are different scientific claims.",
        ha="center",
        va="center",
        fontsize=17,
        color="#111827",
    )

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "graphical_abstract.png", dpi=300)
    plt.close(fig)


def dataset_composition_figure() -> None:
    df = pd.read_csv(DATA_TEMPLATE)

    regime_order = ["Acidic_HighC0", "Acidic_LowC0", "Neutral_HighC0", "Neutral_LowC0"]
    regime_counts = df["Regime"].value_counts().reindex(regime_order).fillna(0).astype(int)

    family_labels = {
        "composite_other": "Composite/other",
        "biomass": "Biomass",
        "carbon_based": "Carbon-based",
        "mineral_oxide": "Mineral/oxide",
        "polymer": "Polymer",
    }
    family_counts = df["Adsorbent_Family"].map(family_labels).value_counts()
    family_order = ["Composite/other", "Biomass", "Carbon-based", "Mineral/oxide", "Polymer"]
    family_counts = family_counts.reindex(family_order).fillna(0).astype(int)

    adsorbate_counts = df["Adsorbate_norm"].value_counts().head(8).sort_values()

    fig = plt.figure(figsize=(12.5, 8.2), facecolor="#f8fafc")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.60, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    colors_regime = ["#2563eb", "#38bdf8", "#16a34a", "#f97316"]
    bars = ax1.bar(regime_counts.index, regime_counts.values, color=colors_regime, edgecolor="#0f172a", linewidth=0.7)
    ax1.set_title("pH-concentration regimes", fontsize=13, fontweight="bold", color="#0f172a")
    ax1.set_ylabel("Records")
    ax1.tick_params(axis="x", labelrotation=18)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_ylim(0, max(regime_counts.values) + 55)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{int(bar.get_height())}", ha="center", fontsize=10)
    ax1.text(
        3,
        regime_counts.iloc[-1] + 25,
        "excluded from\nprimary domain",
        ha="center",
        fontsize=9,
        color="#9a3412",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffedd5", edgecolor="#fdba74"),
    )

    colors_family = ["#0f766e", "#65a30d", "#334155", "#7c3aed", "#db2777"]
    bars = ax2.bar(family_counts.index, family_counts.values, color=colors_family, edgecolor="#0f172a", linewidth=0.7)
    ax2.set_title("Adsorbent-family coverage", fontsize=13, fontweight="bold", color="#0f172a")
    ax2.set_ylabel("Records")
    ax2.tick_params(axis="x", labelrotation=18)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_ylim(0, max(family_counts.values) + 45)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{int(bar.get_height())}", ha="center", fontsize=10)

    ax3.barh(adsorbate_counts.index, adsorbate_counts.values, color="#1d4ed8", edgecolor="#0f172a", linewidth=0.7)
    ax3.set_title("Most represented adsorbates", fontsize=13, fontweight="bold", color="#0f172a", pad=12)
    ax3.set_xlabel("Records")
    ax3.grid(axis="x", alpha=0.25)
    for y, value in enumerate(adsorbate_counts.values):
        ax3.text(value + 2, y, f"{int(value)}", va="center", fontsize=10)

    fig.suptitle("Cleaned adsorption dataset composition (n = 563)", fontsize=17, fontweight="bold", color="#0f172a", y=0.98)
    fig.text(
        0.5,
        0.035,
        "The 537-record primary modeling domain retains the three well-covered pH-concentration regimes and excludes Neutral_LowC0 (n = 26).",
        ha="center",
        fontsize=10.5,
        color="#334155",
    )
    fig.subplots_adjust(top=0.88, bottom=0.12)
    fig.savefig(OUT / "figure_2_dataset_composition.png", dpi=300)
    plt.close(fig)


def validation_cascade_figure() -> None:
    labels = [
        "Track A\nwithin-source\nbest group",
        "Track B\nrandom row-wise\nfull capacity",
        "Track B\nadsorbent-held-out\ncapacity-free",
        "Track B\nreference-held-out\nfull capacity",
        "Track B\nreference-held-out\ncapacity-free",
    ]
    values = [0.9916, 0.8064, 0.5323, 0.3296, 0.3054]
    colors = ["#14b8a6", "#2563eb", "#f59e0b", "#ef4444", "#991b1b"]

    fig, ax = plt.subplots(figsize=(11.0, 5.8), facecolor="#f8fafc")
    ax.set_facecolor("#ffffff")
    bars = ax.bar(range(len(values)), values, color=colors, edgecolor="#0f172a", linewidth=0.8)
    ax.plot(range(len(values)), values, color="#0f172a", linewidth=1.6, marker="o", markersize=5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("$R^2$")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Validation cascade from source interpolation to cross-source transfer", fontsize=16, fontweight="bold", color="#0f172a")
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0, color="#0f172a", linewidth=0.8)

    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=11, fontweight="bold")

    ax.text(
        2.7,
        0.86,
        "Same dataset family,\ndifferent scientific claims",
        fontsize=11,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f1f5f9", edgecolor="#94a3b8"),
    )
    ax.annotate(
        "strictest test:\nentire references unseen",
        xy=(3, values[3]),
        xytext=(3.45, 0.52),
        arrowprops=dict(arrowstyle="->", color="#991b1b", lw=1.2),
        fontsize=10,
        color="#7f1d1d",
    )

    fig.text(
        0.5,
        0.02,
        "Track B values are repeated-split means in the restricted 537-record primary domain; Track A is the best eligible within-source XGBoost result.",
        ha="center",
        fontsize=10,
        color="#334155",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(OUT / "figure_5_validation_cascade.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    workflow_figure()
    graphical_abstract()
    dataset_composition_figure()
    validation_cascade_figure()
    print(f"Wrote submission assets to {OUT}")


if __name__ == "__main__":
    main()
