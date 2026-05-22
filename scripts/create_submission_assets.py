#!/usr/bin/env python3
"""Create submission-grade figures for the JECE manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parents[1] / "paper_draft" / "jece_latex_bundle"


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    workflow_figure()
    graphical_abstract()
    print(f"Wrote submission assets to {OUT}")


if __name__ == "__main__":
    main()
