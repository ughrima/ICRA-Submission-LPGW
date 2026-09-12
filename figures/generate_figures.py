"""
Generate publication-ready figures for LPGW experiments.

Dataset handling
----------------
Each dataset has its own result files:

    UZH:
        *_uzhfpv.csv

    KITTI:
        *_kitti00.csv

Figures are also dataset-specific:

    figures/uzhfpv/
    figures/kitti00/

This prevents UZH and KITTI results from ever being mixed.

The script ONLY reads files from results/ and writes figures.
It does not modify experiment result files.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------

DATASETS = {
    "uzhfpv": {
        "label": "UZH FPV",
        "suffix": "uzhfpv",
    },
    "kitti00": {
        "label": "KITTI 00",
        "suffix": "kitti00",
    },
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def result_path(stem, suffix):
    """
    Return a dataset-specific result path.

    Example:
        full_lpgw_uzhfpv.csv
        full_lpgw_kitti00.csv
    """
    return RESULTS_DIR / f"{stem}_{suffix}.csv"


def figure_dir(suffix):
    """
    Return dataset-specific figure directory.
    """
    path = FIGURES_DIR / suffix
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_result(stem, suffix):
    """
    Load a dataset-specific result file.
    """
    path = result_path(stem, suffix)

    if not path.exists():
        print(f"Skipping {path.name}: file not found")
        return None

    print(f"Reading: {path.name}")
    return pd.read_csv(path)


def save_figure(fig, suffix, filename):
    """
    Save figure inside the dataset-specific figure directory.
    """
    output_dir = figure_dir(suffix)
    output_path = output_dir / filename

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------
# Figure 1
# LPGW performance across percentile thresholds
# ---------------------------------------------------------------------

def figure_percentile_sweep(suffix, dataset_label):

    df = read_result("percentile_sweep", suffix)

    if df is None:
        return

    required = {
        "percentile",
        "precision",
        "recall",
        "f1",
    }

    missing = required - set(df.columns)

    if missing:
        print(
            f"Skipping percentile figure for {suffix}: "
            f"missing columns {sorted(missing)}"
        )
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        df["percentile"],
        df["precision"],
        marker="o",
        label="Precision",
    )

    ax.plot(
        df["percentile"],
        df["recall"],
        marker="o",
        label="Recall",
    )

    ax.plot(
        df["percentile"],
        df["f1"],
        marker="o",
        label="F1",
    )

    ax.set_xlabel("Percentile threshold (%)")
    ax.set_ylabel("Score")
    ax.set_title(
        f"LPGW Performance Across Percentile Thresholds — {dataset_label}"
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(
        fig,
        suffix,
        "figure_percentile_sweep.png",
    )


# ---------------------------------------------------------------------
# Figure 2
# Ground-truth tolerance sweep
# ---------------------------------------------------------------------

def figure_tolerance_sweep(suffix, dataset_label):

    df = read_result("tolerance_sweep", suffix)

    if df is None:
        return

    required = {
        "tolerance_m",
        "precision",
        "recall",
        "f1",
    }

    missing = required - set(df.columns)

    if missing:
        print(
            f"Skipping tolerance figure for {suffix}: "
            f"missing columns {sorted(missing)}"
        )
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        df["tolerance_m"],
        df["precision"],
        marker="o",
        label="Precision",
    )

    ax.plot(
        df["tolerance_m"],
        df["recall"],
        marker="o",
        label="Recall",
    )

    ax.plot(
        df["tolerance_m"],
        df["f1"],
        marker="o",
        label="F1",
    )

    ax.set_xlabel("Ground-truth tolerance (m)")
    ax.set_ylabel("Score")
    ax.set_title(
        f"LPGW Performance Across Ground-Truth Tolerance — {dataset_label}"
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(
        fig,
        suffix,
        "figure_tolerance_sweep.png",
    )


# ---------------------------------------------------------------------
# Figure 3
# Baseline comparison
# ---------------------------------------------------------------------

def figure_baseline_comparison(suffix, dataset_label):

    df = read_result("baseline_accuracy", suffix)

    if df is None:
        return

    required = {
        "method",
        "percentile",
        "f1",
    }

    missing = required - set(df.columns)

    if missing:
        print(
            f"Skipping baseline figure for {suffix}: "
            f"missing columns {sorted(missing)}"
        )
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in df["method"].unique():

        subset = df[df["method"] == method]

        ax.plot(
            subset["percentile"],
            subset["f1"],
            marker="o",
            label=str(method),
        )

    ax.set_xlabel("Percentile threshold (%)")
    ax.set_ylabel("F1 score")
    ax.set_title(
        f"Loop-Closure Detection Performance — {dataset_label}"
    )
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(
        fig,
        suffix,
        "figure_baseline_comparison.png",
    )


# ---------------------------------------------------------------------
# Figure 4
# Reference segment ablation
# ---------------------------------------------------------------------

def figure_reference_ablation(suffix, dataset_label):

    df = read_result("reference_segment_ablation", suffix)

    if df is None:
        return

    if "f1" not in df.columns:
        print(
            f"Skipping reference ablation for {suffix}: "
            "'f1' column missing"
        )
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    if "reference_index" in df.columns:

        ax.plot(
            df["reference_index"],
            df["f1"],
            marker="o",
        )

        ax.set_xlabel("Reference segment index")

    elif "reference_position" in df.columns:

        ax.plot(
            df["reference_position"],
            df["f1"],
            marker="o",
        )

        ax.set_xlabel("Reference position")

    else:

        ax.plot(
            range(len(df)),
            df["f1"],
            marker="o",
        )

        ax.set_xlabel("Reference-segment experiment")

    ax.set_ylabel("F1 score")

    ax.set_title(
        f"Reference Segment Ablation — {dataset_label}"
    )

    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    save_figure(
        fig,
        suffix,
        "figure_reference_segment_ablation.png",
    )


# ---------------------------------------------------------------------
# Figure 5
# Runtime scaling
# ---------------------------------------------------------------------

def figure_runtime_scaling(suffix, dataset_label):

    df = read_result("runtime_scaling", suffix)

    if df is None:
        return

    x_column = None

    for candidate in [
        "num_segments",
        "n_segments",
        "segments",
        "trajectory_length",
    ]:
        if candidate in df.columns:
            x_column = candidate
            break

    if x_column is None:
        print(
            f"Skipping runtime figure for {suffix}: "
            "segment-count column not found"
        )
        return

    runtime_columns = [
        column
        for column in df.columns
        if (
            "runtime" in column.lower()
            or "time" in column.lower()
        )
    ]

    if not runtime_columns:
        print(
            f"Skipping runtime figure for {suffix}: "
            "runtime column not found"
        )
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for column in runtime_columns:

        ax.plot(
            df[x_column],
            df[column],
            marker="o",
            label=column,
        )

    ax.set_xlabel("Number of segments")
    ax.set_ylabel("Runtime (s)")

    ax.set_title(
        f"Runtime Scaling — {dataset_label}"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(
        fig,
        suffix,
        "figure_runtime_scaling.png",
    )


# ---------------------------------------------------------------------
# Figure 6
# Partial-overlap robustness
# ---------------------------------------------------------------------

def figure_overlap(suffix, dataset_label):

    df = read_result("overlap_experiment", suffix)

    if df is None:
        return

    required = {
        "overlap_ratio",
        "method",
        "f1",
    }

    missing = required - set(df.columns)

    if missing:
        print(
            f"Skipping overlap figure for {suffix}: "
            f"missing columns {sorted(missing)}"
        )
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for method in df["method"].unique():

        subset = df[df["method"] == method]

        ax.plot(
            subset["overlap_ratio"],
            subset["f1"],
            marker="o",
            label=str(method),
        )

    ax.set_xlabel("Overlap ratio")
    ax.set_ylabel("F1 score")

    ax.set_title(
        f"Robustness to Partial Observation — {dataset_label}"
    )

    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(
        fig,
        suffix,
        "figure_overlap_experiment.png",
    )


# ---------------------------------------------------------------------
# Figure 7
# Drift robustness
# ---------------------------------------------------------------------

def figure_drift(suffix, dataset_label):

    df = read_result("drift_experiment", suffix)

    if df is None:
        return

    if "f1" not in df.columns:
        print(
            f"Skipping drift figure for {suffix}: "
            "'f1' column missing"
        )
        return

    x_column = None

    for candidate in [
        "drift",
        "drift_m",
        "drift_percent",
        "drift_level",
    ]:
        if candidate in df.columns:
            x_column = candidate
            break

    if x_column is None:
        print(
            f"Skipping drift figure for {suffix}: "
            "drift column not found"
        )
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    if "method" in df.columns:

        for method in df["method"].unique():

            subset = df[df["method"] == method]

            ax.plot(
                subset[x_column],
                subset["f1"],
                marker="o",
                label=str(method),
            )

        ax.legend()

    else:

        ax.plot(
            df[x_column],
            df["f1"],
            marker="o",
        )

    ax.set_xlabel("Drift")
    ax.set_ylabel("F1 score")

    ax.set_title(
        f"Effect of Drift on Loop-Closure Detection — {dataset_label}"
    )

    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    save_figure(
        fig,
        suffix,
        "figure_drift_experiment.png",
    )


# ---------------------------------------------------------------------
# Figure 8
# Full LPGW result
# ---------------------------------------------------------------------

def figure_full_lpgw(suffix, dataset_label):

    df = read_result("full_lpgw", suffix)

    if df is None:
        return

    required = {
        "percentile",
        "precision",
        "recall",
        "f1",
    }

    missing = required - set(df.columns)

    if missing:
        print(
            f"Skipping full LPGW figure for {suffix}: "
            f"missing columns {sorted(missing)}"
        )
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        df["percentile"],
        df["precision"],
        marker="o",
        label="Precision",
    )

    ax.plot(
        df["percentile"],
        df["recall"],
        marker="o",
        label="Recall",
    )

    ax.plot(
        df["percentile"],
        df["f1"],
        marker="o",
        label="F1",
    )

    ax.set_xlabel("Percentile threshold (%)")
    ax.set_ylabel("Score")

    ax.set_title(
        f"Canonical LPGW Performance — {dataset_label}"
    )

    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(
        fig,
        suffix,
        "figure_full_lpgw.png",
    )


# ---------------------------------------------------------------------
# Generate all figures for ONE dataset
# ---------------------------------------------------------------------

def generate_dataset_figures(suffix, dataset_label):

    print()
    print("=" * 70)
    print(f"GENERATING FIGURES: {dataset_label}")
    print("=" * 70)

    print(f"Results: {RESULTS_DIR}")
    print(f"Figures: {FIGURES_DIR / suffix}")
    print()

    figure_full_lpgw(
        suffix,
        dataset_label,
    )

    figure_percentile_sweep(
        suffix,
        dataset_label,
    )

    figure_tolerance_sweep(
        suffix,
        dataset_label,
    )

    figure_baseline_comparison(
        suffix,
        dataset_label,
    )

    figure_reference_ablation(
        suffix,
        dataset_label,
    )

    figure_runtime_scaling(
        suffix,
        dataset_label,
    )

    figure_overlap(
        suffix,
        dataset_label,
    )

    figure_drift(
        suffix,
        dataset_label,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("LPGW PUBLICATION FIGURE GENERATOR")
    print("=" * 70)

    print()
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")
    print()

    # ---------------------------------------------------------------
    # UZH
    # ---------------------------------------------------------------

    generate_dataset_figures(
        "uzhfpv",
        "UZH FPV",
    )

    # ---------------------------------------------------------------
    # KITTI
    # ---------------------------------------------------------------

    generate_dataset_figures(
        "kitti00",
        "KITTI 00",
    )

    print()
    print("=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)

    print()
    print("Figures are stored separately:")
    print(f"  {FIGURES_DIR / 'uzhfpv'}")
    print(f"  {FIGURES_DIR / 'kitti00'}")


if __name__ == "__main__":
    main()