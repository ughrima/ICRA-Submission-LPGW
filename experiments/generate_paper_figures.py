"""Generate manuscript Figures 2 through 5 from trajectories and saved results.

Run from the repository root with:

    python3 experiments/generate_paper_figures.py

Outputs are written to ``results/figures`` as 300 dpi PNG and vector PDF files.
Figures 3 and 5 are generated when their result CSVs are available; Figures 2
and 4 are generated from the active dataset pose files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import config
from core.trajectory_utils import (
    downsample_trajectory,
    load_trajectory,
    segment_trajectory,
)
from loop_closure import LoopClosureDetector


RESULTS_DIR = repo_root / "results"
FIGURE_DIR = RESULTS_DIR / "figures"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})


def save_figure(fig, stem: str) -> None:
    """Save a figure as high-resolution PNG and vector PDF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}.png and {stem}.pdf")


def load_active_trajectories() -> tuple[np.ndarray, np.ndarray]:
    poses_dir = repo_root / config.POSES_DIR
    ref_df = load_trajectory(poses_dir / config.BAG3_CSV)
    query_df = load_trajectory(poses_dir / config.BAG7_CSV)

    ref_xyz = ref_df[["PosX", "PosY", "PosZ"]].to_numpy(dtype=float)
    query_xyz = query_df[["PosX", "PosY", "PosZ"]].to_numpy(dtype=float)
    return ref_xyz, query_xyz


def generate_figure_2(ref_xyz: np.ndarray, query_xyz: np.ndarray) -> None:
    print("Generating Figure 2: trajectory overview")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        ref_xyz[:, 0],
        ref_xyz[:, 1],
        label=config.DATASET_LABEL_REF,
        color="#1769aa",
        linewidth=1.2,
        alpha=0.85,
    )
    ax.plot(
        query_xyz[:, 0],
        query_xyz[:, 1],
        label=config.DATASET_LABEL_QUERY,
        color="#e07a1f",
        linewidth=1.2,
        alpha=0.85,
    )
    ax.set_title("Top-Down Trajectory Overview")
    ax.set_xlabel("X position [m]")
    ax.set_ylabel("Y position [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    save_figure(fig, "fig2_trajectory_overview")


def generate_figure_4(ref_xyz: np.ndarray, query_xyz: np.ndarray) -> None:
    print("Generating Figure 4: LPGW distance matrix")
    ref_ds = downsample_trajectory(ref_xyz, config.TARGET_POINTS)
    query_ds = downsample_trajectory(query_xyz, config.TARGET_POINTS)
    ref_segments = segment_trajectory(
        ref_ds,
        segment_length=config.SEGMENT_LENGTH,
        fps=config.FPS,
        stride=config.STRIDE,
    )
    query_segments = segment_trajectory(
        query_ds,
        segment_length=config.SEGMENT_LENGTH,
        fps=config.FPS,
        stride=config.STRIDE,
    )
    min_len = min(len(ref_segments), len(query_segments))

    detector = LoopClosureDetector(
        segment_length=config.SEGMENT_LENGTH,
        fps=config.FPS,
        stride=config.STRIDE,
        lambdaa=config.LPGW_LAMBDA,
        downsample_points=config.TARGET_POINTS,
        reference_strategy=config.REFERENCE_STRATEGY,
        huber_delta=config.LPGW_HUBER_DELTA,
    )
    matrix = detector.compute_distance_matrix(
        query_segments[:min_len],
        ref_segments[:min_len],
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(
        matrix,
        cmap="viridis",
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, label="LPGW discrepancy")
    ax.set_title("LPGW Pairwise Discrepancy Matrix")
    ax.set_xlabel("Reference segment index")
    ax.set_ylabel("Query segment index")
    save_figure(fig, "fig4_distance_matrix_heatmap")


def generate_figure_3() -> None:
    path = RESULTS_DIR / f"runtime_scaling_{config.DATASET_SHORT}.csv"
    if not path.exists():
        print(f"Skipping Figure 3: missing {path}")
        return

    print("Generating Figure 3: runtime scaling")
    data = pd.read_csv(path)
    runtime_column = "runtime_seconds" if "runtime_seconds" in data else "runtime_sec"
    query_column = "num_query_segments"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        data[query_column],
        data[runtime_column],
        marker="o",
        color="#6a3d9a",
        linewidth=1.8,
    )
    ax.set_yscale("log")
    ax.set_title("LPGW Runtime Scaling")
    ax.set_xlabel("Number of query segments")
    ax.set_ylabel("Runtime [s, log scale]")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    save_figure(fig, "fig3_runtime_scaling")


def _plot_robustness_panel(ax, path: Path, x_column: str, xlabel: str) -> bool:
    if not path.exists():
        return False
    data = pd.read_csv(path)
    if not {x_column, "method", "f1"}.issubset(data.columns):
        return False

    for method, group in data.groupby("method"):
        group = group.sort_values(x_column)
        ax.plot(
            group[x_column],
            group["f1"],
            marker="o",
            linewidth=1.6,
            label=method,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    return True


def generate_figure_5() -> None:
    print("Generating Figure 5: robustness")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    plotted = 0

    overlap_path = RESULTS_DIR / f"overlap_experiment_{config.DATASET_SHORT}.csv"
    if _plot_robustness_panel(
        axes[0],
        overlap_path,
        "overlap_ratio",
        "Contiguous observed fraction",
    ):
        axes[0].set_title("Partial-overlap robustness")
        axes[0].set_xlim(0.35, 1.05)
        axes[0].legend(frameon=False)
        plotted += 1
    else:
        axes[0].set_visible(False)
        print(f"Skipping overlap panel: missing or invalid {overlap_path}")

    drift_path = RESULTS_DIR / f"drift_experiment_{config.DATASET_SHORT}.csv"
    drift_column = "drift_std_per_step_m"
    if _plot_robustness_panel(
        axes[1],
        drift_path,
        drift_column,
        "Drift scale per step [m]",
    ):
        axes[1].set_title("Integrated-drift robustness")
        axes[1].legend(frameon=False)
        plotted += 1
    else:
        axes[1].set_visible(False)
        print(f"Skipping drift panel: missing or invalid {drift_path}")

    if plotted:
        save_figure(fig, "fig5_robustness")
    else:
        plt.close(fig)
        print("Skipping Figure 5: no robustness CSVs available")


def generate_all_figures() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ref_xyz, query_xyz = load_active_trajectories()
    generate_figure_2(ref_xyz, query_xyz)
    generate_figure_3()
    generate_figure_4(ref_xyz, query_xyz)
    generate_figure_5()
    print(f"All available figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    generate_all_figures()
