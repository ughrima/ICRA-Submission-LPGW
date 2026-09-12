
"""
Table III: performance across percentile thresholds.

The LPGW distance matrix is computed once.

Canonical ground truth is loaded from:
    ground_truth/files/gt_<dataset>_2.0m.csv

Only the detection percentile changes across the sweep.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import config

from core.trajectory_utils import (
    load_trajectory,
    segment_trajectory,
    downsample_trajectory,
)
from core.detection import detect_with_percentile
from evaluation.eval_canonical import score_predictions
from loop_closure import LoopClosureDetector


def main():
    # -----------------------------------------------------------------
    # Paths
    # -----------------------------------------------------------------

    poses_dir = repo_root / config.POSES_DIR

    ref_csv = poses_dir / config.BAG3_CSV
    query_csv = poses_dir / config.BAG7_CSV

    # -----------------------------------------------------------------
    # Load trajectories
    # -----------------------------------------------------------------

    ref_df = load_trajectory(ref_csv)
    query_df = load_trajectory(query_csv)

    ref_xyz = ref_df[["PosX", "PosY", "PosZ"]].to_numpy()
    query_xyz = query_df[["PosX", "PosY", "PosZ"]].to_numpy()

    # -----------------------------------------------------------------
    # Canonical preprocessing
    # -----------------------------------------------------------------

    target_points = config.TARGET_POINTS
    segment_length = config.SEGMENT_LENGTH
    fps = config.FPS
    stride = config.STRIDE

    ref_xyz_ds = downsample_trajectory(
        ref_xyz,
        target_points,
    )

    query_xyz_ds = downsample_trajectory(
        query_xyz,
        target_points,
    )

    segments_ref = segment_trajectory(
        ref_xyz_ds,
        segment_length=segment_length,
        fps=fps,
        stride=stride,
    )

    segments_query = segment_trajectory(
        query_xyz_ds,
        segment_length=segment_length,
        fps=fps,
        stride=stride,
    )

    # Match the same segment subset used by the canonical GT generator.
    min_len = min(
        len(segments_ref),
        len(segments_query),
    )

    segments_ref = segments_ref[:min_len]
    segments_query = segments_query[:min_len]

    print(f"Using {min_len} matched segments for LPGW")

    # -----------------------------------------------------------------
    # Load canonical ground truth
    # -----------------------------------------------------------------

    gt_path = (
        repo_root
        / "ground_truth"
        / "files"
        / f"gt_{config.DATASET_SHORT}_2.0m.csv"
    )

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Canonical ground-truth file not found:\n{gt_path}\n\n"
            "Generate it first with:\n"
            "python ground_truth/generate_ground_truth.py"
        )

    gt_df = pd.read_csv(gt_path)

    required_columns = {
        "query_index",
        "nearest_ref_index",
        "nearest_distance_m",
        "label",
    }

    missing = required_columns - set(gt_df.columns)

    if missing:
        raise ValueError(
            "Ground-truth file is missing columns: "
            f"{sorted(missing)}"
        )

    y_true = gt_df["label"].to_numpy(dtype=int)

    if len(y_true) != len(segments_query):
        raise ValueError(
            "GT/query segment mismatch: "
            f"{len(y_true)} GT labels vs "
            f"{len(segments_query)} query segments."
        )

    print(
        f"Canonical GT positives: "
        f"{y_true.sum()} / {len(y_true)}"
    )

    # -----------------------------------------------------------------
    # Compute LPGW distance matrix ONCE
    # -----------------------------------------------------------------

    detector = LoopClosureDetector(
        segment_length=segment_length,
        fps=fps,
        stride=stride,
        lambdaa=config.LPGW_LAMBDA,
        downsample_points=100,
        reference_strategy="robust",
    )

    print("Computing LPGW distance matrix...")

    D = detector.compute_distance_matrix(
        segments_query,
        segments_ref,
    )

    print(f"Distance matrix shape: {D.shape}")

    # -----------------------------------------------------------------
    # Sweep percentiles
    # -----------------------------------------------------------------

    percentiles = config.PERCENTILE_SWEEP

    rows = []

    for percentile in percentiles:
        y_pred, threshold = detect_with_percentile(
            D,
            percentile,
        )

        metrics = score_predictions(
            y_true,
            y_pred,
        )

        row = {
            "percentile": percentile,
            "threshold_tau": threshold,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }

        rows.append(row)

        print(
            f"p={percentile:2d}% → "
            f"tau={threshold:.6f}, "
            f"F1={metrics['f1']:.3f}, "
            f"P={metrics['precision']:.3f}, "
            f"R={metrics['recall']:.3f}"
        )

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(rows)

    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = (
    out_dir
    / f"percentile_sweep_{config.DATASET_SHORT}.csv"
    )

    results_df.to_csv(
        out_path,
        index=False,
    )

    print(f"Percentile sweep saved to {out_path}")


if __name__ == "__main__":
    main()