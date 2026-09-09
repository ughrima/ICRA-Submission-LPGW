
# experiments/run_simple_lpgw_test.py

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Project paths
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

from core.trajectory_utils import (
    load_trajectory,
    segment_trajectory,
    downsample_trajectory,
)

from loop_closure import LoopClosureDetector
repo_root = Path(__file__).resolve().parent.parent
poses_dir = repo_root / config.POSES_DIR

# ------------------------------------------------------------------
# Canonical ground truth
# ------------------------------------------------------------------

def load_canonical_ground_truth(tolerance_m: float):
    gt_dir = PROJECT_ROOT / "ground_truth" / "files"

    gt_csv = (
        gt_dir
        / f"gt_{config.DATASET_SHORT}_{tolerance_m}m.csv"
    )

    meta_json = (
        gt_dir
        / f"gt_{config.DATASET_SHORT}_{tolerance_m}m_meta.json"
    )

    if not gt_csv.exists() or not meta_json.exists():
        raise FileNotFoundError(
            f"Canonical ground truth for "
            f"{config.DATASET_SHORT} at {tolerance_m} m not found. "
            "Run ground_truth/generate_ground_truth.py first."
        )

    gt_df = pd.read_csv(gt_csv)

    with open(meta_json) as f:
        meta = json.load(f)

    y_true = gt_df["label"].to_numpy()

    ref_centers = np.array(meta["ref_centers"])
    query_centers = np.array(meta["query_centers"])

    return (
        y_true,
        ref_centers,
        query_centers,
        meta,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    tolerance = config.SPATIAL_TOLERANCE

    # --------------------------------------------------------------
    # Load canonical ground truth
    # --------------------------------------------------------------

    (
        y_true,
        ref_centers,
        query_centers,
        meta,
    ) = load_canonical_ground_truth(tolerance)

    print(
        f"Loaded canonical GT at "
        f"{tolerance:.1f} m"
    )

    print(
        f"  Query segments: {len(y_true)}"
    )

    print(
        f"  Positive: {y_true.sum()}"
    )

    # --------------------------------------------------------------
    # Load trajectories
    # --------------------------------------------------------------

    poses_dir = PROJECT_ROOT / "data" / "poses"

    ref_csv = poses_dir / config.BAG3_CSV
    query_csv = poses_dir / config.BAG7_CSV

    ref_df = load_trajectory(ref_csv)
    query_df = load_trajectory(query_csv)

    ref_xyz = ref_df[
        ["PosX", "PosY", "PosZ"]
    ].to_numpy()

    query_xyz = query_df[
        ["PosX", "PosY", "PosZ"]
    ].to_numpy()

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------

    target_points = getattr(
        config,
        "TARGET_POINTS",
        5000,
    )

    segment_length = getattr(
        config,
        "SEGMENT_LENGTH",
        5.0,
    )

    fps = getattr(
        config,
        "FPS",
        10,
    )

    stride = getattr(
        config,
        "STRIDE",
        1.0,
    )

    # --------------------------------------------------------------
    # Downsample trajectories
    # --------------------------------------------------------------

    ref_xyz_ds = downsample_trajectory(
        ref_xyz,
        target_points,
    )

    query_xyz_ds = downsample_trajectory(
        query_xyz,
        target_points,
    )

    # --------------------------------------------------------------
    # Segment trajectories
    # --------------------------------------------------------------

    segments_3 = segment_trajectory(
        ref_xyz_ds,
        segment_length=segment_length,
        fps=fps,
        stride=stride,
    )

    segments_7 = segment_trajectory(
        query_xyz_ds,
        segment_length=segment_length,
        fps=fps,
        stride=stride,
    )

    # Keep the same number of segments
    # as the canonical GT generation.

    min_len = min(
        len(segments_7),
        len(segments_3),
    )

    segments_7 = segments_7[:min_len]
    segments_3 = segments_3[:min_len]

    # --------------------------------------------------------------
    # Verify alignment with canonical GT
    # --------------------------------------------------------------

    recomputed_query_centers = np.array(
        [
            seg.mean(axis=0)
            for seg in segments_7
        ]
    )

    recomputed_ref_centers = np.array(
        [
            seg.mean(axis=0)
            for seg in segments_3
        ]
    )

    if (
        recomputed_query_centers.shape
        != query_centers.shape
    ):
        raise RuntimeError(
            "Query segment centers mismatch "
            "with canonical GT. "
            "Check segmentation parameters."
        )

    if (
        recomputed_ref_centers.shape
        != ref_centers.shape
    ):
        raise RuntimeError(
            "Reference segment centers mismatch "
            "with canonical GT. "
            "Check segmentation parameters."
        )

    if not np.allclose(
        recomputed_query_centers,
        query_centers,
    ):
        raise RuntimeError(
            "Query segment centers differ "
            "from canonical GT. "
            "Segmentation may have changed."
        )

    if not np.allclose(
        recomputed_ref_centers,
        ref_centers,
    ):
        raise RuntimeError(
            "Reference segment centers differ "
            "from canonical GT. "
            "Segmentation may have changed."
        )

    print(
        "Segment centers verified against "
        "canonical GT — alignment OK."
    )

    # --------------------------------------------------------------
    # Run LPGW
    # --------------------------------------------------------------

    detector = LoopClosureDetector(
        segment_length=segment_length,
        fps=fps,
        stride=stride,
        lambdaa=getattr(
            config,
            "LPGW_LAMBDA",
            0.5,
        ),
        downsample_points=100,
        reference_strategy="robust",
    )

    print(
        "Computing LPGW distance matrix..."
    )

    D = detector.compute_distance_matrix(
        segments_7,
        segments_3,
    )

    print(
        f"Distance matrix shape: {D.shape}"
    )

    # --------------------------------------------------------------
    # Detect loop closures at configured percentile
    # --------------------------------------------------------------

    tau = detector.choose_threshold(
        D,
        method="percentile",
        percentile=config.PERCENTILE,
    )

    flags, matches, min_distances = (
        detector.detect_loop_closures(
            D,
            threshold=tau,
        )
    )

    y_pred = np.asarray(
        flags,
        dtype=int,
    )

    # --------------------------------------------------------------
    # Score
    # --------------------------------------------------------------

    metrics = detector.evaluate(
        y_true,
        y_pred,
    )

    print(
        "Metrics "
        f"(LPGW, {config.PERCENTILE}th percentile, "
        f"{tolerance:.1f} m GT):"
    )

    print(
        f"  TP={metrics['true_positives']}, "
        f"FP={metrics['false_positives']}, "
        f"FN={metrics['false_negatives']}"
    )

    print(
        f"  Precision={metrics['precision']:.3f}"
    )

    print(
        f"  Recall={metrics['recall']:.3f}"
    )

    print(
        f"  F1={metrics['f1_score']:.3f}"
    )

    print(
        f"  Threshold τ={tau:.6f}"
    )


if __name__ == "__main__":
    main()
