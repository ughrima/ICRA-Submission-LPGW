# experiments/run_drift_experiment.py
"""
Controlled drift/noise experiment for LPGW vs DTW.

Experiment:
    For each noise_std in [0.0, 0.1, 0.2, 0.5, 1.0] meters:

        1. Load clean reference and query trajectories.
        2. Build the canonical clean reference/query segments.
        3. Build GROUND TRUTH ON THE CLEAN TRAJECTORIES ONLY.
        4. Add Gaussian positional noise to the QUERY trajectory.
        5. Rebuild the noisy query segments.
        6. Run LPGW and DTW on the noisy query.
        7. Evaluate predictions against the SAME CLEAN GT.

Important:
    The ground truth is NOT recomputed after adding noise.

This isolates the effect of trajectory noise on the detector rather
than allowing the noise to change the labels.

The same standardized random noise is scaled by sigma:
    sigma = 0.1 -> 0.1 * base_noise
    sigma = 0.2 -> 0.2 * base_noise
    ...

This gives a controlled noise-strength sweep.

Saves:
    results/drift_experiment_{DATASET_SHORT}.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

# ---------------------------------------------------------------------
# Repository path
# ---------------------------------------------------------------------

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

import config

from core.trajectory_utils import load_trajectory
from core.detection import detect_with_percentile
from evaluation.eval_canonical import score_predictions
from loop_closure import LoopClosureDetector
from lpgw_impl import segment_trajectory, downsample_trajectory

from experiments.run_baseline_comparison import compute_dtw_matrix
from experiments.run_simple_lpgw_test import load_canonical_ground_truth


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def build_segments(
    xyz,
    target_points,
    segment_length,
    fps,
    stride,
):
    """
    Downsample and segment a trajectory.
    """
    xyz_ds = downsample_trajectory(
        xyz,
        target_points,
    )

    segments = segment_trajectory(
        xyz_ds,
        segment_length=segment_length,
        fps=fps,
        stride=stride,
    )

    return segments


def compute_center_gt(
    reference_segments,
    query_segments,
    tolerance,
):
    """
    Compute GT from clean trajectory segment centers.

    Each query segment is positive if its nearest reference segment
    center is within the specified spatial tolerance.

    Returns:
        y_true
        nearest_distances
        nearest_indices
    """

    reference_centers = np.array(
        [segment.mean(axis=0) for segment in reference_segments]
    )

    query_centers = np.array(
        [segment.mean(axis=0) for segment in query_segments]
    )

    tree = KDTree(reference_centers)

    distances, indices = tree.query(
        query_centers,
        k=1,
    )

    distances = distances[:, 0]
    indices = indices[:, 0]

    y_true = (
        distances <= tolerance
    ).astype(int)

    return y_true, distances, indices


def select_baseline_segments(
    segments,
    max_segments,
):
    """
    Apply the project's baseline segment policy.

    Returns:
        selected_segments
        selected_indices
    """

    num_segments = len(segments)

    if num_segments <= max_segments:
        indices = np.arange(num_segments)
    else:
        indices = np.linspace(
            0,
            num_segments - 1,
            max_segments,
            dtype=int,
        )

    selected_segments = [
        segments[i]
        for i in indices
    ]

    return selected_segments, indices


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run_drift_experiment():

    poses_dir = Path("data/poses")

    ref_csv = poses_dir / config.BAG3_CSV
    query_csv = poses_dir / config.BAG7_CSV

    print("=" * 70)
    print("CONTROLLED DRIFT / NOISE EXPERIMENT")
    print("=" * 70)

    print(f"Dataset: {config.DATASET_SHORT}")
    print(f"Reference: {ref_csv}")
    print(f"Query:     {query_csv}")

    # -----------------------------------------------------------------
    # Load trajectories
    # -----------------------------------------------------------------

    ref_df = load_trajectory(ref_csv)
    query_df = load_trajectory(query_csv)

    ref_xyz = ref_df[
        ["PosX", "PosY", "PosZ"]
    ].to_numpy()

    query_xyz_clean = query_df[
        ["PosX", "PosY", "PosZ"]
    ].to_numpy()

    print(f"Reference raw points: {len(ref_xyz)}")
    print(f"Query raw points:     {len(query_xyz_clean)}")

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

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

    tolerance = config.SPATIAL_TOLERANCE

    noise_stds = [
        0.0,
        0.1,
        0.2,
        0.5,
        1.0,
    ]

    # -----------------------------------------------------------------
    # Create ONE standardized noise realization.
    #
    # Every sigma uses the same spatial noise pattern, only scaled.
    # -----------------------------------------------------------------

    rng = np.random.default_rng(
        seed=config.RANDOM_SEED
    )

    base_noise = rng.normal(
        loc=0.0,
        scale=1.0,
        size=query_xyz_clean.shape,
    )

    # -----------------------------------------------------------------
    # Build CLEAN reference/query segments ONCE.
    #
    # These are used to create the fixed GT.
    # -----------------------------------------------------------------

    reference_segments_clean = build_segments(
        ref_xyz,
        target_points,
        segment_length,
        fps,
        stride,
    )

    query_segments_clean = build_segments(
        query_xyz_clean,
        target_points,
        segment_length,
        fps,
        stride,
    )

    # Same canonical truncation policy.
    min_len_clean = min(
        len(reference_segments_clean),
        len(query_segments_clean),
    )

    reference_segments_clean = (
        reference_segments_clean[:min_len_clean]
    )

    query_segments_clean = (
        query_segments_clean[:min_len_clean]
    )

    print()
    print(
        f"Clean segments: "
        f"{min_len_clean}"
    )

    # -----------------------------------------------------------------
    # IMPORTANT:
    #
    # Build GT ON CLEAN TRAJECTORIES.
    # This GT stays fixed for every noise level.
    # -----------------------------------------------------------------

    y_true_clean, clean_distances, clean_nearest_indices = (
        compute_center_gt(
            reference_segments_clean,
            query_segments_clean,
            tolerance,
        )
    )

    print(
        f"Clean GT positives: "
        f"{y_true_clean.sum()} / {len(y_true_clean)}"
    )

    print(
        f"GT tolerance: "
        f"{tolerance:.2f} m"
    )

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    rows = []

    # -----------------------------------------------------------------
    # Noise sweep
    # -----------------------------------------------------------------

    for sigma in noise_stds:

        print()
        print("=" * 70)
        print(
            f"NOISE STD = {sigma:.2f} m"
        )
        print("=" * 70)

        # -------------------------------------------------------------
        # Add noise ONLY to query input.
        # -------------------------------------------------------------

        query_xyz_noisy = (
            query_xyz_clean
            + sigma * base_noise
        )

        # -------------------------------------------------------------
        # Rebuild noisy query segments.
        # -------------------------------------------------------------

        query_segments_noisy = build_segments(
            query_xyz_noisy,
            target_points,
            segment_length,
            fps,
            stride,
        )

        # Keep exactly the same number of query segments as GT.
        query_segments_noisy = (
            query_segments_noisy[:min_len_clean]
        )

        if len(query_segments_noisy) != len(y_true_clean):
            raise RuntimeError(
                "Noisy query segment count does not match "
                "clean GT segment count."
            )

        print(
            f"Noisy query segments: "
            f"{len(query_segments_noisy)}"
        )

        # -------------------------------------------------------------
        # LPGW
        # -------------------------------------------------------------

        detector = LoopClosureDetector(
            segment_length=segment_length,
            fps=fps,
            stride=stride,
            lambdaa=config.LPGW_LAMBDA,
            downsample_points=100,
            reference_strategy="robust",
        )

        D_lpgw = detector.compute_distance_matrix(
            query_segments_noisy,
            reference_segments_clean,
        )

        y_pred_lpgw, tau_lpgw = (
            detect_with_percentile(
                D_lpgw,
                config.PERCENTILE,
            )
        )

        metrics_lpgw = score_predictions(
            y_true_clean,
            y_pred_lpgw,
        )

        rows.append(
            {
                "noise_std_m": sigma,
                "method": "LPGW",
                "precision": metrics_lpgw["precision"],
                "recall": metrics_lpgw["recall"],
                "f1": metrics_lpgw["f1"],
                "num_segments": min_len_clean,
                "num_gt_positives": int(y_true_clean.sum()),
                "num_predicted_positives": int(
                    y_pred_lpgw.sum()
                ),
                "threshold_tau": tau_lpgw,
            }
        )

        print(
            f"LPGW: "
            f"P={metrics_lpgw['precision']:.3f}, "
            f"R={metrics_lpgw['recall']:.3f}, "
            f"F1={metrics_lpgw['f1']:.3f}"
        )

        # -------------------------------------------------------------
        # DTW baseline
        # -------------------------------------------------------------

        if config.BASELINE_SEGMENT_POLICY == "subsample_fixed":

            max_baseline_segments = getattr(
                config,
                "MAX_BASELINE_SEGMENTS",
                150,
            )

            if min_len_clean > max_baseline_segments:

                baseline_indices = np.linspace(
                    0,
                    min_len_clean - 1,
                    max_baseline_segments,
                    dtype=int,
                )

            else:

                baseline_indices = np.arange(
                    min_len_clean
                )

        else:

            baseline_indices = np.arange(
                min_len_clean
            )

        seg7_sub = [
            query_segments_noisy[i]
            for i in baseline_indices
        ]

        seg3_sub = [
            reference_segments_clean[i]
            for i in baseline_indices
        ]

        # -------------------------------------------------------------
        # IMPORTANT:
        #
        # Use the CLEAN canonical GT labels corresponding to the
        # selected original query indices.
        #
        # Do NOT recompute GT using noisy query geometry.
        # -------------------------------------------------------------

        y_true_dtw = y_true_clean[
            baseline_indices
        ]

        D_dtw = compute_dtw_matrix(
            seg7_sub,
            seg3_sub,
        )

        y_pred_dtw, tau_dtw = (
            detect_with_percentile(
                D_dtw,
                config.PERCENTILE,
            )
        )

        metrics_dtw = score_predictions(
            y_true_dtw,
            y_pred_dtw,
        )

        rows.append(
            {
                "noise_std_m": sigma,
                "method": "DTW",
                "precision": metrics_dtw["precision"],
                "recall": metrics_dtw["recall"],
                "f1": metrics_dtw["f1"],
                "num_segments": len(baseline_indices),
                "num_gt_positives": int(
                    y_true_dtw.sum()
                ),
                "num_predicted_positives": int(
                    y_pred_dtw.sum()
                ),
                "threshold_tau": tau_dtw,
            }
        )

        print(
            f"DTW:  "
            f"P={metrics_dtw['precision']:.3f}, "
            f"R={metrics_dtw['recall']:.3f}, "
            f"F1={metrics_dtw['f1']:.3f}"
        )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(rows)

    out_dir = Path("results")
    out_dir.mkdir(
        exist_ok=True
    )

    out_path = (
        out_dir
        / f"drift_experiment_{config.DATASET_SHORT}.csv"
    )

    results_df.to_csv(
        out_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Saved to: {out_path}"
    )

    print()
    print(
        results_df.to_string(
            index=False
        )
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_drift_experiment()