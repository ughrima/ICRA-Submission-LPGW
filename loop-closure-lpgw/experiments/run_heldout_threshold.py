'''
Held-out threshold selection.

Select the LPGW detection threshold on one query fold and evaluate it
on the other fold.

This avoids choosing the threshold using the same data being evaluated.

Procedure:
    1. Load canonical ground truth.
    2. Compute the full LPGW distance matrix once.
    3. Split query segments into folds A and B.
    4. Select the best percentile on A and evaluate on B.
    5. Select the best percentile on B and evaluate on A.
    6. Report the average held-out F1.
    7. Also report the original non-held-out percentile result for comparison.

Results are saved to:
    results/heldout_threshold_{DATASET_SHORT}.csv

'''

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Make repository root importable
# ---------------------------------------------------------------------

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import config

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

from core.trajectory_utils import (
    load_trajectory,
    segment_trajectory,
    downsample_trajectory,
)

from evaluation.eval_canonical import score_predictions

from loop_closure import LoopClosureDetector


# Candidate percentile values used for threshold selection.
CANDIDATE_PERCENTILES = [0.5, 1, 2, 5, 10, 20, 50]


# ---------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------

def load_canonical_ground_truth(tolerance_m: float):
    """
    Load canonical ground truth generated for the active dataset.
    """

    gt_dir = repo_root / "ground_truth" / "files"

    gt_csv = gt_dir / (
        f"gt_{config.DATASET_SHORT}_{tolerance_m}m.csv"
    )

    meta_json = gt_dir / (
        f"gt_{config.DATASET_SHORT}_{tolerance_m}m_meta.json"
    )

    if not gt_csv.exists() or not meta_json.exists():
        raise FileNotFoundError(
            f"Canonical ground truth for "
            f"{config.DATASET_SHORT} at {tolerance_m} m not found. "
            f"Run ground_truth/generate_ground_truth.py first."
        )

    gt_df = pd.read_csv(gt_csv)

    with open(meta_json, "r") as f:
        meta = json.load(f)

    y_true = gt_df["label"].to_numpy(dtype=int)

    ref_centers = np.asarray(meta["ref_centers"], dtype=float)
    query_centers = np.asarray(meta["query_centers"], dtype=float)

    return y_true, ref_centers, query_centers


# ---------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------

def best_percentile_for_fold(
    D_fold: np.ndarray,
    y_true_fold: np.ndarray,
):
    """
    Sweep candidate percentiles on a training fold.

    Returns:
        best_percentile
        metrics at the selected percentile
    """

    best_p = None
    best_f1 = -1.0
    best_metrics = None

    for percentile in CANDIDATE_PERCENTILES:

        tau = float(np.percentile(D_fold, percentile))

        # Query segment is predicted as a loop closure if its
        # closest reference segment is within the threshold.
        y_pred = (D_fold.min(axis=1) <= tau).astype(int)

        metrics = score_predictions(
            y_true_fold,
            y_pred,
        )

        if metrics["f1"] > best_f1:
            best_p = percentile
            best_f1 = metrics["f1"]
            best_metrics = metrics

    return best_p, best_metrics


def apply_percentile_to_fold(
    D_train_for_threshold: np.ndarray,
    percentile: float,
    D_eval_fold: np.ndarray,
    y_true_eval_fold: np.ndarray,
):
    """
    Compute the threshold from the training fold and apply it
    to the held-out evaluation fold.
    """

    tau = float(
        np.percentile(
            D_train_for_threshold,
            percentile,
        )
    )

    y_pred = (
        D_eval_fold.min(axis=1) <= tau
    ).astype(int)

    metrics = score_predictions(
        y_true_eval_fold,
        y_pred,
    )

    metrics["threshold_tau"] = tau
    metrics["percentile_used"] = percentile

    return metrics


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("HELD-OUT LPGW THRESHOLD SELECTION")
    print("=" * 70)

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------

    tolerance = config.SPATIAL_TOLERANCE
    target_points = config.TARGET_POINTS
    segment_length = config.SEGMENT_LENGTH
    fps = config.FPS
    stride = config.STRIDE
    lambdaa = config.LPGW_LAMBDA

    print(f"Dataset       : {config.ACTIVE_DATASET}")
    print(f"Reference     : {config.BAG3_CSV}")
    print(f"Query         : {config.BAG7_CSV}")
    print(f"Tolerance     : {tolerance} m")
    print(f"Lambda        : {lambdaa}")
    print(f"Candidates    : {CANDIDATE_PERCENTILES}")

    # -------------------------------------------------------------
    # Load canonical ground truth
    # -------------------------------------------------------------

    y_true, ref_centers, query_centers = (
        load_canonical_ground_truth(tolerance)
    )

    print(
        f"\nLoaded canonical GT: "
        f"{len(y_true)} query segments, "
        f"positives={int(y_true.sum())}"
    )

    # -------------------------------------------------------------
    # Load trajectories
    #
    # IMPORTANT:
    # Paths come entirely from config.py.
    # -------------------------------------------------------------

    ref_csv = (
        repo_root
        / config.POSES_DIR
        / config.BAG3_CSV
    )

    query_csv = (
        repo_root
        / config.POSES_DIR
        / config.BAG7_CSV
    )

    ref_df = load_trajectory(ref_csv)
    query_df = load_trajectory(query_csv)

    ref_xyz = ref_df[
        ["PosX", "PosY", "PosZ"]
    ].to_numpy()

    query_xyz = query_df[
        ["PosX", "PosY", "PosZ"]
    ].to_numpy()

    # -------------------------------------------------------------
    # Downsample trajectories
    # -------------------------------------------------------------

    ref_xyz_ds = downsample_trajectory(
        ref_xyz,
        target_points,
    )

    query_xyz_ds = downsample_trajectory(
        query_xyz,
        target_points,
    )

    # -------------------------------------------------------------
    # Segment trajectories
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Make sure all arrays refer to the same number of segments
    # -------------------------------------------------------------

    n_segments = min(
        len(segments_ref),
        len(segments_query),
        len(y_true),
        len(ref_centers),
        len(query_centers),
    )

    if n_segments == 0:
        raise RuntimeError("No trajectory segments available.")

    segments_ref = segments_ref[:n_segments]
    segments_query = segments_query[:n_segments]

    y_true = y_true[:n_segments]
    ref_centers = ref_centers[:n_segments]
    query_centers = query_centers[:n_segments]

    print(f"Using {n_segments} aligned query/reference segments.")

    # -------------------------------------------------------------
    # Sanity check canonical centers
    # -------------------------------------------------------------

    recomputed_ref_centers = np.asarray(
        [segment.mean(axis=0) for segment in segments_ref]
    )

    recomputed_query_centers = np.asarray(
        [segment.mean(axis=0) for segment in segments_query]
    )

    if not np.allclose(
        recomputed_ref_centers,
        ref_centers,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Reference segment centers differ from canonical GT."
        )

    if not np.allclose(
        recomputed_query_centers,
        query_centers,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Query segment centers differ from canonical GT."
        )

    print("Canonical segment-center checks passed.")

    # -------------------------------------------------------------
    # LPGW detector
    # -------------------------------------------------------------

    detector = LoopClosureDetector(
        segment_length=segment_length,
        fps=fps,
        stride=stride,
        lambdaa=lambdaa,
        downsample_points=100,
        reference_strategy="robust",
    )

    # -------------------------------------------------------------
    # Compute LPGW distance matrix ONCE
    # -------------------------------------------------------------

    print("\nComputing full LPGW distance matrix...")
    start_time = time.time()

    D_full = detector.compute_distance_matrix(
        segments_query,
        segments_ref,
    )

    runtime = time.time() - start_time

    print(
        f"LPGW distance matrix computed in "
        f"{runtime:.2f} seconds."
    )

    if D_full.shape != (n_segments, n_segments):
        raise RuntimeError(
            f"Unexpected distance matrix shape: "
            f"{D_full.shape}; expected "
            f"({n_segments}, {n_segments})"
        )

    # -------------------------------------------------------------
    # Split QUERY segments into two folds
    # -------------------------------------------------------------

    indices = np.arange(n_segments)

    midpoint = n_segments // 2

    fold_A = indices[:midpoint]
    fold_B = indices[midpoint:]

    D_A = D_full[fold_A]
    D_B = D_full[fold_B]

    y_A = y_true[fold_A]
    y_B = y_true[fold_B]

    print(
        f"\nFold A: {len(fold_A)} query segments, "
        f"positives={int(y_A.sum())}"
    )

    print(
        f"Fold B: {len(fold_B)} query segments, "
        f"positives={int(y_B.sum())}"
    )

    # -------------------------------------------------------------
    # A -> B
    #
    # Select percentile on A.
    # Apply threshold to held-out B.
    # -------------------------------------------------------------

    p_from_A, train_metrics_A = (
        best_percentile_for_fold(
            D_A,
            y_A,
        )
    )

    result_B = apply_percentile_to_fold(
        D_A,
        p_from_A,
        D_B,
        y_B,
    )

    result_B["fold_evaluated"] = "B"
    result_B["percentile_selected_on"] = "A"

    print(
        f"\nThreshold selected on A:"
        f" percentile={p_from_A}"
        f" (training F1={train_metrics_A['f1']:.3f})"
    )

    print(
        f"Held-out B:"
        f" P={result_B['precision']:.3f},"
        f" R={result_B['recall']:.3f},"
        f" F1={result_B['f1']:.3f}"
    )

    # -------------------------------------------------------------
    # B -> A
    #
    # Select percentile on B.
    # Apply threshold to held-out A.
    # -------------------------------------------------------------

    p_from_B, train_metrics_B = (
        best_percentile_for_fold(
            D_B,
            y_B,
        )
    )

    result_A = apply_percentile_to_fold(
        D_B,
        p_from_B,
        D_A,
        y_A,
    )

    result_A["fold_evaluated"] = "A"
    result_A["percentile_selected_on"] = "B"

    print(
        f"\nThreshold selected on B:"
        f" percentile={p_from_B}"
        f" (training F1={train_metrics_B['f1']:.3f})"
    )

    print(
        f"Held-out A:"
        f" P={result_A['precision']:.3f},"
        f" R={result_A['recall']:.3f},"
        f" F1={result_A['f1']:.3f}"
    )

    # -------------------------------------------------------------
    # Original non-held-out result
    #
    # This is included ONLY as a comparison.
    # It must not be used to select the final threshold.
    # -------------------------------------------------------------

    original_tau = float(
        np.percentile(
            D_full,
            config.PERCENTILE,
        )
    )

    y_pred_full = (
        D_full.min(axis=1) <= original_tau
    ).astype(int)

    original_metrics = score_predictions(
        y_true,
        y_pred_full,
    )

    original_metrics["fold_evaluated"] = (
        "FULL (not held out)"
    )

    original_metrics["percentile_selected_on"] = (
        "FULL (same data)"
    )

    original_metrics["threshold_tau"] = original_tau
    original_metrics["percentile_used"] = config.PERCENTILE

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------

    rows = [
        result_B,
        result_A,
        original_metrics,
    ]

    results_df = pd.DataFrame(rows)

    out_dir = repo_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = (
        out_dir
        / f"heldout_threshold_{config.DATASET_SHORT}.csv"
    )

    results_df.to_csv(
        out_path,
        index=False,
    )

    avg_heldout_f1 = (
        result_A["f1"] +
        result_B["f1"]
    ) / 2.0

    # -------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Average held-out F1 = "
        f"{avg_heldout_f1:.3f}"
    )

    print(
        f"Original non-held-out F1 "
        f"(percentile={config.PERCENTILE}) = "
        f"{original_metrics['f1']:.3f}"
    )

    print(f"\nSaved to: {out_path}")

    print("\nDetailed results:")
    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
