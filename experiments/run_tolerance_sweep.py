
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Make project root importable
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
from core.detection import detect_with_percentile
from evaluation.eval_canonical import score_predictions
from loop_closure import LoopClosureDetector


# ---------------------------------------------------------------------
# Load canonical ground-truth metadata
# ---------------------------------------------------------------------

def load_canonical_metadata():
    """
    Load the canonical GT metadata.

    The metadata contains the reference/query segment centers produced
    by ground_truth/generate_ground_truth.py.
    """

    gt_dir = repo_root / "ground_truth" / "files"

    meta_path = (
        gt_dir
        / f"gt_{config.DATASET_SHORT}_{config.SPATIAL_TOLERANCE}m_meta.json"
    )

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Canonical GT metadata not found:\n"
            f"  {meta_path}\n\n"
            f"Run:\n"
            f"  python ground_truth/generate_ground_truth.py\n"
            f"first."
        )

    with open(meta_path, "r") as f:
        meta = json.load(f)

    ref_centers = np.asarray(meta["ref_centers"], dtype=float)
    query_centers = np.asarray(meta["query_centers"], dtype=float)

    return ref_centers, query_centers


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("LPGW TOLERANCE SWEEP")
    print("=" * 70)

    print(f"Dataset: {config.DATASET_LABEL_REF} vs {config.DATASET_LABEL_QUERY}")
    print(f"Dataset short name: {config.DATASET_SHORT}")
    print(f"Canonical tolerance: {config.SPATIAL_TOLERANCE} m")
    print(f"Tolerance sweep: {config.TOLERANCE_SWEEP}")
    print(f"Percentile: {config.PERCENTILE}%")

    # ---------------------------------------------------------------
    # 1. Load trajectories
    # ---------------------------------------------------------------

    poses_dir = repo_root / config.POSES_DIR

    ref_csv = poses_dir / config.BAG3_CSV
    query_csv = poses_dir / config.BAG7_CSV

    print("\nLoading trajectories...")

    ref_df = load_trajectory(ref_csv)
    query_df = load_trajectory(query_csv)

    ref_xyz = ref_df[["PosX", "PosY", "PosZ"]].to_numpy()
    query_xyz = query_df[["PosX", "PosY", "PosZ"]].to_numpy()

    print(f"Raw reference points: {len(ref_xyz)}")
    print(f"Raw query points:     {len(query_xyz)}")

    # ---------------------------------------------------------------
    # 2. Canonical preprocessing
    # ---------------------------------------------------------------

    ref_xyz_ds = downsample_trajectory(
        ref_xyz,
        config.TARGET_POINTS,
    )

    query_xyz_ds = downsample_trajectory(
        query_xyz,
        config.TARGET_POINTS,
    )

    print(f"After downsampling:")
    print(f"  Reference: {len(ref_xyz_ds)}")
    print(f"  Query:     {len(query_xyz_ds)}")

    # ---------------------------------------------------------------
    # 3. Canonical segmentation
    # ---------------------------------------------------------------

    segments_ref = segment_trajectory(
        ref_xyz_ds,
        segment_length=config.SEGMENT_LENGTH,
        fps=config.FPS,
        stride=config.STRIDE,
    )

    segments_query = segment_trajectory(
        query_xyz_ds,
        segment_length=config.SEGMENT_LENGTH,
        fps=config.FPS,
        stride=config.STRIDE,
    )

    # Canonical GT truncates both trajectories to the same number
    min_len = min(
        len(segments_ref),
        len(segments_query),
    )

    segments_ref = segments_ref[:min_len]
    segments_query = segments_query[:min_len]

    print(f"Reference segments: {len(segments_ref)}")
    print(f"Query segments:     {len(segments_query)}")

    # ---------------------------------------------------------------
    # 4. Load canonical segment centers
    # ---------------------------------------------------------------

    canonical_ref_centers, canonical_query_centers = (
        load_canonical_metadata()
    )

    # The canonical GT was generated with the same truncation.
    if len(canonical_ref_centers) != min_len:
        raise RuntimeError(
            "Canonical reference-center count does not match "
            "the current segmentation.\n"
            f"Canonical: {len(canonical_ref_centers)}\n"
            f"Current:   {min_len}"
        )

    if len(canonical_query_centers) != min_len:
        raise RuntimeError(
            "Canonical query-center count does not match "
            "the current segmentation.\n"
            f"Canonical: {len(canonical_query_centers)}\n"
            f"Current:   {min_len}"
        )

    # ---------------------------------------------------------------
    # 5. Verify segmentation against canonical GT
    # ---------------------------------------------------------------

    recomputed_ref_centers = np.array(
        [seg.mean(axis=0) for seg in segments_ref]
    )

    recomputed_query_centers = np.array(
        [seg.mean(axis=0) for seg in segments_query]
    )

    if not np.allclose(
        recomputed_ref_centers,
        canonical_ref_centers,
    ):
        raise RuntimeError(
            "Reference segment centers differ from canonical GT.\n"
            "Your preprocessing/segmentation parameters may have changed."
        )

    if not np.allclose(
        recomputed_query_centers,
        canonical_query_centers,
    ):
        raise RuntimeError(
            "Query segment centers differ from canonical GT.\n"
            "Your preprocessing/segmentation parameters may have changed."
        )

    print("\nCanonical segmentation verified.")

    # ---------------------------------------------------------------
    # 6. Compute nearest-reference distance ONCE
    # ---------------------------------------------------------------

    # This is the same geometric relationship used by the canonical
    # ground-truth generator.
    from sklearn.neighbors import KDTree

    tree = KDTree(canonical_ref_centers)

    nearest_distances, nearest_indices = tree.query(
        canonical_query_centers,
        k=1,
    )

    nearest_distances = nearest_distances[:, 0]
    nearest_indices = nearest_indices[:, 0]

    print(
        f"Nearest-reference distances computed for "
        f"{len(nearest_distances)} query segments."
    )

    # ---------------------------------------------------------------
    # 7. Compute LPGW distance matrix ONCE
    # ---------------------------------------------------------------

    detector = LoopClosureDetector(
        segment_length=config.SEGMENT_LENGTH,
        fps=config.FPS,
        stride=config.STRIDE,
        lambdaa=config.LPGW_LAMBDA,
        downsample_points=config.TARGET_POINTS,
        reference_strategy="robust",
    )

    print("\nComputing LPGW distance matrix...")

    D = detector.compute_distance_matrix(
        segments_query,
        segments_ref,
    )

    print(f"LPGW distance matrix shape: {D.shape}")

    # ---------------------------------------------------------------
    # 8. Evaluate every tolerance
    # ---------------------------------------------------------------

    rows = []

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for tolerance in config.TOLERANCE_SWEEP:

        # Ground truth changes ONLY because tolerance changes.
        y_true = (
            nearest_distances <= tolerance
        ).astype(int)

        # LPGW prediction is unchanged across tolerance values.
        y_pred, tau = detect_with_percentile(
            D,
            config.PERCENTILE,
        )

        metrics = score_predictions(
            y_true,
            y_pred,
        )

        row = {
            "dataset": config.DATASET_SHORT,
            "tolerance_m": tolerance,
            "percentile": config.PERCENTILE,
            "lambda": config.LPGW_LAMBDA,
            "num_segments": min_len,
            "num_gt_positives": int(y_true.sum()),
            "num_predicted_positives": int(y_pred.sum()),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "threshold_tau": tau,
        }

        rows.append(row)

        print(
            f"Tolerance={tolerance:.1f} m | "
            f"GT positives={y_true.sum():3d} | "
            f"P={metrics['precision']:.3f} | "
            f"R={metrics['recall']:.3f} | "
            f"F1={metrics['f1']:.3f}"
        )

    # ---------------------------------------------------------------
    # 9. Save results
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(rows)

    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)

    out_path = (
        out_dir
        / f"tolerance_sweep_{config.DATASET_SHORT}.csv"
    )

    results_df.to_csv(
        out_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print(f"Saved to: {out_path}")
    print("=" * 70)

    print("\nFinal table:")
    print(
        results_df[
            [
                "tolerance_m",
                "num_gt_positives",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()