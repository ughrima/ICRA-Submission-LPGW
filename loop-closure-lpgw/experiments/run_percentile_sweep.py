"""Table III: performance across percentile thresholds."""
# experiments/run_percentile_sweep.py
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.neighbors import KDTree

import config
from core.trajectory_utils import load_trajectory
from core.detection import detect_with_percentile
from evaluation.eval_canonical import score_predictions
from loop_closure import LoopClosureDetector
from core.trajectory_utils import segment_trajectory, downsample_trajectory


def main():
    poses_dir = Path("data/poses")
    ref_csv = poses_dir / config.BAG3_CSV
    query_csv = poses_dir / config.BAG7_CSV

    # Load trajectories
    ref_df = load_trajectory(ref_csv)
    query_df = load_trajectory(query_csv)

    ref_xyz = ref_df[["PosX", "PosY", "PosZ"]].to_numpy()
    query_xyz = query_df[["PosX", "PosY", "PosZ"]].to_numpy()

    # Downsample and segment (same as simple test)
    target_points = getattr(config, "TARGET_POINTS", 5000)
    segment_length = getattr(config, "SEGMENT_LENGTH", 5.0)
    fps = getattr(config, "FPS", 10)
    stride = getattr(config, "STRIDE", 1.0)

    ref_xyz_ds = downsample_trajectory(ref_xyz, target_points)
    query_xyz_ds = downsample_trajectory(query_xyz, target_points)

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

    # Match segment counts
    min_len = min(len(segments_7), len(segments_3))
    segments_7 = segments_7[:min_len]
    segments_3 = segments_3[:min_len]

    print(f"Using {min_len} matched segments for LPGW")

    # Precompute LPGW matrix once
    detector = LoopClosureDetector( 
        segment_length=segment_length,
        fps=fps,
        stride=stride,
        lambdaa=config.LPGW_LAMBDA,
        downsample_points=100,
        reference_strategy="robust",
    )

    print("Computing LPGW distance matrix...")
    D = detector.compute_distance_matrix(segments_7, segments_3)
    print(f"Distance matrix shape: {D.shape}")

    # Segment centers for ground truth
    ref_centers = np.array([seg.mean(axis=0) for seg in segments_3])
    query_centers = np.array([seg.mean(axis=0) for seg in segments_7])

    tree = KDTree(ref_centers)
    dists, _ = tree.query(query_centers, k=1)
    dists = dists[:, 0]

    # Fixed tolerance (primary evaluation tolerance)
    tolerance = config.SPATIAL_TOLERANCE  # 2.0 m
    y_true = (dists <= tolerance).astype(int)

    print(f"Positive segments at {tolerance:.1f} m: {y_true.sum()} / {len(y_true)}")

    # Sweep percentiles
    percentiles = config.PERCENTILE_SWEEP  # [1, 5, 10, 20, 50]
    rows = []

    for p in percentiles:
        y_pred, tau = detect_with_percentile(D, p)
        metrics = score_predictions(y_true, y_pred)

        row = {
            "percentile": p,
            "threshold_tau": tau,
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
            f"p={p:2d}% → tau={tau:.6f}, "
            f"F1={metrics['f1']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}"
        )

    results_df = pd.DataFrame(rows)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "percentile_sweep.csv"
    results_df.to_csv(out_path, index=False)
    print(f"Percentile sweep saved to {out_path}")


if __name__ == "__main__":
    main()