# experiments/run_simple_lpgw_test.py

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from loop_closure import (
    LoopClosureDetector,
    load_xyz_trajectory,
)

def load_canonical_ground_truth(tolerance_m: float):
    gt_dir = Path("ground_truth/files")
    gt_csv = gt_dir / f"gt_{config.DATASET_SHORT}_{tolerance_m}m.csv"
    meta_json = gt_dir / f"gt_{config.DATASET_SHORT}_{tolerance_m}m_meta.json"

    if not gt_csv.exists() or not meta_json.exists():
        raise FileNotFoundError(
            f"Canonical ground truth for {config.DATASET_SHORT} at {tolerance_m} m not found. "
            "Run ground_truth/generate_ground_truth.py first."
        )

    gt_df = pd.read_csv(gt_csv)
    with open(meta_json) as f:
        meta = json.load(f)

    y_true = gt_df["label"].to_numpy()
    ref_centers = np.array(meta["ref_centers"])
    query_centers = np.array(meta["query_centers"])

    return y_true, ref_centers, query_centers, meta


def main():
    tolerance = config.SPATIAL_TOLERANCE  # 2.0 m

    # Load canonical ground truth
    y_true, ref_centers, query_centers, meta = load_canonical_ground_truth(tolerance)

    print(f"Loaded canonical GT at {tolerance:.1f} m")
    print(f"  Query segments: {len(y_true)}")
    print(f"  Positive: {y_true.sum()}")

    # Rebuild segments exactly as GT generation did
    poses_dir = Path("data/poses")
    ref_csv = poses_dir / config.BAG3_CSV
    query_csv = poses_dir / config.BAG7_CSV

    ref_df = load_trajectory(ref_csv)
    query_df = load_trajectory(query_csv)

    ref_xyz = ref_df[["PosX", "PosY", "PosZ"]].to_numpy()
    query_xyz = query_df[["PosX", "PosY", "PosZ"]].to_numpy()

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

    min_len = min(len(segments_7), len(segments_3))
    segments_7 = segments_7[:min_len]
    segments_3 = segments_3[:min_len]

    # Verify alignment with GT
    recomputed_query_centers = np.array([seg.mean(axis=0) for seg in segments_7])
    recomputed_ref_centers = np.array([seg.mean(axis=0) for seg in segments_3])

    if recomputed_query_centers.shape != query_centers.shape:
        raise RuntimeError(
            "Query segment centers mismatch with canonical GT. "
            "Check segmentation parameters."
        )
    if recomputed_ref_centers.shape != ref_centers.shape:
        raise RuntimeError(
            "Reference segment centers mismatch with canonical GT. "
            "Check segmentation parameters."
        )

    if not np.allclose(recomputed_query_centers, query_centers):
        raise RuntimeError(
            "Query segment centers differ from canonical GT. "
            "Segmentation may have changed."
        )
    if not np.allclose(recomputed_ref_centers, ref_centers):
        raise RuntimeError(
            "Reference segment centers differ from canonical GT. "
            "Segmentation may have changed."
        )

    print("Segment centers verified against canonical GT — alignment OK.")

    # Run LPGW
    detector = LoopClosureDetector(
        segment_length=segment_length,
        fps=fps,
        stride=stride,
        lambdaa=getattr(config, "LPGW_LAMBDA", 0.5),
        partial=True,
        downsampling=True,
    )

    print("Computing LPGW distance matrix...")
    D = detector.compute_distance_matrix(segments_7, segments_3)
    print(f"Distance matrix shape: {D.shape}")

    # Detect loop closures at 1st percentile
    y_pred, tau = detect_with_percentile(D, config.PERCENTILE)

    # Score
    metrics = score_predictions(y_true, y_pred)

    print("Metrics (LPGW, 1st percentile, 2.0 m GT):")
    print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
    print(f"  Precision={metrics['precision']:.3f}")
    print(f"  Recall={metrics['recall']:.3f}")
    print(f"  F1={metrics['f1']:.3f}")
    print(f"  Threshold τ={tau:.6f}")


if __name__ == "__main__":
    main()