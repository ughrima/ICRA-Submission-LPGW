# experiments/run_reference_segment_ablation.py
"""
Reference-segment sensitivity ablation (Reviewer 5).

LPGW embeds every segment against a single FIXED reference segment X_bar
(currently hardcoded as "the first segment of the reference trajectory" in
lpgw_impl.py). Reviewer 5 asked: how sensitive are results to this arbitrary
choice? This script reruns detection using several different reference segment
choices and reports F1/precision/recall variance across them.

PREREQUISITE: LoopClosureDetector must accept a `reference_index` kwarg
(int, index into the reference-trajectory segment list) that selects which
segment is used as X_bar, instead of always using index 0. If lpgw_impl.py
does not yet support this, add it: wherever the fixed reference segment is
currently selected as segments_ref[0], change it to
segments_ref[self.reference_index] and default self.reference_index = 0 in
__init__. This script will raise a clear error if the parameter is missing.

Saves results/reference_segment_ablation_{DATASET_SHORT}.csv
"""

import sys
from pathlib import Path
import json
import inspect
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import config
from core.trajectory_utils import load_trajectory
from core.detection import detect_with_percentile
from evaluation.eval_canonical import score_predictions
from loop_closure import LoopClosureDetector
from core.trajectory_utils import segment_trajectory, downsample_trajectory

def load_canonical_ground_truth(tolerance_m: float):
    gt_dir = repo_root / "ground_truth" / "files"
    # Use dataset-specific GT files
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
    return y_true, ref_centers, query_centers


def check_reference_index_supported():
    sig = inspect.signature(LoopClosureDetector.__init__)
    if "reference_index" not in sig.parameters:
        raise NotImplementedError(
            "LoopClosureDetector does not accept a `reference_index` kwarg. "
            "Add support for selecting the fixed reference segment by index "
            "before running this ablation. See the module docstring for the "
            "one-line change needed in lpgw_impl.py."
        )


def main():
    check_reference_index_supported()

    tolerance = config.SPATIAL_TOLERANCE
    y_true, ref_centers, query_centers = load_canonical_ground_truth(tolerance)
    n_ref_segments = len(ref_centers)
    print(f"Loaded canonical GT: {len(y_true)} query segments, "
          f"{n_ref_segments} reference segments, positives={y_true.sum()}")

    poses_dir = repo_root / "data" / "poses"
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

    segments_3 = segment_trajectory(ref_xyz_ds, segment_length=segment_length, fps=fps, stride=stride)
    segments_7 = segment_trajectory(query_xyz_ds, segment_length=segment_length, fps=fps, stride=stride)

    min_len = min(len(segments_3), len(segments_7))
    segments_3 = segments_3[:min_len]
    segments_7 = segments_7[:min_len]

    recomputed_ref_centers = np.array([seg.mean(axis=0) for seg in segments_3])
    if not np.allclose(recomputed_ref_centers, ref_centers[:min_len]):
        raise RuntimeError(
            "Reference segment centers differ from canonical GT. "
            "Segmentation parameters may have changed since GT was generated."
        )
    print("Segment centers verified against canonical GT — alignment OK.")

    # Candidate reference-segment indices: first, quartiles, last.
    candidate_indices = sorted(set([
        0,
        min_len // 4,
        min_len // 2,
        (3 * min_len) // 4,
        min_len - 1,
    ]))
    print(f"Testing reference_index in {candidate_indices} "
          f"(out of {min_len} available reference segments)")

    rows = []
    for ref_idx in candidate_indices:
        print(f"\n=== reference_index = {ref_idx} ===")
        detector = LoopClosureDetector(
        segment_length=config.SEGMENT_LENGTH,
        fps=config.FPS,
        stride=config.STRIDE,
        lambdaa=config.LPGW_LAMBDA,
        downsample_points=config.TARGET_POINTS,
        reference_strategy="first",
        reference_index=ref_idx,
        )

        D = detector.compute_distance_matrix(segments_7, segments_3)
        y_pred, tau = detect_with_percentile(D, config.PERCENTILE)
        metrics = score_predictions(y_true, y_pred)

        rows.append({
            "reference_index": ref_idx,
            "reference_index_fraction": ref_idx / max(min_len - 1, 1),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "threshold_tau": tau,
        })
        print(f"  P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    results_df = pd.DataFrame(rows)
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"reference_segment_ablation_{config.DATASET_SHORT}.csv"
    results_df.to_csv(out_path, index=False)

    f1_values = results_df["f1"].to_numpy()
    print(f"\nSaved to {out_path}")
    print(f"F1 across reference choices: min={f1_values.min():.3f}, "
          f"max={f1_values.max():.3f}, std={f1_values.std():.3f}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()