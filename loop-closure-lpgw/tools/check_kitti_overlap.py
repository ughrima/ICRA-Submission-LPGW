# tools/check_kitti_overlap.py
"""
Check spatial overlap between two trajectories (reference vs query).

Usage example:
  python tools/check_kitti_overlap.py \
    --ref data/poses/kitti_00_reference.csv \
    --query data/poses/kitti_00_query.csv \
    --distance_threshold 2.0

Reports:
- min distance
- median distance
- 10th-percentile distance
- fraction of points with distance <= distance_threshold
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def load_xyz(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    return df[["PosX", "PosY", "PosZ"]].to_numpy()

def check_overlap(ref_csv: Path, query_csv: Path, distance_threshold: float = 2.0):
    ref_xyz = load_xyz(ref_csv)
    query_xyz = load_xyz(query_csv)

    # For each reference point, find nearest neighbor in query
    tree = KDTree(query_xyz)
    dists, _ = tree.query(ref_xyz, k=1)

    min_d = float(np.min(dists))
    median_d = float(np.median(dists))
    p10_d = float(np.percentile(dists, 10))
    frac_within = float(np.mean(dists <= distance_threshold))

    print(f"Reference points: {len(ref_xyz)}")
    print(f"Query points:     {len(query_xyz)}")
    print(f"Distance threshold: {distance_threshold} m")
    print(f"Min distance:         {min_d:.4f} m")
    print(f"Median distance:      {median_d:.4f} m")
    print(f"10th-percentile:      {p10_d:.4f} m")
    print(f"Fraction within {distance_threshold} m: {frac_within:.4f} ({frac_within*100:.2f}%)")

    return {
        "min": min_d,
        "median": median_d,
        "p10": p10_d,
        "frac_within": frac_within,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--distance_threshold", type=float, default=2.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    ref_path = args.ref if args.ref.is_absolute() else repo_root / args.ref
    query_path = args.query if args.query.is_absolute() else repo_root / args.query

    check_overlap(ref_path, query_path, args.distance_threshold)