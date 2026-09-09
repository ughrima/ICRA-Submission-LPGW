# tools/check_self_overlap.py
"""
Check self-overlap for a single KITTI sequence CSV.

For a given sequence CSV:
  - Builds a KD-tree on all points.
  - For each point, excludes a temporal window of ±window points around itself,
    then finds the nearest OTHER point on the trajectory outside that window.
  - Reports: min/median/10th-percentile of these self-nearest-neighbor distances,
    and the fraction of points with a same-trajectory neighbor (outside the
    exclusion window) within 2.0 m.
  - Saves an XY plot colored by time index (viridis colormap) to
    figures/self_overlap_{sequence_name}.png.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import config

def check_self_overlap(csv_path, window=300, tolerance=None):
    if tolerance is None:
        tolerance = config.SPATIAL_TOLERANCE  # 2.0 m

    df = pd.read_csv(csv_path)
    xyz = df[["PosX", "PosY", "PosZ"]].to_numpy()
    n = len(xyz)

    print(f"\n=== {csv_path.name} ===")
    print(f"Total points: {n}")

    tree = KDTree(xyz)

    min_dists = []
    for i in range(n):
        # Define exclusion window
        start = max(0, i - window)
        end = min(n, i + window + 1)

        # Query all neighbors, then filter out those in the exclusion window
        # To avoid huge k, we can query a moderate k and increase if needed.
        # Here we query k = min(n, window*4) as a heuristic.
        k_query = min(n, window * 4)
        dists, idxs = tree.query(xyz[i:i+1], k=k_query)
        dists = dists[0]
        idxs = idxs[0]

        # Keep only neighbors outside [start, end)
        valid = (idxs < start) | (idxs >= end)
        if not np.any(valid):
            # If no valid neighbors found with this k, skip (should be rare)
            continue

        min_dists.append(float(dists[valid].min()))

    min_dists = np.array(min_dists)

    if len(min_dists) == 0:
        print("No valid self-neighbors found with this window/k heuristic.")
        return None

    min_val = float(min_dists.min())
    median_val = float(np.median(min_dists))
    p10_val = float(np.percentile(min_dists, 10))
    frac_within_tol = float((min_dists <= tolerance).mean())

    print(f"Self-nearest-neighbor distances (excluding ±{window} points):")
    print(f"  min:        {min_val:.3f} m")
    print(f"  median:     {median_val:.3f} m")
    print(f"  10th perc:  {p10_val:.3f} m")
    print(f"  fraction within {tolerance:.1f} m: {frac_within_tol:.3f} ({frac_within_tol*100:.1f}%)")

    # XY plot colored by time index
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xyz[:, 0], xyz[:, 1], c=np.arange(n), cmap="viridis", s=1)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{csv_path.stem} (colored by time)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Time index")
    ax.grid(True)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"self_overlap_{csv_path.stem}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"XY plot saved to {out_path}")

    return {
        "sequence": csv_path.stem,
        "total_points": n,
        "frac_within_2m": frac_within_tol,
        "median_self_dist_m": median_val,
        "min_self_dist_m": min_val,
        "p10_self_dist_m": p10_val,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequences",
        type=str,
        default="00,02,05,06,08,09",
        help="Comma-separated list of sequence names (e.g. 00,02,05)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=300,
        help="Temporal exclusion window (±window points)",
    )
    args = parser.parse_args()

    poses_dir = Path("data/poses")
    sequences = [s.strip() for s in args.sequences.split(",")]

    results = []
    for seq in sequences:
        csv_path = poses_dir / f"kitti_{seq}.csv"
        if not csv_path.exists():
            print(f"Skipping {seq}: {csv_path} not found")
            continue
        res = check_self_overlap(csv_path, window=args.window)
        if res is not None:
            results.append(res)

    if not results:
        print("\nNo sequences processed.")
        return

    # Print summary table
    print("\n=== Summary table ===")
    print(f"{'Sequence':<10} {'Points':>8} {'Frac ≤2m':>10} {'Median dist (m)':>16} {'Min dist (m)':>14}")
    for r in results:
        print(
            f"{r['sequence']:<10} "
            f"{r['total_points']:>8} "
            f"{r['frac_within_2m']:>10.3f} "
            f"{r['median_self_dist_m']:>16.3f} "
            f"{r['min_self_dist_m']:>14.3f}"
        )

    # Identify strongest self-overlap
    best = max(results, key=lambda r: r["frac_within_2m"])
    print(f"\nStrongest self-overlap: {best['sequence']} "
          f"(fraction within 2.0 m = {best['frac_within_2m']:.3f})")


if __name__ == "__main__":
    main()