# tools/plot_kitti00_split_overlap.py
"""
Plot kitti_00_reference.csv and kitti_00_query.csv overlaid in XZ (bird's-eye view):
- Reference: blue
- Query: orange
- Reference points within 2.0 m of Query: bright red

Uses PosX (horizontal) vs PosZ (vertical, forward driving direction).
Saves to figures/kitti_00_split_overlap_check_v2.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def plot_kitti00_split_overlap():
    # EDIT THESE PATHS IF NEEDED
    repo_root = Path(__file__).resolve().parent.parent
    poses_dir = repo_root / "data" / "poses"
    fig_dir = repo_root / "figures"

    ref_csv = poses_dir / "kitti_00_reference.csv"
    query_csv = poses_dir / "kitti_00_query.csv"
    out_path = fig_dir / "kitti_00_split_overlap_check_v2.png"

    print("repo_root:", repo_root)
    print("poses_dir:", poses_dir)
    print("fig_dir:  ", fig_dir)
    print("ref_csv:  ", ref_csv)
    print("query_csv:", query_csv)
    print("out_path: ", out_path)

    if not ref_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {ref_csv}")
    if not query_csv.exists():
        raise FileNotFoundError(f"Query CSV not found: {query_csv}")

    fig_dir.mkdir(parents=True, exist_ok=True)

    ref_df = pd.read_csv(ref_csv)
    query_df = pd.read_csv(query_csv)

    # Use XZ plane for bird's-eye view
    ref_xz = ref_df[["PosX", "PosZ"]].to_numpy()
    query_xz = query_df[["PosX", "PosZ"]].to_numpy()

    print(f"Loaded reference: {len(ref_xz)} points")
    print(f"Loaded query:     {len(query_xz)} points")

    # For each reference point, find nearest neighbor in query (in XZ)
    tree = KDTree(query_xz)
    dists, _ = tree.query(ref_xz, k=1)

    threshold = 2.0
    within_mask = dists <= threshold
    ref_within = ref_xz[within_mask]

    print(f"Reference points within {threshold} m of query: {len(ref_within)} "
          f"({len(ref_within)/len(ref_xz)*100:.2f}%)")

    # Plot
    plt.figure(figsize=(8, 8))
    plt.plot(ref_xz[:, 0], ref_xz[:, 1], color="#1f77b4", lw=1, label="Reference")
    plt.plot(query_xz[:, 0], query_xz[:, 1], color="#ff7f0e", lw=1, label="Query")

    if len(ref_within) > 0:
        plt.scatter(
            ref_within[:, 0],
            ref_within[:, 1],
            c="#d62728",
            s=8,
            zorder=3,
            label="Ref points within 2.0 m of Query",
        )
    else:
        print("Warning: no reference points within threshold; plot will show only trajectories.")

    plt.xlabel("X (m)")
    plt.ylabel("Z (m) -- forward driving direction")
    plt.title(
        "KITTI 00: Reference vs Query (split at index 2000)\n"
        "Red = ref points within 2.0 m of query"
    )
    plt.legend(markerscale=1.5)
    plt.axis("equal")
    plt.grid(True, ls="--", lw=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot to {out_path}")
    print("Check that this file now exists:")
    print("  ", out_path.absolute())


if __name__ == "__main__":
    plot_kitti00_split_overlap()