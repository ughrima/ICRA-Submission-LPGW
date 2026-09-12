# tools/analyze_self_overlap_kitti00.py
"""
Analyze self-overlap in kitti_00.csv and report WHERE (in index space) the
revisits occur.

For each point i with a same-trajectory neighbor within distance_threshold
(excluding a ±window exclusion), record:
  - i (the point's index)
  - j (the index of its nearest neighbor outside the exclusion window)

Then print:
  - How many such pairs exist.
  - Approximate index ranges where these matches cluster.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

def load_xyz(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    return df[["PosX", "PosY", "PosZ"]].to_numpy()

def analyze_self_overlap(
    csv_path: Path,
    window: int = 300,
    distance_threshold: float = 2.0
):
    xyz = load_xyz(csv_path)
    n = len(xyz)
    tree = KDTree(xyz)

    match_i = []
    match_j = []
    match_dist = []

    for i in range(n):
        # Exclude neighbors within [i - window, i + window]
        min_idx = max(0, i - window)
        max_idx = min(n, i + window + 1)

        # Query enough neighbors to find one outside the exclusion window
        # In worst case, we might need (max_idx - min_idx) + 1 neighbors
        k_needed = (max_idx - min_idx) + 1
        dists, idxs = tree.query(xyz[i:i+1], k=min(k_needed, n))

        dists = dists[0]
        idxs = idxs[0]

        # Find first neighbor outside the exclusion window
        found = False
        for d, j in zip(dists, idxs):
            if j < min_idx or j >= max_idx:
                match_i.append(i)
                match_j.append(j)
                match_dist.append(d)
                found = True
                break

        if not found:
            # No neighbor outside window (should be rare for long sequences)
            continue

    match_i = np.array(match_i, dtype=int)
    match_j = np.array(match_j, dtype=int)
    match_dist = np.array(match_dist)

    within = match_dist <= distance_threshold
    frac_within = float(np.mean(within)) if len(within) > 0 else 0.0

    print(f"Total points: {n}")
    print(f"Points with a same-trajectory neighbor (outside ±{window}): {len(match_i)}")
    print(f"Fraction within {distance_threshold} m: {frac_within:.4f} ({frac_within*100:.2f}%)")

    if len(match_i) == 0:
        print("No self-overlap matches found.")
        return

    # Focus on pairs that are actually within the threshold
    i_within = match_i[within]
    j_within = match_j[within]

    if len(i_within) == 0:
        print(f"No pairs within {distance_threshold} m.")
        return

    print("\nIndex statistics for pairs within threshold:")
    print(f"  i (query point)   min={i_within.min()}, max={i_within.max()}, median={np.median(i_within):.1f}")
    print(f"  j (neighbor point) min={j_within.min()}, max={j_within.max()}, median={np.median(j_within):.1f}")

    # Simple clustering: split trajectory into chunks and count matches per chunk
    chunk_size = max(1, n // 20)  # ~20 chunks
    i_chunks = i_within // chunk_size
    j_chunks = j_within // chunk_size

    print("\nApproximate index ranges where revisits cluster:")
    print(f"  (Using chunk size ~{chunk_size} indices)")

    # Show top chunks by frequency for i and j
    from collections import Counter
    i_counts = Counter(i_chunks)
    j_counts = Counter(j_chunks)

    print("  Top 5 chunks for i (query points):")
    for chunk, count in i_counts.most_common(5):
        start = int(chunk * chunk_size)
        end = int(min((chunk + 1) * chunk_size, n))
        print(f"    indices [{start}, {end}) -> {count} matches")

    print("  Top 5 chunks for j (neighbor points):")
    for chunk, count in j_counts.most_common(5):
        start = int(chunk * chunk_size)
        end = int(min((chunk + 1) * chunk_size, n))
        print(f"    indices [{start}, {end}) -> {count} matches")

    # Also print a simple suggestion for a split point
    # Idea: choose a split between the main clusters of i and j if they are separated
    median_i = float(np.median(i_within))
    median_j = float(np.median(j_within))

    if median_i < median_j:
        suggested_split = int((median_i + median_j) / 2)
        print(f"\nSuggested non-overlapping split:")
        print(f"  Reference = indices [0, {suggested_split})")
        print(f"  Query     = indices [{suggested_split}, {n})")
        print(f"  (Early cluster ~median_i={median_i:.1f}, late cluster ~median_j={median_j:.1f})")
    else:
        print("\nClusters are not clearly separated in index; manual inspection recommended.")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    poses_dir = repo_root / "data" / "poses"
    csv_path = poses_dir / "kitti_00.csv"

    analyze_self_overlap(csv_path, window=300, distance_threshold=2.0)