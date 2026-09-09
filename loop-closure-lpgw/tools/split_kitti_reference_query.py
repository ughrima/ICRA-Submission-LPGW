# tools/split_kitti_reference_query.py
"""
Split kitti_00.csv into reference and query portions with NON-OVERLAPPING
index ranges, while preserving genuine spatial revisits.

Split strategy (to be tuned based on analyze_self_overlap_kitti00.py output):
- Reference = indices [0, S)
- Query     = indices [S, N)

Where S is chosen so that:
  - One part of the self-revisit lies in [0, S)
  - The other part lies in [S, N)
  - There is NO index overlap between the two files.

Output files:
- data/poses/kitti_00_reference.csv
- data/poses/kitti_00_query.csv

Format: Timestamp,PosX,PosY,PosZ,Qx,Qy,Qz,Qw (same as UZH-FPV pose files).
"""

from pathlib import Path
import pandas as pd

# TODO: Set S based on analyze_self_overlap_kitti00.py output.
# Example (replace with your actual value):
SPLIT_INDEX = 2000  # <-- CHANGE THIS after running the analysis

def split_kitti_00():
    repo_root = Path(__file__).resolve().parent.parent
    poses_dir = repo_root / "data" / "poses"

    kitti_00_csv = poses_dir / "kitti_00.csv"
    if not kitti_00_csv.exists():
        raise FileNotFoundError(f"Expected {kitti_00_csv} to exist. Run kitti_to_csv.py first.")

    df = pd.read_csv(kitti_00_csv)
    n = len(df)

    if not (0 < SPLIT_INDEX < n):
        raise ValueError(f"SPLIT_INDEX={SPLIT_INDEX} must be in (0, {n})")

    ref_df = df.iloc[0:SPLIT_INDEX].copy()
    query_df = df.iloc[SPLIT_INDEX:n].copy()

    # Optionally re-index timestamps per split:
    # ref_df["Timestamp"] = range(len(ref_df))
    # query_df["Timestamp"] = range(len(query_df))

    ref_out = poses_dir / "kitti_00_reference.csv"
    query_out = poses_dir / "kitti_00_query.csv"

    ref_df.to_csv(ref_out, index=False)
    query_df.to_csv(query_out, index=False)

    print(f"Reference: {len(ref_df)} points -> {ref_out}")
    print(f"Query:     {len(query_df)} points -> {query_out}")
    print(f"Index ranges: ref=[0,{SPLIT_INDEX}), query=[{SPLIT_INDEX},{n})")


if __name__ == "__main__":
    split_kitti_00()