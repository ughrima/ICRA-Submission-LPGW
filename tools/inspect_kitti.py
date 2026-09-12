# tools/inspect_kitti.py
"""
Inspect KITTI CSVs: print first row XYZ and bounding box for each sequence.
"""

from pathlib import Path
import pandas as pd
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
poses_dir = repo_root / "data" / "poses"

files = {
    "kitti_00": poses_dir / "kitti_00.csv",
    "kitti_02": poses_dir / "kitti_02.csv",
}

for name, path in files.items():
    print(f"\n=== {name} ({path.name}) ===")
    df = pd.read_csv(path)

    first = df.iloc[0]
    print(f"First row XYZ:  X={first['PosX']:.3f}, Y={first['PosY']:.3f}, Z={first['PosZ']:.3f}")

    xmin, ymin, zmin = df[["PosX", "PosY", "PosZ"]].min(axis=0)
    xmax, ymax, zmax = df[["PosX", "PosY", "PosZ"]].max(axis=0)

    print(f"Bounding box X: [{xmin:.3f}, {xmax:.3f}]")
    print(f"Bounding box Y: [{ymin:.3f}, {ymax:.3f}]")
    print(f"Bounding box Z: [{zmin:.3f}, {zmax:.3f}]")