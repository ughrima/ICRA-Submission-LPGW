# tools/kitti_to_csv.py
"""
Convert KITTI pose files to the CSV format expected by load_trajectory().

KITTI pose .txt format (per line):
  [R11, R12, R13, t1, R21, R22, R23, t2, R31, R32, R33, t3]

This script correctly extracts translation (t1, t2, t3) as XYZ positions.
Output columns: Timestamp,PosX,PosY,PosZ,Qx,Qy,Qz,Qw
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def kitti_poses_to_csv(kitti_txt, out_csv):
    data = np.loadtxt(kitti_txt)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    timestamps = np.arange(len(data))

    # Correct extraction: translation is at indices 3, 7, 11
    xyz = data[:, [3, 7, 11]]  # [t1, t2, t3] = [X, Y, Z]

    # Rotation matrix: elements [0,1,2,4,5,6,8,9,10] reshaped to 3x3
    rot_flat = data[:, [0,1,2,4,5,6,8,9,10]]
    rot_mats = rot_flat.reshape(-1, 3, 3)

    quats = []
    for rot in rot_mats:
        r = R.from_matrix(rot)
        q = r.as_quat()  # [x, y, z, w]
        quats.append(q)

    quats = np.array(quats)

    df = pd.DataFrame({
        "Timestamp": timestamps,
        "PosX": xyz[:, 0],
        "PosY": xyz[:, 1],
        "PosZ": xyz[:, 2],
        "Qx": quats[:, 0],
        "Qy": quats[:, 1],
        "Qz": quats[:, 2],
        "Qw": quats[:, 3],
    })

    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    kitti_dir = repo_root / "data" / "kitti_raw" / "poses"
    poses_dir = repo_root / "data" / "poses"

    poses_dir.mkdir(exist_ok=True)

    # Convert all sequences 00–10
    sequences = [f"{i:02d}" for i in range(0, 11)]  # ["00","01",...,"10"]
    for seq in sequences:
        kitti_poses_to_csv(
            kitti_dir / f"{seq}.txt",
            poses_dir / f"kitti_{seq}.csv"
        )