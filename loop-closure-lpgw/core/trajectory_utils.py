# core/trajectory_utils.py

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Segments:
    """
    Metadata describing trajectory segments.
    """

    centers: np.ndarray
    indices: np.ndarray
    timestamps: np.ndarray


def load_trajectory(path: Path) -> pd.DataFrame:
    """
    Load a trajectory CSV.

    Required columns:
        Timestamp
        PosX
        PosY
        PosZ
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Trajectory file not found: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "Timestamp",
        "PosX",
        "PosY",
        "PosZ",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns {missing} in {path}"
        )

    if len(df) == 0:
        raise ValueError(
            f"Trajectory file is empty: {path}"
        )

    return df


def resample_to_target_points(
    df: pd.DataFrame,
    target_points: int,
) -> pd.DataFrame:
    """
    Uniformly subsample a trajectory by sample index.

    If the trajectory already contains fewer than or equal
    to target_points samples, it is returned unchanged.

    This function assumes the input trajectory is sampled
    at approximately uniform temporal intervals.
    """

    if target_points <= 0:
        raise ValueError(
            "target_points must be greater than zero."
        )

    N = len(df)

    if N == 0:
        raise ValueError(
            "Cannot resample an empty trajectory."
        )

    if N <= target_points:
        return df.reset_index(drop=True)

    indices = np.linspace(
        0,
        N - 1,
        target_points,
        dtype=int,
    )

    return df.iloc[indices].reset_index(drop=True)