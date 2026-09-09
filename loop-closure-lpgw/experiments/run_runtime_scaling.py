# experiments/run_runtime_scaling.py
"""
Runtime and scaling experiment for LPGW.

Measures how runtime changes as the number of query/reference
trajectory segments increases.

The experiment separates:

    1. Number of segments
    2. Number of pairwise comparisons
    3. Total LPGW distance-matrix runtime
    4. Average time per pair
    5. Average time per query segment

This is intended to make the computational cost of the complete
LPGW pipeline explicit rather than reporting only an O(K) embedding
claim.

Results are saved to:

    results/runtime_scaling_{DATASET_SHORT}.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Repository path
# ---------------------------------------------------------------------

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

import config

from core.trajectory_utils import load_trajectory
from loop_closure import LoopClosureDetector
from core.trajectory_utils import segment_trajectory, downsample_trajectory



# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def build_segments(
    xyz,
    target_points,
    segment_length,
    fps,
    stride,
):
    """
    Downsample and segment a trajectory.
    """

    xyz_ds = downsample_trajectory(
        xyz,
        target_points,
    )

    segments = segment_trajectory(
        xyz_ds,
        segment_length=segment_length,
        fps=fps,
        stride=stride,
    )

    return segments


def select_evenly_spaced(
    segments,
    n,
):
    """
    Select n approximately evenly spaced segments.

    The original segment indices are preserved conceptually;
    this is only a computational scaling experiment.
    """

    if n > len(segments):
        raise ValueError(
            f"Requested {n} segments but only "
            f"{len(segments)} are available."
        )

    indices = np.linspace(
        0,
        len(segments) - 1,
        n,
        dtype=int,
    )

    return [
        segments[i]
        for i in indices
    ]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_runtime_scaling():

    poses_dir = Path("data/poses")

    ref_csv = poses_dir / config.BAG3_CSV
    query_csv = poses_dir / config.BAG7_CSV

    print("=" * 70)
    print("LPGW RUNTIME / SCALING EXPERIMENT")
    print("=" * 70)

    print(
        f"Dataset: {config.DATASET_SHORT}"
    )

    # -----------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------

    ref_df = load_trajectory(
        ref_csv
    )

    query_df = load_trajectory(
        query_csv
    )

    ref_xyz = ref_df[
        ["PosX", "PosY", "PosZ"]
    ].to_numpy()

    query_xyz = query_df[
        ["PosX", "PosY", "PosZ"]
    ].to_numpy()

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    target_points = getattr(
        config,
        "TARGET_POINTS",
        5000,
    )

    segment_length = getattr(
        config,
        "SEGMENT_LENGTH",
        5.0,
    )

    fps = getattr(
        config,
        "FPS",
        10,
    )

    stride = getattr(
        config,
        "STRIDE",
        1.0,
    )

    # -----------------------------------------------------------------
    # Build full segment sets
    # -----------------------------------------------------------------

    ref_segments = build_segments(
        ref_xyz,
        target_points,
        segment_length,
        fps,
        stride,
    )

    query_segments = build_segments(
        query_xyz,
        target_points,
        segment_length,
        fps,
        stride,
    )

    # Match the canonical truncation policy.
    min_len = min(
        len(ref_segments),
        len(query_segments),
    )

    ref_segments = ref_segments[:min_len]
    query_segments = query_segments[:min_len]

    print(
        f"Available segments: {min_len}"
    )

    # -----------------------------------------------------------------
    # Segment sizes to test
    #
    # You can change this list if desired.
    # -----------------------------------------------------------------

    requested_sizes = [
        25,
        50,
        100,
        150,
        200,
    ]

    sizes = [
        n
        for n in requested_sizes
        if n <= min_len
    ]

    if not sizes:
        raise RuntimeError(
            "None of the requested runtime sizes "
            "are available."
        )

    # -----------------------------------------------------------------
    # Detector
    # -----------------------------------------------------------------

    detector = LoopClosureDetector(
        segment_length=segment_length,
        fps=fps,
        stride=stride,
        lambdaa=config.LPGW_LAMBDA,
        downsample_points=100,
        reference_strategy="robust",
    )

    rows = []

    # -----------------------------------------------------------------
    # Scaling experiment
    # -----------------------------------------------------------------

    for n in sizes:

        print()
        print("-" * 70)
        print(
            f"Testing {n} x {n} segments"
        )
        print("-" * 70)

        ref_subset = select_evenly_spaced(
            ref_segments,
            n,
        )

        query_subset = select_evenly_spaced(
            query_segments,
            n,
        )

        num_pairs = n * n

        # -------------------------------------------------------------
        # Warm-up
        #
        # This avoids measuring only first-call import/setup effects.
        # -------------------------------------------------------------

        warmup_detector = LoopClosureDetector(
            segment_length=segment_length,
            fps=fps,
            stride=stride,
            lambdaa=config.LPGW_LAMBDA,
            downsample_points=100,
            reference_strategy="robust",
        )

        warmup_query = query_subset[:1]
        warmup_ref = ref_subset[:1]

        _ = warmup_detector.compute_distance_matrix(
            warmup_query,
            warmup_ref,
        )

        # -------------------------------------------------------------
        # Timed run
        # -------------------------------------------------------------

        start = time.perf_counter()

        D = detector.compute_distance_matrix(
            query_subset,
            ref_subset,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        # -------------------------------------------------------------
        # Basic validation
        # -------------------------------------------------------------

        expected_shape = (
            n,
            n,
        )

        if D.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected distance matrix shape: "
                f"{D.shape}; expected {expected_shape}"
            )

        # -------------------------------------------------------------
        # Timing statistics
        # -------------------------------------------------------------

        time_per_pair = (
            elapsed / num_pairs
        )

        time_per_query = (
            elapsed / n
        )

        rows.append(
            {
                "num_query_segments": n,
                "num_reference_segments": n,
                "num_pairs": num_pairs,
                "distance_matrix_shape": (
                    f"{n}x{n}"
                ),
                "runtime_seconds": elapsed,
                "time_per_pair_ms": (
                    time_per_pair * 1000.0
                ),
                "time_per_query_ms": (
                    time_per_query * 1000.0
                ),
            }
        )

        print(
            f"Pairs:             {num_pairs:,}"
        )

        print(
            f"Runtime:           {elapsed:.4f} s"
        )

        print(
            f"Time / pair:       "
            f"{time_per_pair * 1000.0:.4f} ms"
        )

        print(
            f"Time / query:      "
            f"{time_per_query * 1000.0:.4f} ms"
        )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    results_df = pd.DataFrame(
        rows
    )

    out_dir = Path("results")
    out_dir.mkdir(
        exist_ok=True
    )

    out_path = (
        out_dir
        / f"runtime_scaling_{config.DATASET_SHORT}.csv"
    )

    results_df.to_csv(
        out_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("RUNTIME EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Saved to: {out_path}"
    )

    print()
    print(
        results_df.to_string(
            index=False
        )
    )


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_runtime_scaling()
