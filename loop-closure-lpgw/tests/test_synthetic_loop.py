from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
from loop_closure import LoopClosureDetector
from lpgw import LPGW


# ============================================================
# 1. Make a synthetic trajectory
# ============================================================

def make_trajectory():
    """
    Create a trajectory with a known repeated path.

    The trajectory has three parts:

        A -> B -> C -> B -> D

    The B section is deliberately repeated.
    """

    # -----------------------------
    # First visit to B
    # -----------------------------
    t1 = np.linspace(0, 1, 100)

    B1 = np.column_stack([
        10 + 10 * t1,
        5 * np.sin(2 * np.pi * t1),
        2 * np.cos(2 * np.pi * t1),
    ])

    # -----------------------------
    # Move away from B
    # -----------------------------
    t2 = np.linspace(0, 1, 100)

    C = np.column_stack([
        20 + 15 * t2,
        8 * np.sin(np.pi * t2),
        3 * t2,
    ])

    # -----------------------------
    # Second visit to B
    #
    # This is the known loop.
    # It has exactly the same shape
    # as B1.
    # -----------------------------

    B2 = B1.copy()

    # -----------------------------
    # Continue somewhere else
    # -----------------------------
    t3 = np.linspace(0, 1, 100)

    D = np.column_stack([
        30 + 10 * t3,
        -5 * np.sin(2 * np.pi * t3),
        2 * t3,
    ])

    trajectory = np.vstack([
        B1,
        C,
        B2,
        D,
    ])

    return trajectory


# ============================================================
# 2. Split trajectory into manageable segments
# ============================================================

def make_segments(trajectory, segment_size=100):
    """
    Split the synthetic trajectory into fixed-size segments.

    We deliberately use 100 points per segment so that:

        segment 0 = B1
        segment 1 = C
        segment 2 = B2
        segment 3 = D

    Therefore the known loop is:

        segment 0 <-> segment 2
    """

    segments = []

    for i in range(0, len(trajectory), segment_size):

        segment = trajectory[i:i + segment_size]

        if len(segment) == segment_size:
            segments.append(segment)

    return segments


# ============================================================
# 3. Main test
# ============================================================

def main():

    print()
    print("=" * 75)
    print("SYNTHETIC LOOP-CLOSURE TEST")
    print("=" * 75)
    print()

    # --------------------------------------------------------
    # Create trajectory
    # --------------------------------------------------------

    trajectory = make_trajectory()

    print("Trajectory shape:", trajectory.shape)

    # --------------------------------------------------------
    # Make segments
    # --------------------------------------------------------

    segments = make_segments(
        trajectory,
        segment_size=100,
    )

    print("Number of segments:", len(segments))
    print()

    # --------------------------------------------------------
    # Show what each segment represents
    # --------------------------------------------------------

    print("Expected structure:")
    print("  Segment 0 = first visit to B")
    print("  Segment 1 = different path C")
    print("  Segment 2 = second visit to B  <-- KNOWN LOOP")
    print("  Segment 3 = different path D")
    print()

    # --------------------------------------------------------
    # Create your ACTUAL LoopClosureDetector
    # --------------------------------------------------------

    detector = LoopClosureDetector(
        segment_length=5.0,
        fps=10.0,
        stride=1.0,
        lambdaa=0.5,
        downsample_points=None,
        reference_strategy="first",
    )

    # --------------------------------------------------------
    # Compute LPGW distance matrix
    #
    # We compare every segment against every other segment.
    # --------------------------------------------------------

    D = detector.compute_distance_matrix(
        segments,
        segments,
    )

    print("Distance matrix shape:", D.shape)
    print()

    # --------------------------------------------------------
    # Print the distance matrix
    # --------------------------------------------------------

    print("LPGW DISTANCE MATRIX")
    print("-" * 75)

    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    print(D)

    print()

    # --------------------------------------------------------
    # Known loop
    #
    # Segment 0 and Segment 2 are identical.
    # Therefore their distance should be very small.
    # --------------------------------------------------------

    loop_distance = float(D[0, 2])

    print("Known loop:")
    print("  Segment 0 <-> Segment 2")

    print(f"  Distance = {loop_distance:.10f}")

    print()

    # --------------------------------------------------------
    # Compare against unrelated segments
    # --------------------------------------------------------

    non_loop_distances = [
        float(D[0, 1]),
        float(D[0, 3]),
    ]

    print("Non-loop comparisons:")
    print(f"  Segment 0 <-> Segment 1 = {D[0, 1]:.10f}")
    print(f"  Segment 0 <-> Segment 3 = {D[0, 3]:.10f}")

    print()

    # --------------------------------------------------------
    # Test whether known loop is the closest match
    # --------------------------------------------------------

    row0 = D[0].copy()

    # Ignore self-match.
    row0[0] = np.inf

    best_match = int(np.argmin(row0))
    best_distance = float(row0[best_match])

    print("Best match for Segment 0:")
    print(f"  Segment {best_match}")
    print(f"  Distance = {best_distance:.10f}")

    print()

    # ========================================================
    # PASS / FAIL checks
    # ========================================================

    print("=" * 75)
    print("RESULTS")
    print("=" * 75)

    # Check 1: identical loop sections should be close.
    check_loop = loop_distance < 0.01

    print()
    print(
        "CHECK 1 - Known loop has small distance:",
        "PASS" if check_loop else "FAIL",
    )

    # Check 2: loop should be closer than unrelated segment 1.
    check_vs_1 = D[0, 2] < D[0, 1]

    print(
        "CHECK 2 - Loop closer than Segment 1:",
        "PASS" if check_vs_1 else "FAIL",
    )

    # Check 3: loop should be closer than unrelated segment 3.
    check_vs_3 = D[0, 2] < D[0, 3]

    print(
        "CHECK 3 - Loop closer than Segment 3:",
        "PASS" if check_vs_3 else "FAIL",
    )

    # Check 4: LPGW should identify segment 2 as best match.
    check_best = best_match == 2

    print(
        "CHECK 4 - Segment 2 is best match:",
        "PASS" if check_best else "FAIL",
    )

    print()

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    all_passed = (
        check_loop
        and check_vs_1
        and check_vs_3
        and check_best
    )

    print("=" * 75)

    if all_passed:
        print("OVERALL RESULT: PASS")
        print()
        print(
            "The LoopClosureDetector successfully detected "
            "the synthetic known loop."
        )
    else:
        print("OVERALL RESULT: FAIL")
        print()
        print(
            "The known loop was not the closest match."
        )

    print("=" * 75)
    print()


if __name__ == "__main__":
    main()
