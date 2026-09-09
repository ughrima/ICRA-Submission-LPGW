from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
from loop_closure import LoopClosureDetector

import numpy as np

from loop_closure import LoopClosureDetector


def make_path(n=100):
    """
    Simple 3D trajectory for testing.
    """
    t = np.linspace(0, 1, n)

    return np.column_stack(
        (
            20 * t,
            2 * np.sin(2 * np.pi * t),
            1.5 * np.cos(2 * np.pi * t),
        )
    )


def main():

    # ---------------------------------------------------------
    # 1. Reference trajectory
    # ---------------------------------------------------------

    reference = make_path(100)

    # ---------------------------------------------------------
    # 2. LPGW detector
    # ---------------------------------------------------------

    detector = LoopClosureDetector(
        segment_length=5.0,
        fps=10.0,
        stride=1.0,
        lambdaa=0.5,
        downsample_points=None,
        reference_strategy="first",
    )

    # ---------------------------------------------------------
    # 3. Different overlap amounts
    # ---------------------------------------------------------

    overlaps = [100, 80, 60, 40, 20, 10]

    print()
    print("=" * 75)
    print("LPGW PARTIAL-OVERLAP TEST")
    print("=" * 75)
    print()

    print(
        f"{'Overlap':>10} | "
        f"{'Distance':>18} | "
        f"{'Gamma mass':>18}"
    )

    print("-" * 75)

    for overlap in overlaps:

        # Number of points included in the query
        n_overlap = int(100 * overlap / 100)

        # -----------------------------------------------------
        # Query = last part of the reference trajectory
        # -----------------------------------------------------

        query = reference[-n_overlap:].copy()

        # -----------------------------------------------------
        # Compute LPGW distance
        #
        # segments1 = query
        # segments2 = reference
        # -----------------------------------------------------

        D = detector.compute_distance_matrix(
            [query],
            [reference],
        )

        distance = float(D[0, 0])

        # -----------------------------------------------------
        # Get the PGW embedding that was just computed
        # -----------------------------------------------------

        embedding = detector.embeddings1[0]

        gamma = embedding["gamma"]

        # This is the amount of mass actually transported
        transported_mass = float(gamma.sum())

        print(
            f"{overlap:>9}% | "
            f"{distance:>18.10f} | "
            f"{transported_mass:>18.10f}"
        )

    print()
    print("=" * 75)
    print("Interpretation")
    print("=" * 75)
    print()
    print("Distance:")
    print("  Smaller = more similar")
    print()
    print("Gamma mass:")
    print("  ~1.0 = nearly all mass transported")
    print("  <1.0 = some mass was left unmatched")
    print()
    print("The important question:")
    print("Does gamma mass become smaller as overlap decreases?")
    print("=" * 75)


if __name__ == "__main__":
    main()
