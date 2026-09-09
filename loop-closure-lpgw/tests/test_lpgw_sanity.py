# tests/test_lpgw_sanity.py
"""
Synthetic sanity checks for the LPGW implementation.

Lower LPGW distance should indicate greater structural similarity.
"""

import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from lpgw_impl import ImprovedLPGWLoopClosure


def make_path(n=100):
    """A non-trivial 3D reference trajectory."""
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack((
        20.0 * t,
        2.0 * np.sin(2.0 * np.pi * t) + 0.5 * np.sin(6.0 * np.pi * t),
        1.5 * np.cos(2.0 * np.pi * t),
    ))


def rotate_z(points, degrees):
    theta = np.deg2rad(degrees)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    return points @ rotation.T


def lpgw_distance(reference, query):
    detector = ImprovedLPGWLoopClosure(
        segment_length=5.0,
        fps=10,
        stride=1.0,
        lambdaa=0.5,
        partial=True,
        downsampling=False,
    )

    return float(detector.compute_distance_matrix([query], [reference])[0, 0])


def main():
    rng = np.random.default_rng(42)
    reference = make_path()

    cases = [
        ("identical", reference.copy()),
        (
            "translated_(10,-7,3)m",
            reference + np.array([10.0, -7.0, 3.0]),
        ),
    ]

    for angle in (15, 30, 45, 90):
        cases.append((f"rotated_{angle}deg_z", rotate_z(reference, angle)))

    # Query is only a contiguous portion of the reference.
    cases.append(("partial_middle_40pct", reference[30:70].copy()))

    for sigma in (0.05, 0.10, 0.25, 0.50, 1.00):
        noisy_query = reference + rng.normal(
            loc=0.0,
            scale=sigma,
            size=reference.shape,
        )
        cases.append((f"gaussian_noise_{sigma:.2f}m", noisy_query))

    # A shape that is structurally unrelated to the reference path.
    t = np.linspace(0.0, 1.0, len(reference))
    unrelated = np.column_stack((
        12.0 * np.cos(4.0 * np.pi * t) + 30.0,
        10.0 * np.sin(3.0 * np.pi * t) - 25.0,
        8.0 * t + 15.0,
    ))
    cases.append(("unrelated", unrelated))

    print("Synthetic LPGW sanity check")
    print("Reference: 100-point curved 3D trajectory")
    print("Partial query: points 30–69 of the reference")
    print("Lower distance means greater structural similarity.\n")

    print(f"{'Case':<30} {'LPGW distance':>16}")
    print("-" * 48)

    results = {}

    for name, query in cases:
        distance = lpgw_distance(reference, query)
        results[name] = distance
        print(f"{name:<30} {distance:>16.8f}")

    print("\nBasic diagnostic checks:")
    print(
        "Identical < unrelated: "
        f"{results['identical'] < results['unrelated']}"
    )
    print(
        "Partial < unrelated:   "
        f"{results['partial_middle_40pct'] < results['unrelated']}"
    )
    print(
        "0.05 m noise < 1.0 m noise: "
        f"{results['gaussian_noise_0.05m'] < results['gaussian_noise_1.00m']}"
    )


if __name__ == "__main__":
    main()
