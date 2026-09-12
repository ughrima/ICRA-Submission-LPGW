from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
from loop_closure import LoopClosureDetector


def make_path(n=100):
    t = np.linspace(0.0, 1.0, n)

    return np.column_stack((
        20.0 * t,
        2.0 * np.sin(2.0 * np.pi * t),
        1.5 * np.cos(2.0 * np.pi * t),
    ))


def rotate_z(points, degrees):
    theta = np.deg2rad(degrees)

    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])

    return points @ R.T


def get_distance(detector, reference, query):
    D = detector.compute_distance_matrix(
        [query],
        [reference],
    )

    return float(D[0, 0])


reference = make_path()
rng = np.random.default_rng(42)

detector = LoopClosureDetector(
    segment_length=5.0,
    fps=10.0,
    stride=1.0,
    lambdaa=0.5,
    downsample_points=None,
    reference_strategy="robust",
)

cases = [
    ("identical", reference.copy()),
    (
        "translated",
        reference + np.array([10.0, -7.0, 3.0]),
    ),
]

for angle in (15, 30, 45, 90):
    cases.append(
        (f"rotated_{angle}deg", rotate_z(reference, angle))
    )

cases.append(
    ("partial_middle", reference[30:70].copy())
)

for sigma in (0.05, 0.10, 0.25, 0.50, 1.00):
    noisy = reference + rng.normal(
        0.0,
        sigma,
        reference.shape,
    )

    cases.append(
        (f"noise_{sigma:.2f}m", noisy)
    )

t = np.linspace(0.0, 1.0, len(reference))

unrelated = np.column_stack((
    12.0 * np.cos(4.0 * np.pi * t) + 30.0,
    10.0 * np.sin(3.0 * np.pi * t) - 25.0,
    8.0 * t + 15.0,
))

cases.append(("unrelated", unrelated))

results = {}

print("case, distance")

for name, query in cases:
    value = get_distance(
        detector,
        reference,
        query,
    )

    results[name] = value
    print(f"{name}, {value:.10f}")

print("\nchecks")

print(
    "identical < unrelated:",
    results["identical"] < results["unrelated"],
)

print(
    "partial < unrelated:",
    results["partial_middle"] < results["unrelated"],
)

print(
    "0.05m noise < 1.0m noise:",
    results["noise_0.05m"] < results["noise_1.00m"],
)

rotation_values = [
    results[f"rotated_{angle}deg"]
    for angle in (15, 30, 45, 90)
]

print(
    "rotation range:",
    max(rotation_values) - min(rotation_values),
)
