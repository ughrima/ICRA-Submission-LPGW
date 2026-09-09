
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np
from loop_closure import LoopClosureDetector
from lpgw import LPGW

def make_shape_1(n=100):
    """Curved 3D trajectory."""
    t = np.linspace(0, 1, n)

    return np.column_stack([
        20 * t,
        2 * np.sin(2 * np.pi * t),
        1.5 * np.cos(2 * np.pi * t),
    ])


def make_unrelated_shape(n=100):
    """Very different curved trajectory."""
    t = np.linspace(0, 1, n)

    return np.column_stack([
        5 * np.sin(6 * np.pi * t),
        5 * np.cos(4 * np.pi * t),
        10 * t,
    ])


# ---------------------------------------------------------
# Two unrelated shapes
# ---------------------------------------------------------

X = make_shape_1()
Y = make_unrelated_shape()


# ---------------------------------------------------------
# Lambda values to test
# ---------------------------------------------------------

lambda_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]


print()
print("=" * 70)
print("LPGW LAMBDA SWEEP")
print("=" * 70)
print()
print(f"{'Lambda':>10} | {'Distance':>18} | {'Gamma mass':>15}")
print("-" * 70)


for lambda_value in lambda_values:

    # New LPGW solver with this Lambda.
    lpgw = LPGW(
        lambdaa=lambda_value,
        seed=42,
    )

    # -----------------------------------------------------
    # Compute embedding of X relative to Y
    # -----------------------------------------------------

    embedding = lpgw.embed(X, Y)

    # Transport plan.
    gamma = embedding["gamma"]

    # How much mass was actually transported?
    transported_mass = float(gamma.sum())

    # -----------------------------------------------------
    # Compute LPGW distance between X and Y
    # -----------------------------------------------------

    # We need a second embedding with the roles reversed
    # because lpgw.distance() compares two LPGW embeddings.
    embedding_reverse = lpgw.embed(Y, X)

    distance = lpgw.distance(
        embedding,
        embedding_reverse,
    )

    print(
        f"{lambda_value:10.3f} | "
        f"{distance:18.10f} | "
        f"{transported_mass:15.10f}"
    )


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)