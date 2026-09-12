import numpy as np

from experiments.run_baseline_comparison import (
    discrete_frechet_distance,
    dtw_distance,
)
from experiments.run_overlap_experiment import truncate_segment
from loop_closure import LoopClosureDetector


def make_line(length, offset=0.0):
    return np.column_stack((
        np.linspace(offset, offset + length, 8),
        np.zeros(8),
        np.zeros(8),
    ))


def test_median_diameter_reference_selection():
    segments = [
        make_line(1.0),
        make_line(3.0),
        make_line(10.0),
    ]

    selected, index = LoopClosureDetector.select_reference(
        segments,
        "median_diameter",
    )

    assert index == 1
    assert np.array_equal(selected, segments[1])


def test_contiguous_truncation_preserves_order():
    segment = np.arange(30).reshape(10, 3)
    truncated = truncate_segment(
        segment,
        ratio=0.4,
        rng=np.random.default_rng(42),
    )

    assert len(truncated) == 5
    assert np.all(np.diff(truncated[:, 0]) == 3)


def test_vectorized_curve_distances_match_identical_curves():
    curve = make_line(2.0)

    assert dtw_distance(curve, curve) == 0.0
    assert discrete_frechet_distance(curve, curve) == 0.0