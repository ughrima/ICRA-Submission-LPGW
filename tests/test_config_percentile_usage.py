import numpy as np

from core.detection import select_adaptive_threshold
from loop_closure import LoopClosureDetector


def test_detector_percentile_threshold_matches_adaptive_helper():
    distances = np.array([
        [1.0, 100.0],
        [2.0, 100.0],
        [3.0, 100.0],
        [4.0, 100.0],
    ])

    expected = select_adaptive_threshold(distances, 68.0)
    actual = LoopClosureDetector.choose_threshold(
        distances,
        method="percentile",
        percentile=68.0,
    )

    assert actual == expected