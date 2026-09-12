import numpy as np

from core.detection import (
    detect_with_percentile,
    select_adaptive_threshold,
)


def test_adaptive_threshold_uses_row_minima():
    distances = np.array([
        [1.0, 100.0],
        [2.0, 100.0],
        [3.0, 100.0],
        [4.0, 100.0],
    ])

    assert select_adaptive_threshold(distances, 50.0) == 2.5


def test_detection_uses_adaptive_threshold_scores():
    distances = np.array([
        [1.0, 100.0],
        [2.0, 100.0],
        [3.0, 100.0],
        [4.0, 100.0],
    ])

    predictions, tau = detect_with_percentile(distances, 50.0)

    assert tau == 2.5
    assert np.array_equal(predictions, np.array([1, 1, 0, 0]))