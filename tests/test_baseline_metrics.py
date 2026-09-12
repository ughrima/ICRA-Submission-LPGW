import numpy as np

from experiments.run_baseline_comparison import evaluate_distance_matrix


def test_threshold_free_metrics_and_recall_at_one():
    distances = np.array([
        [0.1, 0.9, 1.0],
        [0.8, 0.2, 0.9],
        [0.7, 0.6, 0.3],
    ])
    y_true = np.array([1, 1, 0])
    gt_nearest_ref = np.array([0, 1, 2])
    ref_indices = np.array([0, 1, 2])

    metrics = evaluate_distance_matrix(
        distances,
        y_true,
        gt_nearest_ref,
        ref_indices,
    )

    assert metrics["max_f1"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["recall_at_100_precision"] == 1.0
    assert metrics["recall_at_1"] == 1.0


def test_recall_at_one_uses_original_reference_indices():
    distances = np.array([
        [0.1, 0.9],
        [0.2, 0.8],
    ])
    y_true = np.array([1, 0])
    gt_nearest_ref = np.array([4, 0])
    ref_indices = np.array([4, 9])

    metrics = evaluate_distance_matrix(
        distances,
        y_true,
        gt_nearest_ref,
        ref_indices,
    )

    assert metrics["recall_at_1"] == 1.0