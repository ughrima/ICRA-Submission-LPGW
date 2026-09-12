# evaluation/eval_canonical.py

import numpy as np


def score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    """
    Compute binary classification metrics.

    Returns:
        tp
        fp
        fn
        tn
        precision
        recall
        f1
    """

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if y_true.ndim != 1:
        raise ValueError(
            "y_true and y_pred must be 1D arrays."
        )

    # Confusion-matrix counts.
    tp = int(
        ((y_true == 1) & (y_pred == 1)).sum()
    )

    fp = int(
        ((y_true == 0) & (y_pred == 1)).sum()
    )

    fn = int(
        ((y_true == 1) & (y_pred == 0)).sum()
    )

    tn = int(
        ((y_true == 0) & (y_pred == 0)).sum()
    )

    # Precision.
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    # Recall.
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    # F1.
    if precision + recall > 0:
        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 95,
    rng_seed: int = 42,
):
    """Compute bootstrap confidence intervals for precision, recall, and F1.

    Each returned value is a ``(mean, lower, upper)`` tuple, where the bounds
    are percentile bootstrap confidence limits.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )
    if y_true.ndim != 1:
        raise ValueError(
            "y_true and y_pred must be 1D arrays."
        )
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be greater than zero.")
    if not 0 < ci <= 100:
        raise ValueError("ci must be greater than zero and at most 100.")

    rng = np.random.default_rng(rng_seed)
    precisions = []
    recalls = []
    f1s = []

    for _ in range(n_bootstraps):
        indices = rng.integers(0, len(y_true), len(y_true))
        metrics = score_predictions(
            y_true[indices],
            y_pred[indices],
        )
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])

    def get_bounds(values):
        lower_percentile = (100.0 - ci) / 2.0
        upper_percentile = 100.0 - lower_percentile
        return (
            float(np.mean(values)),
            float(np.percentile(values, lower_percentile)),
            float(np.percentile(values, upper_percentile)),
        )

    return {
        "precision_mean_ci": get_bounds(precisions),
        "recall_mean_ci": get_bounds(recalls),
        "f1_mean_ci": get_bounds(f1s),
    }