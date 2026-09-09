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