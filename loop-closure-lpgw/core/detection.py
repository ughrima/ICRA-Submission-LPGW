# core/detection.py

import numpy as np


def detect_with_percentile(
    D: np.ndarray,
    percentile: float,
):
    """
    Convert a pairwise distance matrix into binary loop-closure
    predictions using a global percentile threshold.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix of shape [Nq, Nr].

    percentile : float
        Percentile used to determine the threshold.
        Example: 1, 5, 10, 20, 50.

    Returns
    -------
    y_pred : np.ndarray
        Binary prediction for each query segment, shape [Nq].

    tau : float
        Distance threshold.
    """

    D = np.asarray(D, dtype=float)

    if D.ndim != 2:
        raise ValueError(
            f"D must be a 2D matrix, got shape {D.shape}"
        )

    if D.size == 0:
        raise ValueError(
            "Distance matrix D must not be empty."
        )

    if not np.all(np.isfinite(D)):
        raise ValueError(
            "Distance matrix contains NaN or infinite values."
        )

    if not 0 <= percentile <= 100:
        raise ValueError(
            "percentile must be between 0 and 100."
        )

    # Global threshold from all pairwise distances.
    tau = float(np.percentile(D, percentile))

    # Each query segment is represented by its best
    # (smallest) distance to any database segment.
    min_per_row = D.min(axis=1)

    # A loop is predicted when its best match is at
    # or below the threshold.
    y_pred = (
        min_per_row <= tau
    ).astype(int)

    return y_pred, tau