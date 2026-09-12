import numpy as np
import pytest

from evaluation.eval_canonical import compute_bootstrap_ci


def test_bootstrap_ci_is_deterministic_and_bounded():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])

    first = compute_bootstrap_ci(
        y_true,
        y_pred,
        n_bootstraps=200,
        rng_seed=42,
    )
    second = compute_bootstrap_ci(
        y_true,
        y_pred,
        n_bootstraps=200,
        rng_seed=42,
    )

    assert first == second
    assert set(first) == {
        "precision_mean_ci",
        "recall_mean_ci",
        "f1_mean_ci",
    }

    for mean, lower, upper in first.values():
        assert 0.0 <= lower <= mean <= upper <= 1.0


def test_bootstrap_ci_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        compute_bootstrap_ci([], [], n_bootstraps=10)

    with pytest.raises(ValueError):
        compute_bootstrap_ci([1], [1, 0], n_bootstraps=10)

    with pytest.raises(ValueError):
        compute_bootstrap_ci([1], [1], n_bootstraps=0)

    with pytest.raises(ValueError):
        compute_bootstrap_ci([1], [1], n_bootstraps=10, ci=0)