import numpy as np

from lpgw import LPGW
from loop_closure import LoopClosureDetector


def make_path(n=40):
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack((
        20.0 * t,
        2.0 * np.sin(2.0 * np.pi * t),
        1.5 * np.cos(2.0 * np.pi * t),
    ))


def make_solver():
    return LPGW(
        lambdaa=0.5,
        num_itermax_gw=100,
        seed=42,
    )


def test_huber_distance_quadratic_and_linear_branches():
    delta = 0.15

    assert np.isclose(
        LPGW.huber_distance(np.zeros(3), np.array([0.1, 0.0, 0.0]), delta),
        0.5 * 0.1 ** 2,
    )
    assert np.isclose(
        LPGW.huber_distance(np.zeros(3), np.array([0.3, 0.0, 0.0]), delta),
        delta * (0.3 - 0.5 * delta),
    )


def test_huber_distance_matrix_is_symmetric_with_zero_diagonal():
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ])

    matrix = LPGW.huber_distance_matrix(points, delta=0.15)

    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)


def test_global_scale_preserves_relative_size():
    small = make_path()
    large = 2.0 * small
    solver = LPGW(global_scale=10.0)

    small_scaled = solver.normalize_geometry(small)
    large_scaled = solver.normalize_geometry(large)

    assert np.isclose(
        np.max(np.linalg.norm(small_scaled[:, None] - small_scaled[None, :], axis=2)),
        0.5 * np.max(np.linalg.norm(large_scaled[:, None] - large_scaled[None, :], axis=2)),
    )


def test_detector_calibrates_global_scale_from_all_segments():
    segments = [make_path(), 2.0 * make_path()]
    detector = LoopClosureDetector(
        downsample_points=None,
        reference_strategy="first",
        num_itermax_gw=20,
    )

    detector.compute_distance_matrix(segments, segments)

    diameters = [
        np.max(np.linalg.norm(seg[:, None] - seg[None, :], axis=2))
        for seg in segments + segments
    ]

    assert np.isclose(detector.global_scale, np.median(diameters))
    assert np.isclose(detector.lpgw.global_scale, detector.global_scale)


def test_embedding_vector_matches_fixed_weight_geometric_term():
    solver = make_solver()
    reference = make_path()
    embedding_1 = solver.embed(reference, reference[0:30])
    embedding_2 = solver.embed(reference, reference[5:35])

    vector_distance = np.sum(
        (
            solver.embedding_vector(embedding_1)
            - solver.embedding_vector(embedding_2)
        ) ** 2
    )
    p = solver.uniform_mass(embedding_1["K"].shape[0])
    geometric_term = float(
        p @ (embedding_1["K"] - embedding_2["K"]) ** 2 @ p
    )

    assert np.isclose(vector_distance, geometric_term)


def test_self_distance_zero():
    solver = make_solver()
    reference = make_path()
    embedding = solver.embed(reference, reference[5:35])

    assert solver.distance(embedding, embedding) < 1e-10


def test_distance_is_symmetric():
    solver = make_solver()
    reference = make_path()
    embedding_1 = solver.embed(reference, reference[0:30])
    embedding_2 = solver.embed(reference, reference[5:35])

    distance_12 = solver.distance(embedding_1, embedding_2)
    distance_21 = solver.distance(embedding_2, embedding_1)

    assert abs(distance_12 - distance_21) < 1e-10


def test_identical_shapes_translated_are_close():
    solver = make_solver()
    reference = make_path()
    translated = reference + np.array([100.0, 0.0, 0.0])
    embedding_1 = solver.embed(reference, reference)
    embedding_2 = solver.embed(reference, translated)

    assert solver.distance(embedding_1, embedding_2) < 1e-6
