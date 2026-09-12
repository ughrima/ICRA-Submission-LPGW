"""
loop_closure.py
---------------

Trajectory-specific wrapper around the mathematically faithful LPGW
core in lpgw.py.

This file deliberately separates:
    1. LPGW mathematics          -> lpgw.py
    2. trajectory segmentation  -> this file
    3. reference selection      -> this file
    4. thresholding/evaluation  -> this file

That makes the experimental pipeline easier to describe and reproduce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy import stats
from sklearn.mixture import GaussianMixture

from lpgw import LPGW


class LoopClosureDetector:
    """
    LPGW-based trajectory loop-closure detector.

    Parameters
    ----------
    segment_length:
        Segment duration in seconds.
    fps:
        Sampling frequency used by the trajectory files.
    stride:
        Segment stride in seconds.
    lambdaa:
        LPGW partial-matching penalty.
    downsample_points:
        If not None, each segment is uniformly downsampled to at most
        this many points before LPGW.
    reference_strategy:
        "robust", "first", "middle", or "last".
    """

    def __init__(
        self,
        segment_length: float = 5.0,
        fps: float = 10.0,
        stride: float = 1.0,
        lambdaa: float = 0.5,
        downsample_points: int | None = 100,
        reference_strategy: str = "median_diameter",
        reference_index: int | None = None,
        num_itermax_gw: int = 1000,
        tol: float = 1e-7,
        global_scale: float | None = None,
        scale_aware: bool = True,
        huber_delta: float = 0.15,
    ):
        self.segment_length = float(segment_length)
        self.fps = float(fps)
        self.stride = float(stride)

        self.downsample_points = downsample_points
        self.reference_strategy = reference_strategy
        self.reference_index = reference_index
        self._configured_global_scale = global_scale
        self.global_scale = global_scale
        self.scale_aware = bool(scale_aware)

        self.lpgw = LPGW(
            lambdaa=lambdaa,
            num_itermax_gw=num_itermax_gw,
            tol=tol,
            global_scale=global_scale,
            huber_delta=huber_delta,
        )
    

        self.reference_segment = None
        self.embeddings1 = None
        self.embeddings2 = None
        self.distance_matrix = None

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def segment_trajectory(
        trajectory: np.ndarray,
        segment_length: float = 5.0,
        fps: float = 10.0,
        stride: float = 1.0,
    ) -> list[np.ndarray]:
        trajectory = np.asarray(trajectory, dtype=np.float64)

        if trajectory.ndim != 2:
            raise ValueError("trajectory must have shape (N, D)")
        if len(trajectory) == 0:
            raise ValueError("trajectory cannot be empty")

        num_points = int(round(segment_length * fps))
        stride_points = int(round(stride * fps))

        if num_points <= 0 or stride_points <= 0:
            raise ValueError("segment_length and stride must be positive")

        segments = []

        for start in range(
            0,
            len(trajectory) - num_points + 1,
            stride_points,
        ):
            seg = trajectory[start:start + num_points]

            if len(seg) == num_points:
                segments.append(seg.copy())

        return segments

    @staticmethod
    def downsample_segment(
        segment: np.ndarray,
        target_points: int | None,
    ) -> np.ndarray:
        if target_points is None:
            return np.asarray(segment, dtype=np.float64)

        segment = np.asarray(segment, dtype=np.float64)

        if len(segment) <= target_points:
            return segment.copy()

        indices = np.linspace(
            0,
            len(segment) - 1,
            target_points,
        ).astype(int)

        return segment[indices]

    # ------------------------------------------------------------------
    # Reference selection
    # ------------------------------------------------------------------

    @staticmethod
    def select_reference(
        segments: list[np.ndarray],
        strategy: str = "robust",
    ) -> tuple[np.ndarray, int]:

        if not segments:
            raise ValueError("segments cannot be empty")

        strategy = strategy.lower()

        if strategy == "first":
            return segments[0], 0

        if strategy == "middle":
            idx = len(segments) // 2
            return segments[idx], idx

        if strategy == "last":
            return segments[-1], len(segments) - 1

        if strategy in ("robust", "median_diameter"):
            diameters = np.array(
                [
                    float(np.max(cdist(seg, seg)))
                    for seg in segments
                ],
                dtype=np.float64,
            )
            idx = int(
                np.argmin(
                    np.abs(diameters - np.median(diameters))
                )
            )
            return segments[idx], idx

        if strategy != "median_diameter":
            raise ValueError(
                "strategy must be one of: median_diameter, robust, "
                "first, middle, last"
            )

    # ------------------------------------------------------------------
    # Distance matrix
    # ------------------------------------------------------------------

    def compute_distance_matrix(
        self,
        segments1: list[np.ndarray],
        segments2: list[np.ndarray],
    ) -> np.ndarray:

        if not segments1 or not segments2:
            raise ValueError("Both segment lists must be non-empty")

        # The reference is chosen ONCE.
        #
        # If reference_index is supplied, use that exact segment.
        # Otherwise, use the existing reference-selection strategy.
        if self.reference_index is not None:
            if not 0 <= self.reference_index < len(segments2):
                raise IndexError(
                    f"reference_index={self.reference_index} is out of range "
                    f"for {len(segments2)} reference segments."
                )

            self.reference_segment = segments2[self.reference_index]
        else:
            self.reference_segment, self.reference_index = (
                self.select_reference(
                    segments2,
                    self.reference_strategy,
                )
            )

        # Downsampling is a computational approximation, not part of
        # the LPGW mathematics.
        processed1 = [
            self.downsample_segment(
                seg,
                self.downsample_points,
            )
            for seg in segments1
        ]

        processed2 = [
            self.downsample_segment(
                seg,
                self.downsample_points,
            )
            for seg in segments2
        ]

        if self.scale_aware:
            if self._configured_global_scale is None:
                all_segments = processed1 + processed2
                diameters = [
                    float(np.max(cdist(seg, seg, metric="euclidean")))
                    for seg in all_segments
                ]
                self.global_scale = float(np.median(diameters))
            else:
                self.global_scale = self._configured_global_scale

            if self.global_scale <= 0:
                raise ValueError("global scale must be positive")

            self.lpgw.global_scale = self.global_scale
            print(f"Global scale: {self.global_scale:.3f} m")
        else:
            self.global_scale = None
            self.lpgw.global_scale = None

        print(
            f"Reference segment: index={self.reference_index}, "
            f"points={len(self.reference_segment)}"
        )

        print("Computing LPGW embeddings for query segments...")
        self.embeddings1 = []

        for i, seg in enumerate(processed1):
            self.embeddings1.append(
                self.lpgw.embed(
                    self.reference_segment,
                    seg,
                )
            )

            if (i + 1) % 10 == 0 or i == len(processed1) - 1:
                print(
                    f"  {i + 1}/{len(processed1)}"
                )

        print("Computing LPGW embeddings for database segments...")
        self.embeddings2 = []

        for i, seg in enumerate(processed2):
            self.embeddings2.append(
                self.lpgw.embed(
                    self.reference_segment,
                    seg,
                )
            )

            if (i + 1) % 10 == 0 or i == len(processed2) - 1:
                print(
                    f"  {i + 1}/{len(processed2)}"
                )

        # Pairwise comparison is now cheap relative to solving PGW:
        # no new PGW problem is solved here.
        D = np.zeros(
            (len(self.embeddings1), len(self.embeddings2)),
            dtype=np.float64,
        )

        print("Computing LPGW discrepancy matrix...")

        for i, emb1 in enumerate(self.embeddings1):
            for j, emb2 in enumerate(self.embeddings2):
                D[i, j] = self.lpgw.distance(
                    emb1,
                    emb2,
                )

        self.distance_matrix = D

        return D

    def embedding_vectors(self, embeddings):
        """Return fixed-reference vectors for approximate retrieval."""
        return np.stack(
            [self.lpgw.embedding_vector(embedding) for embedding in embeddings]
        )

    def approximate_nearest_references(
        self,
        query_embeddings,
        reference_embeddings,
    ):
        """Retrieve candidates using the fixed-reference geometric vectors.

        This is an approximate retrieval path. Exact LPGW still includes
        pair-dependent partial-mass terms and should be used to rerank the
        returned candidates when those terms matter.
        """
        reference_vectors = self.embedding_vectors(reference_embeddings)
        query_vectors = self.embedding_vectors(query_embeddings)
        tree = cKDTree(reference_vectors)
        distances, indices = tree.query(query_vectors, k=1)
        return distances, indices

    # ------------------------------------------------------------------
    # Loop-closure detection
    # ------------------------------------------------------------------

    @staticmethod
    def choose_threshold(
        D: np.ndarray,
        method: str = "percentile",
        percentile: float = 1.0,
    ) -> float:

        D = np.asarray(D, dtype=np.float64)
        values = D[np.isfinite(D)]

        if len(values) == 0:
            raise ValueError("Distance matrix contains no finite values")

        method = method.lower()

        if method == "percentile":
            if not 0 <= percentile <= 100:
                raise ValueError(
                    "percentile must be between 0 and 100."
                )

            finite_rows = np.isfinite(D).any(axis=1)
            if not np.any(finite_rows):
                raise ValueError(
                    "Distance matrix contains no finite row scores"
                )

            best_per_query = np.min(
                np.where(np.isfinite(D[finite_rows]), D[finite_rows], np.inf),
                axis=1,
            )
            return float(np.percentile(best_per_query, percentile))

        if method == "gmm":
            if len(values) < 10:
                raise ValueError(
                    "GMM threshold requires more distance samples"
                )

            gmm = GaussianMixture(
                n_components=2,
                random_state=42,
            )
            gmm.fit(values.reshape(-1, 1))

            means = np.sort(gmm.means_.ravel())

            return float(np.mean(means))

        if method == "mad":
            median = float(np.median(values))
            mad = float(stats.median_abs_deviation(values))

            return float(median - 2.0 * mad)

        raise ValueError(
            "Unknown threshold method. "
            "Use 'percentile', 'gmm', or 'mad'."
        )

    @staticmethod
    def detect_loop_closures(
        D: np.ndarray,
        threshold: float,
    ) -> tuple[list[int], list[int], list[float]]:

        D = np.asarray(D, dtype=np.float64)

        flags = []
        matches = []
        min_distances = []

        for i in range(D.shape[0]):

            row = D[i]

            valid = np.isfinite(row)

            if not np.any(valid):
                flags.append(0)
                matches.append(-1)
                min_distances.append(np.inf)
                continue

            valid_indices = np.flatnonzero(valid)

            local_idx = int(
                np.argmin(row[valid])
            )

            best_idx = int(
                valid_indices[local_idx]
            )

            best_distance = float(
                row[best_idx]
            )

            flags.append(
                int(best_distance <= threshold)
            )

            matches.append(best_idx)
            min_distances.append(best_distance)

        return flags, matches, min_distances

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(
        ground_truth: np.ndarray,
        predictions: np.ndarray,
    ) -> dict:

        gt = np.asarray(ground_truth).astype(int)
        pred = np.asarray(predictions).astype(int)

        if gt.shape != pred.shape:
            raise ValueError(
                "ground_truth and predictions must have the same shape"
            )

        tp = int(np.sum((gt == 1) & (pred == 1)))
        fp = int(np.sum((gt == 0) & (pred == 1)))
        fn = int(np.sum((gt == 1) & (pred == 0)))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        f1 = (
            2 * precision * recall
            / max(precision + recall, 1e-12)
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }


def load_xyz_trajectory(csv_path: str):
    """
    Load trajectory CSV with:
        PosX, PosY, PosZ, Timestamp
    """
    df = pd.read_csv(csv_path)

    required = {
        "PosX",
        "PosY",
        "PosZ",
        "Timestamp",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"CSV is missing columns: {sorted(missing)}"
        )

    xyz = df[
        ["PosX", "PosY", "PosZ"]
    ].to_numpy(dtype=np.float64)

    timestamps = df["Timestamp"].to_numpy()

    return xyz, timestamps
