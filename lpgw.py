"""
lpgw.py
-------

Mathematically faithful LPGW core for trajectory loop-closure experiments.

Reference:
    Y. Bai et al.,
    "Linear Partial Gromov-Wasserstein Embedding", ICLR 2025.

This module implements the discrete LPGW embedding described by
Eqs. (13), (23), (24), and the numerical discrepancy in Eq. (25).

IMPORTANT
---------
The PGW solver used here is the Lambda-dependent solver from the
authors' official LPGW repository:

https://github.com/mint-vu/Linearized_Partial_Gromov_Wasserstein

Put that repository on PYTHONPATH (or install/copy its `lib/` package)
so that:

    from lib.gromov import partial_gromov_ver1

works.

Do NOT replace this solver with POT's
`ot.gromov.partial_gromov_wasserstein(..., m=1, ...)` if the goal is
the paper's Lambda-penalized LPGW formulation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

try:
    from lib.gromov import partial_gromov_ver1
except ImportError as exc:
    raise ImportError(
        "\nCould not import the official LPGW PGW solver.\n\n"
        "Install/clone the authors' reference repository and make its "
        "`lib/` directory importable:\n"
        "https://github.com/mint-vu/Linearized_Partial_Gromov_Wasserstein\n\n"
        "For example, if the repository is next to this project:\n"
        "    export PYTHONPATH=/path/to/Linearized_Partial_Gromov_Wasserstein:$PYTHONPATH\n"
    ) from exc


class LPGW:
    """
    Linear Partial Gromov-Wasserstein embedding.

    Each target Y is embedded relative to one fixed reference X as:

        Y -> (K_e, q_e, |gamma_c|)

    where
        gamma  : Lambda-dependent optimal PGW plan
        q_e    : gamma_X = gamma @ 1
        K_e    : projected target metric - reference metric
        |gamma_c| : unmatched/creation mass term

    The pairwise discrepancy follows Eq. (25) of the paper.
    """

    def __init__(
        self,
        lambdaa: float = 0.5,
        num_itermax_gw: int = 1000,
        num_itermax: int | None = None,
        tol: float = 1e-7,
        line_search: bool = True,
        seed: int = 0,
        global_scale: float | None = None,
        huber_delta: float = 0.15,
    ):
        self.lambdaa = float(lambdaa)
        self.num_itermax_gw = int(num_itermax_gw)
        self.num_itermax = num_itermax
        self.tol = float(tol)
        self.line_search = bool(line_search)
        self.seed = int(seed)
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        self.huber_delta = float(huber_delta)
        if global_scale is not None and global_scale <= 0:
            raise ValueError("global_scale must be positive")
        self.global_scale = global_scale

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def geometry_scale(self, points: np.ndarray) -> float:
        """Return the physical scale used to normalize a point cloud."""
        X = np.asarray(points, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("points must have shape (N, D)")
        if X.shape[0] == 0:
            raise ValueError("points cannot be empty")

        if self.global_scale is not None:
            return float(self.global_scale)

        return float(np.max(cdist(X, X, metric="euclidean")))

    def normalize_geometry(self, points: np.ndarray) -> np.ndarray:
        """
        Scale geometry using a shared dataset scale when configured.

        Without a shared scale, use the point cloud's diameter for the
        legacy scale-invariant behavior.

        This is preferable here to independently applying StandardScaler
        to x/y/z because LPGW compares intrinsic pairwise geometry.
        """
        X = np.asarray(points, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("points must have shape (N, D)")
        if X.shape[0] == 0:
            raise ValueError("points cannot be empty")

        scale = self.geometry_scale(X)

        if scale <= 1e-15:
            return np.zeros_like(X)

        return X / scale

    @staticmethod
    def huber_distance(
        p1: np.ndarray,
        p2: np.ndarray,
        delta: float = 0.15,
    ) -> float:
        """Return the Huber loss of the Euclidean point residual."""
        if delta <= 0:
            raise ValueError("delta must be positive")

        residual = float(np.linalg.norm(
            np.asarray(p1, dtype=np.float64)
            - np.asarray(p2, dtype=np.float64)
        ))

        if residual <= delta:
            return 0.5 * residual ** 2
        return delta * (residual - 0.5 * delta)

    @classmethod
    def huber_distance_matrix(
        cls,
        points: np.ndarray,
        delta: float = 0.15,
    ) -> np.ndarray:
        """Return the pairwise Huber loss matrix for a point cloud."""
        if delta <= 0:
            raise ValueError("delta must be positive")

        residuals = cdist(points, points, metric="euclidean")
        return np.where(
            residuals <= delta,
            0.5 * residuals ** 2,
            delta * (residuals - 0.5 * delta),
        )

    def squared_distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """Compatibility wrapper for the LPGW Huber cost matrix."""
        return self.huber_distance_matrix(
            points,
            delta=self.huber_delta,
        )

    @staticmethod
    def uniform_mass(n: int) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be positive")
        return np.full(n, 1.0 / n, dtype=np.float64)

    @staticmethod
    def _normalize_mass(mass: np.ndarray, n: int) -> np.ndarray:
        if mass is None:
            return LPGW.uniform_mass(n)

        mass = np.asarray(mass, dtype=np.float64).reshape(-1)

        if len(mass) != n:
            raise ValueError(
                f"Mass vector has length {len(mass)} but expected {n}"
            )
        if np.any(mass < 0):
            raise ValueError("Masses must be non-negative")

        total = mass.sum()
        if total <= 0:
            raise ValueError("Mass vector must have positive total mass")

        return mass / total

    # ------------------------------------------------------------------
    # PGW
    # ------------------------------------------------------------------

    def solve_pgw(
        self,
        Cx: np.ndarray,
        Cy: np.ndarray,
        p: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        """
        Solve the Lambda-dependent PGW problem using the solver supplied
        by the authors' LPGW repository.

        NOTE:
        `partial_gromov_ver1` is not the same thing as simply fixing
        POT's `m=1`. The Lambda term changes the optimization so the
        resulting plan can carry less than the full mass.
        """
        gamma = partial_gromov_ver1(
            Cx,
            Cy,
            p,
            q,
            Lambda=self.lambdaa,
            numItermax_gw=self.num_itermax_gw,
            numItermax=self.num_itermax,
            tol=self.tol,
            log=False,
            verbose=False,
            line_search=self.line_search,
            seed=self.seed,
        )

        gamma = np.asarray(gamma, dtype=np.float64)

        expected_shape = (len(p), len(q))
        if gamma.shape != expected_shape:
            raise RuntimeError(
                f"PGW returned {gamma.shape}; expected {expected_shape}"
            )

        if not np.all(np.isfinite(gamma)):
            raise RuntimeError("PGW returned non-finite values")

        gamma[gamma < 0] = 0.0
        return gamma

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        p: np.ndarray | None = None,
        q: np.ndarray | None = None,
    ) -> dict:
        """
        Embed Y relative to reference X.

        Returns a dictionary containing:
            K                  Eq. (24)
            q_e                transported source marginal
            gamma_c_abs        |gamma_c|
            gamma              PGW transport plan
            Y_projected        barycentric projection
            transported_mass   |gamma_X|
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must have shape (N, D)")
        if len(X) == 0 or len(Y) == 0:
            raise ValueError("X and Y cannot be empty")

        # 1. Metric-space normalization.
        Xn = self.normalize_geometry(X)
        Yn = self.normalize_geometry(Y)

        # 2. Intrinsic squared-distance matrices.
        x_scale = self.geometry_scale(X)
        y_scale = self.geometry_scale(Y)
        Cx = self.huber_distance_matrix(
            Xn,
            delta=self.huber_delta / max(x_scale, 1e-15),
        )
        Cy = self.huber_distance_matrix(
            Yn,
            delta=self.huber_delta / max(y_scale, 1e-15),
        )

        # 3. Measures.
        p = self._normalize_mass(p, len(Xn))
        q = self._normalize_mass(q, len(Yn))

        # 4. Lambda-dependent partial GW.
        gamma = self.solve_pgw(Cx, Cy, p, q)

        # 5. Source marginal:
        #       q_e = gamma_X = gamma @ 1
        q_e = gamma.sum(axis=1)

        # 6. Barycentric projection:
        #       y_e_i = (1/q_e_i) sum_j gamma_ij y_j
        Y_projected = np.zeros_like(Xn)
        valid = q_e > 1e-15

        if np.any(valid):
            Y_projected[valid] = (
                gamma[valid] @ Yn
            ) / q_e[valid, None]

        # 7. Eq. (24):
        #       K_e[i,j] =
        #       ||y_e_i-y_e_j||^2 - ||x_i-x_j||^2
        Cye = self.huber_distance_matrix(
            Y_projected,
            delta=self.huber_delta / max(y_scale, 1e-15),
        )
        K = Cye - Cx

        # 8. Creation/unmatched mass term:
        #
        #     |gamma_c| = |mu|^2 - |gamma_X|^2
        #
        # for the discrete representation used in the paper.
        source_total = float(p.sum())
        transported_mass = float(q_e.sum())

        gamma_c_abs = max(
            0.0,
            source_total ** 2 - transported_mass ** 2,
        )

        return {
            "K": K,
            "q_e": q_e,
            "gamma_c_abs": gamma_c_abs,
            "gamma": gamma,
            "Y_projected": Y_projected,
            "transported_mass": transported_mass,
        }

    # ------------------------------------------------------------------
    # Eq. (25)
    # ------------------------------------------------------------------

    def distance(self, emb1, emb2):
        K1 = emb1["K"]
        K2 = emb2["K"]

        q1 = emb1["q_e"]
        q2 = emb2["q_e"]

        # Common transported mass
        q12 = np.minimum(q1, q2)
        total_q12 = float(q12.sum())

        # Linearized geometric term
        Kdiff_sq = (K1 - K2) ** 2

        raw_geometric_term = float(
            q12 @ Kdiff_sq @ q12
        )

        # Normalize by common mass squared to keep the geometric term
        # invariant to the total transported point mass.
        if total_q12 > 1e-12:
            geometric_term = raw_geometric_term / (total_q12 ** 2)
        else:
            geometric_term = raw_geometric_term

        # Penalty for different transported masses
        penalty_1 = self.lambdaa * (
            float(q1.sum()) ** 2
            + float(q2.sum()) ** 2
            - 2.0 * (total_q12 ** 2)
        )

        # Compare discarded mass relative to the shared amount. Using the
        # raw sum would add a per-embedding offset and make d(emb, emb) > 0.
        gamma_c_1 = float(emb1["gamma_c_abs"])
        gamma_c_2 = float(emb2["gamma_c_abs"])
        penalty_2 = self.lambdaa * abs(gamma_c_1 - gamma_c_2)

        return max(
            0.0,
            geometric_term + penalty_1 + penalty_2
        )

    @staticmethod
    def embedding_vector(emb, reference_mass=None) -> np.ndarray:
        """Flatten a fixed-reference geometric embedding for approximate search.

        This vector reproduces the geometric term when both embeddings use
        the same fixed reference marginal. It intentionally omits the
        pair-dependent partial-mass terms from :meth:`distance`, so it is an
        approximate retrieval representation rather than the exact LPGW
        discrepancy.
        """
        K = np.asarray(emb["K"], dtype=np.float64)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("Embedding K must be a square matrix")

        if reference_mass is None:
            reference_mass = LPGW.uniform_mass(K.shape[0])
        reference_mass = LPGW._normalize_mass(
            reference_mass,
            K.shape[0],
        )
        weights = np.sqrt(
            np.outer(reference_mass, reference_mass)
        )
        return (weights * K).ravel()
    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def pairwise_distance(
        self,
        reference: np.ndarray,
        Y1: np.ndarray,
        Y2: np.ndarray,
        p=None,
        q1=None,
        q2=None,
    ) -> float:
        emb1 = self.embed(reference, Y1, p=p, q=q1)
        emb2 = self.embed(reference, Y2, p=p, q=q2)
        return self.distance(emb1, emb2)

    def embed_many(
        self,
        reference: np.ndarray,
        segments: list[np.ndarray],
    ) -> list[dict]:
        """
        Compute LPGW embeddings for all segments relative to one fixed
        reference.
        """
        return [self.embed(reference, seg) for seg in segments]
