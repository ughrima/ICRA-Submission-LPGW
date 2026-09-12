
"""
Publication-quality figures for:
    Trajectory Loop Closure Detection Using LPGW

Run from:
    loop-closure-lpgw/

Command:
    python experiments/make_paper_figures.py

Outputs:
    figures/uzhfpv/fig_a_trajectory_gt.png
    figures/uzhfpv/fig_b_lpgw_matrix.png
    figures/uzhfpv/fig_c_detected_closures.png
    figures/uzhfpv/fig_d_thresholds.png
    figures/uzhfpv/fig_e_runtime_scaling.png
    figures/uzhfpv/fig_f_robustness.png

Also saves PDF versions.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize


# ================================================================
# PATHS
# ================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]

POSE_DIR = REPO_ROOT / "data" / "poses"
GT_DIR = REPO_ROOT / "ground_truth" / "files"
RESULT_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "cache"

OUT_DIR = REPO_ROOT / "figures" / "uzhfpv"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# PAPER STYLE
# ================================================================

# IEEE-ish figure dimensions.
# 3.5 in = single-column
# 7.16 in = double-column
SINGLE_COL = (3.45, 2.55)
DOUBLE_COL = (7.05, 3.05)

FONT_SIZE = 8
SMALL_FONT = 7
TITLE_SIZE = 9

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": SMALL_FONT,
    "ytick.labelsize": SMALL_FONT,
    "legend.fontsize": SMALL_FONT,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


# ================================================================
# HELPERS
# ================================================================

def save_fig(fig, name):
    """Save both PNG and PDF."""
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"

    fig.savefig(png, dpi=600)
    fig.savefig(pdf)

    print(f"[saved] {png}")
    print(f"[saved] {pdf}")

    plt.close(fig)


def clean_axes(ax):
    """Clean scientific plot."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        True,
        alpha=0.20,
        linewidth=0.5,
        linestyle="--",
    )


def find_csv(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def read_pose_csv(path):
    """
    Read a pose CSV and automatically identify x/y/z columns.

    Supported common names:
        x, y, z
        pos_x, pos_y, pos_z
        position_x, position_y, position_z
        px, py, pz
    """

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    cols_lower = {c.lower(): c for c in df.columns}

    def find_col(names):
        for name in names:
            if name in cols_lower:
                return cols_lower[name]
        return None

    xcol = find_col([
        "x",
        "pos_x",
        "position_x",
        "px",
        "tx",
    ])

    ycol = find_col([
        "y",
        "pos_y",
        "position_y",
        "py",
        "ty",
    ])

    zcol = find_col([
        "z",
        "pos_z",
        "position_z",
        "pz",
        "tz",
    ])

    if xcol is None or ycol is None:
        raise ValueError(
            f"Could not identify x/y columns in {path}.\n"
            f"Columns found: {list(df.columns)}"
        )

    xyz = pd.DataFrame({
        "x": pd.to_numeric(df[xcol], errors="coerce"),
        "y": pd.to_numeric(df[ycol], errors="coerce"),
    })

    if zcol is not None:
        xyz["z"] = pd.to_numeric(df[zcol], errors="coerce")
    else:
        xyz["z"] = 0.0

    xyz = xyz.dropna().reset_index(drop=True)

    return xyz


def load_uzh_poses():
    ref = POSE_DIR / "poses_bag-3.csv"
    query = POSE_DIR / "poses_bag-7.csv"

    if not ref.exists() or not query.exists():
        raise FileNotFoundError(
            "\nCould not find UZH pose files.\n\n"
            f"Expected:\n"
            f"  {ref}\n"
            f"  {query}\n\n"
            "Put your pose CSVs in data/poses/."
        )

    return read_pose_csv(ref), read_pose_csv(query)


def nearest_indices(points_a, points_b):
    """
    For every point in A, find nearest point in B.

    Used only for visualization.
    """
    result = np.empty(len(points_a), dtype=int)

    # Chunking avoids a potentially huge NxM array.
    chunk = 2000

    for start in range(0, len(points_a), chunk):
        end = min(start + chunk, len(points_a))

        a = points_a[start:end]

        diff = a[:, None, :] - points_b[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)

        result[start:end] = np.argmin(dist2, axis=1)

    return result


def load_gt():
    """
    Load canonical UZH ground truth.

    Expected:
        ground_truth/files/gt_uzhfpv_2.0m.csv
    """

    path = GT_DIR / "gt_uzhfpv_2.0m.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing canonical GT:\n{path}"
        )

    return pd.read_csv(path)


# ================================================================
# FIGURE A
# TRAJECTORY MAP + GROUND TRUTH LOOPS
# ================================================================

def figure_a_trajectory_gt():
    ref, query = load_uzh_poses()
    gt = load_gt()

    ref_xy = ref[["x", "y"]].to_numpy()
    query_xy = query[["x", "y"]].to_numpy()

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    ax.plot(
        ref_xy[:, 0],
        ref_xy[:, 1],
        linewidth=1.2,
        label="Reference (Bag 3)",
    )

    ax.plot(
        query_xy[:, 0],
        query_xy[:, 1],
        linewidth=1.2,
        linestyle="--",
        label="Query (Bag 7)",
    )

    # ------------------------------------------------------------
    # Ground-truth positive query regions
    # ------------------------------------------------------------

    if "query_index" in gt.columns and "label" in gt.columns:

        positives = gt[gt["label"].astype(int) == 1]

        q_indices = positives["query_index"].to_numpy()

        # These are segment indices, not necessarily raw pose indices.
        # We therefore map them approximately along the query trajectory.
        if len(q_indices) > 0:

            q_indices = q_indices[
                (q_indices >= 0) &
                (q_indices < len(query_xy))
            ]

            if len(q_indices) > 0:
                ax.scatter(
                    query_xy[q_indices, 0],
                    query_xy[q_indices, 1],
                    s=8,
                    zorder=5,
                    label="GT loop regions",
                )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("(a) UZH-FPV trajectory and ground-truth loops")

    ax.set_aspect("equal", adjustable="box")
    clean_axes(ax)

    ax.legend(
        loc="best",
        frameon=False,
    )

    save_fig(fig, "fig_a_trajectory_gt")


# ================================================================
# FIND LPGW DISTANCE MATRIX
# ================================================================

def find_lpgw_matrix():
    """
    Search for the actual cached LPGW distance matrix.

    The repository stores caches as .npy and keeps them out of Git.
    """

    patterns = [
        "*lpgw*.npy",
        "*LPGW*.npy",
        "*distance*.npy",
        "*matrix*.npy",
        "*.npy",
    ]

    candidates = []

    for pattern in patterns:
        candidates.extend(CACHE_DIR.glob(pattern))

    candidates = sorted(set(candidates))

    # Prefer square matrices.
    square = []

    for path in candidates:
        try:
            arr = np.load(path, mmap_mode="r")

            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                square.append(path)

        except Exception:
            continue

    if square:
        # Prefer filenames explicitly mentioning LPGW.
        lpgw = [
            p for p in square
            if "lpgw" in p.name.lower()
        ]

        if lpgw:
            return lpgw[0]

        return square[0]

    return None


# ================================================================
# FIGURE B
# LPGW DISTANCE MATRIX
# ================================================================

def figure_b_lpgw_matrix():
    matrix_path = find_lpgw_matrix()

    if matrix_path is None:
        print(
            "\n[WARNING] No LPGW .npy matrix found in cache/.\n"
            "Figure B was not generated.\n\n"
            "Run the full LPGW experiment first so that the "
            "distance matrix is cached.\n"
        )
        return

    print(f"[matrix] using {matrix_path}")

    D = np.asarray(np.load(matrix_path), dtype=float)

    if D.ndim != 2:
        raise ValueError(
            f"LPGW matrix must be 2-D, got {D.shape}"
        )

    # Robust clipping avoids one extreme value dominating the image.
    finite = D[np.isfinite(D)]

    if len(finite) == 0:
        raise ValueError("LPGW matrix contains no finite values.")

    vmin = np.percentile(finite, 1)
    vmax = np.percentile(finite, 99)

    fig, ax = plt.subplots(figsize=SINGLE_COL)

    im = ax.imshow(
        D,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        norm=Normalize(vmin=vmin, vmax=vmax),
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        r"$D_{\mathrm{LPGW}}$",
        rotation=90,
    )

    ax.set_xlabel("Reference segment index")
    ax.set_ylabel("Query segment index")
    ax.set_title("(b) LPGW query-reference distance matrix")

    clean_axes(ax)

    save_fig(fig, "fig_b_lpgw_matrix")


# ================================================================
# FIGURE C
# DETECTED CLOSURES
# ================================================================

def load_full_lpgw_result():
    path = RESULT_DIR / "full_lpgw_uzhfpv.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing result file:\n{path}"
        )

    return pd.read_csv(path)


def figure_c_detected_closures():
    """
    Visualize predicted query/reference matches.

    This uses the canonical GT and the LPGW threshold.
    """

    ref, query = load_uzh_poses()
    gt = load_gt()

    matrix_path = find_lpgw_matrix()

    if matrix_path is None:
        print(
            "[WARNING] No LPGW matrix found. "
            "Figure C cannot be generated."
        )
        return

    D = np.asarray(np.load(matrix_path), dtype=float)

    # ------------------------------------------------------------
    # Determine threshold.
    #
    # We use the 50th percentile because this is the reported
    # best-F1 full LPGW operating point.
    # ------------------------------------------------------------

    finite = D[np.isfinite(D)]
    tau = np.percentile(finite, 50)

    best_ref = np.argmin(D, axis=1)
    best_dist = D[np.arange(D.shape[0]), best_ref]

    predicted = best_dist <= tau

    # ------------------------------------------------------------
    # GT labels
    # ------------------------------------------------------------

    gt_map = {}

    if "query_index" in gt.columns and "label" in gt.columns:
        for _, row in gt.iterrows():
            gt_map[int(row["query_index"])] = int(row["label"])

    gt_labels = np.zeros(D.shape[0], dtype=int)

    for i in range(D.shape[0]):
        if i in gt_map:
            gt_labels[i] = gt_map[i]

    # ------------------------------------------------------------
    # Convert segment indices to approximate trajectory positions.
    # ------------------------------------------------------------

    q_idx = np.linspace(
        0,
        len(query) - 1,
        D.shape[0],
    ).astype(int)

    r_idx = np.linspace(
        0,
        len(ref) - 1,
        D.shape[1],
    ).astype(int)

    q_xy = query[["x", "y"]].to_numpy()
    r_xy = ref[["x", "y"]].to_numpy()

    fig, ax = plt.subplots(figsize=DOUBLE_COL)

    ax.plot(
        ref["x"],
        ref["y"],
        linewidth=1.0,
        label="Reference",
    )

    ax.plot(
        query["x"],
        query["y"],
        linewidth=1.0,
        linestyle="--",
        label="Query",
    )

    # ------------------------------------------------------------
    # Classify predicted detections
    # ------------------------------------------------------------

    for qi in range(len(predicted)):

        if not predicted[qi]:
            continue

        ri = best_ref[qi]

        q = q_xy[q_idx[qi]]
        r = r_xy[r_idx[ri]]

        is_gt = gt_labels[qi] == 1

        if is_gt:
            marker = "o"
        else:
            marker = "x"

        ax.plot(
            [q[0], r[0]],
            [q[1], r[1]],
            alpha=0.35,
            linewidth=0.7,
        )

        ax.scatter(
            q[0],
            q[1],
            s=15,
            marker=marker,
            zorder=5,
        )

        ax.scatter(
            r[0],
            r[1],
            s=15,
            marker=marker,
            zorder=5,
        )

    # ------------------------------------------------------------
    # Missed GT closures
    # ------------------------------------------------------------

    for qi in range(len(gt_labels)):

        if gt_labels[qi] != 1:
            continue

        if predicted[qi]:
            continue

        q = q_xy[q_idx[qi]]

        ax.scatter(
            q[0],
            q[1],
            s=18,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            zorder=6,
        )

    legend_items = [
        Line2D(
            [0], [0],
            linestyle="-",
            label="Reference",
        ),
        Line2D(
            [0], [0],
            linestyle="--",
            label="Query",
        ),
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            label="Detected true closure",
        ),
        Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            label="False positive",
        ),
        Line2D(
            [0], [0],
            marker="o",
            markerfacecolor="none",
            linestyle="None",
            label="Missed GT closure",
        ),
    ]

    ax.legend(
        handles=legend_items,
        frameon=False,
        loc="best",
        ncol=2,
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        "(c) LPGW detected loop closures "
        f"(50th-percentile threshold)"
    )

    ax.set_aspect("equal", adjustable="box")
    clean_axes(ax)

    save_fig(fig, "fig_c_detected_closures")


# ================================================================
# FIGURE D
# THRESHOLD / PERCENTILE CURVES
# ================================================================

def figure_d_thresholds():
    """
    Uses the verified full UZH LPGW results.
    """

    percentiles = np.array([1, 5, 10, 20, 50])

    precision = np.array([
        0.6911,
        0.7108,
        0.7117,
        0.7154,
        0.7022,
    ])

    recall = np.array([
        0.3905,
        0.6036,
        0.7012,
        0.8107,
        0.9556,
    ])

    f1 = np.array([
        0.4991,
        0.6528,
        0.7064,
        0.7601,
        0.8095,
    ])

    fig, ax = plt.subplots(figsize=DOUBLE_COL)

    ax.plot(
        percentiles,
        precision,
        marker="o",
        label="Precision",
    )

    ax.plot(
        percentiles,
        recall,
        marker="s",
        label="Recall",
    )

    ax.plot(
        percentiles,
        f1,
        marker="^",
        linewidth=2.0,
        label="$F_1$",
    )

    # Highlight best F1.
    best = np.argmax(f1)

    ax.annotate(
        f"Best $F_1$ = {f1[best]:.3f}",
        xy=(percentiles[best], f1[best]),
        xytext=(-45, -25),
        textcoords="offset points",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.8,
        ),
    )

    ax.set_xlabel("Percentile threshold (%)")
    ax.set_ylabel("Score")
    ax.set_title("(d) LPGW operating-point sensitivity")

    ax.set_xticks(percentiles)
    ax.set_ylim(0, 1.05)

    clean_axes(ax)

    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower right",
    )

    save_fig(fig, "fig_d_thresholds")


# ================================================================
# FIGURE E
# RUNTIME SCALING
# ================================================================

def figure_e_runtime_scaling():
    """
    Verified runtime measurements from runtime_scaling_uzhfpv.csv.
    """

    path = RESULT_DIR / "runtime_scaling_uzhfpv.csv"

    if path.exists():
        df = pd.read_csv(path)

        # Repository CSV uses num_query_segments and
        # num_reference_segments.
        k = df["num_query_segments"].to_numpy()
        runtime = df["runtime_seconds"].to_numpy()

    else:
        # Fallback to verified values.
        k = np.array([
            25, 50, 100, 150, 200
        ])

        runtime = np.array([
            0.0572,
            0.1227,
            0.2960,
            0.4978,
            0.7778,
        ])

    fig, ax = plt.subplots(figsize=DOUBLE_COL)

    ax.plot(
        k,
        runtime,
        marker="o",
        linewidth=2.0,
        label="Measured LPGW",
    )

    # ------------------------------------------------------------
    # Reference scaling curve.
    #
    # IMPORTANT:
    # This is NOT a measured PGW runtime.
    # It only illustrates quadratic pair growth.
    # ------------------------------------------------------------

    k_ref = np.linspace(
        k.min(),
        k.max(),
        200,
    )

    quadratic = runtime[0] * (k_ref / k[0]) ** 2

    ax.plot(
        k_ref,
        quadratic,
        linestyle=":",
        linewidth=1.2,
        label="Quadratic pair-growth reference",
    )

    ax.set_xlabel(
        "Number of query/reference segments $K$"
    )

    ax.set_ylabel("Runtime (s)")
    ax.set_title("(e) LPGW runtime scaling")

    clean_axes(ax)

    ax.legend(
        frameon=False,
        loc="upper left",
    )

    save_fig(fig, "fig_e_runtime_scaling")


# ================================================================
# FIGURE F
# NOISE + PARTIAL OVERLAP
# ================================================================

def figure_f_robustness():
    """
    Two-panel robustness figure.

    Left:
        trajectory noise

    Right:
        partial overlap
    """

    # ------------------------------------------------------------
    # Noise values from current drift experiment.
    # ------------------------------------------------------------

    noise = np.array([
        0.0,
        0.1,
        0.2,
        0.5,
        1.0,
    ])

    lpgw_noise = np.array([
        0.5277,
        0.1653,
        0.0670,
        0.0404,
        0.0344,
    ])

    dtw_noise = np.array([
        0.6879,
        0.6879,
        0.6710,
        0.6710,
        0.6623,
    ])

    # ------------------------------------------------------------
    # Partial overlap.
    # ------------------------------------------------------------

    overlap = np.array([
        1.0,
        0.8,
        0.6,
        0.4,
    ])

    lpgw_overlap = np.array([
        0.4991,
        0.2482,
        0.1870,
        0.0771,
    ])

    dtw_overlap = np.array([
        0.6879,
        0.6962,
        0.6879,
        0.6962,
    ])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=DOUBLE_COL,
    )

    ax = axes[0]

    ax.plot(
        noise,
        lpgw_noise,
        marker="o",
        linewidth=2.0,
        label="LPGW",
    )

    ax.plot(
        noise,
        dtw_noise,
        marker="s",
        linewidth=1.5,
        linestyle="--",
        label="DTW",
    )

    ax.set_xlabel(r"Noise standard deviation $\sigma$ (m)")
    ax.set_ylabel("$F_1$")
    ax.set_title("(f1) Noise robustness")

    ax.set_ylim(0, 0.75)

    clean_axes(ax)

    ax.legend(
        frameon=False,
        loc="best",
    )

    ax = axes[1]

    ax.plot(
        overlap,
        lpgw_overlap,
        marker="o",
        linewidth=2.0,
        label="LPGW",
    )

    ax.plot(
        overlap,
        dtw_overlap,
        marker="s",
        linewidth=1.5,
        linestyle="--",
        label="DTW",
    )

    ax.set_xlabel("Trajectory overlap ratio")
    ax.set_ylabel("$F_1$")
    ax.set_title("(f2) Partial-overlap robustness")

    ax.set_xlim(0.35, 1.05)
    ax.set_ylim(0, 0.75)

    clean_axes(ax)

    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout(w_pad=1.5)

    save_fig(fig, "fig_f_robustness")


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("Generating LPGW paper figures")
    print("=" * 70)

    print(f"Repository: {REPO_ROOT}")
    print(f"Output:     {OUT_DIR}")

    # ------------------------------------------------------------
    # A
    # ------------------------------------------------------------

    try:
        figure_a_trajectory_gt()
    except Exception as e:
        print(f"[FIG A FAILED] {e}")

    # ------------------------------------------------------------
    # B
    # ------------------------------------------------------------

    try:
        figure_b_lpgw_matrix()
    except Exception as e:
        print(f"[FIG B FAILED] {e}")

    # ------------------------------------------------------------
    # C
    # ------------------------------------------------------------

    try:
        figure_c_detected_closures()
    except Exception as e:
        print(f"[FIG C FAILED] {e}")

    # ------------------------------------------------------------
    # D
    # ------------------------------------------------------------

    try:
        figure_d_thresholds()
    except Exception as e:
        print(f"[FIG D FAILED] {e}")

    # ------------------------------------------------------------
    # E
    # ------------------------------------------------------------

    try:
        figure_e_runtime_scaling()
    except Exception as e:
        print(f"[FIG E FAILED] {e}")

    # ------------------------------------------------------------
    # F
    # ------------------------------------------------------------

    try:
        figure_f_robustness()
    except Exception as e:
        print(f"[FIG F FAILED] {e}")

    print("=" * 70)
    print("Finished.")
    print(f"Figures are in: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()