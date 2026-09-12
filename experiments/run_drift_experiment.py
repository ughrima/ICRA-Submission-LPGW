# experiments/run_drift_experiment.py
"""
Drift robustness: LPGW vs classical trajectory distances.   [CORRECTED]

THE HYPOTHESIS THIS TESTS
=========================
Odometry drift moves absolute coordinates but leaves a segment's INTERNAL
geometry almost unchanged. Gromov-Wasserstein compares intra-segment
distance matrices, so it should be nearly invariant. DTW / Frechet /
Hausdorff compare absolute coordinates, so they should degrade.

This is the one experiment where LPGW's position-invariance is an asset
rather than a handicap, so the protocol has to be airtight.


WHAT WAS WRONG BEFORE
=====================

(1) ZERO-CENTERING DELETED THE DRIFT.
    The old script called zero_center_segments() on both the query and
    the reference before computing distances. Over a 5 s window, drift is
    essentially a constant offset -- exactly what subtracting the segment
    mean removes. Since GW is already translation-invariant, centering was
    a no-op for LPGW but handed DTW the same invariance for free. The
    hypothesis became untestable. Centering is now an OPTIONAL ablation
    (--center), never the default.

(2) THE DRIFT WAS 10-100x TOO LARGE.
    cumsum of N(0,1) has std sqrt(n) at step n, applied to the RAW
    trajectory. With ~5000 raw points, sigma=1.0 displaced the query by
    ~120 m -- on a 20 x 13 m flight area. It also meant the same sigma
    produced different drift on UZH (high-rate Leica) and KITTI (10 Hz).
    Drift is now normalised to a FRACTION OF PATH LENGTH, which is how
    the SLAM literature reports it and is comparable across datasets.

(3) LPGW AND DTW RAN AT DIFFERENT SCALES.
    LPGW used all queries, DTW a 150-subsample, so their percentile
    thresholds were not comparable. Both now run on the same query subset
    against the same full reference database.

(4) THRESHOLD-DEPENDENT METRICS ONLY.
    config.PERCENTILE is tuned to one dataset's base rate, which makes
    F1 track the base rate rather than the ranking. Primary metrics are
    now threshold-free (max F1, AP, Recall@1), with the thresholded
    numbers kept as secondary.


PREREQUISITE
============
Run this only AFTER lpgw.py has been fixed and LPGW.diagnose() reports
transported_mass < 1.0. With the old cost scaling, lambda outweighs the
geometric term by ~1e11, the solver always transports full mass, and
every partial-transport term is identically zero -- so this experiment
would measure plain GW under a perturbation, not LPGW under drift.

Saves: results/drift_experiment_{DATASET_SHORT}.csv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import config
from core.trajectory_utils import load_trajectory, segment_trajectory, downsample_trajectory
from loop_closure import LoopClosureDetector
from experiments.run_baseline_comparison import (
    compute_dtw_matrix,
    compute_frechet_matrix,
    compute_hausdorff_matrix,
)


# ---------------------------------------------------------------------
# Drift model
# ---------------------------------------------------------------------

def make_drift(traj: np.ndarray, drift_fraction: float, rng) -> np.ndarray:
    """
    Integrated random walk normalised to a target fraction of path length.

    Real odometry drift is smooth and accumulating, and is conventionally
    reported as a percentage of distance travelled (typically 1-2%). This
    returns an offset array with the SAME shape as `traj` whose final
    displacement equals `drift_fraction * path_length`.

    Using one fixed realisation scaled by the fraction keeps the sweep
    controlled: every level perturbs the trajectory the same way, only
    harder.
    """
    if drift_fraction <= 0:
        return np.zeros_like(traj)

    path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
    walk = np.cumsum(rng.normal(0.0, 1.0, size=traj.shape), axis=0)
    walk = walk - walk[0]

    final = float(np.linalg.norm(walk[-1]))
    if final < 1e-12:
        return np.zeros_like(traj)

    return walk * (drift_fraction * path_len / final)


def build_segments(xyz, target_points, segment_length, fps, stride):
    xyz_ds = downsample_trajectory(xyz, target_points)
    return segment_trajectory(
        xyz_ds, segment_length=segment_length, fps=fps, stride=stride
    )


def zero_center_segments(segments):
    """Optional ablation only -- see note (1) in the module docstring."""
    return [s - s.mean(axis=0) for s in segments]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def retrieval_metrics(D, y_true, gt_nearest_ref=None, index_tol=2):
    """
    Threshold-free scoring of a query x reference distance matrix.

    Includes the trivial baselines a reviewer will compute anyway:
    a random ranker scores AP ~= base rate, and an all-positive
    classifier scores F1 = 2b/(1+b).
    """
    y_true = np.asarray(y_true, dtype=bool)
    scores = -np.min(D, axis=1)          # higher = more loop-like

    out = {}
    base = float(y_true.mean())
    out["base_rate"] = base
    out["all_positive_f1"] = 2 * base / (1 + base) if base > 0 else 0.0
    out["random_ap"] = base

    if y_true.any() and not y_true.all():
        prec, rec, _ = precision_recall_curve(y_true, scores)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        b = int(np.argmax(f1))
        out["max_f1"] = float(f1[b])
        out["precision_at_max_f1"] = float(prec[b])
        out["recall_at_max_f1"] = float(rec[b])
        out["average_precision"] = float(average_precision_score(y_true, scores))
        perfect = prec >= 0.999
        out["recall_at_100_precision"] = float(rec[perfect].max()) if perfect.any() else 0.0
    else:
        for k in ("max_f1", "precision_at_max_f1", "recall_at_max_f1",
                  "average_precision", "recall_at_100_precision"):
            out[k] = float("nan")

    # Recall@1: for genuine loops, does argmin land near the true reference?
    if gt_nearest_ref is not None and y_true.any():
        hits = 0
        pos = np.flatnonzero(y_true)
        for i in pos:
            if abs(int(D[i].argmin()) - int(gt_nearest_ref[i])) <= index_tol:
                hits += 1
        out["recall_at_1"] = hits / len(pos)
    else:
        out["recall_at_1"] = float("nan")

    # Secondary: the thresholded operating point, for continuity.
    tau = float(np.percentile(-scores, config.PERCENTILE))
    y_pred = (-scores) <= tau
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    out.update({
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / max(p + r, 1e-12),
        "tp": tp, "fp": fp, "fn": fn,
        "threshold_tau": tau,
        "num_predicted_positives": int(y_pred.sum()),
    })
    return out


def load_canonical_ground_truth(n_query):
    gt_path = (
        repo_root / "ground_truth" / "files"
        / f"gt_{config.DATASET_SHORT}_{config.SPATIAL_TOLERANCE}m.csv"
    )
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    gt = pd.read_csv(gt_path).sort_values("query_index").reset_index(drop=True)
    if len(gt) != n_query:
        raise ValueError(
            f"GT has {len(gt)} rows but experiment has {n_query} query segments"
        )
    return gt["label"].to_numpy(dtype=bool), gt["nearest_ref_index"].to_numpy()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

DRIFT_FRACTIONS = [0.0, 0.005, 0.01, 0.02, 0.05]   # 0%, 0.5%, 1%, 2%, 5%


def run_drift_experiment(center: bool = False, fractions=None, skip_slow: bool = False):
    fractions = list(DRIFT_FRACTIONS if fractions is None else fractions)

    poses = repo_root / "data" / "poses"
    ref_csv, query_csv = poses / config.BAG3_CSV, poses / config.BAG7_CSV

    print("=" * 70)
    print("DRIFT ROBUSTNESS EXPERIMENT")
    print("=" * 70)
    print(f"Dataset          : {config.DATASET_SHORT}")
    print(f"Drift fractions  : {[f'{f:.1%}' for f in fractions]}")
    print(f"Zero-centering   : {'ON (ablation)' if center else 'OFF (default)'}")

    ref_xyz = load_trajectory(ref_csv)[["PosX", "PosY", "PosZ"]].to_numpy()
    query_xyz_clean = load_trajectory(query_csv)[["PosX", "PosY", "PosZ"]].to_numpy()

    tp_, sl_ = config.TARGET_POINTS, config.SEGMENT_LENGTH
    fps_, st_ = config.FPS, config.STRIDE

    ref_segs = build_segments(ref_xyz, tp_, sl_, fps_, st_)
    query_segs_clean = build_segments(query_xyz_clean, tp_, sl_, fps_, st_)

    n = min(len(ref_segs), len(query_segs_clean))
    ref_segs, query_segs_clean = ref_segs[:n], query_segs_clean[:n]
    print(f"Segments         : {n}")

    y_true_full, gt_ref_full = load_canonical_ground_truth(n)
    print(f"GT positives     : {y_true_full.sum()} / {n}  "
          f"(base rate {y_true_full.mean():.1%})")

    # Matched query subset for ALL methods; full reference database kept.
    max_q = getattr(config, "MAX_BASELINE_SEGMENTS", 150)
    q_idx = (np.arange(n) if n <= max_q
             else np.linspace(0, n - 1, max_q, dtype=int))
    y_true = y_true_full[q_idx]
    gt_ref = gt_ref_full[q_idx]
    print(f"Query subset     : {len(q_idx)}  (references: {len(ref_segs)})")

    # ONE cost scale, fixed from the CLEAN reference, reused at every level
    # so the sweep measures drift and not a shifting normalisation.
    detector = LoopClosureDetector(
        segment_length=sl_, fps=fps_, stride=st_,
        lambdaa=config.LPGW_LAMBDA,
        downsample_points=getattr(config, "LPGW_SEGMENT_POINTS", None),
        reference_strategy="median_diameter",
    )
    cost_scale = detector.lpgw.calibrate(ref_segs)
    print(f"Cost scale       : {cost_scale:.6g}  (fixed from clean reference)")

    ref_seg_for_embed, ref_idx = detector.select_reference(ref_segs, "median_diameter")
    print(f"Reference segment: index {ref_idx}")

    print("\nVerifying PGW is active before running the sweep...")
    diag = detector.lpgw.diagnose(ref_seg_for_embed, ref_segs)
    if diag["transported_mass_min"] > 0.999:
        raise SystemExit(
            "\nABORT: partial transport is inactive (full mass always "
            "transported).\nEvery lambda term in distance() is identically "
            "zero, so this experiment\nwould measure plain GW. Lower "
            "config.LPGW_LAMBDA and re-run diagnose()."
        )

    rng = np.random.default_rng(config.RANDOM_SEED)
    base_walk = np.cumsum(rng.normal(0.0, 1.0, size=query_xyz_clean.shape), axis=0)
    base_walk -= base_walk[0]
    path_len = float(np.sum(np.linalg.norm(np.diff(query_xyz_clean, axis=0), axis=1)))
    final_mag = float(np.linalg.norm(base_walk[-1]))
    print(f"Query path length: {path_len:.1f} m")

    rows = []
    for frac in fractions:
        print("\n" + "=" * 70)
        print(f"DRIFT = {frac:.1%} of path length  ({frac * path_len:.2f} m total)")
        print("=" * 70)

        offset = (np.zeros_like(query_xyz_clean) if frac <= 0
                  else base_walk * (frac * path_len / max(final_mag, 1e-12)))
        query_noisy = build_segments(
            query_xyz_clean + offset, tp_, sl_, fps_, st_
        )[:n]

        q_segs = [query_noisy[i] for i in q_idx]
        r_segs = list(ref_segs)
        if center:
            q_segs, r_segs = zero_center_segments(q_segs), zero_center_segments(r_segs)

        methods = [("LPGW", None)]
        if not skip_slow:
            methods += [("DTW", compute_dtw_matrix),
                        ("Frechet", compute_frechet_matrix),
                        ("Hausdorff", compute_hausdorff_matrix)]
        else:
            methods += [("Hausdorff", compute_hausdorff_matrix)]

        for name, fn in methods:
            t0 = time.perf_counter()
            if name == "LPGW":
                # Reference segment and cost scale stay fixed across levels.
                detector.reference_index = ref_idx
                D = detector.compute_distance_matrix(q_segs, r_segs)
            else:
                D = fn(q_segs, r_segs)
            elapsed = time.perf_counter() - t0

            m = retrieval_metrics(D, y_true, gt_nearest_ref=gt_ref)
            m.update({
                "drift_fraction": frac,
                "drift_metres": frac * path_len,
                "method": name,
                "runtime_sec": elapsed,
                "num_query_segments": len(q_segs),
                "num_reference_segments": len(r_segs),
                "centered": center,
                "cost_scale": cost_scale,
                "lambda": config.LPGW_LAMBDA,
                "seed": config.RANDOM_SEED,
            })
            rows.append(m)

            print(f"  {name:10s} maxF1={m['max_f1']:.4f}  AP={m['average_precision']:.4f}  "
                  f"R@1={m['recall_at_1']:.4f}  (all-pos F1={m['all_positive_f1']:.4f}, "
                  f"random AP={m['random_ap']:.4f})")

    df = pd.DataFrame(rows)
    out_dir = repo_root / "results"
    out_dir.mkdir(exist_ok=True)
    suffix = "_centered" if center else ""
    out_path = out_dir / f"drift_experiment_{config.DATASET_SHORT}{suffix}.csv"
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY  (max F1 by drift level)")
    print("=" * 70)
    print(df.pivot_table(index="drift_fraction", columns="method",
                         values="max_f1").to_string())
    print("\nSUMMARY  (Recall@1 by drift level)")
    print(df.pivot_table(index="drift_fraction", columns="method",
                         values="recall_at_1").to_string())
    print(f"\nSaved to: {out_path}")
    print("\nWHAT TO LOOK FOR: LPGW roughly flat across drift levels while the")
    print("coordinate-based baselines fall. If LPGW also falls, its invariance")
    print("is not surviving the barycentric projection -- investigate K.")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", action="store_true",
                    help="Zero-center segments (ablation; removes the drift signal)")
    ap.add_argument("--fractions", type=float, nargs="+", default=None,
                    help="Drift levels as fractions of path length")
    ap.add_argument("--skip-slow", action="store_true",
                    help="Skip DTW and Frechet (keep LPGW + Hausdorff)")
    a = ap.parse_args()
    run_drift_experiment(center=a.center, fractions=a.fractions, skip_slow=a.skip_slow)
    