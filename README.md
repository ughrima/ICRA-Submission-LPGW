# Loop Closure Detection with LPGW

This repository implements and evaluates loop-closure detection with a Lambda-dependent Partial Gromov-Wasserstein (LPGW) representation of trajectory segments. The method compares the internal geometry of trajectory windows, so it can match partial or differently sampled motion without requiring pointwise alignment.

The repository contains two layers:

1. The detector: pose loading, preprocessing, segmentation, LPGW embedding, distance-matrix construction, and thresholding.
2. The evaluation pipeline: canonical ground truth, baselines, robustness experiments, result CSVs, tables, and figures.

## The complete pipeline

The main data flow is:

```text
ROS bags or pose CSVs
        |
        v
load_trajectory()
        |
        v
uniform trajectory downsampling to TARGET_POINTS
        |
        v
overlapping 5-second trajectory segments
        |
        +------------------------------+
        |                              |
        v                              v
canonical ground truth          LoopClosureDetector
nearest segment-center labels          |
                                       v
                              select one reference segment
                                       |
                                       v
                              LPGW embedding for every segment
                                       |
                                       v
                              query/reference distance matrix D
                                       |
                                       v
                              nearest reference per query
                                       |
                                       v
                              percentile threshold
                                       |
                                       v
                              precision, recall, F1
```

The important implementation files are:

| File | Responsibility |
| --- | --- |
| `config.py` | Shared dataset, preprocessing, LPGW, threshold, and evaluation settings |
| `core/trajectory_utils.py` | CSV validation, uniform downsampling, and window segmentation |
| `loop_closure.py` | Detector wrapper, reference selection, matrix construction, and detection |
| `lpgw.py` | LPGW geometry, partial GW solve, embedding, and discrepancy |
| `lib/gromov.py` | Lambda-dependent partial Gromov-Wasserstein solver |
| `ground_truth/generate_ground_truth.py` | Canonical segment-center labels and metadata |
| `evaluation/eval_canonical.py` | Shared confusion-matrix metrics |
| `experiments/` | Main evaluation and ablation scripts |
| `results/` | Checked-in numeric outputs from the experiments |
| `tables/` and `figures/` | Paper-ready summaries and visualizations |

## 1. Configuration and datasets

`config.py` is intended to be the single source of truth. The current active configuration is:

```text
Dataset:                KITTI sequence 00
Reference CSV:          data/poses/kitti_00_reference.csv
Query CSV:              data/poses/kitti_00_query.csv
Target trajectory pts:  5000
Segment duration:       5.0 seconds
Sampling rate:          10 Hz
Segment stride:         1.0 second
LPGW lambda:            0.5
Primary percentile:     1%
Ground-truth tolerance: 2.0 meters
Random seed:            42
```

The other supported dataset is UZH-FPV:

```text
Reference: data/poses/poses_bag-3.csv
Query:     data/poses/poses_bag-7.csv
```

Change `ACTIVE_DATASET` to `"uzh_fpv"` or `"kitti"` before running an experiment. The pose CSVs must contain `Timestamp`, `PosX`, `PosY`, and `PosZ`. Only the XYZ columns are used by the detector; timestamps are validated and are available for downstream inspection.

## 2. Pose extraction and loading

When the input is a ROS bag, the separate root-level `extract_poses.py` utility converts the selected pose topic into a CSV. The reference and query bags must use the same timestamp convention.

```bash
python ../extract_poses.py --diagnose --bag-path /path/to/bag.bag
python ../extract_poses.py --bag-path /path/to/bag3.bag --output-name poses_bag-3.csv
python ../extract_poses.py --bag-path /path/to/bag7.bag --output-name poses_bag-7.csv
python ../extract_poses.py --verify-csvs
```

`load_trajectory()` checks that every CSV exists, is non-empty, and has the required columns. It returns a pandas DataFrame; the pipeline then extracts an `N x 3` array of XYZ positions.

## 3. Canonical preprocessing

Every standard experiment uses the same preprocessing so that distances, labels, and reported metrics refer to the same segment indices.

### 3.1 Uniform downsampling

Each full trajectory is uniformly sampled by index to at most 5,000 points. This controls the cost of the experiments and assumes the original trajectory is approximately uniformly sampled in time.

### 3.2 Overlapping windows

With `SEGMENT_LENGTH = 5.0`, `FPS = 10`, and `STRIDE = 1.0`:

```text
window size = 5.0 * 10 = 50 points
window starts = 0, 10, 20, ...
```

Thus consecutive windows overlap by four seconds. A segment is retained only when a complete 50-point window is available. If reference and query produce different numbers of windows, the canonical pipeline truncates both lists to the smaller count. This is why the checked-in runs contain 196 segments for KITTI and 496 for UZH-FPV.

## 4. Canonical ground truth

Ground truth is spatial, not based on a manually supplied timestamp match. Run:

```bash
python ground_truth/generate_ground_truth.py
```

For every query segment:

1. Compute its XYZ center, the mean of all points in the segment.
2. Compute every reference segment center.
3. Find the nearest reference center with a KD-tree.
4. Mark the query segment positive when the nearest-center distance is at most `SPATIAL_TOLERANCE`.

At the primary 2 m tolerance, the saved CSV contains:

```text
query_index, nearest_ref_index, nearest_distance_m, label
```

The accompanying JSON stores the centers and preprocessing parameters. The experiment scripts verify the number and ordering of query segments before scoring, which prevents a silently misaligned ground-truth file from producing misleading metrics. Ground truth at 0.5 m is also generated for the tolerance ablation.

## 5. What LPGW computes

The mathematical core is in `lpgw.py`. For each segment `Y`, the detector uses one selected reference segment `X` and performs the following operations.

### 5.1 Normalize intrinsic geometry

The detector uses one shared scale for a complete distance-matrix run. It computes the diameter of every processed query and reference segment and uses their median as the global scale. This preserves relative physical extent between segments while keeping the cost matrices numerically well-conditioned. Direct `LPGW` use with `global_scale=None` retains the legacy per-segment normalization for scale-invariant ablations.

The method then builds pairwise Huber cost matrices with
`LPGW_HUBER_DELTA = 0.15` m. Residuals below 15 cm use the quadratic branch;
larger residuals use the linear branch, reducing the influence of drift spikes:

```text
C_X[i,j] = Huber(||X_i - X_j||, 0.15 m)
C_Y[i,j] = Huber(||Y_i - Y_j||, 0.15 m)
```

Pairwise geometry remains translation and rotation invariant, while the shared scale preserves relative segment size. `LoopClosureDetector(scale_aware=False)` reproduces the old scale-invariant behavior.

### 5.2 Solve partial Gromov-Wasserstein

Uniform point masses are assigned to both segments. The Lambda-dependent solver in `lib/gromov.py` returns a transport plan `gamma`. Unlike full GW, partial GW is allowed to leave mass unmatched. The parameter `lambda = 0.5` controls the penalty associated with transported and discarded mass.

The source marginal is:

```text
q_e[i] = sum_j gamma[i,j]
```

For each reference point with nonzero transported mass, a barycentric projection of the target points is computed:

```text
y_e[i] = (1 / q_e[i]) * sum_j gamma[i,j] * Y_j
```

The linearized geometric component is the difference between the projected target geometry and the reference geometry:

```text
K_e[i,j] = ||y_e[i] - y_e[j]||^2 - ||X_i - X_j||^2
```

The embedding stores `K_e`, `q_e`, the transport plan, the projected points, and the discarded-mass term.

### 5.3 Compare embeddings

For two segments represented relative to the same `X`, the discrepancy combines:

1. A weighted squared difference between their `K_e` matrices.
2. A penalty when their transported masses differ.
3. A penalty for mass discarded by partial GW.

The result is nonnegative; smaller means more similar. `LoopClosureDetector` embeds every query and reference segment once, then compares the stored embeddings. No new GW optimization is solved inside the pairwise matrix loop.

## 6. Reference selection

The detector chooses one reference segment from the reference segment list and uses it for every embedding. The standard `"median_diameter"` strategy selects the segment whose pairwise diameter is closest to the median diameter across the reference set. `"robust"` remains an alias for this strategy for compatibility.

```text
argmin_i |diameter_i - median(diameters)|
```

The selected index is printed for reproducibility. The detector also supports `first`, `middle`, `last`, and an explicit `reference_index`; the latter is used by the reference-segment ablation.

## 7. From distances to loop closures

The detector returns a matrix `D` with shape:

```text
D[query_segment, reference_segment]
```

For each query segment, the best candidate is the minimum value in its row:

```text
best_distance[i] = min_j D[i,j]
```

An adaptive percentile threshold is computed from the distribution of the per-query best-match scores, rather than from all query-reference matrix entries. A query is predicted to be a loop closure when:

```text
best_distance[i] <= percentile(best_distance, chosen_percentile)
```

The default sweep is 1%, 5%, 10%, 20%, and 50%, while the current primary configured point is `config.PERCENTILE = 68.0`. These percentiles are applied to best-match scores, so the cutoff adapts to the current noisy distance distribution. The detector also contains GMM and MAD threshold helpers, but the checked-in canonical experiments use adaptive percentiles.

## 8. Evaluation metrics

`evaluation/eval_canonical.py` compares the binary predictions against the canonical labels and reports TP, FP, FN, TN, precision, recall, and F1:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

The threshold is part of the result record, so a result can be reproduced and audited instead of being reported only as a rounded metric.

## 9. Experiments and outputs

Run from the repository root after installing `requirements.txt`:

```bash
python ground_truth/generate_ground_truth.py
python experiments/run_full_lpgw-kitti.py
python experiments/run_baseline_comparison.py
python experiments/run_percentile_sweep.py
python experiments/run_heldout_threshold.py
python experiments/run_tolerance_sweep.py
python experiments/run_overlap_experiment.py
python experiments/run_drift_experiment.py
python experiments/run_reference_segment_ablation.py
python experiments/run_runtime_scaling.py
```

The scripts write dataset-specific files such as `results/full_lpgw_kitti00.csv` and `results/full_lpgw_uzhfpv.csv`.

### Main LPGW run

`run_full_lpgw-kitti.py` evaluates the complete canonical segment sets using LPGW only. It writes precision, recall, F1, confusion counts, threshold, runtime, segment counts, and configuration values.

### Baseline comparison

`run_baseline_comparison.py` computes LPGW, DTW, discrete Frechet, and Hausdorff distance matrices. DTW, Frechet, and Hausdorff use the configured fixed baseline subset when necessary, so their matrix sizes are controlled. The comparison is threshold-free: each query is scored by the negative distance to its best reference, and the output reports maximum F1, average precision, recall at 100% precision, and Recall@1. This avoids comparing methods at different operating points caused by thresholding full matrix entries and then applying the threshold to row minima.

### Percentile sweep

`run_percentile_sweep.py` computes the LPGW matrix once and changes only the percentile used to create predictions. This isolates threshold selection from the expensive embedding computation.

### Held-out threshold selection

`run_heldout_threshold.py` splits the query segments into two folds. It chooses the best candidate percentile on one fold, computes the threshold from that fold, and evaluates the other fold. It reports both directions and the non-held-out full-data result.

### Tolerance sweep

`run_tolerance_sweep.py` keeps the LPGW matrix and the configured `config.PERCENTILE` detector operating point fixed while changing the spatial definition of a positive loop: 0.5, 1.0, 1.5, and 2.0 m.

### Overlap experiment

`run_overlap_experiment.py` keeps a contiguous portion of each query segment at overlap ratios of 1.0, 0.8, 0.6, and 0.4. This models entering or leaving a place partway through a window rather than randomly thinning points. It evaluates LPGW and DTW against the same canonical labels.

### Drift experiment

`run_drift_experiment.py` adds integrated random-walk drift to the query trajectory at per-step scales 0.0, 0.1, 0.2, 0.5, and 1.0 m. Ground truth is computed from the clean trajectories and is deliberately not recomputed after drift is added.

### Reference ablation

`run_reference_segment_ablation.py` repeats LPGW using reference indices at the start, quartiles, middle, and end of the reference trajectory. It measures how much the fixed-reference embedding depends on that choice.

### Runtime scaling

`run_runtime_scaling.py` measures the cost of constructing square exact distance matrices as the number of query/reference segments grows. The exact LPGW discrepancy includes pair-dependent transported-mass weights, so it is not an ordinary fixed-vector L2 distance and exact matrix construction remains quadratic. `LoopClosureDetector.approximate_nearest_references()` exposes a fixed-reference geometric vector plus KD-tree retrieval path; use it for approximate retrieval and rerank candidates with exact LPGW when needed.

## 10. Tests and sanity checks

The tests exercise the mathematical behavior on synthetic paths:

```bash
pytest -q
```

They cover one-pair distances, partial overlap, synthetic loops, sanity checks, and lambda sweeps. The experiments themselves are the source of the stored dataset-level result CSVs.

# Results

The following summary is read from the checked-in files under `results/`. Values are rounded here for readability; the CSVs retain full precision.

## Main LPGW results at 10%

The full canonical runs contain 196 KITTI segments and 496 UZH-FPV segments. At the 10% operating point:

| Dataset | Precision | Recall | F1 | Runtime |
| --- | ---: | ---: | ---: | ---: |
| KITTI 00 | 0.133 | 0.759 | 0.227 | 0.95 s |
| UZH-FPV | 0.716 | 0.716 | 0.716 | 3.76 s |

The KITTI result has high recall but low precision, meaning the chosen threshold flags many non-loop segments. UZH-FPV is more balanced at this operating point.

## Percentile behavior

| Dataset | F1 at 1% | F1 at 5% | F1 at 10% | F1 at 20% | F1 at 50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| KITTI 00 | 0.098 | 0.212 | 0.227 | 0.247 | 0.258 |
| UZH-FPV | 0.528 | 0.660 | 0.716 | 0.761 | 0.805 |

Increasing the percentile increases recall on both datasets. It also increases the number of false positives, so the best operating point depends on whether the application values coverage or precision.

## Baseline comparison

With the baseline experiment's fixed 150-segment policy, compare methods using the threshold-free metrics in `results/baseline_accuracy_<dataset>.csv`:

| Dataset | LPGW max F1 | DTW max F1 | Frechet max F1 | Hausdorff max F1 |
| --- | ---: | ---: | ---: | ---: |
| KITTI 00 | regenerate | regenerate | regenerate | regenerate |
| UZH-FPV | regenerate | regenerate | regenerate | regenerate |

The old checked-in baseline CSVs used the invalid full-matrix percentile threshold and are not comparable across methods after this correction. Rerun the baseline experiment before making claims from this table. The separate percentile-sweep results remain useful for studying LPGW's detector threshold, but they are not a fair cross-method comparison.

## Threshold selection without reusing evaluation data

The held-out threshold experiment reports:

| Dataset | Held-out direction | Selected percentile | Held-out F1 |
| --- | --- | ---: | ---: |
| KITTI 00 | A threshold, evaluate B | 5% | 0.364 |
| KITTI 00 | B threshold, evaluate A | 50% | 0.020 |
| UZH-FPV | A threshold, evaluate B | 50% | 0.746 |
| UZH-FPV | B threshold, evaluate A | 50% | 0.836 |

The asymmetric KITTI values show that threshold selection is sensitive to the fold. The held-out results should be preferred when discussing generalization; the full-data 1% rows are useful as a reference but are not held-out estimates.

## Robustness and ablations

### Spatial tolerance

At the fixed 1% operating point, changing the ground-truth tolerance changes the number of positives without changing the LPGW matrix. At 2 m, the positive counts are 29 for KITTI and 338 for UZH-FPV, with F1 values of 0.098 and 0.528. At 0.5 m, KITTI has no positive segments in the checked-in labels, so its F1 is necessarily 0. This is a property of the label definition, not evidence that the distance matrix changed.

### Overlap

LPGW F1 across overlap ratios was:

| Dataset | 1.0 | 0.8 | 0.6 | 0.4 |
| --- | ---: | ---: | ---: | ---: |
| KITTI 00 | 0.098 | 0.116 | 0.200 | 0.103 |
| UZH-FPV | 0.528 | 0.224 | 0.192 | 0.092 |

UZH-FPV degrades as less of a segment is observed. KITTI is already difficult at full overlap, so its values are noisy rather than a monotonic robustness curve.

### Drift/noise

For UZH-FPV, LPGW F1 falls from 0.528 with clean query poses to 0.165, 0.067, 0.040, and 0.034 at noise levels 0.1, 0.2, 0.5, and 1.0 m. For KITTI, the corresponding LPGW F1 values are 0.098, 0.131, 0.136, 0.128, and 0.127. These experiments use clean-trajectory ground truth, so they isolate detector degradation rather than allowing noise to change the labels.

### Reference choice

Across five reference choices, F1 ranged from 0.097 to 0.103 on KITTI and from 0.482 to 0.508 on UZH-FPV. The small KITTI range and moderate UZH-FPV range indicate that the fixed reference matters, but the ablation does not suggest a catastrophic dependence on one particular index for these runs.

### Runtime scaling

For square matrices, runtime increased as follows:

| Segments per side | KITTI runtime | UZH-FPV runtime |
| ---: | ---: | ---: |
| 25 | 0.057 s | 0.053 s |
| 50 | 0.134 s | 0.117 s |
| 100 | 0.309 s | 0.282 s |
| 150 | 0.522 s | 0.486 s |

The number of distance pairs grows from 625 to 22,500 over this range. The implementation avoids repeated PGW solves during pairwise comparison by caching embeddings, but exact matrix construction still has quadratic pair-count growth. The approximate fixed-reference vector path separates embedding cost from KD-tree query cost, while omitting the exact pair-dependent partial-mass terms.

## Reproducible execution order

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Select ACTIVE_DATASET in config.py first.
python ground_truth/generate_ground_truth.py
python experiments/run_full_lpgw-kitti.py
python experiments/run_baseline_comparison.py
python experiments/run_percentile_sweep.py
python experiments/run_heldout_threshold.py
python experiments/run_tolerance_sweep.py
python experiments/run_overlap_experiment.py
python experiments/run_drift_experiment.py
python experiments/run_reference_segment_ablation.py
python experiments/run_runtime_scaling.py
pytest -q
```

The output CSVs are the authoritative numeric artifacts. `tables/` contains formatted summaries for the paper, while `figures/` and `experiments/make_paper_figures_tables.py` provide the presentation layer.

## Paper figures

Generate Figures 2 through 5 from the active dataset trajectories and saved
result CSVs with:

```bash
python3 experiments/generate_paper_figures.py
```

Outputs are written to `results/figures/` as high-resolution PNG and PDF files:

```text
fig2_trajectory_overview
fig3_runtime_scaling
fig4_distance_matrix_heatmap
fig5_robustness
```
