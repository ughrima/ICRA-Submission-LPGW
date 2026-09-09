# Loop Closure Detection via LPGW

This repository contains the implementation and experimental evaluation of **Loop Closure Detection using Lambda-dependent Partial Gromov-Wasserstein (LPGW)**.

The codebase is organized as a reproducible research pipeline for comparing LPGW with standard trajectory-distance baselines and for evaluating robustness to detection thresholds, trajectory overlap, reference-segment selection, drift, and dataset changes.

---

## Project structure

```text
loop-closure-lpgw/
│
├── config.py
│
├── lpgw.py
├── loop_closure.py
│
├── core/
│   ├── trajectory_utils.py
│   └── detection.py
│
├── lib/
│   ├── gromov.py
│   └── opt.py
│
├── evaluation/
│   └── eval_canonical.py
│
├── ground_truth/
│   ├── generate_ground_truth.py
│   └── files/
│
├── experiments/
│   ├── run_full_lpgw-kitti.py
│   ├── run_baseline_comparison.py
│   ├── run_percentile_sweep.py
│   ├── run_heldout_threshold.py
│   ├── run_tolerance_sweep.py
│   ├── run_overlap_experiment.py
│   ├── run_drift_experiment.py
│   ├── run_reference_segment_ablation.py
│   └── run_runtime_scaling.py
│
├── tests/
│
├── data/
│   └── poses/
│
├── results/
│
└── README.md
```

Pose extraction is provided separately at the repository root:

```text
../extract_poses.py
```

---

## Core design

The implementation follows a single canonical pipeline:

```text
config.py
    │
    ▼
trajectory preprocessing
    │
    ▼
LoopClosureDetector
    │
    ▼
lpgw.py
    │
    ▼
lib/gromov.py
    │
    ▼
LPGW distance matrix
    │
    ▼
threshold / percentile detection
    │
    ▼
canonical evaluation
```

### `config.py`

`config.py` is the **single source of truth** for shared experimental constants.

It defines:

* active dataset
* reference and query pose files
* target trajectory size
* segment length
* sampling frequency
* segment stride
* LPGW lambda
* primary percentile
* spatial tolerance
* percentile sweep
* tolerance sweep
* baseline segment policy
* random seed

Experiments should import these values rather than defining their own copies.

---

## Datasets

Two datasets are supported:

### UZH-FPV

```text
Reference: poses_bag-3.csv
Query:     poses_bag-7.csv
```

### KITTI

```text
Reference: kitti_00_reference.csv
Query:     kitti_00_query.csv
```

Select the dataset in `config.py`:

```python
ACTIVE_DATASET = "uzh_fpv"
```

or:

```python
ACTIVE_DATASET = "kitti"
```

The corresponding dataset-specific filenames and labels are selected automatically.

---

## Installation

From the project directory:

```bash
cd loop-closure-lpgw
```

Create and activate a virtual environment if desired:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:

* NumPy
* Pandas
* SciPy
* scikit-learn
* POT
* fastdtw
* rosbags
* Matplotlib

---

## Pose data

Place the required pose CSV files in:

```text
loop-closure-lpgw/data/poses/
```

For UZH-FPV:

```text
data/poses/
├── poses_bag-3.csv
└── poses_bag-7.csv
```

For KITTI:

```text
data/poses/
├── kitti_00_reference.csv
└── kitti_00_query.csv
```

The pose CSVs must contain:

```text
PosX
PosY
PosZ
```

along with the timestamp information required by the trajectory loader.

---

## Pose extraction

If starting from ROS bag files, pose extraction is handled by:

```text
../extract_poses.py
```

Example:

```bash
python ../extract_poses.py --diagnose --bag-path /path/to/bag.bag
```

Extract poses using the same timestamp convention for the reference and query trajectories.

Example:

```bash
python ../extract_poses.py \
    --bag-path /path/to/bag3.bag \
    --output-name poses_bag-3.csv
```

```bash
python ../extract_poses.py \
    --bag-path /path/to/bag7.bag \
    --output-name poses_bag-7.csv
```

Verify the generated CSVs:

```bash
python ../extract_poses.py --verify-csvs
```

---

# Canonical preprocessing

All standard LPGW experiments use the same trajectory preprocessing:

1. Load the reference and query trajectories.
2. Downsample to `TARGET_POINTS`.
3. Segment trajectories using:

   * `SEGMENT_LENGTH`
   * `FPS`
   * `STRIDE`
4. Match the reference/query segment counts.
5. Downsample each segment for LPGW computation.
6. Compute the LPGW distance matrix.

The relevant implementation is in:

```text
core/trajectory_utils.py
```

and:

```text
loop_closure.py
```

---

# LPGW implementation

The canonical LPGW implementation is:

```text
lpgw.py
```

It calls the partial Gromov-Wasserstein solver implemented in:

```text
lib/gromov.py
```

The detector wrapper is:

```text
loop_closure.py
```

The normal experiments use:

```python
reference_strategy="robust"
```

Reference-segment selection can also be explicitly controlled for the reference-ablation experiment.

---

# Ground truth

Ground truth is generated centrally rather than independently by each experiment.

Run:

```bash
python ground_truth/generate_ground_truth.py
```

The generator:

1. loads the reference and query trajectories;
2. applies the canonical preprocessing;
3. computes segment centers;
4. finds the nearest reference segment center for every query segment;
5. records the nearest distance;
6. assigns the loop-closure label using the configured spatial tolerance;
7. saves both labels and metadata.

The generated files are stored in:

```text
ground_truth/files/
```

For example:

```text
gt_uzhfpv_2.0m.csv
gt_uzhfpv_2.0m_meta.json

gt_kitti00_2.0m.csv
gt_kitti00_2.0m_meta.json
```

The canonical GT CSV contains:

```text
query_index
nearest_ref_index
nearest_distance_m
label
```

Standard experiments should use these canonical labels rather than independently reconstructing ground truth.

---

# Evaluation

The common evaluation function is:

```text
evaluation/eval_canonical.py
```

It reports:

* TP
* FP
* FN
* TN
* precision
* recall
* F1

This keeps evaluation consistent across experiments.

---

# Experiments

## 1. Full LPGW experiment

```bash
python experiments/run_full_lpgw-kitti.py
```

Runs the full LPGW evaluation for the configured dataset.

---

## 2. Baseline comparison

```bash
python experiments/run_baseline_comparison.py
```

Compares:

* LPGW
* Dynamic Time Warping
* Discrete Fréchet distance
* Hausdorff distance

The experiment uses a fixed baseline segment subset so that the methods are compared under controlled conditions.

Each distance matrix is computed once.

---

## 3. Percentile sweep

```bash
python experiments/run_percentile_sweep.py
```

Evaluates the LPGW distance matrix at:

```text
1%
5%
10%
20%
50%
```

The LPGW distance matrix is computed once.

Only the detection percentile changes.

The experiment uses the canonical 2 m ground truth.

Results are written to:

```text
results/percentile_sweep.csv
```

---

## 4. Held-out threshold experiment

```bash
python experiments/run_heldout_threshold.py
```

Tests threshold selection and evaluation using separate data partitions.

The threshold is selected on one partition and evaluated on held-out data.

---

## 5. Spatial tolerance sweep

```bash
python experiments/run_tolerance_sweep.py
```

Evaluates:

```text
0.5 m
1.0 m
1.5 m
2.0 m
```

The LPGW distance matrix remains fixed.

Only the ground-truth spatial tolerance changes.

This separates the effect of the ground-truth definition from the detector itself.

---

## 6. Overlap experiment

```bash
python experiments/run_overlap_experiment.py
```

Evaluates loop-closure detection under different query/reference trajectory overlap ratios.

The canonical ground truth is kept fixed while the observed query trajectory is partially reduced.

---

## 7. Drift experiment

```bash
python experiments/run_drift_experiment.py
```

Adds controlled trajectory drift/noise to evaluate robustness of LPGW detection.

The clean trajectory provides the ground-truth reference while the perturbed trajectory is used for detection.

---

## 8. Reference-segment ablation

```bash
python experiments/run_reference_segment_ablation.py
```

Evaluates sensitivity to the choice of reference segment.

Specific reference indices are supplied explicitly rather than relying on automatic reference selection.

---

## 9. Runtime scaling

```bash
python experiments/run_runtime_scaling.py
```

Measures LPGW runtime as the number of trajectory segments increases.

The experiment reports runtime as a function of problem size.

---

# Reproducibility rules

The following rules are important for reproducing the reported experiments.

### 1. Use `config.py`

Do not independently redefine:

```text
segment length
stride
FPS
percentile
tolerance
LPGW lambda
```

inside individual experiment scripts.

### 2. Use canonical preprocessing

All experiments should use:

```text
core/trajectory_utils.py
```

for trajectory loading, downsampling, and segmentation.

### 3. Use canonical ground truth

Standard experiments should load:

```text
ground_truth/files/
```

rather than recreating labels independently.

### 4. Compute distance matrices once

When an experiment sweeps a threshold, percentile, or evaluation condition, the LPGW distance matrix should not be recomputed unnecessarily.

### 5. Keep reference selection explicit

Normal experiments use the robust reference strategy.

Reference ablation explicitly specifies the reference index.

### 6. Keep evaluation consistent

All reported metrics should use:

```text
evaluation/eval_canonical.py
```

---

# Results

Experiment outputs are written to:

```text
results/
```

Generated matrices and other temporary computational artifacts should remain outside version control where appropriate.

Typical result files include:

```text
results/
├── percentile_sweep.csv
├── tolerance_sweep.csv
├── baseline_comparison.csv
├── overlap_experiment.csv
├── reference_segment_ablation.csv
└── runtime_scaling.csv
```

Actual filenames may vary slightly between experiments.

---

# Recommended execution order

For a fresh reproduction, use this order:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify pose data
python ../extract_poses.py --verify-csvs

# 3. Generate canonical ground truth
python ground_truth/generate_ground_truth.py

# 4. Run the main LPGW experiment
python experiments/run_full_lpgw-kitti.py

# 5. Run baseline comparison
python experiments/run_baseline_comparison.py

# 6. Run percentile sweep
python experiments/run_percentile_sweep.py

# 7. Run held-out threshold evaluation
python experiments/run_heldout_threshold.py

# 8. Run robustness experiments
python experiments/run_tolerance_sweep.py
python experiments/run_overlap_experiment.py
python experiments/run_drift_experiment.py

# 9. Run ablation and runtime experiments
python experiments/run_reference_segment_ablation.py
python experiments/run_runtime_scaling.py
```

For UZH-FPV, set:

```python
ACTIVE_DATASET = "uzh_fpv"
```

For KITTI, set:

```python
ACTIVE_DATASET = "kitti"
```

before running the relevant experiments.

---

# Research pipeline summary

The intended experimental architecture is:

```text
                    config.py
                       │
                       ▼
              trajectory preprocessing
                       │
                       ▼
              canonical segmentation
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
    canonical GT              LPGW detector
          │                         │
          │                         ▼
          │                   lpgw.py
          │                         │
          │                         ▼
          │                  lib/gromov.py
          │                         │
          │                         ▼
          │                 distance matrix
          │                         │
          └────────────┬────────────┘
                       ▼
                canonical scoring
                       │
                       ▼
             precision / recall / F1
```

This separation ensures that preprocessing, ground truth, LPGW computation, threshold selection, and evaluation are controlled independently and can be audited across experiments.

```
```
