# Loop Closure Detection via LPGW

Canonical reproduction codebase for the ICRA paper. See [`Paper_Context.md`](../Paper_Context.md) for exact table/figure definitions.

## Layout

```
loop-closure-lpgw/
├── config.py                 # ALL shared constants — single source of truth
├── data/poses/               # poses_bag-3.csv, poses_bag-7.csv
├── core/                     # trajectory utils, distance matrices, detection
├── ground_truth/             # GT generator + ground_truth/files/ output
├── evaluation/               # eval_canonical.py (only scoring function)
├── experiments/              # Table II/III/V + robustness ablations
├── figures/                  # Fig. 8 detection map
├── cache/                    # D matrices (.npy), gitignored
├── results/                  # experiment outputs, gitignored
└── tests/
```

Pose extraction lives one level up: [`../extract_poses.py`](../extract_poses.py).

## Setup

```bash
cd loop-closure-lpgw
pip install -r requirements.txt
```

`extract_poses.py` requires `rosbags` (`pip install rosbags`) — no full ROS install needed.

## Pose extraction workflow

**Do not run the pipeline until both pose CSVs share the same timestamp convention.**

```bash
# 1. Diagnose both bags (replace paths with your UZH-FPV bag files)
python ../extract_poses.py --diagnose --bag-path /path/to/bag3.bag
python ../extract_poses.py --diagnose --bag-path /path/to/bag7.bag

# 2. Extract (same TIMESTAMP_SOURCE for every bag)
python ../extract_poses.py --bag-path /path/to/bag3.bag --output-name poses_bag-3.csv
python ../extract_poses.py --bag-path /path/to/bag7.bag --output-name poses_bag-7.csv

# 3. Verify
python ../extract_poses.py --verify-csvs
```

## Config rule

**Every module must `import config` (or `from config import …`).**  
Hardcoding `SEGMENT_LENGTH`, `STRIDE`, `PERCENTILE`, tolerances, or sweeps outside `config.py` is a review error.

## Status

Skeleton only — implementation to be ported from the legacy repo in subsequent phases.
