"""
Single source of truth for shared constants.

Every other module in this repo must import from here.
Do NOT hardcode segment length, stride, percentile, or tolerance elsewhere.
"""

# ----------------- Dataset selection -----------------

ACTIVE_DATASET = "kitti"  # choose: "uzh_fpv" or "kitti"

DATASETS = {
    "uzh_fpv": {
        "ref_csv": "poses_bag-3.csv",
        "query_csv": "poses_bag-7.csv",
        "label_ref": "Reference (Bag 3)",
        "label_query": "Query (Bag 7)",
        "short_name": "uzhfpv",
    },
    "kitti": {
        "ref_csv": "kitti_00_reference.csv",
        "query_csv": "kitti_00_query.csv",
        "label_ref": "Reference (KITTI Seq 00)",
        "label_query": "Query (KITTI Seq 00 query)",
        "short_name": "kitti00",
    },
}

# Convenience accessors
_ref_query = DATASETS[ACTIVE_DATASET]
BAG3_CSV = _ref_query["ref_csv"]
BAG7_CSV = _ref_query["query_csv"]
DATASET_LABEL_REF = _ref_query["label_ref"]
DATASET_LABEL_QUERY = _ref_query["label_query"]
DATASET_SHORT = _ref_query["short_name"]

POSES_DIR = "data/poses"


# ----------------- Trajectory preprocessing -----------------

TARGET_POINTS = 5000
SEGMENT_LENGTH = 5.0  # seconds
FPS = 10
STRIDE = 1.0  # seconds


# ----------------- Detection / evaluation operating points -----------------

PERCENTILE_UZH = 68.0  # Match the true positive density (~68% of queries are matches)
PERCENTILE_KITTI = 15.0  # Match the true positive density (~15% of queries are matches)
SPATIAL_TOLERANCE = 2.0  # meters, primary reported tolerance
TOLERANCE_SWEEP = [0.5, 1.0, 1.5, 2.0]
PERCENTILE_SWEEP = [1, 5, 10, 20, 50]


# ----------------- Ground truth labeling -----------------

GT_SEGMENT_CENTER_THRESHOLD = 0.5  # meters


# ----------------- LPGW -----------------

# Canonical LPGW configuration
LPGW_LAMBDA = 0.5
LPGW_HUBER_DELTA = 0.15  # meters in the LPGW geometry cost


# ----------------- Reproducibility -----------------

RANDOM_SEED = 42


# ----------------- Temporal matching tolerance -----------------

TIME_TOLERANCE = 2.0


# ----------------- Default ROS topic -----------------

DEFAULT_POSE_TOPIC = "/groundtruth/pose"


# ----------------- Baseline experiment settings -----------------

# Policy for baseline query count. The reference database remains complete:
#   "full"  -> use all query segments
#   "subsample_fixed" -> subsample queries to MAX_BASELINE_SEGMENTS
BASELINE_SEGMENT_POLICY = "subsample_fixed"
MAX_BASELINE_SEGMENTS = 150  # used only if BASELINE_SEGMENT_POLICY == "subsample_fixed"

REFERENCE_STRATEGY = "robust"
