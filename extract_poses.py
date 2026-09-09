#!/usr/bin/env python3
"""
Extract ground-truth poses from a ROS bag into a CSV.

Uses the `rosbags` library (no full ROS installation required).
Both Bag 3 and Bag 7 must use the same timestamp source (TIMESTAMP_SOURCE).

Usage:
  pip install rosbags

  # Step 1 — diagnose timestamp conventions before extracting:
  python extract_poses.py --diagnose --bag-path /path/to/bag.bag

  # Step 2 — extract with the chosen convention:
  python extract_poses.py --bag-path /path/to/bag.bag --output-name poses_bag-3.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Iterator, Literal

try:
    from rosbags.highlevel import AnyReader
except ImportError as exc:
    AnyReader = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

# ---------------------------------------------------------------------------
# Timestamp convention — set AFTER running --diagnose on all bags.
# Both Bag 3 and Bag 7 must use this same source.
# ---------------------------------------------------------------------------
TimestampSource = Literal["header", "bag_record"]

# Default: message header stamp (ROS sim / message time).
# Re-run --diagnose and update if bag files indicate otherwise.
TIMESTAMP_SOURCE: TimestampSource = "header"

DEFAULT_TOPIC = "/groundtruth/pose"
OUTPUT_DIR = Path(__file__).resolve().parent / "loop-closure-lpgw" / "data" / "poses"


def _require_rosbags() -> None:
    if AnyReader is None:
        print(
            "ERROR: rosbags is not installed.\n"
            "  pip install rosbags",
            file=sys.stderr,
        )
        if _IMPORT_ERROR is not None:
            print(f"  Import error: {_IMPORT_ERROR}", file=sys.stderr)
        sys.exit(1)


def _classify_range(first: float, last: float) -> str:
    """Heuristic: ~50–70 s elapsed vs Unix-epoch absolute time."""
    span = last - first
    looks_like_epoch = first > 1e8
    if looks_like_epoch:
        return f"absolute / Unix-epoch-like (span={span:.2f}s)"
    if span < 500:
        return f"elapsed / sim-clock-like (span={span:.2f}s)"
    return f"unclear (span={span:.2f}s, first={first:.3f})"


def _header_stamp_sec(msg) -> float:
    """PoseStamped header time in seconds."""
    stamp = msg.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _bag_record_sec(timestamp_ns: int) -> float:
    """Bag-record time from AnyReader message timestamp (nanoseconds)."""
    return timestamp_ns * 1e-9


def _iter_poses(
    bag_path: Path,
    topic: str,
) -> Iterator[tuple[object, float, float]]:
    """
    Yield (msg, header_stamp_sec, bag_record_sec) for each PoseStamped on topic.
    """
    _require_rosbags()
    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            available = sorted({c.topic for c in reader.connections})
            raise RuntimeError(
                f"No connections for topic {topic!r} in {bag_path}.\n"
                f"  Available topics ({len(available)}): "
                + ", ".join(available[:20])
                + (" ..." if len(available) > 20 else "")
            )
        for _connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, _connection.msgtype)
            yield msg, _header_stamp_sec(msg), _bag_record_sec(timestamp)


def diagnose_timestamps(bag_path: str | Path, topic: str = DEFAULT_TOPIC) -> dict:
    """
    For a bag file and topic, print first/last header.stamp and bag-record times
    and classify which looks like ~50s elapsed vs Unix epoch.

    Returns a summary dict with the collected values.
    """
    bag_path = Path(bag_path)
    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    header_stamps: list[float] = []
    bag_times: list[float] = []

    for _msg, header_sec, bag_sec in _iter_poses(bag_path, topic):
        header_stamps.append(header_sec)
        bag_times.append(bag_sec)

    if not header_stamps:
        raise RuntimeError(f"No messages on topic {topic!r} in {bag_path}")

    summary = {
        "bag_path": str(bag_path),
        "topic": topic,
        "n_messages": len(header_stamps),
        "header_first": header_stamps[0],
        "header_last": header_stamps[-1],
        "header_span": header_stamps[-1] - header_stamps[0],
        "bag_record_first": bag_times[0],
        "bag_record_last": bag_times[-1],
        "bag_record_span": bag_times[-1] - bag_times[0],
    }

    print("=" * 72)
    print(f"TIMESTAMP DIAGNOSTIC: {bag_path.name}")
    print(f"  topic : {topic}")
    print(f"  count : {summary['n_messages']} messages")
    print("-" * 72)
    print("msg.header.stamp  (sec + nanosec * 1e-9, message / sim time)")
    print(f"  first : {summary['header_first']:.9f}")
    print(f"  last  : {summary['header_last']:.9f}")
    print(f"  span  : {summary['header_span']:.3f} s")
    print(f"  → {_classify_range(summary['header_first'], summary['header_last'])}")
    print("-" * 72)
    print("bag-record timestamp  (AnyReader message time, nanoseconds → seconds)")
    print(f"  first : {summary['bag_record_first']:.9f}")
    print(f"  last  : {summary['bag_record_last']:.9f}")
    print(f"  span  : {summary['bag_record_span']:.3f} s")
    print(f"  → {_classify_range(summary['bag_record_first'], summary['bag_record_last'])}")
    print("=" * 72)

    return summary


def _pick_timestamp(
    header_stamp_sec: float,
    bag_record_sec: float,
    source: TimestampSource,
) -> float:
    if source == "header":
        return header_stamp_sec
    if source == "bag_record":
        return bag_record_sec
    raise ValueError(f"Unknown timestamp source: {source!r}")


def extract_poses(
    bag_path: str | Path,
    output_path: str | Path,
    topic: str = DEFAULT_TOPIC,
    timestamp_source: TimestampSource | None = None,
) -> Path:
    """
    Extract poses from a bag file to CSV using a unified timestamp source.
    """
    bag_path = Path(bag_path)
    output_path = Path(output_path)
    source = timestamp_source or TIMESTAMP_SOURCE

    if not bag_path.is_file():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")

    rows: list[list[float]] = []
    for msg, header_sec, bag_sec in _iter_poses(bag_path, topic):
        ts = _pick_timestamp(header_sec, bag_sec, source)
        pos = msg.pose.position
        ori = msg.pose.orientation
        rows.append([ts, pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])

    if not rows:
        raise RuntimeError(f"No messages on topic {topic!r} in {bag_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["Timestamp", "PosX", "PosY", "PosZ", "OriX", "OriY", "OriZ", "OriW"]
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    first_ts, last_ts = rows[0][0], rows[-1][0]
    span = last_ts - first_ts
    print(f"Wrote {len(rows)} poses → {output_path}")
    print(f"  timestamp source : {source}")
    print(f"  first / last       : {first_ts:.6f} / {last_ts:.6f}")
    print(f"  span               : {span:.3f} s")
    return output_path


def _verify_csv_convention(csv_path: Path, label: str) -> None:
    """Quick check on an existing CSV without needing the bag file."""
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        timestamps = [float(row["Timestamp"]) for row in reader]
    if not timestamps:
        print(f"  {label}: EMPTY")
        return
    first, last = timestamps[0], timestamps[-1]
    span = last - first
    print(f"  {label}: n={len(timestamps)}, first={first:.6f}, last={last:.6f}, span={span:.3f}s")
    print(f"    → {_classify_range(first, last)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose and extract ground-truth poses from ROS bags (via rosbags).",
    )
    parser.add_argument(
        "--bag-path",
        type=Path,
        help="Path to the .bag file",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Output CSV filename (saved under loop-closure-lpgw/data/poses/)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_TOPIC,
        help=f"Pose topic (default: {DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Print timestamp diagnostics and exit (no extraction)",
    )
    parser.add_argument(
        "--timestamp-source",
        choices=["header", "bag_record"],
        default=None,
        help=f"Override TIMESTAMP_SOURCE constant (default: {TIMESTAMP_SOURCE})",
    )
    parser.add_argument(
        "--verify-csvs",
        action="store_true",
        help="Inspect existing pose CSVs in data/poses/ (no bag required)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.verify_csvs:
        poses_dir = OUTPUT_DIR
        print("Existing pose CSV timestamp conventions:")
        for name in ("poses_bag-3.csv", "poses_bag-7.csv"):
            path = poses_dir / name
            if path.is_file():
                _verify_csv_convention(path, name)
            else:
                print(f"  {name}: not found at {path}")
        return

    if not args.bag_path:
        parser.error("--bag-path is required unless using --verify-csvs")

    if args.diagnose:
        diagnose_timestamps(args.bag_path, topic=args.topic)
        return

    if not args.output_name:
        parser.error("--output-name is required for extraction (omit --diagnose)")

    output_path = OUTPUT_DIR / args.output_name
    extract_poses(
        args.bag_path,
        output_path,
        topic=args.topic,
        timestamp_source=args.timestamp_source,
    )


if __name__ == "__main__":
    main()
