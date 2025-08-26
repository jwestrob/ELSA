#!/usr/bin/env python3
"""
Merge default and adaptive cluster assignments with sink-adoption rules.

Usage:
  python tools/merge_cluster_assignments.py \
    --default default_assignments.csv \
    --adaptive adaptive_assignments.csv \
    --out merged_assignments.csv \
    [--min_adaptive_cluster_size 2]

CSV format expected:
  block_id,cluster_id

Merging rule:
  - Keep default assignments for all blocks except those in sink (cluster_id==0).
  - For default-sink blocks, adopt adaptive cluster if >0 and the adaptive cluster
    size >= min_adaptive_cluster_size (default: 1).

This script is file-based and does not touch the database.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict


def read_assignments(path: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                bid = int(row[0])
                cid = int(row[1])
            except Exception:
                continue
            mapping[bid] = cid
    return mapping


def write_assignments(path: str, mapping: dict[int, int]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block_id", "cluster_id"])
        for bid in sorted(mapping.keys()):
            w.writerow([bid, mapping[bid]])


def main():
    ap = argparse.ArgumentParser(description="Merge cluster assignments from default and adaptive runs")
    ap.add_argument("--default", required=True, help="CSV of default assignments: block_id,cluster_id")
    ap.add_argument("--adaptive", required=True, help="CSV of adaptive assignments: block_id,cluster_id")
    ap.add_argument("--out", required=True, help="Output CSV for merged assignments")
    ap.add_argument("--min_adaptive_cluster_size", type=int, default=1, help="Require adaptive cluster size >= N")
    args = ap.parse_args()

    default_map = read_assignments(args.default)
    adaptive_map = read_assignments(args.adaptive)

    # Compute adaptive cluster sizes
    adaptive_sizes: dict[int, int] = defaultdict(int)
    for cid in adaptive_map.values():
        if cid and cid > 0:
            adaptive_sizes[cid] += 1

    merged: dict[int, int] = {}
    all_blocks = set(default_map.keys()) | set(adaptive_map.keys())
    for bid in all_blocks:
        dcid = int(default_map.get(bid, 0))
        acid = int(adaptive_map.get(bid, 0))
        if dcid == 0 and acid > 0 and adaptive_sizes.get(acid, 0) >= args.min_adaptive_cluster_size:
            merged[bid] = acid
        else:
            merged[bid] = dcid if bid in default_map else 0

    write_assignments(args.out, merged)
    print(f"Merged {len(merged)} assignments -> {args.out}")


if __name__ == "__main__":
    main()

