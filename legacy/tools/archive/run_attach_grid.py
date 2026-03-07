#!/usr/bin/env python3
"""
ARCHIVED: Batch runner for PFAM-agnostic attach experiments.

This script is retained for historical reference. The current workflow favors a
single, well-tuned attach pass (exp8-like) rather than gridsearching parameters.

Original help:
  python tools/run_attach_grid.py     --config elsa.config.yaml     --blocks syntenic_analysis/syntenic_blocks.csv     --db genome_browser/genome_browser.db
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import subprocess


def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return (res.stdout or '') + (res.stderr or '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=Path('elsa.config.yaml'))
    ap.add_argument('--blocks', type=Path, default=Path('syntenic_analysis/syntenic_blocks.csv'))
    ap.add_argument('--db', type=Path, default=Path('genome_browser/genome_browser.db'))
    args = ap.parse_args()

    base = args.blocks.with_suffix('.base.csv')
    if not base.exists():
        shutil.copyfile(args.blocks, base)

    # Minimal demonstration run (formerly a grid)
    out_csv = args.blocks.with_name(f"{args.blocks.stem}.exp_demo.csv")
    shutil.copyfile(base, out_csv)
    print("Running archived demo attach…")
    attach_cmd = [
        'python', 'tools/attach_by_cluster_signatures.py',
        '--config', str(args.config),
        '--blocks', str(out_csv),
        '--out', str(out_csv),
        '--enable_stitch', '--stitch_gap', '2', '--stitch_max_neighbors', '2',
        '--tiny_window_cap', '3', '--bandset_contain_tau_tiny', '0.55', '--k1_contain_tau_tiny', '0.55',
        '--k1_inter_min_tiny', '1', '--triangle_min_tiny', '1', '--triangle_member_tau_tiny', '0.50', '--margin_min_tiny', '0.05'
    ]
    print(run(attach_cmd))
    eval_cmd = [
        'python', 'tools/evaluate_curated_rp_purity.py',
        '--blocks', str(out_csv),
        '--db', str(args.db),
    ]
    print(run(eval_cmd))


if __name__ == '__main__':
    main()
