#!/usr/bin/env python3
"""
Batch runner for the PFAM-agnostic attach experiments.

Runs multiple configurations of tools/attach_by_cluster_signatures.py
on copies of syntenic_blocks, then evaluates compactness and curated RP purity
via tools/evaluate_curated_rp_purity.py. Prints per-case summaries.

Usage:
  python tools/run_attach_grid.py \
    --config elsa.config.yaml \
    --blocks syntenic_analysis/syntenic_blocks.csv \
    --db genome_browser/genome_browser.db

Outputs new CSVs alongside the input, suffixed by case names (exp1..).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import subprocess


def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    out = (res.stdout or '') + (res.stderr or '')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=Path('elsa.config.yaml'))
    ap.add_argument('--blocks', type=Path, default=Path('syntenic_analysis/syntenic_blocks.csv'))
    ap.add_argument('--db', type=Path, default=Path('genome_browser/genome_browser.db'))
    args = ap.parse_args()

    base = args.blocks.with_suffix('.base.csv')
    if not base.exists():
        shutil.copyfile(args.blocks, base)

    cases: list[tuple[str, list[str]]] = [
        (
            'exp1',
            [
                '--enable_stitch', '--stitch_gap', '2', '--stitch_max_neighbors', '2',
                '--tiny_window_cap', '2', '--bandset_contain_tau_tiny', '0.60', '--k1_contain_tau_tiny', '0.60',
                '--triangle_min_tiny', '2', '--triangle_member_tau_tiny', '0.55', '--margin_min_tiny', '0.08',
            ],
        ),
        (
            'exp2',
            [
                '--enable_stitch', '--stitch_gap', '2', '--stitch_max_neighbors', '2',
                '--tiny_window_cap', '2', '--bandset_contain_tau_tiny', '0.55', '--k1_contain_tau_tiny', '0.55',
                '--triangle_min_tiny', '1', '--triangle_member_tau_tiny', '0.50', '--margin_min_tiny', '0.05',
            ],
        ),
        (
            'exp3',
            [
                '--enable_stitch', '--stitch_gap', '3', '--stitch_max_neighbors', '3',
                '--tiny_window_cap', '2', '--bandset_contain_tau_tiny', '0.55', '--k1_contain_tau_tiny', '0.55',
                '--triangle_min_tiny', '1', '--triangle_member_tau_tiny', '0.50', '--margin_min_tiny', '0.05',
            ],
        ),
        (
            'exp4',
            [
                '--enable_stitch', '--stitch_gap', '2', '--stitch_max_neighbors', '2',
                '--tiny_window_cap', '2', '--bandset_contain_tau_tiny', '0.60', '--k1_contain_tau_tiny', '0.60',
                '--triangle_min_tiny', '2', '--triangle_member_tau_tiny', '0.55', '--margin_min_tiny', '0.08',
                '--k1_method', 'icws', '--icws_r', '8', '--icws_bbit', '0',
            ],
        ),
        (
            'exp5',
            [
                '--enable_stitch', '--stitch_gap', '2', '--stitch_max_neighbors', '2',
                '--tiny_window_cap', '2', '--bandset_contain_tau_tiny', '0.60', '--k1_contain_tau_tiny', '0.60',
                '--triangle_min_tiny', '2', '--triangle_member_tau_tiny', '0.55', '--margin_min_tiny', '0.08',
                '--k1_method', 'icws', '--icws_r', '12', '--icws_bbit', '0',
            ],
        ),
        (
            'exp6',
            [
                '--enable_stitch', '--stitch_gap', '2', '--stitch_max_neighbors', '2',
                '--tiny_window_cap', '2', '--bandset_contain_tau_tiny', '0.58', '--k1_contain_tau_tiny', '0.58',
                '--triangle_min_tiny', '1', '--triangle_member_tau_tiny', '0.55', '--margin_min_tiny', '0.10',
            ],
        ),
    ]

    print('Running attach grid...')

    # Precompute and cache union signatures to avoid recomputation per run
    sig_dir = args.blocks.parent
    sig_xor = sig_dir / 'attach_sigs_xor.pkl'
    sig_icws_r8 = sig_dir / 'attach_sigs_icws_r8.pkl'
    sig_icws_r12 = sig_dir / 'attach_sigs_icws_r12.pkl'

    def ensure_sig(path: Path, k1_method: str, icws_r: int = 0):
        if path.exists():
            return
        # Build once from base blocks; save and exit quickly
        cmd = [
            'python', 'tools/attach_by_cluster_signatures.py',
            '--config', str(args.config),
            '--blocks', str(base),
            '--out', str(base),
            '--save_signatures', str(path),
            '--signatures_only',
            '--member_sample', '8',
            '--k1_method', k1_method,
        ]
        if k1_method == 'icws':
            cmd += ['--icws_r', str(icws_r), '--icws_bbit', '0']
        print('Precomputing signatures:', ' '.join(cmd))
        out = run(cmd)
        print(out.strip())

    ensure_sig(sig_xor, 'xor')
    ensure_sig(sig_icws_r8, 'icws', 8)
    ensure_sig(sig_icws_r12, 'icws', 12)
    for name, extra in cases:
        out_csv = args.blocks.with_name(f"{args.blocks.stem}.{name}.csv")
        shutil.copyfile(base, out_csv)
        print(f"\n== {name} ==")
        attach_cmd = [
            'python', 'tools/attach_by_cluster_signatures.py',
            '--config', str(args.config),
            '--blocks', str(out_csv),
            '--out', str(out_csv),
        ] + extra

        # Attach appropriate precomputed signatures
        if '--k1_method' in extra:
            # pick icws_r
            try:
                idx = extra.index('--k1_method')
                method = extra[idx + 1]
            except Exception:
                method = 'xor'
        else:
            method = 'xor'
        sig_path = sig_xor
        if method == 'icws':
            # default r=8; check if r=12 passed
            r = 8
            if '--icws_r' in extra:
                try:
                    r = int(extra[extra.index('--icws_r') + 1])
                except Exception:
                    r = 8
            sig_path = sig_icws_r12 if r >= 12 else sig_icws_r8
        attach_cmd += ['--load_signatures', str(sig_path), '--limit_member_sample', '5']
        attach_out = run(attach_cmd)
        print(attach_out.strip())
        eval_cmd = [
            'python', 'tools/evaluate_curated_rp_purity.py',
            '--blocks', str(out_csv),
            '--db', str(args.db),
        ]
        eval_out = run(eval_cmd)
        # Print just the key lines
        lines = [ln for ln in eval_out.splitlines() if ln.strip()]
        for ln in lines[:12]:
            print(ln)


if __name__ == '__main__':
    main()

