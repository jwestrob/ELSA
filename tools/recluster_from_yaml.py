#!/usr/bin/env python3
"""
Run SRP mutual-Jaccard reclustering using a YAML config profile.
This script maps relevant fields from analyze.clustering into the recluster_srp CLI.
"""
from pathlib import Path
import argparse
import subprocess
import sys
import json

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML not installed; please `pip install pyyaml` or use direct CLI flags.")
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='YAML file with analyze.clustering settings')
    p.add_argument('--blocks', default='syntenic_analysis/syntenic_blocks.csv')
    p.add_argument('--windows', default='elsa_index/shingles/windows.parquet')
    p.add_argument('--db', default='genome_browser/genome_browser.db')
    args = p.parse_args()

    cfg = load_yaml(Path(args.config))
    clustering = ((cfg or {}).get('analyze') or {}).get('clustering') or {}

    # Build command line for recluster tool
    cmd = [
        sys.executable, 'tools/recluster_srp.py',
        '--blocks', args.blocks,
        '--windows', args.windows,
        '--db', args.db,
        '--tau', str(clustering.get('jaccard_tau', 0.5)),
        '--mutual_k', str(clustering.get('mutual_k', 3)),
        '--df_max', str(clustering.get('df_max', 200)),
        '--degree_cap', '10',
        '--shingle_k', str(clustering.get('shingle_k', 3)),
        '--shingle_method', str(clustering.get('shingle_method', 'xor')),
        '--bands_per_window', str(clustering.get('bands_per_window', 4)),
        '--band_stride', str(clustering.get('band_stride', 7)),
    ]

    # Hybrid switches
    if clustering.get('enable_hybrid_bandset', False):
        cmd.append('--enable_hybrid_bandset')
    cmd += [
        '--bandset_tau', str(clustering.get('bandset_tau', 0.25)),
        '--bandset_df_max', str(clustering.get('bandset_df_max', 3000)),
        '--bandset_min_len', str(clustering.get('bandset_min_len', 20)),
        '--bandset_min_identity', str(clustering.get('bandset_min_identity', 0.98)),
    ]

    # Pruning controls
    if clustering.get('enable_mutual_topk_filter', False):
        cmd.append('--enable_mutual_topk_filter')
    cmd += [
        '--max_candidates_per_block', str(clustering.get('max_candidates_per_block', 100000)),
        '--min_shared_shingles', str(clustering.get('min_shared_shingles', 1)),
        '--bandset_topk_candidates', str(clustering.get('bandset_topk_candidates', 100000)),
        '--min_shared_band_tokens', str(clustering.get('min_shared_band_tokens', 1)),
    ]

    print('Running:', ' '.join(cmd))
    res = subprocess.run(cmd, check=False)
    sys.exit(res.returncode)


if __name__ == '__main__':
    main()

