#!/usr/bin/env python3
"""
Grid search clustering parameters (XOR path) starting from a base YAML
profile, and evaluate whether a single cluster captures canonical
ribosomal protein loci across all genomes.

Reference profile: configs/_composed_control_strandcanon.yaml

Evaluation:
- Define RP markers (PFAM-like names).
- For each (block_id, genome_id), count distinct RP markers in that block's
  genes for that genome.
- Mark (block, genome) as canonical if count >= rp_marker_threshold.
- A cluster is a SUCCESS if, across its blocks, canonical hits cover all
  genomes present in the database.

Outputs: prints table of param settings → success/coverage stats; highlights
the best.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

BASE_CONFIG = Path("configs/_composed_control_strandcanon.yaml")
DB_PATH = Path("genome_browser/genome_browser.db")

RP_MARKERS = [
    "Ribosomal_L2","Ribosomal_L3","Ribosomal_L4","Ribosomal_L5","Ribosomal_L6",
    "Ribosomal_L14","Ribosomal_L16","Ribosomal_L18p","Ribosomal_L22","Ribosomal_L23",
    "Ribosomal_L24","Ribosomal_L29","Ribosomal_L30","Ribosomal_L34",
    "Ribosomal_S3","Ribosomal_S5","Ribosomal_S7","Ribosomal_S8","Ribosomal_S10",
    "Ribosomal_S14","Ribosomal_S17","Ribosomal_S19",
]


def load_yaml(p: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML not installed.")
    with open(p, 'r') as f:
        return yaml.safe_load(f) or {}


def dump_yaml(obj: dict, p: Path) -> None:
    with open(p, 'w') as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run_analyze(config_path: Path) -> int:
    print(f"\n>>> Running elsa analyze with {config_path}")
    # Disable genome browser setup to speed up grid (uses new dual-flag)
    res = subprocess.run(['elsa', 'analyze', '-c', str(config_path), '--no-setup-genome-browser'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    return res.returncode


def evaluate_db(rp_marker_threshold: int = 4, exclude_sink: bool = True, sink_id: int = 0):
    if not DB_PATH.exists():
        return {"success": False, "reason": "DB missing"}
    conn = sqlite3.connect(str(DB_PATH))
    try:
        # Fetch genomes set
        genomes = [row[0] for row in conn.execute("SELECT DISTINCT genome_id FROM genes")] 
        genomes = [g for g in genomes if g is not None]
        n_genomes = len(set(genomes))

        # Gather RP hits per (block, genome)
        terms = [f"%{m.lower()}%" for m in RP_MARKERS]
        like_clause = " OR ".join(["LOWER(g.pfam_domains) LIKE ?" for _ in RP_MARKERS])
        sql = f"""
        SELECT gbm.block_id, sb.cluster_id, g.genome_id, g.pfam_domains
        FROM gene_block_mappings gbm
        JOIN genes g ON gbm.gene_id = g.gene_id
        JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id
        WHERE {like_clause}
        """
        rows = conn.execute(sql, terms).fetchall()
        # block->cluster
        block2cluster = {}
        # cluster -> genome -> count of markers
        cluster_genome_markers: dict[int, dict[str, set[str]]] = {}
        for block_id, cluster_id, genome_id, pf in rows:
            if cluster_id is None:
                continue
            block2cluster[int(block_id)] = int(cluster_id)
            cid = int(cluster_id)
            if cid not in cluster_genome_markers:
                cluster_genome_markers[cid] = {}
            gmap = cluster_genome_markers[cid]
            gmap.setdefault(genome_id, set())
            pf_l = str(pf or '').lower()
            for m in RP_MARKERS:
                if m.lower() in pf_l:
                    gmap[genome_id].add(m)

        # Evaluate clusters
        best = None
        successes = []
        for cid, gmap in cluster_genome_markers.items():
            if exclude_sink and int(cid) == int(sink_id):
                continue
            covered = [g for g, ms in gmap.items() if len(ms) >= rp_marker_threshold]
            coverage = len(set(covered))
            ok = (coverage >= n_genomes)
            rec = {"cluster_id": cid, "coverage": coverage, "n_genomes": n_genomes}
            if ok:
                successes.append(rec)
            if best is None or coverage > best["coverage"]:
                best = rec
        return {"success": bool(successes), "successes": successes, "best": best}
    finally:
        conn.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=BASE_CONFIG, help='Base YAML config')
    p.add_argument('--grid', type=str, default='{"jaccard_tau":[0.45,0.5],"df_max":[200,400],"min_shared_shingles":[1]}',
                   help='JSON dict of analyze.clustering overrides to grid-search')
    p.add_argument('--rp_marker_threshold', type=int, default=4)
    p.add_argument('--exclude_sink', action='store_true', default=True, help='Exclude sink cluster (cluster_id=0) from success evaluation (default: True)')
    p.add_argument('--sink_id', type=int, default=0, help='Sink cluster id to exclude when --exclude_sink is set')
    p.add_argument('--max_clusters_cap', type=int, default=200, help='Skip settings yielding > this many clusters')
    args = p.parse_args()

    base = load_yaml(args.base)
    grid = json.loads(args.grid)

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    print(f"Grid has {len(combos)} combos: {keys} → {values}")

    results = []
    for vals in combos:
        overrides = dict(zip(keys, vals))
        cfg = deepcopy(base)
        clustering = cfg.get('analyze', {}).get('clustering', {})
        for k, v in overrides.items():
            clustering[k] = v
        cfg['analyze']['clustering'] = clustering
        # Write temp config
        tmp = Path(tempfile.mkdtemp())
        cfg_path = tmp / 'config.yaml'
        dump_yaml(cfg, cfg_path)
        # Run analyze
        code = run_analyze(cfg_path)
        if code != 0:
            print(f"Run failed for overrides={overrides}")
            continue
        # Quick check: read cluster count from syntenic_clusters.csv
        try:
            import pandas as pd
            df = pd.read_csv('syntenic_analysis/syntenic_clusters.csv')
            n_clusters = len(df)
        except Exception:
            n_clusters = -1
        if n_clusters > args.max_clusters_cap and args.max_clusters_cap > 0:
            print(f"Skipping evaluation; too many clusters ({n_clusters}) for overrides={overrides}")
            results.append({"overrides": overrides, "n_clusters": n_clusters, "skipped": True})
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        # Evaluate DB
        eval_res = evaluate_db(args.rp_marker_threshold, exclude_sink=args.exclude_sink, sink_id=args.sink_id)
        rec = {"overrides": overrides, "n_clusters": n_clusters, **eval_res}
        results.append(rec)
        print(f"Eval: overrides={overrides} → success={eval_res['success']} best={eval_res.get('best')}")
        shutil.rmtree(tmp, ignore_errors=True)

    # Summarize
    print("\n=== Grid Summary ===")
    for rec in results:
        print(json.dumps(rec, indent=2))


if __name__ == '__main__':
    main()
