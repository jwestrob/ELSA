#!/usr/bin/env python3
"""
Diagnose why a specific block fails to join a target cluster under mutual-Jaccard clustering.

Computes order-aware shingles for the focus block and all members of the target
cluster, applies DF filtering, and evaluates weighted Jaccard, low-DF support,
and IDF means to identify which thresholds block edges. Also probes orientation
effects (normal vs reversed) and canonical-strand shingles.

Usage:
  python tools/diagnose_block_vs_cluster.py \
    --config elsa.config.yaml \
    --blocks syntenic_analysis/syntenic_blocks.csv \
    --focus 288 \
    --cluster 1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import math
import numpy as np
import pandas as pd

from elsa.params import load_config as load_cfg
from elsa.manifest import ELSAManifest
from elsa.analysis import SyntenicAnalyzer
from elsa.analyze.shingles import srp_tokens, block_shingles


def load_lookup(cfg_path: Path):
    cfg = load_cfg(cfg_path)
    manifest = ELSAManifest(cfg.data.work_dir)
    analyzer = SyntenicAnalyzer(cfg, manifest)
    return analyzer._create_window_embed_lookup(), cfg


def parse_windows(row) -> List[str]:
    q = str(getattr(row, 'query_windows_json') or '')
    t = str(getattr(row, 'target_windows_json') or '')
    qw = [x for x in q.split(';') if x]
    tw = [x for x in t.split(';') if x]
    n = min(len(qw), len(tw))
    return qw[:n]


def parse_windows_both(row) -> List[str]:
    q = str(getattr(row, 'query_windows_json') or '')
    t = str(getattr(row, 'target_windows_json') or '')
    qw = [x for x in q.split(';') if x]
    tw = [x for x in t.split(';') if x]
    n = min(len(qw), len(tw))
    # Union both sides up to n on each to keep balance
    return (qw[:n] + tw[:n])


def tokens_for_wids(wids: List[str], lookup, cfg) -> List[List[int]]:
    embs = []
    for wid in wids:
        v = lookup(wid)
        if v is not None:
            embs.append(np.asarray(v).flatten())
    if not embs:
        return []
    mat = np.stack(embs, axis=0)
    return srp_tokens(
        mat,
        n_bits=getattr(cfg.analyze.clustering, 'srp_bits', 256),
        n_bands=getattr(cfg.analyze.clustering, 'srp_bands', 32),
        band_bits=getattr(cfg.cfg.analyze.clustering, 'srp_band_bits', 8) if hasattr(cfg, 'cfg') else getattr(cfg.analyze.clustering, 'srp_band_bits', 8),
        seed=getattr(cfg.analyze.clustering, 'srp_seed', 1337),
    )


def build_shingles(window_tokens: List[List[int]], cfg, reverse: bool = False, canonical: bool = False) -> Set[int]:
    if not window_tokens:
        return set()
    seq = window_tokens[::-1] if reverse else window_tokens
    method = getattr(cfg.analyze.clustering, 'shingle_method', 'xor')
    k = getattr(cfg.analyze.clustering, 'shingle_k', 3)
    # Optional fixed-subset logic for long blocks mirrors clusterer (approximate)
    enable_fixed = getattr(cfg.analyze.clustering, 'enable_fixed_subset_for_long', False)
    if enable_fixed:
        n_aln = len(seq)
        long_min_len = getattr(cfg.analyze.clustering, 'long_min_len', 20)
        if n_aln >= long_min_len:
            method = 'fixed_subset'
            k = getattr(cfg.analyze.clustering, 'fixed_subset_k', 2)
    return block_shingles(
        seq,
        k=k,
        method=method,
        bands_per_window=getattr(cfg.analyze.clustering, 'bands_per_window', 4),
        band_stride=getattr(cfg.analyze.clustering, 'band_stride', 7),
        fixed_bands=getattr(cfg.analyze.clustering, 'fixed_subset_bands', [0, 8, 16, 24]),
        icws_r=getattr(cfg.analyze.clustering, 'icws_r', 8),
        icws_bbit=getattr(cfg.analyze.clustering, 'icws_bbit', 0),
        seed=getattr(cfg.analyze.clustering, 'srp_seed', 1337),
        skipgram_offsets=None,
        strand_canonical_shingles=canonical,
    )


def weighted_jaccard(A: Set[int], B: Set[int], idf: Dict[int, float]) -> Tuple[float, int, float]:
    inter = A & B
    union = A | B
    if not union:
        return 0.0, 0, 0.0
    inter_w = sum(idf.get(s, 0.0) for s in inter)
    union_w = sum(idf.get(s, 0.0) for s in union)
    sim = (inter_w / union_w) if union_w > 0 else 0.0
    mean_idf = (inter_w / max(1, len(inter))) if inter else 0.0
    return sim, len(inter), mean_idf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, default=Path('elsa.config.yaml'))
    ap.add_argument('--blocks', type=Path, default=Path('syntenic_analysis/syntenic_blocks.csv'))
    ap.add_argument('--focus', type=int, required=True)
    ap.add_argument('--cluster', type=int, required=True)
    args = ap.parse_args()

    lookup, cfg = load_lookup(args.config)
    df = pd.read_csv(args.blocks)
    focus_row = df[df['block_id'] == int(args.focus)].iloc[0]
    target_rows = df[df['cluster_id'] == int(args.cluster)]
    if target_rows.empty:
        print('No blocks in target cluster')
        return

    # Build shingles for focus in modes
    w_focus = parse_windows(focus_row)
    w_focus_both = parse_windows_both(focus_row)
    tok_focus = tokens_for_wids(w_focus, lookup, cfg)
    tok_focus_both = tokens_for_wids(w_focus_both, lookup, cfg)
    focus_modes = {
        'normal': build_shingles(tok_focus, cfg, reverse=False, canonical=False),
        'reversed': build_shingles(tok_focus, cfg, reverse=True, canonical=False),
        'canonical': build_shingles(tok_focus, cfg, reverse=False, canonical=True),
        'both_sides': build_shingles(tok_focus_both, cfg, reverse=False, canonical=False),
    }

    # Build shingles for cluster members (normal only; try reversed per pair later by symmetric best)
    shingle_map: Dict[int, Set[int]] = {}
    for r in target_rows.itertuples(index=False):
        wid = parse_windows(r)
        wid_both = parse_windows_both(r)
        tok = tokens_for_wids(wid, lookup, cfg)
        tok_b = tokens_for_wids(wid_both, lookup, cfg)
        # Store both normal and both_sides for members too
        shingle_map[int(getattr(r, 'block_id'))] = build_shingles(tok, cfg, reverse=False, canonical=False)
        shingle_map[(int(getattr(r, 'block_id')), 'both')] = build_shingles(tok_b, cfg, reverse=False, canonical=False)

    # Build DF and IDF over focus+cluster sets
    all_sets = list(focus_modes.values()) + [v for k, v in shingle_map.items()]
    df_map: Dict[int, int] = {}
    for S in all_sets:
        for s in S:
            df_map[s] = df_map.get(s, 0) + 1
    # Apply DF filter with cluster df_max
    df_max = getattr(cfg.analyze.clustering, 'df_max', 200)
    filt = lambda S: {s for s in S if df_map.get(s, 0) <= df_max}
    focus_f = {k: filt(v) for k, v in focus_modes.items()}
    shingle_map_f = {bid: filt(S) for bid, S in shingle_map.items()}
    # IDF map
    n_docs = max(1, len(all_sets))
    idf = {s: math.log(1.0 + (n_docs / max(1, df_map.get(s, 1)))) for s in df_map.keys()}

    # Thresholds
    j_tau = getattr(cfg.analyze.clustering, 'jaccard_tau', 0.5)
    low_df_thr = max(10, df_max // 5)
    min_low_df = getattr(cfg.analyze.clustering, 'min_low_df_anchors', 3)
    idf_mean_min = getattr(cfg.analyze.clustering, 'idf_mean_min', 1.0)
    size_ratio_min = getattr(cfg.analyze.clustering, 'size_ratio_min', 0.5)
    size_ratio_max = getattr(cfg.analyze.clustering, 'size_ratio_max', 2.0)
    min_shared = getattr(cfg.analyze.clustering, 'min_shared_shingles', 2)

    # Evaluate
    print(f"Focus block {args.focus}: shingles sizes (raw→filt):",
          {k: f"{len(v)}→{len(focus_f[k])}" for k, v in focus_modes.items()})
    print(f"Cluster {args.cluster} members: {len(shingle_map)} blocks")

    rows = []
    # Precompute focus length and S sizes for size-ratio filter
    focus_len = int(getattr(focus_row, 'length') or len(w_focus))
    for key, S_c in shingle_map_f.items():
        if isinstance(key, tuple):
            cid, tag = key
        else:
            cid, tag = key, 'normal'
        c_row = target_rows[target_rows['block_id'] == cid].iloc[0]
        c_len = int(getattr(c_row, 'length') or len(parse_windows(c_row)))
        for mode, S_b in focus_f.items():
            # size ratio prefilter
            s_b = len(S_b)
            s_c = len(S_c)
            sh_ratio = (s_b / max(1, s_c)) if s_c else 0.0
            len_ratio = (focus_len / max(1, c_len)) if c_len else 0.0
            # shared count for candidate min_shared
            shared = len(S_b & S_c)
            sim, inter_count, mean_idf = weighted_jaccard(S_b, S_c, idf)
            low_df_count = sum(1 for s in (S_b & S_c) if df_map.get(s, 0) <= low_df_thr)
            rows.append({
                'cluster_member': cid,
                'member_mode': tag,
                'mode': mode,
                'shared': shared,
                'j_weighted': sim,
                'low_df_shared': low_df_count,
                'idf_mean_inter': mean_idf,
                'size_ratio_ok': (size_ratio_min <= sh_ratio <= size_ratio_max) and (size_ratio_min <= len_ratio <= size_ratio_max),
                'min_shared_ok': shared >= min_shared,
                'j_tau_ok': sim >= j_tau,
                'low_df_ok': low_df_count >= min_low_df,
                'idf_mean_ok': mean_idf >= idf_mean_min,
            })
    out = pd.DataFrame(rows)
    # Summarize best mode per member
    best = out.sort_values(['cluster_member', 'j_weighted'], ascending=[True, False]).groupby('cluster_member').head(1)
    print("\nTop 10 closest cluster members for focus:")
    print(best.sort_values('j_weighted', ascending=False).head(10).to_string(index=False))
    # Identify which gate fails most often among top-5
    topf = best.sort_values('j_weighted', ascending=False).head(5)
    if not topf.empty:
        fails = {
            'size_ratio_ok': int((~topf['size_ratio_ok']).sum()),
            'min_shared_ok': int((~topf['min_shared_ok']).sum()),
            'j_tau_ok': int((~topf['j_tau_ok']).sum()),
            'low_df_ok': int((~topf['low_df_ok']).sum()),
            'idf_mean_ok': int((~topf['idf_mean_ok']).sum()),
        }
        print("\nFailure counts among top-5 (by weighted Jaccard):", fails)
    # Save detailed CSV
    diag_path = Path('syntenic_analysis') / f'diag_block_{args.focus}_vs_cluster_{args.cluster}.csv'
    out.to_csv(diag_path, index=False)
    print(f"\nDetailed diagnostics written to {diag_path}")


if __name__ == '__main__':
    main()
