#!/usr/bin/env python3
"""
General, PFAM-agnostic attach step: promote sink blocks into existing clusters
using cluster-level union signatures (bandset and k=1 shingles) and strict
containment thresholds. Keeps global clustering compact; attaches only when a
single cluster stands out with strong evidence.

Algorithm (fast):
- Load existing syntenic_blocks.csv and a config (for SRP params)
- Build per-cluster union signatures:
  U_bandset[c] = ∪ bandset(block)
  U_k1[c]      = ∪ k1(block)
  Also sample up to M member blocks' k1 sets for triangle checks
- For each sink block b (cluster_id==0):
  - Compute bandset B_b and k1 S_b
  - For each cluster c, score:
      contain_b = |B_b ∩ U_bandset[c]| / |B_b|
      contain_s = |S_b ∩ U_k1[c]| / |S_b|
      score(c) = max(contain_b, contain_s)
  - Let c* = argmax score(c); c2 = 2nd best
  - Accept if score(c*) >= tau_main, margin (score(c*)-score(c2)) >= margin_min,
    inter_k1 with U_k1[c*] >= inter_min, and triangle support: b overlaps (k1)
    with at least tri_min member blocks in c* with per-member contain >= tau_member.
  - Reassign b to c*

Parameters tune trade-off between recall and purity.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
import random
import pickle

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


def block_window_ids(row) -> List[str]:
    q = str(getattr(row, 'query_windows_json') or '')
    t = str(getattr(row, 'target_windows_json') or '')
    qw = [x for x in q.split(';') if x]
    tw = [x for x in t.split(';') if x]
    n = min(len(qw), len(tw))
    return qw[:n]


def tokens_for_block(row, lookup, cfg) -> List[int]:
    wins = block_window_ids(row)
    embs = []
    for wid in wins:
        v = lookup(wid)
        if v is not None:
            embs.append(np.asarray(v).flatten())
    if not embs:
        return []
    mat = np.stack(embs, axis=0)
    # Optionally ignore strand dimension for tokenization
    try:
        if getattr(cfg.analyze.clustering, 'ignore_strand_in_tokens', False) and mat.shape[1] > 0:
            mat[:, -1] = 0
    except Exception:
        pass
    return srp_tokens(
        mat,
        n_bits=getattr(cfg.analyze.clustering, 'srp_bits', 256),
        n_bands=getattr(cfg.analyze.clustering, 'srp_bands', 32),
        band_bits=getattr(cfg.analyze.clustering, 'srp_band_bits', 8),
        seed=getattr(cfg.analyze.clustering, 'srp_seed', 1337),
    )


def union_signatures_for_clusters(
    df: pd.DataFrame,
    lookup,
    cfg,
    member_sample: int = 5,
    k1_method: str = 'xor',
    icws_r: int = 8,
    icws_bbit: int = 0,
):
    clusters = sorted([c for c in df['cluster_id'].unique() if c != 0])
    U_bandset: Dict[int, Set[int]] = {}
    U_k1: Dict[int, Set[int]] = {}
    members_k1: Dict[int, List[Set[int]]] = {}
    for cid in clusters:
        sub = df[df['cluster_id'] == cid]
        Ub: Set[int] = set()
        Us: Set[int] = set()
        mk1: List[Set[int]] = []
        rows = list(sub.itertuples(index=False))
        if member_sample > 0 and len(rows) > member_sample:
            rows = random.sample(rows, member_sample)
        for row in rows:
            toks = tokens_for_block(row, lookup, cfg)
            if not toks:
                continue
            bset = block_shingles(toks, k=1, method='bandset')
            if k1_method == 'icws':
                k1 = block_shingles(toks, k=1, method='icws', icws_r=icws_r, icws_bbit=icws_bbit)
            else:
                k1 = block_shingles(toks, k=1, method='xor')
            Ub |= bset
            Us |= k1
            mk1.append(k1)
        U_bandset[cid] = Ub
        U_k1[cid] = Us
        members_k1[cid] = mk1
    return U_bandset, U_k1, members_k1


def window_tokens_for_wids(wids, lookup, cfg):
    embs = []
    for wid in wids:
        v = lookup(wid)
        if v is not None:
            embs.append(np.asarray(v).flatten())
    if not embs:
        return []
    mat = np.stack(embs, axis=0)
    try:
        if getattr(cfg.analyze.clustering, 'ignore_strand_in_tokens', False) and mat.shape[1] > 0:
            mat[:, -1] = 0
    except Exception:
        pass
    return srp_tokens(
        mat,
        n_bits=getattr(cfg.analyze.clustering, 'srp_bits', 256),
        n_bands=getattr(cfg.analyze.clustering, 'srp_bands', 32),
        band_bits=getattr(cfg.analyze.clustering, 'srp_band_bits', 8),
        seed=getattr(cfg.analyze.clustering, 'srp_seed', 1337),
    )


def stitched_window_ids(row, sink_df: pd.DataFrame, max_gap: int = 2, max_neighbors: int = 2) -> list[str]:
    """Return a stitched list of window IDs around this row within small gaps.

    Stitches neighbors in the same query/target locus whose query and target
    window starts are within max_gap of this row's ends. Limits to max_neighbors
    total to keep locality and safety.
    """
    q = str(getattr(row, 'query_windows_json') or '')
    t = str(getattr(row, 'target_windows_json') or '')
    qw = [x for x in q.split(';') if x]
    tw = [x for x in t.split(';') if x]
    n = min(len(qw), len(tw))
    base_qw = qw[:n]

    qloc = getattr(row, 'query_locus')
    tloc = getattr(row, 'target_locus')
    q_end = int(getattr(row, 'query_window_end') or 0)
    t_end = int(getattr(row, 'target_window_end') or 0)

    neighbors = []
    sub = sink_df[(sink_df['query_locus'] == qloc) & (sink_df['target_locus'] == tloc)]
    sub = sub.assign(_qstart=sub['query_window_start'].astype(int))
    sub = sub.sort_values(by='_qstart')
    for r in sub.itertuples(index=False):
        if int(getattr(r, 'block_id')) == int(getattr(row, 'block_id')):
            continue
        qs = int(getattr(r, 'query_window_start') or 0)
        ts = int(getattr(r, 'target_window_start') or 0)
        if 0 <= qs - q_end <= max_gap and 0 <= ts - t_end <= max_gap:
            neighbors.append(r)
        if len(neighbors) >= max_neighbors:
            break

    wins = list(base_qw)
    for r in neighbors:
        qj = str(getattr(r, 'query_windows_json') or '')
        tj = str(getattr(r, 'target_windows_json') or '')
        qwj = [x for x in qj.split(';') if x]
        twj = [x for x in tj.split(';') if x]
        m = min(len(qwj), len(twj))
        wins.extend(qwj[:m])
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', '-c', type=Path, default=Path('elsa.config.yaml'))
    ap.add_argument('--blocks', type=Path, default=Path('syntenic_analysis/syntenic_blocks.csv'))
    ap.add_argument('--out', type=Path, default=Path('syntenic_analysis/syntenic_blocks.csv'))
    # thresholds
    ap.add_argument('--bandset_contain_tau', type=float, default=0.65)
    ap.add_argument('--k1_contain_tau', type=float, default=0.65)
    ap.add_argument('--k1_inter_min', type=int, default=2)
    ap.add_argument('--margin_min', type=float, default=0.10)
    ap.add_argument('--triangle_min', type=int, default=1)
    ap.add_argument('--triangle_member_tau', type=float, default=0.50)
    ap.add_argument('--member_sample', type=int, default=5)
    ap.add_argument('--limit_member_sample', type=int, default=0, help='If loading signatures, cap triangle members to this many (0=no cap)')
    # Signatures cache
    ap.add_argument('--save_signatures', type=Path, default=None, help='Path to save precomputed union signatures (pickle)')
    ap.add_argument('--load_signatures', type=Path, default=None, help='Path to load precomputed union signatures (pickle)')
    ap.add_argument('--signatures_only', action='store_true', help='Only build/save signatures and exit (no attachments)')
    # Tiny-block handling
    ap.add_argument('--tiny_window_cap', type=int, default=2)
    ap.add_argument('--bandset_contain_tau_tiny', type=float, default=0.60)
    ap.add_argument('--k1_contain_tau_tiny', type=float, default=0.60)
    ap.add_argument('--k1_inter_min_tiny', type=int, default=2)
    ap.add_argument('--margin_min_tiny', type=float, default=0.08)
    ap.add_argument('--triangle_min_tiny', type=int, default=2)
    ap.add_argument('--triangle_member_tau_tiny', type=float, default=0.55)
    # Stitching
    ap.add_argument('--enable_stitch', action='store_true')
    ap.add_argument('--stitch_gap', type=int, default=2)
    ap.add_argument('--stitch_max_neighbors', type=int, default=2)
    # K1 method
    ap.add_argument('--k1_method', type=str, default='xor', choices=['xor', 'icws'])
    ap.add_argument('--icws_r', type=int, default=8)
    ap.add_argument('--icws_bbit', type=int, default=0)
    # Target cluster constraints
    ap.add_argument('--max_target_cluster_size', type=int, default=0, help='If >0, only consider clusters with size <= this many blocks')

    args = ap.parse_args()

    df = pd.read_csv(args.blocks)
    non_sink = df[df['cluster_id'] != 0]
    sink = df[df['cluster_id'] == 0]
    if non_sink.empty or sink.empty:
        print('Nothing to attach (no clusters or no sink blocks).')
        return

    lookup, cfg = load_lookup(args.config)
    # Load or build signatures
    if args.load_signatures is not None and Path(args.load_signatures).exists():
        with open(args.load_signatures, 'rb') as f:
            payload = pickle.load(f)
        U_bandset = payload['U_bandset']
        U_k1 = payload['U_k1']
        members_k1 = payload['members_k1']
        meta = payload.get('meta', {})
        # Validate method compatibility
        if meta.get('k1_method') != args.k1_method:
            raise ValueError(f"Signatures k1_method={meta.get('k1_method')} != requested {args.k1_method}")
        if args.k1_method == 'icws':
            if int(meta.get('icws_r', -1)) != int(args.icws_r) or int(meta.get('icws_bbit', -1)) != int(args.icws_bbit):
                raise ValueError("ICWS signatures parameters mismatch")
        # Optionally cap member samples per cluster
        if args.limit_member_sample and args.limit_member_sample > 0:
            cap = int(args.limit_member_sample)
            for cid in list(members_k1.keys()):
                members_k1[cid] = members_k1[cid][:cap]
    else:
        U_bandset, U_k1, members_k1 = union_signatures_for_clusters(
            non_sink, lookup, cfg, args.member_sample, args.k1_method, args.icws_r, args.icws_bbit
        )
        # Save if requested
        if args.save_signatures is not None:
            meta = {
                'k1_method': args.k1_method,
                'icws_r': int(args.icws_r),
                'icws_bbit': int(args.icws_bbit),
            }
            with open(args.save_signatures, 'wb') as f:
                pickle.dump({'U_bandset': U_bandset, 'U_k1': U_k1, 'members_k1': members_k1, 'meta': meta}, f)
            if args.signatures_only:
                print(f"Saved union signatures to {args.save_signatures}")
                return
    if args.signatures_only:
        raise ValueError("--signatures_only requires --save_signatures to be set")

    # Optional candidate target cluster filter by size
    allowed_cids = set()
    if args.max_target_cluster_size and args.max_target_cluster_size > 0:
        sizes = non_sink.groupby('cluster_id').size().to_dict()
        allowed_cids = {int(cid) for cid, sz in sizes.items() if int(cid) != 0 and int(sz) <= int(args.max_target_cluster_size)}
    attached = []
    for row in sink.itertuples(index=False):
        block_id = int(getattr(row, 'block_id'))
        # Optionally stitch nearby windows to strengthen tiny blocks
        if args.enable_stitch:
            wids = stitched_window_ids(row, sink, max_gap=args.stitch_gap, max_neighbors=args.stitch_max_neighbors)
            toks = window_tokens_for_wids(wids, lookup, cfg)
        else:
            toks = tokens_for_block(row, lookup, cfg)
        if not toks:
            continue
        B = block_shingles(toks, k=1, method='bandset')
        if args.k1_method == 'icws':
            S = block_shingles(toks, k=1, method='icws', icws_r=args.icws_r, icws_bbit=args.icws_bbit)
        else:
            S = block_shingles(toks, k=1, method='xor')
        if not B and not S:
            continue
        # Determine if tiny and select thresholds accordingly
        n_wins = len(toks)
        tiny = n_wins <= args.tiny_window_cap
        band_tau = args.bandset_contain_tau_tiny if tiny else args.bandset_contain_tau
        k1_tau = args.k1_contain_tau_tiny if tiny else args.k1_contain_tau
        inter_min = args.k1_inter_min_tiny if tiny else args.k1_inter_min
        margin_min = args.margin_min_tiny if tiny else args.margin_min
        tri_min = args.triangle_min_tiny if tiny else args.triangle_min
        tri_member_tau = args.triangle_member_tau_tiny if tiny else args.triangle_member_tau
        scores: List[Tuple[int, float, int]] = []  # (cid, score, inter_k1)
        for cid in U_bandset.keys():
            if allowed_cids and cid not in allowed_cids:
                continue
            Ub = U_bandset[cid]
            Us = U_k1[cid]
            contain_b = (len(B & Ub) / max(1, len(B))) if B else 0.0
            inter_s = len(S & Us) if S else 0
            contain_s = (inter_s / max(1, len(S))) if S else 0.0
            sc = max(contain_b, contain_s)
            scores.append((cid, sc, inter_s))
        if not scores:
            continue
        scores.sort(key=lambda x: -x[1])
        cid_best, score_best, inter_best = scores[0]
        score_second = scores[1][1] if len(scores) > 1 else 0.0
        if score_best < max(band_tau, k1_tau):
            continue
        if score_best - score_second < margin_min:
            continue
        if inter_best < inter_min:
            continue
        # triangle support with sampled members (k1 containment per member)
        tri = 0
        for mset in members_k1.get(cid_best, []):
            if not mset:
                continue
            contain_member = len(S & mset) / max(1, len(S)) if S else 0.0
            if contain_member >= tri_member_tau:
                tri += 1
        if tri < tri_min:
            continue
        attached.append((block_id, cid_best))

    if not attached:
        print('No sink blocks met attachment thresholds.')
        return

    # Reassign
    attach_ids = {bid: cid for bid, cid in attached}
    df['cluster_id'] = [attach_ids.get(int(b), c) for b, c in zip(df['block_id'], df['cluster_id'])]
    df.to_csv(args.out, index=False)
    print(f'Attached {len(attached)} sink blocks. Clusters unchanged; reassignments applied.')

if __name__ == '__main__':
    main()
