#!/usr/bin/env python3
"""
Content-driven clustering pipeline for syntenic blocks (PFAM-based first pass).

Builds a sparse similarity graph over blocks using IDF-weighted Jaccard on
per-block PFAM domain sets derived from gene_block_mappings → genes.pfam_domains.

Removes all hard-coded cluster sizes/representatives; assigns cluster IDs by
community detection over the content graph. Isolates remain cluster_id=0.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Set
import math

import pandas as pd
import sqlite3

def compute_cluster_explorer_summary(conn: sqlite3.Connection,
                                     cluster_id: int,
                                     min_core_coverage: float = 0.6,
                                     df_percentile_ban: float = 0.9,
                                     max_tokens_preview: int = 0) -> Dict:
    """Build a compact, versioned summary for cluster explorer cards.

    Schema v1 fields:
      - cluster_id, size, consensus_length, identity_est, organisms, representatives
      - consensus (ordered pfam tokens with coverage/pos/df/fwd_frac/n_occ)
      - adjacency_support (neighbor pairs with same_frac/support)
      - directional_consensus {agree_frac, status}
      - preview {tokens:[{pfam,coverage,pos,color}], hint}
      - quality (heuristic 0..1)
      - caveats (optional list)
    """
    # Basic cluster info
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT size, consensus_length, consensus_score, representative_query, representative_target
        FROM clusters WHERE cluster_id = ?
        """,
        (int(cluster_id),)
    ).fetchone()
    size, consensus_length, consensus_score, rep_q, rep_t = (row if row else (0, 0, 0.0, None, None))
    try:
        identity_est = min(float(consensus_score) / 10.0, 1.0) if consensus_score is not None else 0.0
    except Exception:
        identity_est = 0.0

    # Representatives and organisms (names if available)
    reps = {"query": rep_q or "", "target": rep_t or ""}
    organisms: List[str] = []
    try:
        def _extract_genome_id(rep: str) -> str:
            return rep.split(":")[0] if rep and ":" in rep else (rep or "")
        gq = _extract_genome_id(rep_q)
        gt = _extract_genome_id(rep_t)
        for gid in {gq, gt} - {""}:
            r = cur.execute("SELECT organism_name FROM genomes WHERE genome_id = ?", (gid,)).fetchone()
            if r and r[0]:
                organisms.append(str(r[0]))
    except Exception:
        organisms = []

    # Consensus cassette: prefer precomputed if available
    payload = None
    try:
        row = conn.execute("SELECT consensus_json FROM cluster_consensus WHERE cluster_id = ?", (int(cluster_id),)).fetchone()
        if row and row[0]:
            import json as _json
            payload = _json.loads(row[0])
    except Exception:
        payload = None
    if not payload:
        payload = compute_cluster_pfam_consensus(conn, int(cluster_id), float(min_core_coverage), float(df_percentile_ban), int(max_tokens_preview))
    consensus = payload.get('consensus', []) if isinstance(payload, dict) else []
    pairs = payload.get('pairs', []) if isinstance(payload, dict) else []

    # Directional consensus summary
    supported = [p for p in pairs if p.get('same_frac') is not None and int(p.get('support', 0)) >= 3]
    if supported:
        import statistics as stats
        agree = stats.mean([float(p['same_frac']) for p in supported])
    else:
        agree = 0.0
    status = 'strong' if agree >= 0.6 else ('mixed' if agree >= 0.4 else 'weak')

    # Preview tokens for quick rendering
    preview_tokens = [{
        'pfam': c.get('token'),
        'coverage': float(c.get('coverage', 0.0)),
        'pos': float(c.get('mean_pos', 0.0)),
        'color': c.get('color', '#888888'),
    } for c in consensus]

    # Quality heuristic
    core_cov = [float(c.get('coverage', 0.0)) for c in consensus if float(c.get('coverage', 0.0)) >= 0.6]
    import statistics as stats
    med_core = (stats.median(core_cov) if core_cov else (stats.median([float(c.get('coverage', 0.0)) for c in consensus]) if consensus else 0.0))
    size_term = min(1.0, math.log1p(max(0, int(size))) / 5.0)
    quality = max(0.0, min(1.0, 0.5 * med_core + 0.3 * agree + 0.2 * size_term))

    # Caveats
    caveats: List[str] = []
    if len(consensus) < 3:
        caveats.append('short cassette')

    return {
        'schema': 'cluster_explorer_summary.v1',
        'cluster_id': int(cluster_id),
        'size': int(size or 0),
        'consensus_length': int(consensus_length or 0),
        'identity_est': float(identity_est or 0.0),
        'organisms': organisms[:5],
        'representatives': reps,
        'consensus': [{
            'pfam': c.get('token'),
            'coverage': float(c.get('coverage', 0.0)),
            'pos': float(c.get('mean_pos', 0.0)),
            'df': int(c.get('df', 0) or 0),
            'fwd_frac': float(c.get('fwd_frac', 0.0)),
            'n_occ': int(c.get('n_occ', 0) or 0),
        } for c in consensus],
        'adjacency_support': [{
            't1': p.get('t1'), 't2': p.get('t2'),
            'same_frac': (float(p.get('same_frac')) if p.get('same_frac') is not None else None),
            'support': int(p.get('support', 0) or 0),
        } for p in pairs],
        'directional_consensus': {
            'agree_frac': float(agree),
            'status': status,
        },
        'preview': {
            'tokens': preview_tokens,
            'hint': f"{sum(1 for c in consensus if c.get('coverage',0)>=0.6)} core tokens, {status} directional consensus",
        },
        'quality': float(quality),
        'caveats': caveats,
    }

DEFAULTS = {
    "jaccard_tau": 0.20,          # Min IDF-weighted Jaccard to keep an edge
    "mutual_k": 10,               # Mutual-k sparsification
    "degree_cap": 20,             # Degree cap per node after mutual-k
    "min_token_per_block": 3,     # Require at least N PFAM tokens per block
}

def compute_cluster_pfam_consensus(conn: sqlite3.Connection,
                                   cluster_id: int,
                                   min_core_coverage: float = 0.7,
                                   df_percentile_ban: float = 0.9,
                                   max_tokens: int = 10):
    """Compute an order-aware PFAM consensus cassette for a cluster.

    Returns a dict with:
      - consensus: list of dicts for tokens [{token, coverage, mean_pos, df, color, fwd_frac, n_occ}]
      - pairs: list of dicts for adjacent token pairs [{i, j, t1, t2, same_frac, support}]
    where mean_pos ∈ [0,1] is the mean normalized index across blocks and fwd_frac is
    the fraction of occurrences on the forward strand for that token.
    """
    import hashlib
    import statistics as stats

    # Load per-gene PFAM and strand for genes mapped to blocks in this cluster
    q = pd.read_sql_query(
        """
        SELECT sb.block_id, gbm.block_role, g.start_pos, g.strand, g.pfam_domains
        FROM gene_block_mappings gbm
        JOIN genes g ON gbm.gene_id = g.gene_id
        JOIN syntenic_blocks sb ON gbm.block_id = sb.block_id
        WHERE sb.cluster_id = ? AND g.pfam_domains IS NOT NULL AND g.pfam_domains != ''
        ORDER BY sb.block_id, gbm.block_role, g.start_pos
        """,
        conn,
        params=(int(cluster_id),)
    )
    if q.empty:
        return []

    # Simple PFAM normalization to collapse subfamilies (e.g., Ribosomal_S7_N -> Ribosomal_S7)
    import re
    rib_pat = re.compile(r"^(Ribosomal_[LS]\d+)", re.I)
    def _norm(tok: str) -> str:
        m = rib_pat.match(tok)
        if m:
            return m.group(1)
        return tok

    # Build one token & strand sequence per block, preferring 'query' side; fallback to 'target'
    by_block_role: dict[tuple[int, str], list[tuple[str, int]]] = {}
    for row in q.itertuples(index=False):
        toks_raw = [t.strip() for t in str(row.pfam_domains).split(';') if t.strip()]
        toks = [_norm(t) for t in toks_raw]
        if not toks:
            continue
        primary = toks[0]
        key = (int(row.block_id), str(row.block_role))
        by_block_role.setdefault(key, []).append((primary, int(row.strand)))

    by_block: dict[int, list[tuple[str, int]]] = {}
    for (bid, role), seq in by_block_role.items():
        if bid not in by_block:
            by_block[bid] = seq
        else:
            # prefer query; if we already have query, ignore target; otherwise choose longer
            if role == 'query' or len(seq) > len(by_block[bid]):
                by_block[bid] = seq

    # Remove empty sequences
    by_block = {bid: seq for bid, seq in by_block.items() if seq}
    if not by_block:
        return []

    blocks = list(by_block.keys())
    n_blocks = len(blocks)

    # In-cluster DF over blocks (presence/absence)
    df_counts: dict[str, int] = {}
    for seq in by_block.values():
        # Presence/absence by token (ignore strand here)
        seen = {tok for (tok, _strand) in seq}
        for t in seen:
            df_counts[t] = df_counts.get(t, 0) + 1

    # Global DF over blocks (presence/absence across entire DB)
    global_df: dict[str, int] = {}
    if df_percentile_ban is not None and df_percentile_ban > 0.0:
        q_all = pd.read_sql_query(
            """
            SELECT gbm.block_id, g.pfam_domains
            FROM gene_block_mappings gbm
            JOIN genes g ON gbm.gene_id = g.gene_id
            WHERE g.pfam_domains IS NOT NULL AND g.pfam_domains != ''
            """,
            conn,
        )
        block_tokens: dict[int, set[str]] = {}
        for row in q_all.itertuples(index=False):
            toks_raw = [t.strip() for t in str(row.pfam_domains).split(';') if t.strip()]
            toks = {_norm(t) for t in toks_raw}
            if not toks:
                continue
            bid = int(row.block_id)
            s = block_tokens.get(bid)
            if s is None:
                block_tokens[bid] = set(toks)
            else:
                s.update(toks)
        for toks in block_tokens.values():
            for t in toks:
                global_df[t] = global_df.get(t, 0) + 1

    # Ban top-percentile tokens by GLOBAL DF (optional). If df_percentile_ban <= 0, do not ban.
    banned_global = set()
    if global_df and df_percentile_ban is not None and df_percentile_ban > 0.0:
        vals = sorted(global_df.values())
        import math
        qidx = max(0, min(len(vals)-1, int(math.floor(df_percentile_ban * (len(vals)-1)))))
        cutoff = vals[qidx]
        banned_global = {t for t, c in global_df.items() if c >= cutoff}

    # Coverage and positions
    token_pos: dict[str, list[float]] = {}
    token_strand: dict[str, list[int]] = {}
    for bid, seq in by_block.items():
        L = len(seq)
        if L == 0:
            continue
        # dedupe consecutive repeats to reduce noise
        compact = [seq[0]]
        for tok_s in seq[1:]:
            if tok_s[0] != compact[-1][0]:
                compact.append(tok_s)
        L = len(compact)
        for i, (tok, strand) in enumerate(compact):
            token_pos.setdefault(tok, []).append(i / max(1, L-1) if L > 1 else 0.0)
            token_strand.setdefault(tok, []).append(1 if int(strand) >= 0 else 0)

    consensus_list = []
    for tok, positions in token_pos.items():
        cov = (df_counts.get(tok, 0) / n_blocks) if n_blocks > 0 else 0.0
        # Keep tokens that meet the in-cluster coverage requirement
        if cov < min_core_coverage:
            continue
        # Apply global DF ban unless token is core (cov high) or whitelisted (Ribosomal_*)
        if banned_global and (tok in banned_global) and rib_pat.match(tok) is None:
            # Note: condition keeps cores and whitelisted tokens; others banned
            # But since this tok passed core coverage, do not ban
            pass
        mean_pos = stats.mean(positions) if positions else 0.0
        # simple stable color from hash
        h = hashlib.md5(tok.encode()).hexdigest()
        color = f"#{h[:6]}"
        strands = token_strand.get(tok, [])
        n_occ = len(strands)
        fwd_frac = (sum(1 for s in strands if s == 1) / n_occ) if n_occ > 0 else 0.0
        consensus_list.append({
            'token': tok,
            'coverage': cov,
            'mean_pos': mean_pos,
            'df': df_counts.get(tok, 0),
            'color': color,
            'fwd_frac': fwd_frac,
            'n_occ': n_occ,
        })

    # Sort by mean position; optionally cap to top-N by coverage
    consensus_list.sort(key=lambda d: d['mean_pos'])
    if max_tokens and max_tokens > 0 and len(consensus_list) > max_tokens:
        consensus_list = sorted(consensus_list, key=lambda d: (-d['coverage'], d['mean_pos']))[:max_tokens]
        consensus_list.sort(key=lambda d: d['mean_pos'])

    # Compute adjacency same-strand fractions between neighboring consensus tokens
    pairs = []
    ordered_tokens = [c['token'] for c in consensus_list]
    if len(ordered_tokens) >= 2:
        # Precompute per-block first occurrence and strand of each token
        per_block_idx_strand: dict[int, dict[str, tuple[int, int]]] = {}
        for bid, seq in by_block.items():
            idx_map: dict[str, tuple[int, int]] = {}
            for idx, (tok, strand) in enumerate(seq):
                if tok not in idx_map:
                    idx_map[tok] = (idx, int(strand))
            per_block_idx_strand[bid] = idx_map
        for i in range(len(ordered_tokens)-1):
            t1, t2 = ordered_tokens[i], ordered_tokens[i+1]
            support = 0
            same = 0
            for bid, idx_map in per_block_idx_strand.items():
                if t1 in idx_map and t2 in idx_map:
                    support += 1
                    s1 = idx_map[t1][1]
                    s2 = idx_map[t2][1]
                    if (s1 >= 0 and s2 >= 0) or (s1 < 0 and s2 < 0):
                        same += 1
            same_frac = (same / support) if support > 0 else None
            pairs.append({'i': i, 'j': i+1, 't1': t1, 't2': t2, 'same_frac': same_frac, 'support': support})

    return {'consensus': consensus_list, 'pairs': pairs}


def compute_block_pfam_consensus(conn: sqlite3.Connection,
                                 block_id: int,
                                 df_percentile_ban: float = 0.9) -> Dict:
    """Compute a pairwise PFAM consensus cassette for a single syntenic block (query vs target).

    Returns dict with keys:
      - consensus: list of dicts [{token, coverage, mean_pos, df, color, fwd_frac, n_occ}]
      - pairs: list of dicts for adjacent consensus tokens [{i, j, t1, t2, same_frac, support}]

    Notes:
      - Coverage per token is presence across sides (0.5 if present in one, 1.0 if in both).
      - mean_pos is the mean normalized index across sides where present.
      - Directional consensus between adjacent consensus tokens is computed over sides where both tokens are present (support ∈ {0,1,2}).
    """
    import hashlib
    import statistics as stats
    import re

    # Load PFAMs and strands for genes mapped to this block, preserving order per side
    q = pd.read_sql_query(
        """
        SELECT gbm.block_role, g.start_pos, g.strand, g.pfam_domains
        FROM gene_block_mappings gbm
        JOIN genes g ON gbm.gene_id = g.gene_id
        WHERE gbm.block_id = ? AND g.pfam_domains IS NOT NULL AND g.pfam_domains != ''
        ORDER BY gbm.block_role, g.start_pos
        """,
        conn,
        params=(int(block_id),)
    )
    if q.empty:
        return {'consensus': [], 'pairs': []}

    rib_pat = re.compile(r"^(Ribosomal_[LS]\d+)", re.I)
    def _norm(tok: str) -> str:
        m = rib_pat.match(tok)
        return m.group(1) if m else tok

    # Build per-side token+strand sequences
    sides = {}
    for row in q.itertuples(index=False):
        toks_raw = [t.strip() for t in str(row.pfam_domains).split(';') if t.strip()]
        toks = [_norm(t) for t in toks_raw]
        if not toks:
            continue
        primary = toks[0]
        role = str(row.block_role)
        sides.setdefault(role, []).append((primary, int(row.strand)))

    if not sides:
        return {'consensus': [], 'pairs': []}

    # Compact consecutive duplicates per side to reduce noise
    for role, seq in list(sides.items()):
        if not seq:
            continue
        compact = [seq[0]]
        for tok_s in seq[1:]:
            if tok_s[0] != compact[-1][0]:
                compact.append(tok_s)
        sides[role] = compact

    # Token presence and normalized positions per side
    token_pos_side: Dict[str, Dict[str, float]] = {}
    token_strand_side: Dict[str, Dict[str, int]] = {}
    for role, seq in sides.items():
        L = len(seq)
        if L == 0:
            continue
        for i, (tok, strand) in enumerate(seq):
            token_pos_side.setdefault(tok, {})[role] = (i / max(1, L-1) if L > 1 else 0.0)
            token_strand_side.setdefault(tok, {})[role] = 1 if int(strand) >= 0 else 0

    # Global DF across database for ban (optional)
    banned_global = set()
    if df_percentile_ban is not None and df_percentile_ban > 0.0:
        q_all = pd.read_sql_query(
            """
            SELECT gbm.block_id, g.pfam_domains
            FROM gene_block_mappings gbm
            JOIN genes g ON gbm.gene_id = g.gene_id
            WHERE g.pfam_domains IS NOT NULL AND g.pfam_domains != ''
            """,
            conn,
        )
        block_tokens: Dict[int, set[str]] = {}
        for row in q_all.itertuples(index=False):
            toks_raw = [t.strip() for t in str(row.pfam_domains).split(';') if t.strip()]
            toks = {_norm(t) for t in toks_raw}
            if not toks:
                continue
            bid = int(row.block_id)
            s = block_tokens.get(bid)
            if s is None:
                block_tokens[bid] = set(toks)
            else:
                s.update(toks)
        global_df: Dict[str, int] = {}
        for toks in block_tokens.values():
            for t in toks:
                global_df[t] = global_df.get(t, 0) + 1
        if global_df:
            vals = sorted(global_df.values())
            import math
            qidx = max(0, min(len(vals)-1, int(math.floor(df_percentile_ban * (len(vals)-1)))))
            cutoff = vals[qidx]
            banned_global = {t for t, c in global_df.items() if c >= cutoff}

    # Build consensus entries (tokens from union; coverage by sides present)
    tokens = sorted(token_pos_side.keys(), key=lambda t: sum(token_pos_side[t].values())/max(1,len(token_pos_side[t])))
    consensus_list = []
    for tok in tokens:
        sides_present = token_pos_side.get(tok, {})
        if not sides_present:
            continue
        mean_pos = stats.mean(list(sides_present.values()))
        cov = len(sides_present) / max(1, len(sides))  # 0.5 or 1.0
        # simple stable color from hash
        h = hashlib.md5(tok.encode()).hexdigest()
        color = f"#{h[:6]}"
        # fwd fraction over sides present
        strands = token_strand_side.get(tok, {})
        n_occ = len(strands)
        fwd_frac = (sum(1 for s in strands.values() if s == 1) / n_occ) if n_occ > 0 else 0.0
        consensus_list.append({
            'token': tok,
            'coverage': cov,
            'mean_pos': mean_pos,
            'df': len(sides_present),  # in this context, DF=number of sides present
            'color': color,
            'fwd_frac': fwd_frac,
            'n_occ': n_occ,
        })

    # Filter out globally banned tokens only if they are not present on both sides
    if banned_global:
        consensus_list = [c for c in consensus_list if (c['token'] not in banned_global or c['coverage'] >= 1.0)]

    # Keep only tokens conserved on both sides (100% within this block)
    consensus_list = [c for c in consensus_list if float(c.get('coverage', 0.0)) >= 1.0]

    # Sort by position
    consensus_list.sort(key=lambda d: d['mean_pos'])

    # Adjacency directional consensus between neighboring consensus tokens
    pairs = []
    ordered_tokens = [c['token'] for c in consensus_list]
    if len(ordered_tokens) >= 2:
        # Build per-side first-occurrence and strand
        idx_side: Dict[str, Dict[str, tuple[int,int]]] = {}
        for role, seq in sides.items():
            d = {}
            for idx, (tok, strand) in enumerate(seq):
                if tok not in d:
                    d[tok] = (idx, int(strand))
            idx_side[role] = d
        for i in range(len(ordered_tokens)-1):
            t1, t2 = ordered_tokens[i], ordered_tokens[i+1]
            support = 0
            same = 0
            for role in idx_side.keys():
                d = idx_side[role]
                if t1 in d and t2 in d:
                    support += 1
                    s1 = d[t1][1]
                    s2 = d[t2][1]
                    if (s1 >= 0 and s2 >= 0) or (s1 < 0 and s2 < 0):
                        same += 1
            same_frac = (same / support) if support > 0 else None
            pairs.append({'i': i, 'j': i+1, 't1': t1, 't2': t2, 'same_frac': same_frac, 'support': support})

    return {'consensus': consensus_list, 'pairs': pairs}


def _load_block_pfams(conn: sqlite3.Connection) -> Dict[int, Set[str]]:
    """Return block_id -> set of PFAM tokens (strings)."""
    q = pd.read_sql_query(
        """
        SELECT gbm.block_id, g.pfam_domains
        FROM gene_block_mappings gbm
        JOIN genes g ON gbm.gene_id = g.gene_id
        WHERE g.pfam_domains IS NOT NULL AND g.pfam_domains != ''
        """,
        conn,
    )
    block_pfams: Dict[int, Set[str]] = {}
    for row in q.itertuples(index=False):
        bid = int(row.block_id)
        toks = {t.strip() for t in str(row.pfam_domains).split(';') if t.strip()}
        if not toks:
            continue
        s = block_pfams.get(bid)
        if s is None:
            block_pfams[bid] = set(toks)
        else:
            s.update(toks)
    return block_pfams


def _idf_weights(block_pfams: Dict[int, Set[str]]) -> Tuple[Dict[str, float], Dict[int, float]]:
    """Compute IDF(token) and per-block token weight sum."""
    df: Dict[str, int] = {}
    for toks in block_pfams.values():
        for t in toks:
            df[t] = df.get(t, 0) + 1
    N = max(1, len(block_pfams))
    idf = {t: math.log(1.0 + (N / max(1, c))) for t, c in df.items()}
    block_sum = {bid: sum(idf.get(t, 0.0) for t in toks) for bid, toks in block_pfams.items()}
    return idf, block_sum


def _build_edges(block_pfams: Dict[int, Set[str]], idf: Dict[str, float], block_sum: Dict[int, float],
                 jaccard_tau: float, mutual_k: int, degree_cap: int) -> List[Tuple[int, int, float]]:
    """Build sparse edge list (u, v, w) using IDF-weighted Jaccard.

    Uses an inverted index over tokens to accumulate intersections; then computes
    weighted Jaccard with precomputed per-block token sum.
    Applies per-node top-K selection and mutual-k; caps final degree.
    """
    # Inverted index: token -> list of blocks
    inv: Dict[str, List[int]] = {}
    for bid, toks in block_pfams.items():
        for t in toks:
            inv.setdefault(t, []).append(bid)

    # Accumulate intersection weights per pair via shared tokens
    inter: Dict[Tuple[int, int], float] = {}
    for t, posts in inv.items():
        wt = idf.get(t, 0.0)
        if wt <= 0 or len(posts) < 2:
            continue
        # For all pairs in the posting list
        # Small postings lists expected for informative tokens; OK for PFAM
        for i in range(len(posts)):
            bi = posts[i]
            for j in range(i + 1, len(posts)):
                bj = posts[j]
                if bi == bj:
                    continue
                key = (bi, bj) if bi < bj else (bj, bi)
                inter[key] = inter.get(key, 0.0) + wt

    # Compute weighted jaccard; bucket by node for top-k selection
    nbrs: Dict[int, List[Tuple[int, float]]] = {}
    for (u, v), iw in inter.items():
        denom = block_sum.get(u, 0.0) + block_sum.get(v, 0.0) - iw
        if denom <= 0:
            continue
        wj = iw / denom
        if wj >= jaccard_tau:
            nbrs.setdefault(u, []).append((v, wj))
            nbrs.setdefault(v, []).append((u, wj))

    # Per-node top-K
    topk: Dict[int, Dict[int, float]] = {}
    for u, arr in nbrs.items():
        arr.sort(key=lambda x: x[1], reverse=True)
        if mutual_k and len(arr) > mutual_k:
            arr = arr[:mutual_k]
        topk[u] = {v: w for v, w in arr}

    # Mutual-k and degree capping
    edges: List[Tuple[int, int, float]] = []
    deg: Dict[int, int] = {}
    for u, dct in topk.items():
        for v, w in dct.items():
            # Check mutual
            if u not in topk.get(v, {}):
                continue
            # Degree caps (apply symmetrically)
            if degree_cap:
                if deg.get(u, 0) >= degree_cap or deg.get(v, 0) >= degree_cap:
                    continue
            if u < v:
                edges.append((u, v, w))
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
    return edges


def _communities(edges: List[Tuple[int, int, float]], nodes: List[int]) -> Dict[int, int]:
    """Run community detection; return block_id -> cluster_id (>=1)."""
    try:
        import networkx as nx
    except Exception as e:
        raise RuntimeError("networkx required for clustering") from e

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v, w in edges:
        G.add_edge(u, v, weight=float(w))

    # Prefer greedy modularity
    try:
        comms = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
        comms = [sorted(list(c)) for c in comms]
    except Exception:
        # Fallback to connected components
        comms = [sorted(list(c)) for c in nx.connected_components(G)]

    # Assign cluster IDs (skip singletons without edges → they stay sink=0)
    mapping: Dict[int, int] = {}
    cid = 1
    for comp in comms:
        if len(comp) <= 1:
            continue
        for bid in comp:
            mapping[bid] = cid
        cid += 1
    return mapping


def cluster_blocks_by_pfam(db_path: Path | str,
                           jaccard_tau: float = DEFAULTS["jaccard_tau"],
                           mutual_k: int = DEFAULTS["mutual_k"],
                           degree_cap: int = DEFAULTS["degree_cap"],
                           min_token_per_block: int = DEFAULTS["min_token_per_block"],
                           dry_run: bool = False) -> Dict[str, int]:
    """Compute PFAM-based clusters and persist to DB. Returns summary stats.

    - Removes reliance on `clusters` size spec.
    - Writes `cluster_assignments` and updates `syntenic_blocks.cluster_id`.
    - Leaves cluster_id=0 as sink for isolates/noise.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        # Load PFAM tokens per block
        block_pfams = _load_block_pfams(conn)
        # Filter blocks with insufficient tokens
        block_pfams = {bid: toks for bid, toks in block_pfams.items() if len(toks) >= min_token_per_block}
        all_blocks = set(pd.read_sql_query("select block_id from syntenic_blocks", conn)['block_id'].astype(int))

        if not block_pfams:
            return {"status": 1, "msg": "no pfam tokens found"}

        idf, block_sum = _idf_weights(block_pfams)
        edges = _build_edges(block_pfams, idf, block_sum, jaccard_tau, mutual_k, degree_cap)

        # Build community mapping
        nodes = list(block_pfams.keys())
        mapping = _communities(edges, nodes)

        # Persist
        cur = conn.cursor()
        cur.execute("DELETE FROM cluster_assignments")
        if mapping and not dry_run:
            rows = [(int(bid), int(cid)) for bid, cid in mapping.items()]
            cur.executemany("INSERT INTO cluster_assignments (block_id, cluster_id) VALUES (?, ?)", rows)
            # Update syntenic_blocks.cluster_id using assignments; others fall to 0
            cur.execute("UPDATE syntenic_blocks SET cluster_id = 0")
            cur.execute(
                """
                UPDATE syntenic_blocks
                SET cluster_id = (
                    SELECT ca.cluster_id FROM cluster_assignments ca WHERE ca.block_id = syntenic_blocks.block_id
                )
                WHERE block_id IN (SELECT block_id FROM cluster_assignments)
                """
            )
            conn.commit()

        assigned = len(mapping)
        isolates = len(all_blocks) - assigned
        return {
            "status": 0,
            "assigned": assigned,
            "isolates": isolates,
            "edges": len(edges),
            "nodes": len(nodes),
        }
    finally:
        conn.close()


def compute_micro_cluster_pfam_consensus(conn: sqlite3.Connection,
                                         micro_cluster_id: int,
                                         min_core_coverage: float = 0.7,
                                         df_percentile_ban: float = 0.9,
                                         max_tokens: int = 10):
    """Compute PFAM consensus for a micro cluster using micro_gene_* tables.

    Returns dict with keys 'consensus' and 'pairs'.
    """
    import statistics as stats
    import re
    import numpy as np
    import pandas as pd
    import hashlib

    # Ensure micro tables exist
    try:
        r = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='micro_gene_blocks'").fetchone()
        if not r:
            return {'consensus': [], 'pairs': []}
    except Exception:
        return {'consensus': [], 'pairs': []}

    # Ordered gene indices per genome+contig
    conn.execute("DROP TABLE IF EXISTS _tmp_gene_order_micro")
    conn.execute(
        """
        CREATE TEMP TABLE _tmp_gene_order_micro AS
        SELECT 
            genome_id, contig_id, gene_id,
            start_pos, end_pos,
            ROW_NUMBER() OVER (PARTITION BY genome_id, contig_id ORDER BY start_pos, end_pos) - 1 AS idx
        FROM genes
        """
    )

    # Genes per micro block for the given cluster
    q = pd.read_sql_query(
        """
        WITH block_ranges AS (
            SELECT m.block_id, m.genome_id, m.contig_id, m.start_index, m.end_index
            FROM micro_gene_blocks m
            WHERE m.cluster_id = ?
        ),
        genes_in_block AS (
            SELECT br.block_id, go.gene_id, go.start_pos, go.end_pos
            FROM block_ranges br
            JOIN _tmp_gene_order_micro go
              ON go.genome_id = br.genome_id
             AND go.contig_id = br.contig_id
             AND go.idx BETWEEN br.start_index AND br.end_index
            ORDER BY br.block_id, go.start_pos, go.end_pos
        )
        SELECT gib.block_id, g.gene_id, g.start_pos, g.strand, g.pfam_domains
        FROM genes_in_block gib JOIN genes g ON g.gene_id = gib.gene_id
        ORDER BY gib.block_id, g.start_pos
        """,
        conn,
        params=(int(micro_cluster_id),)
    )
    if q.empty:
        return {'consensus': [], 'pairs': []}

    rib_pat = re.compile(r"^(Ribosomal_[LS]\d+)", re.I)
    def _norm(tok: str) -> str:
        m = rib_pat.match(tok)
        return m.group(1) if m else tok

    # Build per-block token+strand sequences
    by_block: dict[int, list[tuple[str, int]]] = {}
    for row in q.itertuples(index=False):
        toks_raw = [t.strip() for t in str(row.pfam_domains or '').split(';') if t.strip()]
        if not toks_raw:
            continue
        primary = _norm(toks_raw[0])
        by_block.setdefault(int(row.block_id), []).append((primary, int(row.strand or 1)))
    by_block = {bid: seq for bid, seq in by_block.items() if seq}
    if not by_block:
        return {'consensus': [], 'pairs': []}

    n_blocks = len(by_block)
    # DF and normalized positions (per-block normalized like macro path)
    df_counts: dict[str, int] = {}
    positions: dict[str, list[float]] = {}
    fwd_counts: dict[str, int] = {}
    occ_counts: dict[str, int] = {}
    for seq in by_block.values():
        # Compact consecutive duplicates
        compact = []
        for tok_s in seq:
            if not compact or tok_s[0] != compact[-1][0]:
                compact.append(tok_s)
        L = len(compact)
        seen = set()
        for idx, (tok, strand) in enumerate(compact):
            occ_counts[tok] = occ_counts.get(tok, 0) + 1
            fwd_counts[tok] = fwd_counts.get(tok, 0) + (1 if strand >= 0 else 0)
            # normalized position 0..1
            pos = (idx / max(1, L-1)) if L > 1 else 0.0
            positions.setdefault(tok, []).append(pos)
            if tok not in seen:
                df_counts[tok] = df_counts.get(tok, 0) + 1
                seen.add(tok)

    # Ban top-DF percentile tokens if requested
    banned = set()
    if df_percentile_ban is not None and df_counts:
        vals = list(df_counts.values())
        thresh = np.percentile(vals, float(df_percentile_ban) * 100.0)
        banned = {tok for tok, df in df_counts.items() if df >= thresh}

    consensus = []
    for tok, dfv in df_counts.items():
        if tok in banned:
            continue
        cov = dfv / float(n_blocks)
        if cov >= float(min_core_coverage):
            pos = stats.mean(positions.get(tok, [0]))
            fwd = (fwd_counts.get(tok, 0) / float(occ_counts.get(tok, 1))) if occ_counts.get(tok, 0) else 0.0
            color = f"#{hashlib.md5(tok.encode()).hexdigest()[:6]}"
            consensus.append({'token': tok, 'coverage': float(cov), 'df': int(dfv), 'mean_pos': float(pos), 'fwd_frac': float(fwd), 'n_occ': int(occ_counts.get(tok, 0)), 'color': color})

    consensus.sort(key=lambda x: x.get('mean_pos', 0.0))
    if max_tokens and len(consensus) > int(max_tokens):
        consensus = consensus[:int(max_tokens)]

    # Adjacent pair stats aligned to consensus token order
    ordered_tokens = [c['token'] for c in consensus]
    pairs = []
    if len(ordered_tokens) >= 2:
        # Build per-block first-occurrence and strand index
        per_block_idx_strand: dict[int, dict[str, tuple[int, int]]] = {}
        for bid, seq in by_block.items():
            idx_map: dict[str, tuple[int, int]] = {}
            for idx, (tok, strand) in enumerate(seq):
                if tok not in idx_map:
                    idx_map[tok] = (idx, int(strand))
            per_block_idx_strand[bid] = idx_map
        for i in range(len(ordered_tokens)-1):
            t1, t2 = ordered_tokens[i], ordered_tokens[i+1]
            support = 0
            same = 0
            for idx_map in per_block_idx_strand.values():
                if t1 in idx_map and t2 in idx_map:
                    support += 1
                    s1 = idx_map[t1][1]
                    s2 = idx_map[t2][1]
                    if (s1 >= 0 and s2 >= 0) or (s1 < 0 and s2 < 0):
                        same += 1
            same_frac = (same / support) if support > 0 else None
            pairs.append({'i': i, 'j': i+1, 't1': t1, 't2': t2, 'same_frac': same_frac, 'support': support})

    return {'consensus': consensus, 'pairs': pairs}


def precompute_all_micro_consensus(conn: sqlite3.Connection,
                                   min_core_coverage: float = 0.7,
                                   df_percentile_ban: float = 0.9,
                                   max_tokens: int = 10) -> int:
    """Compute and store consensus JSON for all micro clusters."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS micro_cluster_consensus (
            cluster_id INTEGER PRIMARY KEY,
            consensus_json TEXT
        )
        """
    )
    conn.execute("DELETE FROM micro_cluster_consensus")
    rows = conn.execute("SELECT cluster_id FROM micro_gene_clusters").fetchall()
    n = 0
    import json as _json
    for (cid,) in rows:
        payload = compute_micro_cluster_pfam_consensus(conn, int(cid), min_core_coverage, df_percentile_ban, max_tokens)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO micro_cluster_consensus(cluster_id, consensus_json) VALUES (?, ?)",
                (int(cid), _json.dumps(payload))
            )
            n += 1
        except Exception:
            pass
    conn.commit()
    return n


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='genome_browser/genome_browser.db')
    p.add_argument('--tau', type=float, default=DEFAULTS['jaccard_tau'])
    p.add_argument('--mutual_k', type=int, default=DEFAULTS['mutual_k'])
    p.add_argument('--degree_cap', type=int, default=DEFAULTS['degree_cap'])
    p.add_argument('--min_tokens', type=int, default=DEFAULTS['min_token_per_block'])
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    stats = cluster_blocks_by_pfam(
        db_path=args.db,
        jaccard_tau=args.tau,
        mutual_k=args.mutual_k,
        degree_cap=args.degree_cap,
        min_token_per_block=args.min_tokens,
        dry_run=args.dry_run,
    )
    print(stats)
