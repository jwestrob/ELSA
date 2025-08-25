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
                                   max_tokens: int = 10) -> list[dict]:
    """Compute an order-aware PFAM consensus cassette for a cluster.

    Returns a list of dicts: [{token, coverage, mean_pos, df, color}]
    where mean_pos ∈ [0,1] is the mean normalized index across blocks.
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

    # Build one token sequence per block, preferring 'query' side; fallback to 'target'
    by_block_role: dict[tuple[int, str], list[str]] = {}
    for row in q.itertuples(index=False):
        toks_raw = [t.strip() for t in str(row.pfam_domains).split(';') if t.strip()]
        toks = [_norm(t) for t in toks_raw]
        if not toks:
            continue
        primary = toks[0]
        key = (int(row.block_id), str(row.block_role))
        by_block_role.setdefault(key, []).append(primary)

    by_block: dict[int, list[str]] = {}
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
        seen = set(seq)
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
    for bid, seq in by_block.items():
        L = len(seq)
        if L == 0:
            continue
        # dedupe consecutive repeats to reduce noise
        compact = [seq[0]]
        for tok in seq[1:]:
            if tok != compact[-1]:
                compact.append(tok)
        L = len(compact)
        for i, tok in enumerate(compact):
            token_pos.setdefault(tok, []).append(i / max(1, L-1) if L > 1 else 0.0)

    consensus = []
    for tok, positions in token_pos.items():
        cov = (df_counts.get(tok, 0) / n_blocks) if n_blocks > 0 else 0.0
        # Keep tokens that meet the in-cluster coverage requirement
        if cov < min_core_coverage:
            continue
        # Apply global DF ban unless token is core (cov high) or whitelisted (Ribosomal_*)
        if banned_global and (tok in banned_global) and (cov < min_core_coverage) and rib_pat.match(tok) is None:
            # Note: condition keeps cores and whitelisted tokens; others banned
            continue
        mean_pos = stats.mean(positions) if positions else 0.0
        # simple stable color from hash
        h = hashlib.md5(tok.encode()).hexdigest()
        color = f"#{h[:6]}"
        consensus.append({
            'token': tok,
            'coverage': cov,
            'mean_pos': mean_pos,
            'df': df_counts.get(tok, 0),
            'color': color,
        })

    # Sort by mean position; pick top-N by coverage if too many
    consensus.sort(key=lambda d: d['mean_pos'])
    if len(consensus) > max_tokens:
        consensus = sorted(consensus, key=lambda d: (-d['coverage'], d['mean_pos']))[:max_tokens]
        consensus.sort(key=lambda d: d['mean_pos'])
    return consensus


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
