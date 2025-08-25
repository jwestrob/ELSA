"""
Cluster robust blocks using a sparse mutual-k graph with Jaccard over shingles.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Iterable, Callable, Any
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
import re
import logging

from .shingles import srp_tokens, block_shingles, df_filter, jaccard

logger = logging.getLogger(__name__)


def cluster_blocks_jaccard(blocks: Iterable, window_embed_lookup: Callable, cfg: Any) -> Dict[int, int]:
    """
    Cluster robust blocks using Mutual-k Jaccard over order-aware shingles.
    
    Args:
        blocks: Iterable of SyntenicBlock objects with .matches, .query_windows, .target_windows, .strand, .id
        window_embed_lookup: Callable that takes window_id and returns np.ndarray (1 x d) embedding
        cfg: Configuration object with analyze.clustering.* parameters
        
    Returns:
        Dictionary mapping block_id (or index) -> cluster_id (0 = sink, 1..K = clusters)
    """
    # Convert blocks to list and assign indices if they don't have .id
    blocks_list = list(blocks)
    if not blocks_list:
        return {}
    
    console_print = lambda msg: print(f"[Clustering] {msg}")
    
    # Extract configuration (stricter defaults to reduce fold-only clustering)
    min_anchors = getattr(cfg, 'min_anchors', 4)
    min_span_genes = getattr(cfg, 'min_span_genes', 8)
    v_mad_max_genes = getattr(cfg, 'v_mad_max_genes', 0.5)
    
    srp_bits = getattr(cfg, 'srp_bits', 256)
    srp_bands = getattr(cfg, 'srp_bands', 32)
    srp_band_bits = getattr(cfg, 'srp_band_bits', 8)
    srp_seed = getattr(cfg, 'srp_seed', 1337)
    shingle_k = getattr(cfg, 'shingle_k', 3)
    shingle_method = getattr(cfg, 'shingle_method', 'xor')
    # New: ICWS + skip-gram parameters (defaults preserve legacy behavior)
    icws_r = getattr(cfg, 'icws_r', 8)
    icws_bbit = getattr(cfg, 'icws_bbit', 0)
    icws_weighting = getattr(cfg, 'icws_weighting', 'uniform')
    shingle_pattern = getattr(cfg, 'shingle_pattern', None)
    strand_canonical_shingles = getattr(cfg, 'strand_canonical_shingles', False)
    skipgram_offsets = None
    if isinstance(shingle_pattern, str) and shingle_pattern:
        try:
            parts = [int(x) for x in shingle_pattern.split(',')]
            if len(parts) == 3:
                skipgram_offsets = (parts[0], parts[1], parts[2])
        except Exception:
            skipgram_offsets = None
    elif isinstance(shingle_pattern, (list, tuple)) and len(shingle_pattern) == 3:
        skipgram_offsets = (int(shingle_pattern[0]), int(shingle_pattern[1]), int(shingle_pattern[2]))
    bands_per_window = getattr(cfg, 'bands_per_window', 4)
    band_stride = getattr(cfg, 'band_stride', 7)
    # Fixed-subset fallback for long, high-identity blocks
    enable_fixed_subset_for_long = getattr(cfg, 'enable_fixed_subset_for_long', False)
    long_min_len = getattr(cfg, 'long_min_len', 20)
    long_min_identity = getattr(cfg, 'long_min_identity', 0.98)
    fixed_subset_k = getattr(cfg, 'fixed_subset_k', 2)
    fixed_subset_bands = getattr(cfg, 'fixed_subset_bands', [0, 8, 16, 24])
    
    jaccard_tau = getattr(cfg, 'jaccard_tau', 0.75)
    mutual_k = getattr(cfg, 'mutual_k', 3)
    df_max = getattr(cfg, 'df_max', 30)
    use_weighted_jaccard = getattr(cfg, 'use_weighted_jaccard', True)
    low_df_threshold = getattr(cfg, 'low_df_threshold', max(10, df_max // 5))
    min_low_df_anchors = getattr(cfg, 'min_low_df_anchors', 3)
    idf_mean_min = getattr(cfg, 'idf_mean_min', 1.0)
    # Optional: ban top-percentile DF shingles entirely (e.g., 0.9)
    max_df_percentile = getattr(cfg, 'max_df_percentile', None)
    
    size_ratio_min = getattr(cfg, 'size_ratio_min', 0.5)
    size_ratio_max = getattr(cfg, 'size_ratio_max', 2.0)
    keep_singletons = getattr(cfg, 'keep_singletons', False)
    sink_label = getattr(cfg, 'sink_label', 0)
    k_core_min_degree = getattr(cfg, 'k_core_min_degree', 3)
    enable_cassette_mode = getattr(cfg, 'enable_cassette_mode', True)
    cassette_max_len = getattr(cfg, 'cassette_max_len', 4)
    # New: graph coherence and community detection controls
    degree_cap = getattr(cfg, 'degree_cap', 10)  # keep top-N edges per node by weight
    triangle_support_min = getattr(cfg, 'triangle_support_min', 1)  # require >= T triangles per edge
    use_community_detection = getattr(cfg, 'use_community_detection', True)
    community_method = getattr(cfg, 'community_method', 'greedy')  # 'greedy' (networkx)

    # Hybrid bandset augmentation
    enable_hybrid_bandset = getattr(cfg, 'enable_hybrid_bandset', False)
    bandset_tau = getattr(cfg, 'bandset_tau', 0.25)
    bandset_df_max = getattr(cfg, 'bandset_df_max', 2000)
    bandset_min_len = getattr(cfg, 'bandset_min_len', 20)
    bandset_min_identity = getattr(cfg, 'bandset_min_identity', 0.98)

    # Targeted bandset acceptor (non-hybrid, narrowly gated)
    enable_targeted_bandset_acceptor = getattr(cfg, 'enable_targeted_bandset_acceptor', False)
    targeted_bandset_tau = getattr(cfg, 'targeted_bandset_tau', 0.25)
    targeted_bandset_df_max = getattr(cfg, 'targeted_bandset_df_max', 2000)
    targeted_bandset_min_len = getattr(cfg, 'targeted_bandset_min_len', 20)
    targeted_bandset_min_identity = getattr(cfg, 'targeted_bandset_min_identity', 0.98)
    targeted_bandset_topk_cap = getattr(cfg, 'targeted_bandset_topk_cap', 100)
    targeted_bandset_low_df_max = getattr(cfg, 'targeted_bandset_low_df_max', 300)
    targeted_bandset_min_low_df = getattr(cfg, 'targeted_bandset_min_low_df', 3)
    targeted_bandset_idf_mean_min = getattr(cfg, 'targeted_bandset_idf_mean_min', 1.0)
    
    console_print(f"Processing {len(blocks_list)} blocks with mutual-k Jaccard clustering")
    
    # Step 1: Robustness gate per block
    robust_blocks = []
    sink_blocks = []
    
    for idx, block in enumerate(blocks_list):
        block_id = getattr(block, 'id', idx)
        
        # Parse integer window indices from block.matches
        window_indices = _extract_window_indices(block)
        if not window_indices:
            console_print(f"Block {block_id}: No valid window indices found")
            sink_blocks.append((block_id, block))
            continue
            
        query_indices, target_indices = window_indices
        
        # Compute robustness metrics
        n = len(query_indices)  # alignment_length
        span_q = max(query_indices) - min(query_indices) + 1 if query_indices else 0
        span_t = max(target_indices) - min(target_indices) + 1 if target_indices else 0
        
        # Compute diagonal variance (MAD of v = q_idx - t_idx)
        v_values = [q_idx - t_idx for q_idx, t_idx in zip(query_indices, target_indices)]
        v_median = np.median(v_values) if v_values else 0
        v_mad = np.median([abs(v - v_median) for v in v_values]) if v_values else 0
        
        # Apply robustness criteria (allow very small, tightly collinear cassettes)
        is_robust = (n >= min_anchors and 
                    span_q >= min_span_genes and 
                    span_t >= min_span_genes and 
                    v_mad <= v_mad_max_genes)

        if not is_robust and enable_cassette_mode:
            if 2 <= n <= cassette_max_len and v_mad <= 0.0:
                is_robust = True

        if is_robust:
            robust_blocks.append((block_id, block))
        else:
            sink_blocks.append((block_id, block))
    
    console_print(f"Robustness gate: {len(robust_blocks)} robust, {len(sink_blocks)} sink")
    
    if len(robust_blocks) == 0:
        # All blocks go to sink
        return {block_id: sink_label for block_id, _ in sink_blocks}
    
    # Step 2 & 3: Build per-block shingles with orientation normalization
    block_shingles_map = {}
    need_bandset = bool(enable_hybrid_bandset or enable_targeted_bandset_acceptor)
    bandset_map = {} if need_bandset else None
    
    for block_id, block in robust_blocks:
        try:
            # Get window sequence respecting strand orientation
            strand = getattr(block, 'strand', 1)
            if hasattr(block, 'query_windows'):
                window_ids = block.query_windows
            else:
                # Fallback to extracting from matches
                window_ids = [match.query_window_id for match in block.matches]
            
            # For strand == -1, reverse the order for tokenization/shingling
            if strand == -1:
                window_ids = window_ids[::-1]
            
            # Fetch embeddings for each window
            window_embeddings = []
            for window_id in window_ids:
                try:
                    emb = window_embed_lookup(window_id)
                    if emb is not None:
                        # Ensure 1D array
                        emb = np.asarray(emb).flatten()
                        window_embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Failed to fetch embedding for window {window_id}: {e}")
            
            if not window_embeddings:
                console_print(f"Block {block_id}: No valid embeddings found")
                sink_blocks.append((block_id, block))
                continue
            
            # Stack into matrix (n_windows, d)
            emb_matrix = np.stack(window_embeddings, axis=0)
            
            # Make SRP tokens strand-insensitive by nulling the strand-sign dimension
            # (last column added by shingling). This preserves content while removing
            # orientation from per-window tokens so reverse/forward loci can collide.
            try:
                if emb_matrix.shape[1] > 0:
                    emb_matrix[:, -1] = 0
            except Exception:
                pass
            # Compute SRP tokens
            window_tokens = srp_tokens(
                emb_matrix,
                n_bits=srp_bits,
                n_bands=srp_bands,
                band_bits=srp_band_bits,
                seed=srp_seed
            )
            # Track length for small-block logic
            # window_len_map defined above loop
            try:
                window_len_map
            except NameError:
                window_len_map = {}
            window_len_map[block_id] = len(window_tokens)
            
            # Decide shingling method per block
            use_method = shingle_method
            use_k = shingle_k
            if enable_fixed_subset_for_long:
                # Determine block robustness (alignment length ~ len(query_windows))
                n_aln = len(window_tokens)
                ident = getattr(block, 'identity', None)
                if n_aln >= long_min_len and (ident is not None and float(ident) >= float(long_min_identity)):
                    use_method = 'fixed_subset'
                    use_k = fixed_subset_k

            # Generate k-gram shingles (adaptive for short blocks if enabled)
            eff_k = use_k
            eff_offsets = skipgram_offsets
            if getattr(cfg, 'enable_adaptive_shingles', False):
                Lw = len(window_tokens)
                if Lw < 4:
                    eff_k = 2
                    eff_offsets = None
                elif Lw < 6:
                    eff_k = 3
                    eff_offsets = None
            shingles = block_shingles(
                window_tokens,
                k=eff_k,
                method=use_method,
                bands_per_window=bands_per_window,
                band_stride=band_stride,
                fixed_bands=fixed_subset_bands if use_method == 'fixed_subset' else None,
                icws_r=icws_r,
                icws_bbit=icws_bbit,
                icws_weighting=icws_weighting,
                seed=srp_seed,
                skipgram_offsets=eff_offsets,
                strand_canonical_shingles=strand_canonical_shingles,
            )
            block_shingles_map[block_id] = shingles
            if need_bandset:
                # Build bandset (order-agnostic) too
                bandset = block_shingles(
                    window_tokens,
                    k=1,
                    method='bandset',
                )
                bandset_map[block_id] = bandset
            
        except Exception as e:
            logger.warning(f"Failed to process block {block_id}: {e}")
            sink_blocks.append((block_id, block))
    
    console_print(f"Generated shingles for {len(block_shingles_map)} blocks")
    
    # Step 4: Inverted index and document frequencies
    shingle_to_blocks = defaultdict(set)
    for block_id, shingles in block_shingles_map.items():
        for shingle_id in shingles:
            shingle_to_blocks[shingle_id].add(block_id)
    
    # Compute document frequencies
    shingle_df = {shingle_id: len(blocks) for shingle_id, blocks in shingle_to_blocks.items()}
    
    # Filter high-DF shingles
    filtered_shingles_map = {}
    for block_id, shingles in block_shingles_map.items():
        filtered_shingles = df_filter(shingle_df, df_max, shingles)
        filtered_shingles_map[block_id] = filtered_shingles
    
    # Update inverted index with filtered shingles
    filtered_shingle_to_blocks = defaultdict(set)
    for block_id, shingles in filtered_shingles_map.items():
        for shingle_id in shingles:
            filtered_shingle_to_blocks[shingle_id].add(block_id)
    
    # Optional additional ban of top-percentile DF shingles
    if max_df_percentile is not None and 0.0 < max_df_percentile < 1.0:
        # Compute DF values list
        dfs = np.array([shingle_df.get(sid, 0) for sid in filtered_shingle_to_blocks.keys()])
        cutoff = np.quantile(dfs, max_df_percentile)
        banned = {sid for sid, dfv in shingle_df.items() if dfv >= cutoff}
        if banned:
            new_map = {}
            for block_id, shingles in filtered_shingles_map.items():
                new_map[block_id] = {s for s in shingles if s not in banned}
            filtered_shingles_map = new_map
            # Rebuild postings
            filtered_shingle_to_blocks = defaultdict(set)
            for block_id, shingles in filtered_shingles_map.items():
                for shingle_id in shingles:
                    filtered_shingle_to_blocks[shingle_id].add(block_id)

    console_print(f"After DF filtering: {len(filtered_shingle_to_blocks)} unique shingles")

    # Pre-index robust blocks for quick lookup
    robust_map = {bid: b for bid, b in robust_blocks}

    # Build IDF map for weighted Jaccard: idf = log(1 + N / df)
    import math
    n_docs = max(1, len(filtered_shingles_map))
    shingle_idf = {sid: math.log(1.0 + (n_docs / max(1, shingle_df.get(sid, 1)))) for sid in filtered_shingle_to_blocks.keys()}

    # Optional bandset postings/IDF (used by hybrid or targeted acceptor)
    if need_bandset:
        bandset_to_blocks = defaultdict(set)
        for block_id, bandset in bandset_map.items():
            for token in bandset:
                bandset_to_blocks[token].add(block_id)
        bandset_df = {tok: len(v) for tok, v in bandset_to_blocks.items()}
        # Filter high-DF bandset tokens more permissively
        filtered_bandset_map = {}
        # Choose DF cap depending on which pathway is active
        active_bandset_df_max = bandset_df_max if enable_hybrid_bandset else targeted_bandset_df_max
        for block_id, bandset in bandset_map.items():
            filtered_bandset = {t for t in bandset if bandset_df.get(t, 0) <= active_bandset_df_max}
            filtered_bandset_map[block_id] = filtered_bandset
        filtered_bandset_to_blocks = defaultdict(set)
        for block_id, bandset in filtered_bandset_map.items():
            for t in bandset:
                filtered_bandset_to_blocks[t].add(block_id)
        n_docs_b = max(1, len(filtered_bandset_map))
        bandset_idf = {tok: math.log(1.0 + (n_docs_b / max(1, bandset_df.get(tok, 1)))) for tok in filtered_bandset_to_blocks.keys()}
    
    # Step 5: Candidate generation with shared-count and size filters
    block_candidates = {}
    min_shared_sh = getattr(cfg, 'min_shared_shingles', 2)
    max_cands = getattr(cfg, 'max_candidates_per_block', 500)
    min_shared_bt = getattr(cfg, 'min_shared_band_tokens', 2)
    band_topk = getattr(cfg, 'bandset_topk_candidates', 100)
    # Clamp overly large bandset candidate lists when using targeted acceptor (non-hybrid)
    if enable_targeted_bandset_acceptor and not enable_hybrid_bandset:
        band_topk = min(band_topk, max(1, int(targeted_bandset_topk_cap)))

    for block_id, shingles in filtered_shingles_map.items():
        # Count shared shingles per candidate
        cand_counts = defaultdict(int)
        for shingle_id in shingles:
            for cand_id in filtered_shingle_to_blocks[shingle_id]:
                if cand_id != block_id:
                    cand_counts[cand_id] += 1

        # Hybrid: add bandset candidates for qualifying blocks (bounded top-K by shared tokens)
        band_candidates = []
        if enable_hybrid_bandset or enable_targeted_bandset_acceptor:
            try:
                block_ref = robust_map[block_id]
                if enable_hybrid_bandset:
                    req_len = bandset_min_len
                    req_ident = bandset_min_identity
                else:
                    req_len = targeted_bandset_min_len
                    req_ident = targeted_bandset_min_identity
                length_ok = len(getattr(block_ref, 'matches', [])) >= req_len
                ident_ok = getattr(block_ref, 'identity', None)
                identity_ok = (ident_ok is not None and ident_ok >= req_ident)
            except Exception:
                length_ok = False
                identity_ok = False
            if length_ok and identity_ok:
                bandset = filtered_bandset_map.get(block_id, set())
                cand_band_counts = defaultdict(int)
                for tok in bandset:
                    for cid in filtered_bandset_to_blocks.get(tok, set()):
                        if cid != block_id:
                            # For targeted (non-hybrid), gate candidates by their own length/identity too
                            if enable_targeted_bandset_acceptor and not enable_hybrid_bandset:
                                c_ref = robust_map.get(cid)
                                if c_ref is None:
                                    continue
                                c_len_ok = len(getattr(c_ref, 'matches', [])) >= targeted_bandset_min_len
                                c_ident = getattr(c_ref, 'identity', None)
                                c_id_ok = (c_ident is not None and c_ident >= targeted_bandset_min_identity)
                                if not (c_len_ok and c_id_ok):
                                    continue
                            cand_band_counts[cid] += 1
                band_list = [(cid, cnt) for cid, cnt in cand_band_counts.items() if cnt >= min_shared_bt]
                band_list.sort(key=lambda x: (-x[1], x[0]))
                band_candidates = [cid for cid, _ in band_list[:band_topk]]

        # Combine candidates
        candidates = set([cid for cid, cnt in cand_counts.items() if cnt >= min_shared_sh])
        candidates.update(band_candidates)

        # Apply size ratio prefilter and cap candidate list
        n_b = len(robust_map[block_id].matches)
        s_b_size = len(shingles)
        filtered_list = []
        for cand_id in candidates:
            if cand_id in filtered_shingles_map and cand_id in robust_map:
                n_c = len(robust_map[cand_id].matches)
                s_c_size = len(filtered_shingles_map[cand_id])
                shingle_ratio = s_b_size / max(s_c_size, 1)
                length_ratio = n_b / max(n_c, 1)
                if (size_ratio_min <= shingle_ratio <= size_ratio_max and 
                    size_ratio_min <= length_ratio <= size_ratio_max):
                    # Prefer higher shared count (0 if from bandset-only)
                    filtered_list.append((cand_id, cand_counts.get(cand_id, 0)))
        filtered_list.sort(key=lambda x: (-x[1], x[0]))
        block_candidates[block_id] = [cid for cid, _ in filtered_list[:max_cands]]
    
    # Step 6: Similarity computation and mutual-k graph construction (parallelizable)
    edges = set()
    edge_weights: Dict[Tuple[int, int], float] = {}

    # Helper for parallel scoring
    def _score_block(bid: int, cands: List[int]) -> List[Tuple[int, int, float]]:
        if bid not in filtered_shingles_map:
            return []
        S_b = filtered_shingles_map[bid]
        sims = []
        for cid in cands:
            if cid in filtered_shingles_map:
                S_c = filtered_shingles_map[cid]
                if use_weighted_jaccard:
                    inter = S_b & S_c
                    union = S_b | S_c
                    if union:
                        inter_w = sum(shingle_idf.get(s, 0.0) for s in inter)
                        union_w = sum(shingle_idf.get(s, 0.0) for s in union)
                        j_sim = (inter_w / union_w) if union_w > 0 else 0.0
                    else:
                        j_sim = 0.0
                else:
                    j_sim = jaccard(S_b, S_c)
                sims.append((cid, j_sim))
        # Sort and optionally mutual-top-k gate
        sims.sort(key=lambda x: (-x[1], x[0]))
        if getattr(cfg, 'enable_mutual_topk_filter', False):
            k = int(getattr(cfg, 'mutual_k', 3))
            sims = sims[:max(1, k)]
        out = []
        for cid, j_sim in sims:
            if j_sim >= jaccard_tau:
                S_c = filtered_shingles_map[cid]
                inter = S_b & S_c
                low_df_count = sum(1 for s in inter if shingle_df.get(s, 0) <= low_df_threshold)
                mean_idf = (sum(shingle_idf.get(s, 0.0) for s in inter) / max(1, len(inter))) if inter else 0.0
                accept = (low_df_count >= min_low_df_anchors and mean_idf >= idf_mean_min)
                if getattr(cfg, 'enable_small_path', False):
                    small_len_thresh = int(getattr(cfg, 'small_len_thresh', 6))
                    Lb = window_len_map.get(bid, 999)
                    Lc = window_len_map.get(cid, 999)
                    if len(inter) <= 2 and (Lb < small_len_thresh or Lc < small_len_thresh):
                        accept = True
                if accept:
                    u, v = sorted((bid, cid))
                    out.append((u, v, j_sim))
        return out

    # Determine parallelism
    n_jobs = 1
    syscfg = getattr(cfg, 'system', None)
    if syscfg is not None:
        jobs_val = getattr(syscfg, 'jobs', 'auto')
        if jobs_val == 'auto':
            n_jobs = max(1, (os.cpu_count() or 1) - 0)
        else:
            try:
                n_jobs = int(jobs_val)
            except Exception:
                n_jobs = 1
    n_jobs = max(1, n_jobs)

    if n_jobs == 1 or len(block_candidates) < 50:
        # Small graphs: stay serial to avoid overhead
        for bid, cands in block_candidates.items():
            for u, v, w in _score_block(bid, cands):
                edges.add((u, v))
                edge_weights[(u, v)] = w
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {ex.submit(_score_block, bid, cands): bid for bid, cands in block_candidates.items()}
            for fut in as_completed(futures):
                res = fut.result()
                for u, v, w in res:
                    edges.add((u, v))
                    edge_weights[(u, v)] = w
                    edge_weights[edge] = max(edge_weights.get(edge, 0.0), j_sim)
                    added = True
            # Hybrid fall-back: bandset Jaccard for qualifying blocks/pairs
            if enable_hybrid_bandset and not added:
                # Qualify both blocks
                try:
                    b_ref = [b for b_id, b in robust_blocks if b_id == block_id][0]
                    c_ref = [b for b_id, b in robust_blocks if b_id == cand_id][0]
                    len_ok = (len(getattr(b_ref, 'matches', [])) >= bandset_min_len and
                              len(getattr(c_ref, 'matches', [])) >= bandset_min_len)
                    id_ok = (getattr(b_ref, 'identity', None) is not None and getattr(b_ref, 'identity') >= bandset_min_identity and
                             getattr(c_ref, 'identity', None) is not None and getattr(c_ref, 'identity') >= bandset_min_identity)
                except Exception:
                    len_ok = False
                    id_ok = False
                if len_ok and id_ok:
                    B = filtered_bandset_map.get(block_id, set())
                    C = filtered_bandset_map.get(cand_id, set())
                    if B and C:
                        inter = B & C
                        union = B | C
                        if union:
                            inter_w = sum(bandset_idf.get(s, 0.0) for s in inter)
                            union_w = sum(bandset_idf.get(s, 0.0) for s in union)
                            j_bw = (inter_w / union_w) if union_w > 0 else 0.0
                        else:
                            j_bw = 0.0
                        if j_bw >= bandset_tau:
                            edge = tuple(sorted([block_id, cand_id]))
                            edges.add(edge)
                            edge_weights[edge] = max(edge_weights.get(edge, 0.0), float(j_bw))
            # Targeted non-hybrid acceptor: narrowly gated bandset Jaccard
            if enable_targeted_bandset_acceptor and not added:
                try:
                    b_ref = [b for b_id, b in robust_blocks if b_id == block_id][0]
                    c_ref = [b for b_id, b in robust_blocks if b_id == cand_id][0]
                    len_ok = (len(getattr(b_ref, 'matches', [])) >= targeted_bandset_min_len and
                              len(getattr(c_ref, 'matches', [])) >= targeted_bandset_min_len)
                    id_ok = (getattr(b_ref, 'identity', None) is not None and getattr(b_ref, 'identity') >= targeted_bandset_min_identity and
                             getattr(c_ref, 'identity', None) is not None and getattr(c_ref, 'identity') >= targeted_bandset_min_identity)
                except Exception:
                    len_ok = False
                    id_ok = False
                if len_ok and id_ok:
                    B = filtered_bandset_map.get(block_id, set())
                    C = filtered_bandset_map.get(cand_id, set())
                    if B and C:
                        inter = B & C
                        union = B | C
                        if union:
                            inter_w = sum(bandset_idf.get(s, 0.0) for s in inter)
                            union_w = sum(bandset_idf.get(s, 0.0) for s in union)
                            j_bw = (inter_w / union_w) if union_w > 0 else 0.0
                        else:
                            j_bw = 0.0
                        # Additional informative-overlap checks on bandset intersection
                        low_df_count_b = sum(1 for s in inter if bandset_df.get(s, 0) <= targeted_bandset_low_df_max)
                        mean_idf_b = (sum(bandset_idf.get(s, 0.0) for s in inter) / max(1, len(inter))) if inter else 0.0
                        if j_bw >= targeted_bandset_tau and low_df_count_b >= targeted_bandset_min_low_df and mean_idf_b >= targeted_bandset_idf_mean_min:
                            edge = tuple(sorted([block_id, cand_id]))
                            edges.add(edge)
                            edge_weights[edge] = max(edge_weights.get(edge, 0.0), float(j_bw))
    
    console_print(f"Constructed graph with {len(edges)} mutual-k edges")

    # Optional: cap per-node degree to break hubs (keep top-N edges by weight)
    def _degree_cap(edge_set: Set[Tuple[int, int]], weights: Dict[Tuple[int, int], float], cap: int) -> Set[Tuple[int, int]]:
        if cap is None or cap <= 0:
            return edge_set
        inc: Dict[int, List[Tuple[Tuple[int, int], float]]] = defaultdict(list)
        for e in edge_set:
            u, v = e
            w = weights.get(e, 0.0)
            inc[u].append((e, w))
            inc[v].append((e, w))
        keep: Set[Tuple[int, int]] = set()
        for node, lst in inc.items():
            lst.sort(key=lambda x: -x[1])
            for e, _ in lst[:cap]:
                keep.add(e)
        return keep

    capped_edges = _degree_cap(edges, edge_weights, degree_cap)

    # Optional: prune graph by k-core to reduce transitive glue
    def _k_core(edge_set: Set[Tuple[int, int]], k: int) -> Set[Tuple[int, int]]:
        if k <= 0:
            return edge_set
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        changed = True
        active = set(adj.keys())
        while changed:
            changed = False
            to_remove = [n for n in list(active) if len(adj[n]) < k]
            if to_remove:
                changed = True
                for n in to_remove:
                    for nbr in list(adj[n]):
                        adj[nbr].discard(n)
                    adj.pop(n, None)
                active = set(adj.keys())
        new_edges = set()
        for u in adj:
            for v in adj[u]:
                if u < v:
                    new_edges.add((u, v))
        return new_edges

    # Remove k-core pruning (too strict for cassette exploration)
    pruned_edges = capped_edges

    # Optional: require edges to be supported by triangles (shared neighbors)
    def _triangle_filter(edge_set: Set[Tuple[int, int]], min_tri: int) -> Set[Tuple[int, int]]:
        if min_tri is None or min_tri <= 0:
            return edge_set
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        keep = set()
        for u, v in edge_set:
            common = adj[u].intersection(adj[v])
            if len(common) >= min_tri:
                keep.add((u, v))
        return keep

    # Optionally require triangles only for small-block edges (adaptive path)
    coherent_edges = pruned_edges
    if getattr(cfg, 'enable_small_path', False):
        small_len_thresh = int(getattr(cfg, 'small_len_thresh', 6))
        small_tri_min = int(getattr(cfg, 'small_edge_triangle_min', 1))
        if small_tri_min > 0:
            tri_edges = _triangle_filter(pruned_edges, small_tri_min)
            def is_small_edge(e):
                u, v = e
                Lu = window_len_map.get(u, 999)
                Lv = window_len_map.get(v, 999)
                return (Lu < small_len_thresh) or (Lv < small_len_thresh)
            coherent_edges = set()
            for e in pruned_edges:
                if is_small_edge(e):
                    if e in tri_edges:
                        coherent_edges.add(e)
                else:
                    coherent_edges.add(e)

    # Step 7: Community detection (fallback to connected components if unavailable)
    nodes = list(filtered_shingles_map.keys())
    components: List[Set[int]] = []
    if use_community_detection:
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(nodes)
            for u, v in coherent_edges:
                w = edge_weights.get((u, v), 1.0)
                G.add_edge(u, v, weight=w)
            # Greedy modularity communities (weight-aware)
            comms = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
            components = [set(c) for c in comms if len(c) > 0]
        except Exception as e:
            console_print(f"Community detection unavailable or failed ({e}), using connected components")
            components = _find_connected_components(coherent_edges, nodes)
    else:
        components = _find_connected_components(coherent_edges, nodes)
    
    # Step 8: Assign cluster IDs deterministically
    cluster_assignment = {}
    
    # Reserve 0 for sink
    next_cluster_id = 1
    
    # Add all non-robust blocks to sink
    for block_id, _ in sink_blocks:
        cluster_assignment[block_id] = sink_label
    
    # Process components for robust blocks
    component_keys = []
    for component in components:
        if len(component) == 1 and not keep_singletons:
            # Singleton goes to sink
            cluster_assignment[list(component)[0]] = sink_label
        else:
            # Multi-block component or kept singleton
            component_size = len(component)
            representative = min(component)  # Lowest block_id as representative
            key = (-component_size, representative)  # Sort by size desc, then by rep block id asc
            component_keys.append((key, component))
    
    # Sort components and assign cluster IDs
    component_keys.sort()
    for (key, component) in component_keys:
        for block_id in component:
            cluster_assignment[block_id] = next_cluster_id
        next_cluster_id += 1
    
    console_print(f"Final clusters: {max(cluster_assignment.values()) if cluster_assignment else 0} non-sink + sink")
    
    # Ensure all blocks have assignments
    for block_id, _ in robust_blocks:
        if block_id not in cluster_assignment:
            cluster_assignment[block_id] = sink_label
    
    return cluster_assignment


def _extract_window_indices(block) -> Tuple[List[int], List[int]]:
    """Extract integer window indices from block matches."""
    query_indices = []
    target_indices = []
    
    if hasattr(block, 'matches') and block.matches:
        for match in block.matches:
            # Extract indices from window IDs like 'sample_locus_123' -> 123
            q_idx = _parse_window_index(match.query_window_id)
            t_idx = _parse_window_index(match.target_window_id)
            
            if q_idx is not None and t_idx is not None:
                query_indices.append(q_idx)
                target_indices.append(t_idx)
    
    return query_indices, target_indices


def _parse_window_index(window_id: str) -> int:
    """Parse window index from window ID like 'sample_locus_123' -> 123."""
    try:
        # Window IDs typically end with '_<number>'
        parts = window_id.split('_')
        if len(parts) >= 2:
            return int(parts[-1])
    except (ValueError, IndexError):
        pass
    
    # Fallback: try to find any number in the string
    match = re.search(r'(\d+)$', window_id)
    if match:
        return int(match.group(1))
    
    return None


def _find_connected_components(edges: Set[Tuple], nodes: List) -> List[Set]:
    """Find connected components in an undirected graph."""
    # Build adjacency list
    graph = defaultdict(set)
    all_nodes = set(nodes)
    
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
        all_nodes.add(u)
        all_nodes.add(v)
    
    # DFS to find components
    visited = set()
    components = []
    
    for node in all_nodes:
        if node not in visited:
            component = set()
            stack = [node]
            
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.add(current)
                    
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            components.append(component)
    
    return components
