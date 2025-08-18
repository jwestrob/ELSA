"""
Cluster robust blocks using a sparse mutual-k graph with Jaccard over shingles.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Iterable, Callable, Any
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
            
            # Compute SRP tokens
            window_tokens = srp_tokens(
                emb_matrix, 
                n_bits=srp_bits,
                n_bands=srp_bands, 
                band_bits=srp_band_bits,
                seed=srp_seed
            )
            
            # Generate k-gram shingles
            shingles = block_shingles(window_tokens, k=shingle_k)
            block_shingles_map[block_id] = shingles
            
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

    # Build IDF map for weighted Jaccard: idf = log(1 + N / df)
    import math
    n_docs = max(1, len(filtered_shingles_map))
    shingle_idf = {sid: math.log(1.0 + (n_docs / max(1, shingle_df.get(sid, 1)))) for sid in filtered_shingle_to_blocks.keys()}
    
    # Step 5: Candidate generation with size ratio prefilter
    block_candidates = {}
    
    for block_id, shingles in filtered_shingles_map.items():
        candidates = set()
        
        # Union postings of all shingles for this block
        for shingle_id in shingles:
            candidates.update(filtered_shingle_to_blocks[shingle_id])
        
        # Remove self
        candidates.discard(block_id)
        
        # Apply size ratio prefilter
        n_b = len([b for b_id, b in robust_blocks if b_id == block_id][0].matches)  # alignment length
        s_b_size = len(shingles)
        
        filtered_candidates = []
        for cand_id in candidates:
            if cand_id in filtered_shingles_map:
                n_c = len([b for b_id, b in robust_blocks if b_id == cand_id][0].matches)
                s_c_size = len(filtered_shingles_map[cand_id])
                
                # Size ratio checks
                shingle_ratio = s_b_size / max(s_c_size, 1)
                length_ratio = n_b / max(n_c, 1)
                
                if (size_ratio_min <= shingle_ratio <= size_ratio_max and 
                    size_ratio_min <= length_ratio <= size_ratio_max):
                    filtered_candidates.append(cand_id)
        
        block_candidates[block_id] = filtered_candidates
    
    # Step 6: Similarity computation and mutual-k graph construction
    edges = set()
    edge_weights: Dict[Tuple[int, int], float] = {}
    
    for block_id, candidates in block_candidates.items():
        if block_id not in filtered_shingles_map:
            continue
            
        shingles_b = filtered_shingles_map[block_id]
        
        # Compute Jaccard similarities with candidates
        similarities = []
        for cand_id in candidates:
            if cand_id in filtered_shingles_map:
                shingles_c = filtered_shingles_map[cand_id]
                if use_weighted_jaccard:
                    inter = shingles_b & shingles_c
                    union = shingles_b | shingles_c
                    if union:
                        inter_w = sum(shingle_idf.get(s, 0.0) for s in inter)
                        union_w = sum(shingle_idf.get(s, 0.0) for s in union)
                        j_sim = (inter_w / union_w) if union_w > 0 else 0.0
                    else:
                        j_sim = 0.0
                else:
                    j_sim = jaccard(shingles_b, shingles_c)
                similarities.append((cand_id, j_sim))
        
        # Keep top-k by Jaccard similarity (ties broken by lower block id)
        similarities.sort(key=lambda x: (-x[1], x[0]))
        top_k = similarities[:mutual_k]
        
        # Check mutual-k condition and similarity threshold
        for cand_id, j_sim in top_k:
            if j_sim >= jaccard_tau:
                # Check if the edge is mutual
                if (cand_id in block_candidates and 
                    block_id in block_candidates[cand_id]):
                    
                    # Verify reciprocal top-k
                    cand_similarities = []
                    shingles_c = filtered_shingles_map[cand_id]
                    for other_id in block_candidates[cand_id]:
                        if other_id in filtered_shingles_map:
                            shingles_other = filtered_shingles_map[other_id]
                            other_j_sim = jaccard(shingles_c, shingles_other)
                            cand_similarities.append((other_id, other_j_sim))
                    
                    cand_similarities.sort(key=lambda x: (-x[1], x[0]))
                    cand_top_k = cand_similarities[:mutual_k]

                    # Check if block_id is in candidate's top-k
                    if any(other_id == block_id and other_j_sim >= jaccard_tau 
                          for other_id, other_j_sim in cand_top_k):
                        # Enforce informative-overlap checks before adding edge
                        inter = shingles_b & shingles_c
                        low_df_count = sum(1 for s in inter if shingle_df.get(s, 0) <= low_df_threshold)
                        mean_idf = (sum(shingle_idf.get(s, 0.0) for s in inter) / max(1, len(inter))) if inter else 0.0
                        if low_df_count >= min_low_df_anchors and mean_idf >= idf_mean_min:
                            edge = tuple(sorted([block_id, cand_id]))
                            edges.add(edge)
                            # Store weight as the similarity value
                            edge_weights[edge] = max(edge_weights.get(edge, 0.0), j_sim)
    
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

    # Remove triangle support requirement (overly strict)
    coherent_edges = pruned_edges

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
