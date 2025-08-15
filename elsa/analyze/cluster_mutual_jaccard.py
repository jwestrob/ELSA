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
    
    # Extract configuration
    min_anchors = getattr(cfg, 'min_anchors', 4)
    min_span_genes = getattr(cfg, 'min_span_genes', 8)
    v_mad_max_genes = getattr(cfg, 'v_mad_max_genes', 1)
    
    srp_bits = getattr(cfg, 'srp_bits', 256)
    srp_bands = getattr(cfg, 'srp_bands', 32)
    srp_band_bits = getattr(cfg, 'srp_band_bits', 8)
    srp_seed = getattr(cfg, 'srp_seed', 1337)
    shingle_k = getattr(cfg, 'shingle_k', 3)
    
    jaccard_tau = getattr(cfg, 'jaccard_tau', 0.5)
    mutual_k = getattr(cfg, 'mutual_k', 3)
    df_max = getattr(cfg, 'df_max', 200)
    
    size_ratio_min = getattr(cfg, 'size_ratio_min', 0.5)
    size_ratio_max = getattr(cfg, 'size_ratio_max', 2.0)
    keep_singletons = getattr(cfg, 'keep_singletons', False)
    sink_label = getattr(cfg, 'sink_label', 0)
    
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
        
        # Apply robustness criteria
        is_robust = (n >= min_anchors and 
                    span_q >= min_span_genes and 
                    span_t >= min_span_genes and 
                    v_mad <= v_mad_max_genes)
        
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
    
    console_print(f"After DF filtering: {len(filtered_shingle_to_blocks)} unique shingles")
    
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
    
    for block_id, candidates in block_candidates.items():
        if block_id not in filtered_shingles_map:
            continue
            
        shingles_b = filtered_shingles_map[block_id]
        
        # Compute Jaccard similarities with candidates
        similarities = []
        for cand_id in candidates:
            if cand_id in filtered_shingles_map:
                shingles_c = filtered_shingles_map[cand_id]
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
                        # Add undirected edge
                        edge = tuple(sorted([block_id, cand_id]))
                        edges.add(edge)
    
    console_print(f"Constructed graph with {len(edges)} mutual-k edges")
    
    # Step 7: Connected components
    components = _find_connected_components(edges, list(filtered_shingles_map.keys()))
    
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