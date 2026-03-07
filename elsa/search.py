"""
Locus-level search: find syntenic blocks matching a query locus.

Reuses the same seed -> chain -> extract pipeline but with a single
query locus against the full indexed database.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Any, Tuple, Dict, Set

import numpy as np
import pandas as pd

from .seed import GeneAnchor
from .chain import ChainedBlock, chain_anchors_lis, extract_nonoverlapping_chains


def embed_query_proteins(
    proteins: list,
    plm_config: Any,
    work_dir: str,
    query_name: str = "query",
) -> pd.DataFrame:
    """Embed query proteins on-the-fly and return a DataFrame for search_locus().

    Embeds proteins with the configured PLM, optionally projects through
    PCA (when ``plm_config.project_to_D > 0``), and L2-normalizes to
    match the indexed embeddings.

    Args:
        proteins: Parsed ProteinSequence objects to embed.
        plm_config: PLMConfig instance (model, device, etc.).
        work_dir: Path to the ELSA index work directory.
        query_name: Label used as ``sample_id`` (defaults to "query").

    Returns:
        DataFrame with columns expected by :func:`search_locus`.
    """
    from .embeddings import ProteinEmbedder, AggregationStrategy

    work_path = Path(work_dir)

    # Embed proteins
    embedder = ProteinEmbedder(
        plm_config,
        window_size=1024,
        overlap=256,
        aggregation=AggregationStrategy.MAX_POOL,
    )
    raw_embeddings = list(embedder.embed_sequences(proteins))
    emb_matrix = np.array([e.embedding for e in raw_embeddings])

    # Project through PCA only when projection is enabled
    project_to_D = getattr(plm_config, 'project_to_D', 0)
    if project_to_D and project_to_D > 0:
        pca_path = work_path / "ingest" / "pca_model.pkl"
        scaler_path = work_path / "ingest" / "scaler.pkl"

        if not pca_path.exists():
            raise FileNotFoundError(
                f"PCA model not found at {pca_path}. Run 'elsa embed' first."
            )

        with open(pca_path, "rb") as fh:
            pca_model = pickle.load(fh)

        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as fh:
                scaler = pickle.load(fh)

        if scaler is not None:
            emb_matrix = scaler.transform(emb_matrix)
        projected = pca_model.transform(emb_matrix)
    else:
        # No PCA — use raw embeddings directly
        projected = emb_matrix

    # L2-normalize to match indexed embeddings
    if getattr(plm_config, 'l2_normalize', True):
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        projected = projected / (norms + 1e-8)

    # Build DataFrame matching search_locus() expectations
    dim = projected.shape[1]
    data = {
        "sample_id": [p.sample_id for p in proteins],
        "contig_id": [p.contig_id for p in proteins],
        "gene_id": [p.gene_id for p in proteins],
        "start": [p.start for p in proteins],
        "end": [p.end for p in proteins],
        "strand": [p.strand for p in proteins],
    }
    for i in range(dim):
        data[f"emb_{i:03d}"] = projected[:, i]

    df = pd.DataFrame(data)
    df = df.sort_values(["contig_id", "start"])
    df["position_index"] = df.groupby(["sample_id", "contig_id"]).cumcount()

    return df


def search_locus(
    query_genes: pd.DataFrame,
    index_tuple: Any,
    target_genes: pd.DataFrame,
    target_embeddings: np.ndarray,
    k: int = 50,
    similarity_threshold: float = 0.85,
    max_gap: int = 2,
    min_chain_size: int = 2,
    gap_penalty_scale: float = 0.0,
    max_results: int = 50,
) -> List[ChainedBlock]:
    """
    Search for syntenic blocks matching a query locus.

    Args:
        query_genes: DataFrame for query locus with gene_id, sample_id, contig_id,
                     position_index, and emb_* columns
        index_tuple: Pre-built (type, index) from build_gene_index
        target_genes: Full target gene DataFrame (same schema as query)
        target_embeddings: (n_target, dim) array of target gene embeddings
        k: Number of neighbors to search
        similarity_threshold: Minimum cosine similarity
        max_gap: Maximum gap in chain
        min_chain_size: Minimum anchors per block
        gap_penalty_scale: Concave gap penalty
        max_results: Maximum blocks to return

    Returns:
        List of ChainedBlock objects sorted by chain_score descending
    """
    index_type, index = index_tuple
    if index is None:
        return []

    emb_cols = [c for c in query_genes.columns if c.startswith("emb_")]
    if not emb_cols:
        return []

    query_embeddings = query_genes[emb_cols].values.astype(np.float32)
    n_query = len(query_genes)

    # Normalize query embeddings
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_normalized = query_embeddings / (norms + 1e-9)

    # Query the index
    k_query = min(k + 10, len(target_genes))

    if index_type == "hnsw":
        labels, distances = index.knn_query(query_normalized, k=k_query)
        similarities = 1 - distances
    elif index_type == "faiss":
        similarities, labels = index.search(query_normalized.astype(np.float32), k_query)
        labels = labels.astype(np.intp)
    else:  # sklearn
        distances, labels = index.kneighbors(query_normalized, n_neighbors=k_query)
        similarities = 1 - distances

    # Build target lookup arrays
    t_genome_arr = target_genes['sample_id'].values
    t_contig_arr = target_genes['contig_id'].values
    t_pos_arr = target_genes['position_index'].values
    t_gene_id_arr = target_genes['gene_id'].values
    t_strand_arr = target_genes['strand'].values if 'strand' in target_genes.columns else None

    q_genome_arr = query_genes['sample_id'].values
    q_contig_arr = query_genes['contig_id'].values
    q_pos_arr = query_genes['position_index'].values
    q_gene_id_arr = query_genes['gene_id'].values
    q_strand_arr = query_genes['strand'].values if 'strand' in query_genes.columns else None

    query_genome = str(q_genome_arr[0])

    # Collect cross-genome anchors from query to targets
    anchors: List[GeneAnchor] = []
    seen: Set[Tuple[int, int]] = set()

    for qi in range(n_query):
        for j_pos in range(k_query):
            tj = int(labels[qi, j_pos])
            if tj < 0:  # FAISS returns -1 when no result found
                continue
            sim = float(similarities[qi, j_pos])

            if sim < similarity_threshold:
                continue

            if str(t_genome_arr[tj]) == query_genome:
                continue

            pair_key = (qi, tj)
            if pair_key in seen:
                continue
            seen.add(pair_key)

            # Compute relative orientation
            if q_strand_arr is not None and t_strand_arr is not None:
                sq = int(q_strand_arr[qi]) if q_strand_arr[qi] != 0 else 1
                st = int(t_strand_arr[tj]) if t_strand_arr[tj] != 0 else 1
                rel_orient = 1 if sq == st else -1
            else:
                rel_orient = 0

            anchor = GeneAnchor(
                query_idx=int(q_pos_arr[qi]),
                target_idx=int(t_pos_arr[tj]),
                query_genome=query_genome,
                target_genome=str(t_genome_arr[tj]),
                query_contig=str(q_contig_arr[qi]),
                target_contig=str(t_contig_arr[tj]),
                query_gene_id=str(q_gene_id_arr[qi]),
                target_gene_id=str(t_gene_id_arr[tj]),
                similarity=sim,
                orientation=rel_orient,
            )
            anchors.append(anchor)

    if not anchors:
        return []

    # Group by target contig
    groups: Dict[Tuple[str, str], List[GeneAnchor]] = {}
    for anchor in anchors:
        key = (anchor.target_genome, anchor.target_contig)
        groups.setdefault(key, []).append(anchor)

    # Chain each group
    all_blocks: List[ChainedBlock] = []
    block_id = 0

    for key, group_anchors in groups.items():
        if len(group_anchors) < min_chain_size:
            continue

        chains = chain_anchors_lis(
            group_anchors,
            max_gap=max_gap,
            min_size=min_chain_size,
            gap_penalty_scale=gap_penalty_scale,
        )
        if not chains:
            continue

        blocks = extract_nonoverlapping_chains(chains, block_id_start=block_id)
        all_blocks.extend(blocks)
        block_id += len(blocks)

    # Sort by score and limit
    all_blocks.sort(key=lambda b: -b.chain_score)
    return all_blocks[:max_results]
