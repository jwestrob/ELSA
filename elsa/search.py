"""
Locus-level search: find syntenic blocks matching a query locus.

Reuses the same seed -> chain -> extract pipeline but with a single
query locus against the full indexed database.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Any, Tuple, Dict, Set

import numpy as np
import pandas as pd

from .chain import ChainedBlock, chain_groups_batched
from .seed import ANCHOR_COLS, group_anchors_by_contig_pair


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
    neighborhood_retrieval: bool = True,
    n_jobs: int | None = None,
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
        neighborhood_retrieval: Select a fixed target budget using original-seed
            support within the supplied query locus and full reference contigs.
        n_jobs: Threads for retrieval, neighborhood selection and chaining.

    Returns:
        List of ChainedBlock objects sorted by chain_score descending
    """
    index_type, index = index_tuple
    if index is None or query_genes.empty or target_genes.empty or max_results < 1:
        return []
    if k < 1 or max_gap < 0 or min_chain_size < 1:
        raise ValueError("k and min_chain_size must be positive; max_gap must be nonnegative")
    if target_embeddings.ndim != 2 or len(target_embeddings) != len(target_genes):
        raise ValueError("Target embeddings must align with target metadata rows")

    emb_cols = [c for c in query_genes.columns if c.startswith("emb_")]
    if not emb_cols:
        return []
    query_vectors = query_genes[emb_cols].to_numpy(dtype=np.float32)
    if query_vectors.shape[1] != target_embeddings.shape[1]:
        raise ValueError("Query and target embedding dimensions differ")
    norms = np.linalg.norm(query_vectors, axis=1)
    eligible = np.isfinite(query_vectors).all(axis=1) & np.isfinite(norms) & (norms > 0)
    query_vectors /= norms[:, None] + np.float32(1e-9)
    active = np.flatnonzero(eligible)
    if len(active) == 0:
        return []
    k_query = min(k + 10, len(target_genes))
    if index_type == "hnsw":
        options = {"num_threads": n_jobs} if n_jobs is not None else {}
        labels, distances = index.knn_query(query_vectors[active], k=k_query, **options)
        similarities = 1 - distances
    elif index_type == "faiss":
        if n_jobs is not None:
            import faiss
            faiss.omp_set_num_threads(n_jobs)
        similarities, labels = index.search(query_vectors[active], k_query)
    else:
        distances, labels = index.kneighbors(query_vectors[active], n_neighbors=k_query)
        similarities = 1 - distances

    qr = np.repeat(active, k_query)
    tr = np.asarray(labels, dtype=np.int64).ravel()
    scores = similarities.ravel()
    valid = (tr >= 0) & (tr < len(target_genes)) & np.isfinite(scores) & (scores >= similarity_threshold)
    qr, tr, scores = qr[valid], tr[valid], scores[valid]
    qsample, tsample = query_genes.sample_id.to_numpy(), target_genes.sample_id.to_numpy()
    cross = qsample[qr] != tsample[tr]
    qr, tr, scores = qr[cross], tr[cross], scores[cross]
    if neighborhood_retrieval:
        from .neighborhood import select_neighborhood_candidates
        columns = ['sample_id', 'contig_id', 'position_index']
        query_metadata = query_genes[columns + (['strand'] if 'strand' in query_genes else [])]
        reference_metadata = target_genes[columns + (['strand'] if 'strand' in target_genes else [])]
        selected = select_neighborhood_candidates(
            query_vectors, target_embeddings, qr, tr, scores,
            query_metadata=query_metadata, reference_metadata=reference_metadata,
            k=k_query, threshold=similarity_threshold, max_gap=max_gap,
            n_jobs=n_jobs, normalize_reference=True,
        )
        qr, tr, scores = selected.query_rows, selected.target_rows, selected.similarities
    if len(qr) == 0:
        return []
    order = np.lexsort((tr, qr))
    qr, tr, scores = qr[order], tr[order], scores[order]
    # Repeated ANN neighbors represent one candidate protein pair.
    unique = np.r_[True, (qr[1:] != qr[:-1]) | (tr[1:] != tr[:-1])]
    qr, tr, scores = qr[unique], tr[unique], scores[unique]
    q = query_genes.iloc[qr]
    t = target_genes.iloc[tr]
    qs = q.strand.to_numpy() if 'strand' in q else np.zeros(len(q), dtype=np.int8)
    ts = t.strand.to_numpy() if 'strand' in t else np.zeros(len(t), dtype=np.int8)
    orientation = np.where(np.isin(qs, [-1, 1]) & np.isin(ts, [-1, 1]), qs*ts, 0).astype(np.int8)
    anchors = pd.DataFrame(dict(
        query_idx=q.position_index.to_numpy(), target_idx=t.position_index.to_numpy(),
        query_genome=q.sample_id.to_numpy(), target_genome=t.sample_id.to_numpy(),
        query_contig=q.contig_id.to_numpy(), target_contig=t.contig_id.to_numpy(),
        query_gene_id=q.gene_id.to_numpy(), target_gene_id=t.gene_id.to_numpy(),
        similarity=scores, orientation=orientation,
    ), columns=ANCHOR_COLS)
    if n_jobs is not None:
        from .chain import _NUMBA_AVAILABLE
        if _NUMBA_AVAILABLE:
            import numba
            numba.set_num_threads(n_jobs)
    blocks = chain_groups_batched(
        group_anchors_by_contig_pair(anchors, canonicalize=False),
        max_gap=max_gap, min_size=min_chain_size, gap_penalty_scale=gap_penalty_scale,
        extract_blocks=True,
    )
    output = []
    for row in blocks.itertuples(index=False):
        output.append(ChainedBlock(
            block_id=int(row.block_id), query_genome=str(row.query_genome),
            target_genome=str(row.target_genome), query_contig=str(row.query_contig),
            target_contig=str(row.target_contig), query_start=int(row.query_start),
            query_end=int(row.query_end), target_start=int(row.target_start),
            target_end=int(row.target_end), n_anchors=int(row.n_anchors),
            chain_score=float(row.chain_score), orientation=int(row.orientation),
            **{name: json.loads(getattr(row, name)) for name in (
                'anchor_query_ids', 'anchor_target_ids', 'anchor_query_gene_ids', 'anchor_target_gene_ids')},
        ))
    output.sort(key=lambda block: -block.chain_score)
    return output[:max_results]
