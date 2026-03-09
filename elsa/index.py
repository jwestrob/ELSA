"""
HNSW / FAISS / sklearn index construction for gene-level kNN search.
"""

from __future__ import annotations

from typing import Tuple, Any
import os
import sys

import numpy as np

# Prevent OpenMP duplicate-library crash on macOS (pip faiss-cpu links its own
# libomp which conflicts with the conda/system copy).  Must be set before
# faiss is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# NOTE: Do NOT set OMP_NUM_THREADS=1 here — it bakes into FAISS's thread pool
# at import time and cannot be reliably overridden later.  Instead, we control
# FAISS threading via faiss.omp_set_num_threads() in seed.py at search time.

# Optional HNSW import — will fall back to FAISS or sklearn if not available
try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False

# Optional FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def _resolve_backend(index_backend: str) -> str:
    """Resolve 'auto' to the best available backend."""
    if index_backend != "auto":
        return index_backend

    if HNSWLIB_AVAILABLE:
        return "hnsw"
    if FAISS_AVAILABLE:
        return "faiss_flat"
    return "sklearn"


def build_gene_index(
    embeddings: np.ndarray,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 128,
    index_backend: str = "faiss_ivfflat",
    faiss_nprobe: int = 32,
) -> Tuple[str, Any]:
    """
    Build ANN index for gene-level kNN search.

    Args:
        embeddings: (n_genes, dim) array of L2-normalized gene embeddings
        m: HNSW M parameter (connections per node)
        ef_construction: HNSW build quality parameter
        ef_search: HNSW query quality parameter
        index_backend: "auto" | "hnsw" | "faiss_ivfflat" | "faiss_ivfpq" | "faiss_ivfsq" | "faiss_flat" | "sklearn"
        faiss_nprobe: IVF clusters to search (higher = better recall, slower)

    Returns:
        Tuple of (index_type, index) for use with query functions.
        index_type is one of: "hnsw", "faiss", "sklearn", "empty"
    """
    n, dim = embeddings.shape
    if n == 0:
        return ("empty", None)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-9)

    backend = _resolve_backend(index_backend)

    # --- HNSW ---
    if backend == "hnsw":
        if not HNSWLIB_AVAILABLE:
            raise ImportError("hnswlib is not installed. Install with: pip install hnswlib")
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
        ids = np.arange(n, dtype=np.int32)
        index.add_items(normalized, ids)
        index.set_ef(ef_search)
        return ("hnsw", index)

    # --- FAISS backends ---
    if backend.startswith("faiss"):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is not installed. Install with: pip install faiss-cpu")
        return _build_faiss_index(normalized, dim, n, backend, faiss_nprobe)

    # --- sklearn fallback ---
    if backend == "sklearn":
        print("[GeneIndex] Using sklearn NearestNeighbors (slower)",
              file=sys.stderr, flush=True)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(50, n), metric="cosine", algorithm="brute")
        nn.fit(normalized)
        return ("sklearn", nn)

    raise ValueError(f"Unknown index_backend: {index_backend!r}")


def _build_faiss_index(
    normalized: np.ndarray,
    dim: int,
    n: int,
    backend: str,
    faiss_nprobe: int,
) -> Tuple[str, Any]:
    """Build a FAISS index (inner-product on L2-normalized vectors = cosine)."""
    # Ensure contiguous float32 for FAISS
    normalized = np.ascontiguousarray(normalized, dtype=np.float32)

    if backend == "faiss_flat":
        index = faiss.IndexFlatIP(dim)
        index.add(normalized)
        return ("faiss", index)

    # IVF variants need a minimum number of vectors for training
    nlist = max(16, min(int(np.sqrt(n)), 4096))
    # FAISS requires at least nlist training points
    if n < nlist:
        print(f"[GeneIndex] Too few vectors ({n}) for IVF (nlist={nlist}), falling back to faiss_flat",
              file=sys.stderr, flush=True)
        index = faiss.IndexFlatIP(dim)
        index.add(normalized)
        return ("faiss", index)

    quantizer = faiss.IndexFlatIP(dim)

    if backend == "faiss_ivfflat":
        index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                   faiss.METRIC_INNER_PRODUCT)
        index.train(normalized)
        index.add(normalized)
        index.nprobe = faiss_nprobe
        return ("faiss", index)

    if backend == "faiss_ivfpq":
        pq_m = dim // 8  # 32 subvectors for 256D
        if pq_m < 1:
            pq_m = 1
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, 8,
                                 faiss.METRIC_INNER_PRODUCT)
        index.train(normalized)
        index.add(normalized)
        index.nprobe = faiss_nprobe
        return ("faiss", index)

    if backend == "faiss_ivfsq":
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, dim, nlist,
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.train(normalized)
        index.add(normalized)
        index.nprobe = faiss_nprobe
        return ("faiss", index)

    raise ValueError(f"Unknown FAISS backend: {backend!r}")
