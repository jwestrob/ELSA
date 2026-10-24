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


def _resolve_backend(index_backend: str, n: int = 0) -> str:
    """Resolve 'auto' to the best available backend for a corpus of ``n`` genes.

    The ranking depends on corpus size, and it inverts between the two scales
    ELSA runs at. Compare on the recall/time frontier, since either backend can
    buy accuracy with time — reading off a single (ef, nprobe) pair says more
    about the settings than the backends.

    At 90,566 genes (``index_backend_recall.csv``) IVF-Flat leads: recall@60
    0.9895 at nprobe=32 against 0.9813 for hnsw at ef=128. That is why
    ``faiss_ivfflat`` became the documented default.

    At 3,395,472 genes (``index_backend_recall_at_scale.csv``, k=60, totals
    covering build plus the full all-vs-all pass ELSA actually performs):

        backend                    total    recall@60
        hnsw ef=128                 371s       0.9914
        hnsw ef=256                 500s       0.9968
        faiss_ivfflat nprobe=32    1093s       0.9913
        faiss_ivfflat nprobe=64    2163s       0.9976

    Matched on recall, hnsw is 2.9x faster at ~0.991 and 4.3x faster at ~0.997.
    The mechanism is IVF granularity: ``nlist`` grows as sqrt(n), so cells hold
    ~300 vectors at 90k and ~1,840 at 3.4M, and a fixed ``nprobe`` scans a
    steadily smaller share of the corpus. HNSW has no such term and its graph
    improves as the corpus densifies.

    So ``auto`` takes hnsw above ``FLAT_MAX`` and exact search below it, where an
    exact index builds in a fraction of a second and leaves nothing to
    approximate. ``auto`` is now the configured default too; runs made before
    2026-08 used ``faiss_ivfflat``, which must be named explicitly to reproduce
    them.

    Two lessons from getting here. Selecting on build time alone is wrong: an
    earlier revision preferred IVF-Flat for its 27.8x faster build, checked
    end-to-end only at 90,589 genes where search time hides, and inverted the
    ranking at scale. And a single measurement is weak against a parameter
    sweep — the fix was to repeat the sweep at the scale in question, not to
    argue from one point.
    """
    if index_backend != "auto":
        return index_backend

    FLAT_MAX = 50_000
    if n < FLAT_MAX and FAISS_AVAILABLE:
        return "faiss_flat"
    if HNSWLIB_AVAILABLE:
        return "hnsw"
    if FAISS_AVAILABLE:
        return "faiss_flat" if n < FLAT_MAX else "faiss_ivfflat"
    return "sklearn"


def build_gene_index(
    embeddings: np.ndarray,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 256,
    index_backend: str = "auto",
    faiss_nprobe: int = 32,
) -> Tuple[str, Any]:
    """
    Build ANN index for gene-level kNN search.

    Args:
        embeddings: (n_genes, dim) array of L2-normalized gene embeddings
        m: HNSW M parameter (connections per node)
        ef_construction: HNSW build quality parameter
        ef_search: HNSW query quality parameter. 256 is the measured knee at
            3.4M genes: recall@60 0.9914 at 128, 0.9968 at 256, 0.9967 at 512,
            for +129s and +178s respectively on a full all-vs-all pass.
        index_backend: "auto" | "hnsw" | "faiss_ivfflat" | "faiss_ivfpq" | "faiss_ivfsq" | "faiss_flat" | "sklearn"
        faiss_nprobe: IVF clusters to search (higher = better recall, slower)

    Returns:
        Tuple of (index_type, index) for use with query functions.
        index_type is one of: "hnsw", "faiss", "sklearn", "empty"
    """
    n, dim = embeddings.shape
    if n == 0:
        return ("empty", None)

    # Normalize embeddings for cosine similarity.
    #
    # This repeats work: the pipeline hands over vectors projection.py already
    # L2-normalized, and find_cross_genome_anchors normalizes again on its own
    # path, so the corpus is divided through twice and a second full copy
    # (4.3 GB at DPANN scale) is allocated to produce roughly the array we were
    # given. Skipping it when the input is already unit-length was measured and
    # rejected: it saves 5.6s of an 858s run, and real embeddings arrive at
    # norms of 0.99974-1.00026, so the guard only fires with a tolerance loose
    # enough to change retrieval at the last ulp and reorder tied neighbours.
    # Perturbing every downstream number to save 0.65% is not a trade worth
    # making; revisit if the index build ever dominates.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-9)

    backend = _resolve_backend(index_backend, n)

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
