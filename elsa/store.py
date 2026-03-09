"""
Persistent FAISS index + gene metadata store for synteny discovery.

Directory layout:
    store_dir/
        index.faiss        # FAISS IVF-Flat (or Flat) index
        metadata.parquet   # gene_id, sample_id, contig_id, start, end, strand
        embeddings.npy     # raw embedding matrix (N, D) — for index rebuilds on add
        config.json        # dim, nlist, nprobe, genomes list, creation info
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Any, List

import numpy as np
import pandas as pd


def _require_faiss():
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import importlib
    faiss = importlib.import_module("faiss")
    return faiss


class SyntenyStore:
    """Persistent FAISS index with gene metadata sidecar."""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.index_path = self.store_dir / "index.faiss"
        self.metadata_path = self.store_dir / "metadata.parquet"
        self.embeddings_path = self.store_dir / "embeddings.npy"
        self.config_path = self.store_dir / "config.json"
        self._index = None
        self._metadata: Optional[pd.DataFrame] = None
        self._embeddings: Optional[np.ndarray] = None
        self._config: dict = {}

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        store_dir: Path,
        genes_df: pd.DataFrame,
        nprobe: int = 32,
    ) -> "SyntenyStore":
        """Build a new store from a genes DataFrame with emb_* columns.

        The DataFrame must contain sample_id, contig_id, gene_id, start,
        end, strand, and emb_* embedding columns (already L2-normalized).
        """
        store = cls(store_dir)
        store.store_dir.mkdir(parents=True, exist_ok=True)

        emb_cols = sorted(c for c in genes_df.columns if c.startswith("emb_"))
        if not emb_cols:
            raise ValueError("genes_df has no emb_* columns")

        meta_cols = ["sample_id", "contig_id", "gene_id", "start", "end", "strand"]
        missing = set(meta_cols) - set(genes_df.columns)
        if missing:
            raise ValueError(f"genes_df missing columns: {missing}")

        # Canonical sort order (same as pipeline)
        genes_df = genes_df.sort_values(
            ["sample_id", "contig_id", "start", "end"]
        ).reset_index(drop=True)

        embeddings = genes_df[emb_cols].values.astype(np.float32)
        metadata = genes_df[meta_cols].copy()

        # Build FAISS index
        index = _build_ivfflat(embeddings, nprobe)

        # Persist
        faiss = _require_faiss()
        faiss.write_index(index, str(store.index_path))
        metadata.to_parquet(store.metadata_path, index=False)
        np.save(store.embeddings_path, embeddings)

        config = {
            "dim": int(embeddings.shape[1]),
            "n_vectors": int(embeddings.shape[0]),
            "nprobe": nprobe,
            "genomes": sorted(metadata["sample_id"].unique().tolist()),
            "emb_cols": emb_cols,
        }
        store.config_path.write_text(json.dumps(config, indent=2))

        store._index = index
        store._metadata = metadata
        store._embeddings = embeddings
        store._config = config

        n_genomes = len(config["genomes"])
        print(
            f"[Store] Created {store.store_dir}: "
            f"{config['n_vectors']} vectors, {config['dim']}D, "
            f"{n_genomes} genomes",
            file=sys.stderr, flush=True,
        )
        return store

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, store_dir: Path, nprobe: Optional[int] = None) -> "SyntenyStore":
        """Load an existing store from disk."""
        store = cls(store_dir)

        if not store.config_path.exists():
            raise FileNotFoundError(f"No store found at {store_dir}")

        store._config = json.loads(store.config_path.read_text())

        faiss = _require_faiss()
        store._index = faiss.read_index(str(store.index_path))

        if nprobe is not None:
            store._index.nprobe = nprobe
        elif hasattr(store._index, "nprobe"):
            store._index.nprobe = store._config.get("nprobe", 32)

        store._metadata = pd.read_parquet(store.metadata_path)
        store._embeddings = np.load(store.embeddings_path)

        print(
            f"[Store] Loaded {store.store_dir}: "
            f"{store._config['n_vectors']} vectors, {store._config['dim']}D, "
            f"{len(store._config['genomes'])} genomes",
            file=sys.stderr, flush=True,
        )
        return store

    # ------------------------------------------------------------------
    # Incremental add
    # ------------------------------------------------------------------

    def add_genes(
        self,
        new_genes_df: pd.DataFrame,
        nprobe: Optional[int] = None,
    ) -> None:
        """Add new genes + embeddings. Rebuilds the FAISS index.

        Index rebuild is fast (seconds for 100k+ vectors). The value of
        persistence is avoiding re-embedding and re-ingesting metadata.
        """
        emb_cols = self._config["emb_cols"]
        meta_cols = ["sample_id", "contig_id", "gene_id", "start", "end", "strand"]

        new_emb = new_genes_df[emb_cols].values.astype(np.float32)
        new_meta = new_genes_df[meta_cols].copy()

        # Check for duplicate gene_ids
        existing_ids = set(self._metadata["gene_id"])
        new_ids = set(new_meta["gene_id"])
        overlap = existing_ids & new_ids
        if overlap:
            n = len(overlap)
            print(
                f"[Store] Skipping {n} duplicate gene_ids already in store",
                file=sys.stderr, flush=True,
            )
            mask = ~new_meta["gene_id"].isin(existing_ids)
            new_meta = new_meta[mask].reset_index(drop=True)
            new_emb = new_emb[mask.values]

        if len(new_meta) == 0:
            print("[Store] No new genes to add", file=sys.stderr, flush=True)
            return

        # Concatenate
        combined_meta = pd.concat(
            [self._metadata, new_meta], ignore_index=True
        )
        combined_emb = np.vstack([self._embeddings, new_emb])

        # Re-sort to canonical order
        sort_idx = combined_meta.sort_values(
            ["sample_id", "contig_id", "start", "end"]
        ).index.values
        combined_meta = combined_meta.iloc[sort_idx].reset_index(drop=True)
        combined_emb = combined_emb[sort_idx]

        # Rebuild index
        _nprobe = nprobe or self._config.get("nprobe", 32)
        index = _build_ivfflat(combined_emb, _nprobe)

        # Update state
        self._metadata = combined_meta
        self._embeddings = combined_emb
        self._index = index

        new_genomes = sorted(combined_meta["sample_id"].unique().tolist())
        added_genomes = set(new_genomes) - set(self._config["genomes"])

        self._config.update({
            "n_vectors": int(combined_emb.shape[0]),
            "genomes": new_genomes,
        })

        # Persist
        faiss = _require_faiss()
        faiss.write_index(index, str(self.index_path))
        combined_meta.to_parquet(self.metadata_path, index=False)
        np.save(self.embeddings_path, combined_emb)
        self.config_path.write_text(json.dumps(self._config, indent=2))

        print(
            f"[Store] Added {len(new_meta)} genes "
            f"({len(added_genomes)} new genomes: {sorted(added_genomes) if added_genomes else 'none'}). "
            f"Total: {self._config['n_vectors']} vectors, "
            f"{len(new_genomes)} genomes",
            file=sys.stderr, flush=True,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_index_tuple(self) -> Tuple[str, Any]:
        """Return ("faiss", index) for the chain pipeline."""
        return ("faiss", self._index)

    def get_genes_df(self) -> pd.DataFrame:
        """Return genes DataFrame with metadata + emb_* columns."""
        emb_cols = self._config["emb_cols"]
        emb_df = pd.DataFrame(self._embeddings, columns=emb_cols)
        return pd.concat([self._metadata.reset_index(drop=True), emb_df], axis=1)

    @property
    def genomes(self) -> List[str]:
        return self._config.get("genomes", [])

    @property
    def n_vectors(self) -> int:
        return self._config.get("n_vectors", 0)

    @property
    def dim(self) -> int:
        return self._config.get("dim", 0)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_ivfflat(
    embeddings: np.ndarray,
    nprobe: int = 32,
) -> Any:
    """Build a FAISS IVF-Flat index from L2-normalized embeddings."""
    faiss = _require_faiss()

    n, dim = embeddings.shape
    normalized = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Re-normalize to be safe
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    normalized = normalized / (norms + 1e-9)

    nlist = max(16, min(int(np.sqrt(n)), 4096))

    if n < nlist:
        index = faiss.IndexFlatIP(dim)
        index.add(normalized)
        return index

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(
        quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT
    )
    index.train(normalized)
    index.add(normalized)
    index.nprobe = nprobe
    return index
