"""
Adapter for loading gene metadata and embeddings from external sources.

Supports:
  - Sharur DuckDB (proteins table) or Prodigal-header FASTA files
  - HDF5 embeddings (Sharur format) or parquet with emb_* columns
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def load_proteins_from_duckdb(db_path: Union[str, Path]) -> pd.DataFrame:
    """Read protein metadata from a Sharur DuckDB database.

    Returns a DataFrame with ELSA-compatible columns:
        sample_id, contig_id, gene_id, start, end, strand
    """
    import duckdb

    db_path = str(db_path)
    conn = duckdb.connect(db_path, read_only=True)
    try:
        df = conn.execute("""
            SELECT protein_id, contig_id, bin_id, start, end_coord, strand
            FROM proteins
            ORDER BY bin_id, contig_id, start
        """).fetchdf()
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(f"No proteins found in {db_path}")

    df = df.rename(columns={
        "protein_id": "gene_id",
        "bin_id": "sample_id",
        "end_coord": "end",
    })

    # Convert strand: "+"/"-" → 1/-1
    strand_map = {"+": 1, "-": -1}
    df["strand"] = df["strand"].map(strand_map).fillna(1).astype(int)

    return df[["sample_id", "contig_id", "gene_id", "start", "end", "strand"]]


def load_proteins_from_fasta(proteins_dir: Union[str, Path]) -> pd.DataFrame:
    """Parse protein metadata from Prodigal-style FASTA headers.

    Expects headers like:
        >PROTEIN_ID # START # END # STRAND # ...
    or Sharur-style:
        >PROTEIN_ID|CONTIG_ID|GENOME_ID # START # END # STRAND # ...

    Falls back to sequential indexing if headers aren't parseable.
    """
    proteins_dir = Path(proteins_dir)
    faa_files = sorted(proteins_dir.glob("*.faa"))
    if not faa_files:
        raise FileNotFoundError(f"No .faa files found in {proteins_dir}")

    rows = []
    for faa in faa_files:
        sample_id = faa.stem
        with open(faa) as f:
            for line in f:
                if not line.startswith(">"):
                    continue
                header = line[1:].strip()
                row = _parse_prodigal_header(header, sample_id)
                if row:
                    rows.append(row)

    if not rows:
        raise RuntimeError(f"No proteins parsed from {proteins_dir}")

    return pd.DataFrame(rows)


def _parse_prodigal_header(header: str, default_sample: str) -> Optional[dict]:
    """Parse a single Prodigal or Sharur-style FASTA header.

    Supports:
        Sharur:   >PROTEIN|CONTIG|GENOME # START # END # STRAND # ...
        Prodigal: >accn|CONTIG_N # START # END # STRAND # ...
        Prodigal: >CONTIG_N # START # END # STRAND # ...
    """
    parts = header.split("#")
    if len(parts) < 4:
        return None

    id_field = parts[0].strip()
    try:
        start = int(parts[1].strip())
        end = int(parts[2].strip())
        strand_raw = int(parts[3].strip())
    except (ValueError, IndexError):
        return None

    # Sharur format: PROTEIN_ID|CONTIG_ID|GENOME_ID (3 pipe-delimited fields)
    pipe_parts = id_field.split("|")
    if len(pipe_parts) >= 3:
        gene_id = pipe_parts[0]
        contig_id = pipe_parts[1]
        sample_id = pipe_parts[2]
    else:
        # Prodigal format: gene_id is the full first token (may contain |)
        # e.g., "accn|CONTIG_N" or just "CONTIG_N"
        gene_id = id_field.split()[0]
        # Contig is gene_id minus the trailing _N (gene ordinal)
        if "_" in gene_id:
            contig_id = gene_id.rsplit("_", 1)[0]
        else:
            contig_id = gene_id
        sample_id = default_sample

    return {
        "sample_id": sample_id,
        "contig_id": contig_id,
        "gene_id": gene_id,
        "start": start,
        "end": end,
        "strand": 1 if strand_raw >= 0 else -1,
    }


def load_embeddings_h5(h5_path: Union[str, Path]) -> pd.DataFrame:
    """Load embeddings from a Sharur-format HDF5 file.

    Expected datasets:
        protein_ids: array of protein ID strings
        embeddings:  (N, D) float matrix

    Returns DataFrame with gene_id + emb_000..emb_NNN columns.
    """
    import h5py

    h5_path = str(h5_path)
    with h5py.File(h5_path, "r") as f:
        protein_ids = f["protein_ids"][:]
        embeddings = f["embeddings"][:]

    # Decode bytes if needed
    if protein_ids.dtype.kind == "S" or protein_ids.dtype.kind == "O":
        protein_ids = [
            pid.decode("utf-8") if isinstance(pid, bytes) else str(pid)
            for pid in protein_ids
        ]

    n, dim = embeddings.shape
    emb_cols = [f"emb_{i:03d}" for i in range(dim)]
    emb_df = pd.DataFrame(embeddings.astype(np.float32), columns=emb_cols)
    emb_df.insert(0, "gene_id", protein_ids)

    return emb_df


def load_embeddings_parquet(parquet_path: Union[str, Path]) -> pd.DataFrame:
    """Load embeddings from a parquet file with emb_* columns.

    Must contain a protein/gene ID column (auto-detected) and emb_* columns.
    """
    df = pd.read_parquet(parquet_path)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError(f"No emb_* columns found in {parquet_path}")

    # Find the ID column
    id_col = None
    for candidate in ["gene_id", "protein_id", "id"]:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise RuntimeError(
            f"No ID column (gene_id/protein_id/id) found in {parquet_path}. "
            f"Columns: {list(df.columns)}"
        )

    result = df[[id_col] + emb_cols].copy()
    if id_col != "gene_id":
        result = result.rename(columns={id_col: "gene_id"})

    return result


def build_genes_dataframe(
    proteins_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    normalize: bool = True,
) -> pd.DataFrame:
    """Merge protein metadata with embeddings into ELSA genes format.

    Args:
        proteins_df: DataFrame with sample_id, contig_id, gene_id, start, end, strand
        embeddings_df: DataFrame with gene_id + emb_* columns
        normalize: L2-normalize embedding vectors (recommended)

    Returns:
        DataFrame ready for run_chain_pipeline
    """
    emb_cols = [c for c in embeddings_df.columns if c.startswith("emb_")]

    merged = proteins_df.merge(embeddings_df, on="gene_id", how="inner")

    n_proteins = len(proteins_df)
    n_embeddings = len(embeddings_df)
    n_merged = len(merged)
    n_dropped = n_proteins - n_merged

    if n_merged == 0:
        raise RuntimeError(
            f"No protein IDs matched between metadata ({n_proteins} proteins) "
            f"and embeddings ({n_embeddings} vectors). "
            f"Check that protein IDs are consistent."
        )

    if n_dropped > 0:
        import sys
        print(
            f"[Adapter] Matched {n_merged}/{n_proteins} proteins "
            f"({n_dropped} without embeddings dropped)",
            file=sys.stderr, flush=True,
        )

    if normalize:
        emb_vals = merged[emb_cols].values.astype(np.float32)
        norms = np.linalg.norm(emb_vals, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        merged[emb_cols] = emb_vals / norms

    return merged
