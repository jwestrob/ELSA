#!/usr/bin/env python3
"""Fit a frozen PCA model on UniRef50 embeddings.

Usage:
    # Embed UniRef50 sample and fit PCA (run this first):
    python scripts/fit_uniref50_pca.py --fasta data/uniref50_sample.fasta \
        --output data/frozen_pca/ --model esm2_t12 --dim 256

    # Then use the frozen PCA in your config:
    # plm:
    #   frozen_pca_path: data/frozen_pca/pca_model.pkl

    # Or just fit PCA from pre-computed raw embeddings:
    python scripts/fit_uniref50_pca.py --raw-parquet data/frozen_pca/uniref50_raw.parquet \
        --output data/frozen_pca/ --dim 256
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from sklearn.decomposition import PCA

console = Console()

CHUNK_SIZE = 2000  # sequences per disk flush


def embed_uniref50(fasta_path: Path, output_dir: Path, model: str,
                   batch_aa: int, fp16: bool, max_seqs: int = None) -> Path:
    """Embed UniRef50 sequences and stream raw embeddings to disk in chunks."""
    from elsa.embeddings import ProteinEmbedder, parse_fasta_to_proteins
    from elsa.params import PLMConfig

    console.print(f"[bold]Loading sequences from {fasta_path}[/bold]")
    proteins = parse_fasta_to_proteins(fasta_path, sample_id="uniref50")

    if max_seqs and len(proteins) > max_seqs:
        console.print(f"Subsampling {max_seqs} from {len(proteins)} sequences")
        rng = np.random.default_rng(42)
        indices = rng.choice(len(proteins), max_seqs, replace=False)
        proteins = [proteins[i] for i in sorted(indices)]

    total_aa = sum(p.length for p in proteins)
    console.print(f"Sequences: {len(proteins):,}  Total AA: {total_aa:,}")

    plm_config = PLMConfig(
        model=model,
        device="auto",
        batch_amino_acids=batch_aa,
        fp16=fp16,
        project_to_D=0,  # No PCA during embedding
        l2_normalize=False,
    )

    embedder = ProteinEmbedder(plm_config)
    # Override the conservative sliding window batch halving — <1% of UniRef50
    # sequences need sliding window, so the penalty tanks throughput for nothing.
    embedder.batch_size_aa = batch_aa
    emb_dim = embedder.embedding_dim
    console.print(f"Embedding dim: {emb_dim}")
    console.print(f"Batch size override: {embedder.batch_size_aa} AA")

    # Build pyarrow schema once
    fields = [pa.field("gene_id", pa.string())]
    for i in range(emb_dim):
        fields.append(pa.field(f"raw_{i:04d}", pa.float32()))
    schema = pa.schema(fields)

    raw_path = output_dir / "uniref50_raw.parquet"
    progress_file = output_dir / "embed_progress.txt"

    # Stream embeddings to parquet via chunked row-group writes
    writer = pq.ParquetWriter(raw_path, schema, compression="snappy")
    chunk_ids = []
    chunk_vecs = []
    total_written = 0

    try:
        for emb in embedder.embed_sequences(proteins, progress_file=progress_file):
            chunk_ids.append(emb.gene_id)
            chunk_vecs.append(emb.embedding.astype(np.float32))

            if len(chunk_ids) >= CHUNK_SIZE:
                _flush_chunk(writer, chunk_ids, chunk_vecs, emb_dim)
                total_written += len(chunk_ids)
                chunk_ids.clear()
                chunk_vecs.clear()

        # Final partial chunk
        if chunk_ids:
            _flush_chunk(writer, chunk_ids, chunk_vecs, emb_dim)
            total_written += len(chunk_ids)
    finally:
        writer.close()

    size_mb = raw_path.stat().st_size / 1e6
    console.print(f"Saved {total_written:,} raw embeddings: {raw_path} ({size_mb:.1f} MB)")
    return raw_path


def _flush_chunk(writer: pq.ParquetWriter, gene_ids: list,
                 vecs: list, emb_dim: int):
    """Write a chunk of embeddings as a parquet row group."""
    mat = np.stack(vecs)  # (N, emb_dim)
    arrays = [pa.array(gene_ids, type=pa.string())]
    for i in range(emb_dim):
        arrays.append(pa.array(mat[:, i], type=pa.float32()))

    batch = pa.RecordBatch.from_arrays(
        arrays,
        names=["gene_id"] + [f"raw_{i:04d}" for i in range(emb_dim)],
    )
    writer.write_batch(batch)


def fit_pca(raw_path: Path, output_dir: Path, dim: int, subsample: int = 50000) -> Path:
    """Fit PCA on raw embeddings and save the model."""
    console.print(f"\n[bold]Fitting PCA: -> {dim}D[/bold]")

    pf = pq.ParquetFile(raw_path)
    raw_cols = [c for c in pf.schema.names if c.startswith("raw_")]
    raw_cols.sort()

    # Read only embedding columns into memory
    table = pf.read(columns=raw_cols)
    matrix = np.column_stack([col.to_numpy(zero_copy_only=False) for col in table.columns]).astype(np.float32)
    del table
    console.print(f"Loaded {matrix.shape[0]:,} embeddings, dim={matrix.shape[1]}")

    if matrix.shape[0] > subsample:
        console.print(f"Subsampling {subsample:,} for PCA fitting")
        rng = np.random.default_rng(42)
        indices = rng.choice(matrix.shape[0], subsample, replace=False)
        fit_matrix = matrix[indices]
    else:
        fit_matrix = matrix

    pca = PCA(n_components=dim, random_state=42)
    pca.fit(fit_matrix)

    explained = pca.explained_variance_ratio_.sum()
    console.print(f"Explained variance: {explained:.4f}")
    console.print(f"Top-10 components: {pca.explained_variance_ratio_[:10].round(4).tolist()}")

    # Save PCA model
    pca_path = output_dir / "pca_model.pkl"
    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    console.print(f"Saved PCA model: {pca_path}")

    # Quick sanity check: project and check norms
    projected = pca.transform(matrix[:100])
    norms = np.linalg.norm(projected, axis=1)
    console.print(f"Sanity check (first 100): projected norm mean={norms.mean():.3f} std={norms.std():.3f}")

    return pca_path


def main():
    parser = argparse.ArgumentParser(description="Fit frozen PCA on UniRef50 embeddings")
    parser.add_argument("--fasta", type=Path, help="UniRef50 FASTA file")
    parser.add_argument("--raw-parquet", type=Path, help="Pre-computed raw embeddings parquet")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/frozen_pca"),
                        help="Output directory")
    parser.add_argument("--model", default="esm2_t12", help="PLM model name")
    parser.add_argument("--dim", type=int, default=256, help="PCA target dimension")
    parser.add_argument("--batch-aa", type=int, default=16000, help="Batch size in amino acids")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--max-seqs", type=int, default=None,
                        help="Max sequences to embed (subsample)")
    parser.add_argument("--subsample-pca", type=int, default=50000,
                        help="Max sequences for PCA fitting")

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.raw_parquet:
        raw_path = args.raw_parquet
    elif args.fasta:
        raw_path = embed_uniref50(
            args.fasta, args.output, args.model,
            args.batch_aa, args.fp16, args.max_seqs,
        )
    else:
        parser.error("Provide either --fasta or --raw-parquet")

    pca_path = fit_pca(raw_path, args.output, args.dim, args.subsample_pca)

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"Add to your config:")
    console.print(f"  plm:")
    console.print(f"    frozen_pca_path: {pca_path}")


if __name__ == "__main__":
    main()
