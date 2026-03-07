#!/usr/bin/env python3
"""Embed cross-species genes with ProtT5-XL-U50.

Reads protein sequences from the existing ESM2 genes.parquet,
embeds them with ProtT5, and writes a new parquet with ProtT5 embeddings.
Checkpoints every N batches so it can be resumed.

Usage:
    python benchmarks/scripts/embed_prott5.py [--batch-aa 6000] [--resume]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from transformers import T5Tokenizer, T5EncoderModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ESM2_PARQUET = PROJECT_ROOT / "benchmarks/elsa_output/cross_species/ingest/genes.parquet"
OUTPUT_DIR = PROJECT_ROOT / "benchmarks/elsa_output/cross_species_prott5/ingest"
OUTPUT_PARQUET = OUTPUT_DIR / "genes.parquet"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
GENOMES_DIR = PROJECT_ROOT / "benchmarks/data/cross_species/genomes"


def write_protein_fasta(genes_df: pd.DataFrame, output_path: Path) -> int:
    """Extract protein sequences from genome FASTAs and write a single FASTA.

    Returns number of sequences written.
    """
    from Bio import SeqIO
    from Bio.Seq import Seq
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="Bio")

    # Process one genome at a time to limit memory
    n_written = 0
    with open(output_path, "w") as out:
        for fasta in sorted(GENOMES_DIR.glob("*.fna")):
            sample_id = fasta.stem
            # Load this genome's contigs
            contigs = {}
            for record in SeqIO.parse(fasta, "fasta"):
                contigs[record.id] = str(record.seq)

            # Extract genes for this genome
            genome_genes = genes_df[genes_df["sample_id"] == sample_id]
            for _, row in genome_genes.iterrows():
                contig_seq = contigs.get(row["contig_id"])
                if contig_seq is None:
                    continue
                nuc = contig_seq[int(row["start"]) - 1:int(row["end"])]
                if int(row["strand"]) == -1:
                    nuc = str(Seq(nuc).reverse_complement())
                aa = str(Seq(nuc).translate(to_stop=True))
                if len(aa) >= 20:
                    out.write(f">{row['gene_id']}\n{aa}\n")
                    n_written += 1
            del contigs  # free genome contigs

    return n_written


def iter_fasta_sequences(fasta_path: Path) -> dict[str, str]:
    """Read sequences from a FASTA into a dict (only called in batches)."""
    from Bio import SeqIO
    seqs = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seqs[rec.id] = str(rec.seq)
    return seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-aa", type=int, default=6000,
                        help="Target amino acids per batch")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ESM2 parquet for metadata (drop embedding columns to save RAM)
    print("Loading ESM2 genes.parquet for metadata...")
    esm2_df = pd.read_parquet(ESM2_PARQUET)
    meta_cols = [c for c in esm2_df.columns if not c.startswith("emb_")]
    meta_df = esm2_df[meta_cols].copy()
    del esm2_df
    print(f"  {len(meta_df):,} genes, {meta_df['sample_id'].nunique()} genomes")

    # Check for completed output
    if args.resume and OUTPUT_PARQUET.exists():
        print(f"Output already exists: {OUTPUT_PARQUET}")
        return

    # Extract protein sequences to a FASTA on disk (avoids holding in RAM)
    proteins_fasta = OUTPUT_DIR / "proteins_prott5.faa"
    if not proteins_fasta.exists():
        print("Extracting protein sequences from genome FASTAs...")
        n_written = write_protein_fasta(meta_df, proteins_fasta)
        print(f"  Wrote {n_written:,} sequences to {proteins_fasta}")
    else:
        n_written = sum(1 for line in open(proteins_fasta) if line.startswith(">"))
        print(f"  Using existing {proteins_fasta} ({n_written:,} sequences)")

    # Build gene_id -> sequence length lookup (scan FASTA, don't hold seqs)
    gene_lengths = {}
    with open(proteins_fasta) as f:
        current_id = None
        current_len = 0
        for line in f:
            if line.startswith(">"):
                if current_id:
                    gene_lengths[current_id] = current_len
                current_id = line[1:].strip()
                current_len = 0
            else:
                current_len += len(line.strip())
        if current_id:
            gene_lengths[current_id] = current_len

    # Find which genes still need embedding
    done_genes = set()
    checkpoint_files = sorted(CHECKPOINT_DIR.glob("batch_*.npz"))
    if args.resume and checkpoint_files:
        print(f"Found {len(checkpoint_files)} checkpoints, loading gene IDs...")
        for cf in checkpoint_files:
            data = np.load(cf, allow_pickle=True)
            done_genes.update(data["gene_ids"].tolist())
        print(f"  {len(done_genes):,} genes already embedded")

    gene_ids = meta_df["gene_id"].values
    work_indices = [i for i, gid in enumerate(gene_ids)
                    if gid not in done_genes and gid in gene_lengths]
    print(f"  {len(work_indices):,} genes to embed")

    if not work_indices:
        print("All genes already embedded, assembling output...")
        _assemble_output(meta_df, 1024)
        return

    # Load the full FASTA into a dict now (we need random access by gene_id).
    # At ~44MB for 139k proteins, this is fine.
    print("Loading protein sequences for embedding...")
    seq_map = {}
    with open(proteins_fasta) as f:
        current_id = None
        current_seq = []
        for line in f:
            if line.startswith(">"):
                if current_id:
                    seq_map[current_id] = "".join(current_seq)
                current_id = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_id:
            seq_map[current_id] = "".join(current_seq)
    print(f"  {len(seq_map):,} sequences loaded")

    mem = psutil.virtual_memory()
    print(f"  Memory before model: {mem.used / 1e9:.1f} GB used, {mem.available / 1e9:.1f} GB avail")

    # Load model
    print("Loading ProtT5-XL-half on MPS (float32)...")
    t0 = time.time()
    model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device).half().eval()
    emb_dim = model.config.d_model
    print(f"  Loaded in {time.time() - t0:.1f}s, dim={emb_dim}, device={device}")

    mem = psutil.virtual_memory()
    print(f"  Memory after model: {mem.used / 1e9:.1f} GB used, {mem.available / 1e9:.1f} GB avail")

    # Batch and embed
    batch_aa = args.batch_aa
    total_aa_processed = 0
    total_genes_processed = 0
    batch_count = len(checkpoint_files)
    start_time = time.time()

    current_batch_indices = []
    current_batch_aa = 0

    def _embed_single(gid, seq):
        """Embed a single protein — no padding overhead."""
        spaced = " ".join(seq)
        inputs = tokenizer(
            [spaced],
            add_special_tokens=True,
            padding=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pool over sequence length (exclude nothing — single seq, no pad)
        emb = outputs.last_hidden_state[0].float().mean(dim=0).cpu().numpy()

        # Free MPS memory
        del inputs, outputs
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

        return emb

    # Checkpoint every N genes
    checkpoint_interval = batch_aa // 300  # roughly same checkpoint freq
    checkpoint_interval = max(checkpoint_interval, 10)
    buffer_gids = []
    buffer_embs = []

    for pos, idx in enumerate(work_indices):
        gid = gene_ids[idx]
        seq = seq_map[gid]
        emb = _embed_single(gid, seq)

        buffer_gids.append(gid)
        buffer_embs.append(emb)
        total_aa_processed += len(seq)
        total_genes_processed += 1

        if len(buffer_gids) >= checkpoint_interval or pos == len(work_indices) - 1:
            batch_count += 1
            emb_array = np.stack(buffer_embs)
            # L2 normalize
            norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
            emb_array = emb_array / np.maximum(norms, 1e-8)

            np.savez_compressed(
                CHECKPOINT_DIR / f"batch_{batch_count:06d}.npz",
                gene_ids=np.array(buffer_gids),
                embeddings=emb_array,
            )
            buffer_gids = []
            buffer_embs = []

        if (total_genes_processed % 500 == 0) or pos == len(work_indices) - 1:
            elapsed = time.time() - start_time
            rate = total_aa_processed / elapsed if elapsed > 0 else 0
            remaining_genes = len(work_indices) - (pos + 1)
            if total_genes_processed > 0:
                avg_aa_per_gene = total_aa_processed / total_genes_processed
                remaining_aa = remaining_genes * avg_aa_per_gene
                eta_sec = remaining_aa / rate if rate > 0 else 0
                eta_str = f"{eta_sec / 3600:.1f}h" if eta_sec > 3600 else f"{eta_sec / 60:.1f}m"
            else:
                eta_str = "?"

            mem = psutil.virtual_memory()
            print(
                f"  {total_genes_processed:,}/{len(work_indices):,} genes "
                f"({100 * total_genes_processed / len(work_indices):.1f}%), "
                f"{rate:.0f} AA/s, "
                f"ETA {eta_str}, "
                f"mem {mem.available / 1e9:.1f}GB avail"
            )

    elapsed = time.time() - start_time
    print(f"\nEmbedding complete: {total_genes_processed:,} genes in {elapsed / 3600:.1f}h "
          f"({total_aa_processed / elapsed:.0f} AA/s)")

    # Free model before assembly
    del model, tokenizer, seq_map
    if device.type == "mps":
        torch.mps.empty_cache()
    import gc; gc.collect()

    # Assemble final parquet
    _assemble_output(meta_df, emb_dim)


def _assemble_output(meta_df: pd.DataFrame, emb_dim: int):
    """Assemble checkpoint files into final parquet."""
    print("Assembling final parquet...")

    # Load all checkpoints
    gene_embeddings = {}
    for cf in sorted(CHECKPOINT_DIR.glob("batch_*.npz")):
        data = np.load(cf, allow_pickle=True)
        gids = data["gene_ids"]
        embs = data["embeddings"]
        for gid, emb in zip(gids, embs):
            gene_embeddings[str(gid)] = emb

    print(f"  {len(gene_embeddings):,} embeddings loaded from checkpoints")

    # Determine embedding dim from first embedding
    first_emb = next(iter(gene_embeddings.values()))
    emb_dim = len(first_emb)

    # Build embedding matrix aligned to meta_df order
    n = len(meta_df)
    emb_matrix = np.zeros((n, emb_dim), dtype=np.float32)
    matched = 0
    for i, gid in enumerate(meta_df["gene_id"].values):
        if gid in gene_embeddings:
            emb_matrix[i] = gene_embeddings[gid]
            matched += 1

    print(f"  {matched:,}/{n:,} genes have ProtT5 embeddings")

    # Build output dataframe
    emb_col_names = [f"emb_{i}" for i in range(emb_dim)]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_col_names, index=meta_df.index)
    out_df = pd.concat([meta_df.reset_index(drop=True), emb_df], axis=1)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"  Wrote {OUTPUT_PARQUET} ({len(out_df):,} rows, {emb_dim}D embeddings)")


if __name__ == "__main__":
    main()
