#!/usr/bin/env python3
"""Fit universal PCA projections from diverse UniRef50 protein embeddings.

Fits three PCA variants and evaluates each on the E. coli operon benchmark:
  1. Standard PCA (480D → 256D)
  2. PCA + whitening (equalizes component variance)
  3. PCA + whitening + drop top-1 component ("all-but-the-top")

Usage:
    python scripts/fit_universal_pca.py --fasta data/uniref50_sample.fasta --device cuda

Output:
    elsa/data/universal_pca.pkl
    elsa/data/universal_pca_whitened.pkl
    elsa/data/universal_pca_abt.pkl   (all-but-the-top)
    benchmarks/evaluation/pca_variant_comparison.md
"""

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Download ───────────────────────────────────────────────────────

def download_uniref50_sample(n_total: int = 50000, output_fasta: Path = None) -> Path:
    """Download a taxonomically stratified sample from UniRef50 via UniProt API.

    Uses paginated search API (500 per page) for reliable, fast downloads.

    Stratification:
      - 40% Bacteria (taxid 2)
      - 30% Eukaryota (taxid 2759)
      - 15% Archaea (taxid 2157)
      - 15% Viruses (taxid 10239)
    """
    strata = [
        ("Bacteria", 2, int(n_total * 0.40)),
        ("Eukaryota", 2759, int(n_total * 0.30)),
        ("Archaea", 2157, int(n_total * 0.15)),
        ("Viruses", 10239, int(n_total * 0.15)),
    ]

    if output_fasta is None:
        output_fasta = ROOT / "data" / "uniref50_sample.fasta"
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    PAGE_SIZE = 500  # Small pages download quickly

    with open(output_fasta, "w") as out_f:
        total_written = 0

        for name, taxid, n_wanted in strata:
            print(f"Downloading {n_wanted:,} {name} sequences (taxid={taxid})...")
            n_got = 0
            # Use paginated search endpoint (returns quickly)
            url = (
                f"https://rest.uniprot.org/uniref/search"
                f"?format=fasta"
                f"&query=(identity:0.5)+AND+(taxonomy_id:{taxid})"
                f"&size={min(PAGE_SIZE, n_wanted)}"
            )

            while url and n_got < n_wanted:
                retries = 3
                for attempt in range(retries):
                    try:
                        resp = requests.get(url, timeout=120)
                        resp.raise_for_status()
                        fasta_text = resp.text
                        n_seqs = fasta_text.count(">")
                        out_f.write(fasta_text)
                        if not fasta_text.endswith("\n"):
                            out_f.write("\n")
                        n_got += n_seqs
                        total_written += n_seqs

                        if n_got % 2000 < PAGE_SIZE:
                            print(f"  {name}: {n_got:,}/{n_wanted:,} sequences...")

                        # Follow pagination Link header
                        link = resp.headers.get("Link", "")
                        if 'rel="next"' in link and n_got < n_wanted:
                            # Parse: <URL>; rel="next"
                            url = link.split(">")[0].lstrip("<")
                        else:
                            url = None
                        break
                    except Exception as e:
                        if attempt < retries - 1:
                            wait = 5 * (attempt + 1)
                            print(f"  Retry {attempt+1}/{retries}: {e}")
                            time.sleep(wait)
                        else:
                            print(f"  WARNING: Failed {name} at {n_got}: {e}")
                            url = None

            print(f"  Got {n_got:,} {name} sequences")

    print(f"\nTotal: {total_written:,} sequences saved to {output_fasta}")
    return output_fasta


# ── FASTA parsing ───────────────────────────────────────────────────

def parse_fasta(fasta_path: Path) -> list:
    """Parse FASTA file into list of (id, sequence) tuples."""
    sequences = []
    current_id = None
    current_seq = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences.append((current_id, "".join(current_seq)))
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        sequences.append((current_id, "".join(current_seq)))

    return sequences


# ── Embedding ───────────────────────────────────────────────────────

def embed_proteins(sequences: list, device: str = "cuda", batch_aa: int = 16000):
    """Embed protein sequences with ESM2-t12, return raw 480D embeddings."""
    from elsa.embeddings import ProteinEmbedder, ProteinSequence
    from elsa.params import PLMConfig

    config = PLMConfig(
        model="esm2_t12",
        device=device,
        fp16=True,
        batch_amino_acids=batch_aa,
        project_to_D=0,
        l2_normalize=False,
    )

    embed_system = ProteinEmbedder(config)

    protein_seqs = [
        ProteinSequence(
            sample_id="uniref50",
            contig_id="uniref50",
            gene_id=sid,
            start=0,
            end=len(seq),
            strand=1,
            sequence=seq,
        )
        for sid, seq in sequences
    ]

    print(f"\nEmbedding {len(protein_seqs):,} proteins on {device}...")

    chunk_size = 5000
    all_embeddings = []

    for i in range(0, len(protein_seqs), chunk_size):
        chunk = protein_seqs[i:i + chunk_size]
        chunk_embeddings = list(embed_system.embed_sequences(chunk))
        all_embeddings.extend(chunk_embeddings)
        n_done = min(i + chunk_size, len(protein_seqs))
        print(f"  Embedded {n_done:,}/{len(protein_seqs):,} proteins")

    print(f"Embedding dimension: {all_embeddings[0].embedding.shape[0]}")
    return all_embeddings


# ── PCA variant fitting ─────────────────────────────────────────────

def fit_all_pca_variants(matrix: np.ndarray, target_dim: int = 256,
                         output_dir: Path = None) -> dict:
    """Fit three PCA variants and save all to disk.

    Returns dict mapping variant name -> (pca_model, metadata_dict).
    """
    from sklearn.decomposition import PCA

    if output_dir is None:
        output_dir = ROOT / "elsa" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {}

    # ── 1. Standard PCA ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Fitting STANDARD PCA: {matrix.shape[1]}D → {target_dim}D")
    print(f"{'='*60}")
    pca_std = PCA(n_components=target_dim, whiten=False, random_state=42)
    pca_std.fit(matrix)
    ev_std = pca_std.explained_variance_ratio_.sum()
    print(f"  Explained variance: {ev_std:.4f} ({ev_std*100:.2f}%)")

    path_std = output_dir / "universal_pca.pkl"
    with open(path_std, "wb") as f:
        pickle.dump(pca_std, f)
    print(f"  Saved: {path_std}")

    variants["standard"] = {
        "model": pca_std,
        "path": path_std,
        "explained_variance": float(ev_std),
        "desc": f"PCA {target_dim}D",
    }

    # ── 2. PCA + Whitening ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Fitting WHITENED PCA: {matrix.shape[1]}D → {target_dim}D")
    print(f"{'='*60}")
    pca_wh = PCA(n_components=target_dim, whiten=True, random_state=42)
    pca_wh.fit(matrix)
    ev_wh = pca_wh.explained_variance_ratio_.sum()
    print(f"  Explained variance: {ev_wh:.4f} ({ev_wh*100:.2f}%)")

    path_wh = output_dir / "universal_pca_whitened.pkl"
    with open(path_wh, "wb") as f:
        pickle.dump(pca_wh, f)
    print(f"  Saved: {path_wh}")

    variants["whitened"] = {
        "model": pca_wh,
        "path": path_wh,
        "explained_variance": float(ev_wh),
        "desc": f"PCA {target_dim}D + whitening",
    }

    # ── 3. All-but-the-Top (whitened, drop PC1) ──────────────────────
    # Fit to target_dim+1, then we'll drop the first component at transform time
    print(f"\n{'='*60}")
    print(f"Fitting ALL-BUT-THE-TOP PCA: {matrix.shape[1]}D → {target_dim+1}D (drop PC1 → {target_dim}D)")
    print(f"{'='*60}")
    pca_abt = PCA(n_components=target_dim + 1, whiten=True, random_state=42)
    pca_abt.fit(matrix)
    ev_abt = pca_abt.explained_variance_ratio_[1:].sum()  # Excluding PC1
    pc1_var = pca_abt.explained_variance_ratio_[0]
    print(f"  PC1 variance (dropped): {pc1_var:.4f} ({pc1_var*100:.2f}%)")
    print(f"  Remaining explained variance: {ev_abt:.4f} ({ev_abt*100:.2f}%)")

    path_abt = output_dir / "universal_pca_abt.pkl"
    with open(path_abt, "wb") as f:
        pickle.dump(pca_abt, f)
    print(f"  Saved: {path_abt}")

    variants["abt"] = {
        "model": pca_abt,
        "path": path_abt,
        "explained_variance": float(ev_abt),
        "pc1_variance": float(pc1_var),
        "desc": f"PCA {target_dim+1}D whitened, drop PC1 → {target_dim}D",
    }

    # ── Save combined metadata ───────────────────────────────────────
    meta = {
        "input_dim": int(matrix.shape[1]),
        "target_dim": target_dim,
        "n_training_proteins": int(matrix.shape[0]),
        "plm_model": "esm2_t12",
        "source": "UniRef50 random subsample (50k)",
        "variants": {
            name: {
                "path": str(v["path"]),
                "explained_variance": v["explained_variance"],
                "desc": v["desc"],
            }
            for name, v in variants.items()
        },
    }
    meta_path = output_dir / "universal_pca_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved metadata: {meta_path}")

    return variants


# ── E. coli projection + evaluation ─────────────────────────────────

def project_ecoli(variant_name: str, pca_model, drop_first: bool = False):
    """Project E. coli 480D embeddings with a frozen PCA model, save parquet."""
    import pandas as pd

    ecoli_parquet = ROOT / "benchmarks/elsa_output/ecoli/elsa_index_nopca/ingest/genes.parquet"
    if not ecoli_parquet.exists():
        print(f"  SKIP: E. coli no-PCA parquet not found at {ecoli_parquet}")
        return None

    df = pd.read_parquet(ecoli_parquet)
    emb_cols = sorted([c for c in df.columns if c.startswith("emb_")])
    raw_matrix = df[emb_cols].values.astype(np.float32)
    print(f"  Loaded {len(df):,} E. coli genes ({raw_matrix.shape[1]}D)")

    # Apply frozen PCA transform
    projected = pca_model.transform(raw_matrix)

    # Drop first component for all-but-the-top
    if drop_first:
        projected = projected[:, 1:]
        print(f"  Dropped PC1 → {projected.shape[1]}D")

    # L2 normalize
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    projected = projected / (norms + 1e-8)

    output_dim = projected.shape[1]
    print(f"  Projected to {output_dim}D, L2-normalized")

    # Save parquet
    out_dir = ROOT / f"benchmarks/elsa_output/ecoli/elsa_index_universal_{variant_name}/ingest"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = [c for c in df.columns if not c.startswith("emb_")]
    new_df = df[meta_cols].copy()
    for i in range(output_dim):
        new_df[f"emb_{i:03d}"] = projected[:, i].astype(np.float16)

    parquet_path = out_dir / "genes.parquet"
    new_df.to_parquet(parquet_path, compression="snappy", index=False)
    print(f"  Saved: {parquet_path} ({parquet_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Write MANIFEST.json
    manifest = {
        "artifacts": {
            "genes": {"path": str(parquet_path), "stage": "projection"}
        }
    }
    manifest_path = out_dir.parent / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return parquet_path


def write_config(variant_name: str):
    """Write a config YAML for this variant."""
    import yaml

    config = {
        "data": {
            "work_dir": f"benchmarks/elsa_output/ecoli/elsa_index_universal_{variant_name}",
        },
        "plm": {
            "model": "esm2_t12",
            "project_to_D": 256,
            "l2_normalize": True,
        },
        "system": {"jobs": 1, "rng_seed": 17},
        "chain": {
            "index_backend": "faiss_ivfflat",
            "faiss_nprobe": 32,
            "hnsw_k": 50,
            "similarity_threshold": 0.85,
            "max_gap_genes": 2,
            "min_chain_size": 2,
            "gap_penalty_scale": 0.0,
            "jaccard_tau": 0.3,
            "min_genome_support": 2,
        },
    }

    config_path = ROOT / f"benchmarks/configs/ecoli_universal_{variant_name}.config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, indent=2)
    print(f"  Config: {config_path}")
    return config_path


def run_analyze(variant_name: str, config_path: Path):
    """Run elsa analyze for a variant."""
    output_dir = f"benchmarks/results/ecoli_universal_{variant_name}_chain"
    cmd = [
        sys.executable, "-m", "elsa.cli", "analyze",
        "-c", str(config_path),
        "-o", output_dir,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
    return output_dir


def run_operon_eval(variant_name: str, blocks_dir: str):
    """Run operon recall evaluation."""
    blocks_csv = ROOT / blocks_dir / "micro_chain" / "micro_chain_blocks.csv"
    if not blocks_csv.exists():
        # Try without micro_chain subdir
        blocks_csv = ROOT / blocks_dir / "micro_chain_blocks.csv"
    if not blocks_csv.exists():
        print(f"  SKIP eval: blocks not found at {blocks_csv}")
        return None

    output_md = ROOT / f"benchmarks/evaluation/operon_recall_universal_{variant_name}.md"
    cmd = [
        sys.executable, str(ROOT / "benchmarks/scripts/evaluate_operon_recall.py"),
        "--operon-gt", str(ROOT / "benchmarks/ground_truth/ecoli_operon_gt_v2.tsv"),
        "--elsa-blocks", str(blocks_csv),
        "--output", str(output_md),
        "--threshold", "0.5",
    ]
    print(f"  Evaluating operon recall...")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-300:]}")
        return None

    # Parse recall values from output
    recalls = {}
    for line in result.stdout.split("\n"):
        if "Strict recall:" in line:
            recalls["strict"] = line.split(":")[1].strip()
        elif "Independent recall:" in line:
            recalls["independent"] = line.split(":")[1].strip()
        elif "Any coverage:" in line:
            recalls["any"] = line.split(":")[1].strip()

    # Also count blocks
    import pandas as pd
    n_blocks = len(pd.read_csv(blocks_csv)) - 1  # subtract header... actually pandas handles this
    n_blocks = len(pd.read_csv(blocks_csv))
    recalls["n_blocks"] = n_blocks

    return recalls


# ── Main pipeline ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fit universal PCA variants from UniRef50")
    parser.add_argument("--n-proteins", type=int, default=50000)
    parser.add_argument("--target-dim", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fasta", type=Path, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    # ── Step 1: Get sequences ────────────────────────────────────────
    default_fasta = ROOT / "data" / "uniref50_sample.fasta"
    if args.fasta and args.fasta.exists():
        fasta_path = args.fasta
        print(f"Using provided FASTA: {fasta_path}")
    elif args.skip_download and default_fasta.exists():
        fasta_path = default_fasta
        print(f"Using cached FASTA: {fasta_path}")
    else:
        fasta_path = download_uniref50_sample(args.n_proteins, default_fasta)

    # ── Step 2: Parse and filter ─────────────────────────────────────
    sequences = parse_fasta(fasta_path)
    print(f"Parsed {len(sequences):,} sequences")

    original_count = len(sequences)
    sequences = [(sid, seq) for sid, seq in sequences if 10 <= len(seq) <= 2000]
    if len(sequences) < original_count:
        print(f"Filtered to {len(sequences):,} sequences (removed {original_count - len(sequences)})")

    # ── Step 3: Embed ────────────────────────────────────────────────
    embeddings = embed_proteins(sequences, device=args.device)

    # Save raw embedding matrix for reuse
    matrix = np.array([emb.embedding for emb in embeddings], dtype=np.float32)
    raw_path = ROOT / "elsa" / "data" / "uniref50_raw_embeddings.npy"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(raw_path, matrix)
    print(f"\nSaved raw embeddings: {raw_path} ({raw_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ── Step 4: Fit all PCA variants ─────────────────────────────────
    variants = fit_all_pca_variants(matrix, target_dim=args.target_dim)

    if args.skip_validation:
        print("\nSkipping validation. Done!")
        return

    # ── Step 5: Evaluate each variant on E. coli ─────────────────────
    print(f"\n{'#'*60}")
    print(f"# EVALUATING ALL VARIANTS ON E. COLI OPERON BENCHMARK")
    print(f"{'#'*60}")

    results = {}
    variant_configs = [
        ("standard", False),    # Standard PCA, don't drop PC1
        ("whitened", False),    # Whitened PCA, don't drop PC1
        ("abt", True),          # All-but-the-top, drop PC1
    ]

    for variant_name, drop_first in variant_configs:
        print(f"\n{'─'*60}")
        print(f"VARIANT: {variants[variant_name]['desc']}")
        print(f"{'─'*60}")

        # Project E. coli embeddings
        parquet_path = project_ecoli(
            variant_name,
            variants[variant_name]["model"],
            drop_first=drop_first,
        )
        if parquet_path is None:
            continue

        # Write config
        config_path = write_config(variant_name)

        # Run analysis
        output_dir = run_analyze(variant_name, config_path)

        # Evaluate operon recall
        recalls = run_operon_eval(variant_name, output_dir)
        if recalls:
            results[variant_name] = {
                "desc": variants[variant_name]["desc"],
                "explained_variance": variants[variant_name]["explained_variance"],
                **recalls,
            }

    # ── Step 6: Print comparison table ───────────────────────────────
    print(f"\n{'='*70}")
    print("COMPARISON: PCA VARIANT OPERON RECALL")
    print(f"{'='*70}")
    print(f"{'Variant':<35} {'Blocks':>8} {'Strict':>10} {'Indep':>10} {'Any':>10}")
    print(f"{'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    # Baselines
    print(f"{'Dataset-specific PCA 256D':<35} {'19,279':>8} {'47.2%':>10} {'82.6%':>10} {'98.4%':>10}")
    print(f"{'No-PCA 480D':<35} {'25,702':>8} {'12.4%':>10} {'76.1%':>10} {'98.0%':>10}")
    print(f"{'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    for name, r in results.items():
        print(f"{r['desc']:<35} {r['n_blocks']:>8,} {r.get('strict','?'):>10} {r.get('independent','?'):>10} {r.get('any','?'):>10}")

    # Save comparison report
    report_path = ROOT / "benchmarks/evaluation/pca_variant_comparison.md"
    with open(report_path, "w") as f:
        f.write("# PCA Variant Comparison: Operon Recall\n\n")
        f.write("All variants use frozen PCA weights fit on 50k UniRef50 proteins.\n")
        f.write("Evaluated on 20-genome E. coli dataset (10,182 operon instances).\n\n")
        f.write("| Variant | Blocks | Strict | Independent | Any Coverage |\n")
        f.write("|---------|--------|--------|-------------|-------------|\n")
        f.write(f"| Dataset-specific PCA 256D (baseline) | 19,279 | 47.2% | 82.6% | 98.4% |\n")
        f.write(f"| No-PCA 480D | 25,702 | 12.4% | 76.1% | 98.0% |\n")
        for name, r in results.items():
            f.write(f"| {r['desc']} | {r['n_blocks']:,} | {r.get('strict','?')} | {r.get('independent','?')} | {r.get('any','?')} |\n")
        f.write(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\nReport saved: {report_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
