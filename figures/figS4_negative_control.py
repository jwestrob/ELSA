#!/usr/bin/env python3
"""Figure S4: Negative Control -- Cross-phylum embedding similarity.

Shows the distribution of cross-genome cosine similarity between
E. coli and B. subtilis protein embeddings, demonstrating that
unrelated genomes produce near-zero similarity scores.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

setup_style()

# ── Data ────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent

# Try to load precomputed negative control data
neg_data_paths = [
    BASE / 'benchmarks/evaluation/negative_control_similarity.csv',
    BASE / 'benchmarks/evaluation/cross_phylum_similarity.csv',
    BASE / 'benchmarks/evaluation/ecoli_bsubtilis_similarity.csv',
]

precomputed = None
for p in neg_data_paths:
    if p.exists():
        precomputed = pd.read_csv(p)
        print(f'Loaded precomputed data: {p}')
        break

# Check if we can compute from embeddings
ecoli_parquet = BASE / 'benchmarks/elsa_output/ecoli/elsa_index/ingest/genes.parquet'
bsub_parquet = BASE / 'benchmarks/elsa_output/bacillus/elsa_index/ingest/genes.parquet'

can_compute = ecoli_parquet.exists() and bsub_parquet.exists()

if precomputed is not None:
    # Use precomputed data
    similarities = precomputed['similarity'].values if 'similarity' in precomputed.columns else None
elif can_compute:
    print('Computing cross-phylum similarities from embeddings...')
    # Load a sample of embeddings from each
    ecoli_df = pd.read_parquet(ecoli_parquet)
    bsub_df = pd.read_parquet(bsub_parquet)

    # Extract embedding columns
    emb_cols = [c for c in ecoli_df.columns if c.startswith('emb_') or c.startswith('d')]
    if not emb_cols:
        # Try numeric columns
        emb_cols = [c for c in ecoli_df.columns if isinstance(c, (int, float)) or
                    (isinstance(c, str) and c.isdigit())]
    if not emb_cols:
        # Columns might just be integers
        emb_cols = [c for c in ecoli_df.columns if str(c).isdigit()]

    if emb_cols:
        ecoli_embs = ecoli_df[emb_cols].values.astype(np.float32)
        bsub_embs = bsub_df[emb_cols].values.astype(np.float32)

        # Sample to keep computation tractable
        rng = np.random.RandomState(42)
        n_sample = min(2000, len(ecoli_embs), len(bsub_embs))
        ecoli_sample = ecoli_embs[rng.choice(len(ecoli_embs), n_sample, replace=False)]
        bsub_sample = bsub_embs[rng.choice(len(bsub_embs), n_sample, replace=False)]

        # L2 normalize
        ecoli_norms = np.linalg.norm(ecoli_sample, axis=1, keepdims=True)
        bsub_norms = np.linalg.norm(bsub_sample, axis=1, keepdims=True)
        ecoli_sample = ecoli_sample / np.maximum(ecoli_norms, 1e-8)
        bsub_sample = bsub_sample / np.maximum(bsub_norms, 1e-8)

        # Compute all pairwise cosine similarities
        cos_sim = ecoli_sample @ bsub_sample.T  # (n_sample, n_sample)
        similarities = cos_sim.flatten()

        # Save for future use
        out_csv = BASE / 'benchmarks/evaluation/cross_phylum_similarity.csv'
        pd.DataFrame({'similarity': similarities}).to_csv(out_csv, index=False)
        print(f'Saved {len(similarities):,} similarity values to {out_csv}')
    else:
        similarities = None
        print('Could not identify embedding columns. Creating placeholder.')
else:
    similarities = None
    print('No precomputed data and cannot compute. Creating placeholder.')


# ── Figure ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL, SINGLE_COL * 0.85))

if similarities is not None and len(similarities) > 0:
    # Actual histogram
    bins = np.linspace(-0.3, 1.0, 120)
    ax.hist(similarities, bins=bins, color=ELSA_COLOR, edgecolor='none',
            alpha=0.8, density=True)

    mean_sim = np.mean(similarities)
    frac_gt08 = np.mean(similarities > 0.8) * 100

    # Mark the threshold
    ax.axvline(x=0.8, color='#EE6677', linestyle='--', linewidth=0.8,
               label=r'$\tau$ = 0.8 threshold')

    # Annotation box
    stats_text = (f'Mean similarity = {mean_sim:.3f}\n'
                  f'Pairs > 0.8: {frac_gt08:.2f}%\n'
                  f'n = {len(similarities):,} pairs')
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=6.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='lightgray', alpha=0.9))

    ax.set_xlabel('Cosine similarity')
    ax.set_ylabel('Density')
    ax.set_title('E. coli vs. B. subtilis\n(cross-phylum negative control)', fontsize=8)
    ax.legend(fontsize=6, loc='upper left')

else:
    # Placeholder
    ax.text(0.5, 0.5,
            'PLACEHOLDER\n\n'
            'Cross-phylum embedding similarity\n'
            '(E. coli vs. B. subtilis)\n\n'
            'Expected results:\n'
            r'  Mean similarity $\approx$ 0.001' + '\n'
            '  Only 0.21% of pairs > 0.8\n\n'
            'Requires: ecoli + bacillus\n'
            'genes.parquet with embeddings',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=7, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                      edgecolor='gray'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Cosine similarity')
    ax.set_ylabel('Density')
    ax.set_title('E. coli vs. B. subtilis\n(cross-phylum negative control)', fontsize=8)

add_panel_label(ax, 'a', x=-0.18, y=1.08)

plt.tight_layout()
save_figure(fig, 'figS4_negative_control')
plt.close()
print('Done: Figure S4')
