#!/bin/bash
set -e
cd "/media/jacob/Crucial X9/Sandbox/elsa_test/ELSA"

UNIREF_GZ="data/uniref50.fasta.gz"
SAMPLE_FASTA="data/uniref50_sample.fasta"
N_SAMPLE=50000

echo "=== Step 1: Download UniRef50 FASTA (~13GB) ==="
mkdir -p data
if [ ! -f "$UNIREF_GZ" ]; then
    wget -c -O "$UNIREF_GZ" \
        "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
    echo "Download complete: $(ls -lh $UNIREF_GZ)"
else
    echo "Already downloaded: $(ls -lh $UNIREF_GZ)"
fi

echo ""
echo "=== Step 2: Random subsample $N_SAMPLE sequences ==="
if [ ! -f "$SAMPLE_FASTA" ]; then
    # Count total sequences (streaming through gz)
    echo "Counting sequences in UniRef50..."
    TOTAL=$(zcat "$UNIREF_GZ" | grep -c '^>')
    echo "Total sequences: $TOTAL"
    
    # Calculate sampling rate (oversample 2x then truncate)
    # Use reservoir sampling via awk for true random sample
    echo "Extracting random $N_SAMPLE sequences..."
    python3 -c "
import gzip
import random
import sys

random.seed(42)
n_sample = $N_SAMPLE
gz_path = '$UNIREF_GZ'
out_path = '$SAMPLE_FASTA'

# First pass: count sequences
print('Counting sequences...')
n_total = 0
with gzip.open(gz_path, 'rt') as f:
    for line in f:
        if line.startswith('>'):
            n_total += 1
print(f'Total: {n_total:,} sequences')

# Generate random indices to keep
keep = set(random.sample(range(n_total), min(n_sample, n_total)))
print(f'Sampling {len(keep):,} sequences...')

# Second pass: extract selected sequences
with gzip.open(gz_path, 'rt') as f, open(out_path, 'w') as out:
    seq_idx = -1
    writing = False
    for line in f:
        if line.startswith('>'):
            seq_idx += 1
            writing = seq_idx in keep
            if writing and seq_idx % 10000 == 0:
                kept_so_far = sum(1 for k in keep if k <= seq_idx)
                print(f'  Progress: seq {seq_idx:,}, kept {kept_so_far:,}/{len(keep):,}', flush=True)
        if writing:
            out.write(line)

print(f'Saved {n_sample:,} sequences to {out_path}')
"
    echo "Sample FASTA: $(ls -lh $SAMPLE_FASTA)"
else
    echo "Already sampled: $(ls -lh $SAMPLE_FASTA)"
fi

echo ""
echo "=== Step 3: Embed + Fit PCA + Validate ==="
python3 scripts/fit_universal_pca.py \
    --fasta "$SAMPLE_FASTA" \
    --n-proteins $N_SAMPLE \
    --device cuda

echo ""
echo "=== ALL DONE ==="
date
