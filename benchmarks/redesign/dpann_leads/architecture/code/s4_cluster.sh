#!/bin/bash
# Architecture screen, step 4: group screen candidates by full-length homology (MMseqs2, bidirectional coverage >= 0.7,
# E <= 1e-3, no identity floor, -s 7.5; 4 threads). Bidirectional 70% coverage across genes that are >= 1.4x the carrier
# length forces the extra region to align inside a group.
set -euo pipefail
cd "$(dirname "$0")/.."
SP=${SP:-/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad}
mmseqs easy-cluster data/cand.faa $SP/mm/cand $SP/mm/tmp --min-seq-id 0 -c 0.7 --cov-mode 0 -e 1e-3 -s 7.5 --threads 4 -v 1
cp $SP/mm/cand_cluster.tsv data/cand_cluster.tsv
# all-vs-all within the candidate set for pairwise identity / coverage
mmseqs easy-search data/cand.faa data/cand.faa $SP/mm/cand_ava.m8 $SP/mm/tmp -e 1e-3 -s 7.5 --threads 4 -v 1 \
   --format-output query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits
gzip -c $SP/mm/cand_ava.m8 > data/cand_ava.m8.gz
wc -l data/cand_cluster.tsv
