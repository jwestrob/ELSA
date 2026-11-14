#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J rd_seqfam_dp
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/seqfam_dpann_%j.log

# usage: sbatch --export=ALL,ARM=<name>,COV=<0.8|0.5>,MINID=<0|0.2> run_seq_families_dpann.sbatch
# Permissive sequence families on the DPANN package, same parameters as run_seq_families.sbatch
# (sensitivity 7.5, E <= 1e-3, bidirectional coverage, optional identity floor).
# Tokens go to engine/dpann_tokens/ (the package directory and its manifest stay as built).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
P=$RD/ws_dpann/package
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
W=$RD/engine/seqfam_dpann/${ARM:?}
rm -rf "$W"; mkdir -p "$W" "$RD/engine/dpann_tokens"
"$MM" createdb "$RD/ws_dpann/work/proteins_gid.faa" "$W/seqdb" -v 2
PARAMS="-s 7.5 -e 1e-3 --min-seq-id ${MINID:?} -c ${COV:?} --cov-mode 0 --cluster-mode 0"
t0=$(date +%s)
"$MM" cluster "$W/seqdb" "$W/clu" "$W/tmp" $PARAMS --threads "$T" -v 2
t1=$(date +%s)
"$MM" createtsv "$W/seqdb" "$W/seqdb" "$W/clu" "$W/clu.tsv" --threads "$T" -v 2
J="{\"tool\": \"mmseqs cluster\", \"mmseqs_version\": \"$($MM version)\", \"params\": \"$PARAMS\", \"input_fasta\": \"$RD/ws_dpann/work/proteins_gid.faa\", \"threads\": $T}"
"$PY" -u "$P/code/mmseqs_tokens.py" --clusters "$W/clu.tsv" --universe "$P/universe/genes.parquet" \
    --out "$RD/engine/dpann_tokens/$ARM" --threads "$T" --params "$J" --mmseqs-seconds $((t1 - t0))
rm -rf "$W/tmp"
