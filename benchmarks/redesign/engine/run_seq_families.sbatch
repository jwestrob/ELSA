#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J rd_seqfam
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/seqfam_%j.log

# usage: sbatch --export=ALL,ARM=<name>,COV=<0.8|0.5>,MINID=<0|0.2> run_seq_families.sbatch
# Permissive sequence families: sensitivity 7.5, E <= 1e-3, bidirectional coverage, optional identity floor.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
DB=$RD/ws_tokens/work/mmseqs/seqdb
W=$RD/engine/seqfam/${ARM:?}
rm -rf "$W"; mkdir -p "$W"
PARAMS="-s 7.5 -e 1e-3 --min-seq-id ${MINID:?} -c ${COV:?} --cov-mode 0 --cluster-mode 0"
t0=$(date +%s)
"$MM" cluster "$DB" "$W/clu" "$W/tmp" $PARAMS --threads "$T" -v 2
t1=$(date +%s)
"$MM" createtsv "$DB" "$DB" "$W/clu" "$W/clu.tsv" --threads "$T" -v 2
"$PY" -u "$RD/engine/code/seq_family_tokens.py" --tsv "$W/clu.tsv" --arm "$ARM" --params "$PARAMS" \
    --seconds $((t1 - t0)) --threads "$T"
rm -rf "$W/tmp"
