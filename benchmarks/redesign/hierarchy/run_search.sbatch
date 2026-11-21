#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J hier_search
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/logs/hier_search_%j.log
# Task 4 nestedness + nested levels, then task 3 tuning (odd genomes) and finals (even genomes, 200-replicate null).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
PY=$RD/ws_tokens/env/bin/python
C=$RD/hierarchy/code
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$RD/tmp"
export NUMBA_NUM_THREADS=$T NUMBA_CACHE_DIR=$RD/tmp/numba_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR"
echo "$(date -Is) host=$(hostname) cpus=$T job=$SLURM_JOB_ID stages=${STAGES:-nested,tune,final,ntune,nfinal}"
cd $C
IFS=',' read -ra ST <<< "${STAGES:-nested,tune,final,ntune,nfinal}"
for s in "${ST[@]}"; do
  echo "$(date -Is) stage $s"
  case $s in
    nested) "$PY" -u $C/nested.py L8 L13 S3 ;;
    tune)   "$PY" -u $C/hier_search.py tune --threads $T ;;
    final)  "$PY" -u $C/hier_search.py final --threads $T ;;
    ntune)  "$PY" -u $C/hier_search.py tune --threads $T --nested --only N_emb_L8,N_emb_L13 ;;
    nfinal) "$PY" -u $C/hier_search.py final --threads $T --nested --only N_emb_L8,N_emb_L13 ;;
    rate)   "$PY" -u $C/hier_search.py rate --threads $T --only H_emb_L13,H_emb_L8,H_seq_S3,H_both_L13S3,H_both_L8S3,chk_e3c50s75 ;;
    rfinal) "$PY" -u $C/hier_search.py rfinal --threads $T --only H_emb_L13,H_emb_L8,H_seq_S3,H_both_L13S3,H_both_L8S3,chk_e3c50s75 ;;
    unl)    "$PY" -u $C/hier_unlabelled.py ;;
  esac
done
echo "$(date -Is) end"
