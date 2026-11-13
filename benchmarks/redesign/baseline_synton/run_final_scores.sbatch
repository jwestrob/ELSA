#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J bs_final
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/baseline_synton/logs/bs_final_%j.log
# baseline_synton: joint scoring of faithful + matched synton loci on each dataset, with paired differences between them
# and against the word method (Spacedust: transfer_spacedust per-region frames, read-only; DPANN: vocab V-coh0.1).
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
BS=$RD/baseline_synton; C=$BS/code; TS=$RD/transfer_spacedust/results
T="${SLURM_CPUS_ON_NODE:?}"
EPY=/home/jwestrob/miniconda3/envs/elsa/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID"
if [[ "${SPACEDUST:-1}" == 1 ]]; then
  CMP=()
  for r in P2__V-coh0.1 P2__CC1 P2__CC2 P1__V-coh0.1; do
    [[ -s "$TS/pr_$r.parquet" ]] && CMP+=(--compare "word $r=$TS/pr_$r.parquet")
  done
  "$EPY" -u "$C/score_spacedust.py" score --loci "$BS/runs/sd_faithful/post/loci.parquet" --label "synton faithful" \
      --loci "$BS/runs/sd_matched/post/loci.parquet" --label "synton matched (rescue3)" "${CMP[@]}" --tag final --threads "$T"
fi
if [[ "${DPANN:-1}" == 1 ]]; then
  ARGS=(--loci "$BS/runs/dp_faithful/post/loci.parquet" --label "synton faithful")
  [[ -s "$BS/runs/dp_matched/post/loci.parquet" ]] && ARGS+=(--loci "$BS/runs/dp_matched/post/loci.parquet" --label "synton matched (rescue3)")
  NUMBA_NUM_THREADS="$T" NUMBA_CACHE_DIR="$BS/tmp/numba_$SLURM_JOB_ID" \
    "$EPY" -u "$C/score_dpann.py" score "${ARGS[@]}" --tag final --threads "$T"
fi
stamp "DONE"
