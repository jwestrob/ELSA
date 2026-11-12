#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J elsa_seqinvis
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/seq_invisible_20261111/logs/seqinvis_%j.log

set -euo pipefail
RUN=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/seq_invisible_20261111
T="${SLURM_CPUS_ON_NODE:?}"
trap 'status=$?; printf "%s exit=%s\n" "$(date -u +%FT%TZ)" "$status" > "$RUN/logs/job_${SLURM_JOB_ID}.exit"' EXIT

# End-to-end smoke on 300 blocks of one model; a failure here ends the job in about a minute.
bash "$RUN/code/pipeline.sh" "$RUN/smoke" "$T" 300 esm2_8m
date -u +%FT%TZ > "$RUN/smoke/SMOKE_PASSED"

# Full run: seven models, all chained pairs.
bash "$RUN/code/pipeline.sh" "$RUN" "$T" 0
date -u +%FT%TZ > "$RUN/results/COMPLETED"
