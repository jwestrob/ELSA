#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J eq_final
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/logs/%x_%j.log
# Assemble a drained run: row-aligned matrix, QC, independent float32 single-protein audit, manifest.
# Then cancel this run's worker jobs that are still pending. Usage: sbatch finalize.sbatch <run dir>
set -uo pipefail
RUN=${1:?run directory}
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/code/env.sh
echo "[$(date -Is)] host $(hostname) cpus $SLURM_CPUS_ON_NODE run $RUN"
$PYBIN -u $EQ/code/embed_queue.py finalize --run $RUN --threads $SLURM_CPUS_ON_NODE
rc=$?
if [ $rc -eq 0 ] || [ $rc -eq 4 ]; then
  for j in $(sort -u $RUN/worker_jobs.txt 2>/dev/null); do
    [ "$(squeue -h -j $j -o %T 2>/dev/null)" = PENDING ] && { scancel $j; echo "cancelled pending worker $j"; }
  done
fi
echo "[$(date -Is)] finalize exit code $rc"
exit $rc
