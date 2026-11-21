#!/bin/bash
# One embed_queue GPU worker. Args: <run dir> <asc|desc>. Launched by worker_h200.sbatch / worker_a5000.sbatch.
# It claims shards until the queue is drained or the wall limit is near, then
#   - submits its own successor when it stopped for the wall limit (exit code 75 of `work`) and EQ_RESUBMIT=1,
#   - submits finalize.sbatch (CPU, standard_expand) once when every shard is done (atomic mkdir lock).
# On a GPU listed in the Talea bad_gpus.txt (thermal slowdown): EQ_ON_BAD_GPU=requeue does no work and requeues the
# job in 20 min (needs EQ_RESUBMIT=1); EQ_ON_BAD_GPU=work embeds on it anyway (slower, numerically the same).
# The H200 wrapper sets EQ_RESUBMIT=0 EQ_ON_BAD_GPU=work, so a worker never adds a job under gpu_h200_4: the Talea
# lanes submit their successors under that QOS and a refused submission ends a lane chain (10 submitted jobs per user).
set -uo pipefail
RUN=${1:?run directory}; ORDER=${2:-asc}
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/code/env.sh
SELF=${EQ_SBATCH:?set by the sbatch wrapper}; RESUB=${EQ_RESUBMIT:-1}; ONBAD=${EQ_ON_BAD_GPU:-requeue}
JID=${SLURM_JOB_ID:-nojob}
log(){ echo "[$(date -Is)] job $JID: $*"; }
resubmit(){ local j; j=$(clean_sbatch sbatch --parsable "$@" "$SELF" "$RUN" "$ORDER") && echo "$j" >> $RUN/worker_jobs.txt; echo "${j:-FAILED}"; }
log "start on $(hostname), ${SLURM_CPUS_ON_NODE:-?} CPUs, partition ${SLURM_JOB_PARTITION:-?}; GPU: $(nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader)"
[ -e $RUN/STOP ] && { log "STOP file present; exiting"; exit 0; }
mypci=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [ -n "$mypci" ] && awk -v n="${SLURMD_NODENAME:-$(hostname -s)}" -v p="$mypci" '$1==n && $2==p {f=1} END{exit !f}' $BAD_GPUS 2>/dev/null; then
  if [ "$ONBAD" = work ]; then log "on excluded GPU $mypci (thermal slowdown); embedding on it anyway"
  elif [ "$RESUB" = 1 ]; then log "on excluded GPU $mypci; no work done; requeued as $(resubmit --begin=now+1200)"; exit 0
  else log "on excluded GPU $mypci; no work done; EQ_RESUBMIT=0, so no requeue"; exit 0; fi
fi
$PYBIN -u $EQ/code/embed_queue.py work --run $RUN --order $ORDER --threads ${SLURM_CPUS_ON_NODE:-8}
rc=$?
log "work exit code $rc"
if [ $rc -eq 75 ]; then
  if [ "$RESUB" = 1 ] && [ ! -e $RUN/STOP ]; then log "wall limit near; successor $(resubmit)"; else log "wall limit near; no successor (EQ_RESUBMIT=$RESUB)"; fi
  rc=0
fi
if $PYBIN $EQ/code/embed_queue.py status --run $RUN --check-done; then
  if mkdir $RUN/finalize.lock 2>/dev/null; then
    f=$(clean_sbatch sbatch --parsable $EQ/code/finalize.sbatch $RUN) && echo $f > $RUN/finalize.jobid
    log "every shard is done; finalize job ${f:-FAILED}"
  fi
fi
exit $rc
