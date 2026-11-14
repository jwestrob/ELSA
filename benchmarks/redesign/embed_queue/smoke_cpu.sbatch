#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J eq_smoke
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/logs/%x_%j.log
# CPU smoke test of the whole path before any GPU time is spent, plus the sharding of the full run.
#   1. prep of the full run directory (sharding only; no embedding)
#   2. 320 sampled proteins + 20 duplicate rows through prep -> work -> finalize with ESM++ small, float32, 24 threads
#      gates: self-check (single vs padded batch) and finalize audit, min cosine >= 0.999
#   3. the same sample through native transformers ESM-C 300M and a row-wise comparison (reported, no gate)
# Usage: sbatch smoke_cpu.sbatch <fasta> <full run dir>
set -uo pipefail
FASTA=${1:?fasta}; FULL=${2:?full run directory}
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/code/env.sh
T=$SLURM_CPUS_ON_NODE; E=$EQ/code/embed_queue.py; TAG=$SLURM_JOB_ID
echo "[$(date -Is)] host $(hostname) cpus $T"
if [ ! -e $FULL/run.json ]; then
  $PYBIN -u $E prep --fasta $FASTA --run $FULL --backend esmpp --model-dir $ESMPP_SMALL \
      --model-id Synthyra/ESMplusplus_small --model-revision $ESMPP_SMALL_REV --dtype bf16 || exit 10
fi
S1=$EQ/runs/smoke_esmpp_$TAG
$PYBIN -u $E prep --fasta $FASTA --run $S1 --sample 320 --sample-duplicates 20 --shard-tokens 20000 --backend esmpp \
    --model-dir $ESMPP_SMALL --model-id Synthyra/ESMplusplus_small --model-revision $ESMPP_SMALL_REV --dtype fp32 || exit 11
$PYBIN -u $E work --run $S1 --device cpu --threads $T || exit 12
$PYBIN -u $E finalize --run $S1 --threads $T --audit 60 || exit 13
echo "[$(date -Is)] SMOKE_ESMPP_OK"
S2=$EQ/runs/smoke_esmc_native_$TAG
NATIVE=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/model_scaling_20260913/models/esmc300m
( $ELSA_PY -u $E prep --fasta $FASTA --run $S2 --sample 320 --sample-duplicates 20 --shard-tokens 20000 --backend esmc_native \
      --model-dir $NATIVE --model-id esmc300m --dtype fp32 \
  && $ELSA_PY -u $E work --run $S2 --device cpu --threads $T \
  && $ELSA_PY -u $E finalize --run $S2 --threads $T --audit 60 \
  && $ELSA_PY -u $E compare --a $S1 --b $S2 && echo "[$(date -Is)] NATIVE_COMPARE_OK" ) || echo "[$(date -Is)] NATIVE_COMPARE_UNAVAILABLE"
echo "[$(date -Is)] done"
