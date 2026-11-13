#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -J sb_esmpp
#SBATCH --exclude=node-224-2t-8gpu-1
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_srvp_bac/logs/embed_gpu_%j.log
# ESM++ small on the SR-VP bacteria proteins: one embed_queue worker on the gpu partition, A5000 node only (the H200
# node, also in partition gpu, is excluded), the DPANN
# recipe (embed_queue run dpann__esmpp_small__20261111: bf16 autocast, float32 residue-only mean, whole protein
# in one pass, worker self-check and per-shard float32 audit on the GPU). This wrapper never submits another
# job: no successor at the wall limit and no automatic finalize (finalize runs from finalize_zca.sbatch).
set -uo pipefail
RUN=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_srvp_bac/embed/srvp_bac__esmpp_small__20261113
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/code/env.sh
echo "[$(date -Is)] job ${SLURM_JOB_ID} on $(hostname), ${SLURM_CPUS_ON_NODE} CPUs; GPU: $(nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,temperature.gpu --format=csv,noheader)"
[ -e $RUN/STOP ] && { echo "STOP file present"; exit 0; }
$PYBIN -u $EQ/code/embed_queue.py work --run $RUN --order asc --threads ${SLURM_CPUS_ON_NODE}
rc=$?
echo "[$(date -Is)] work exit code $rc"
$PYBIN $EQ/code/embed_queue.py status --run $RUN
exit $rc
