#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu_h200
#SBATCH --qos=gpu_h200_4
#SBATCH --gres=gpu:1
#SBATCH -c 28
#SBATCH -J annot_val_gpu
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu/validation/val_gpu_%j.log

# Validation gate, GPU arm: Pfam and KOfam with --cut_ga on the 4,000-protein testset3 sample through Aksha 0.2.3
# GPU mode (--gpu-manifest, manifests made with the installed runtime's create_pressed_manifest). One H200.
set -euo pipefail
for v in $(compgen -e | grep -E '^(ASTRA_|_ASTRA_|PLAN7_)' || true); do unset "$v"; done
export PYTHONDONTWRITEBYTECODE=1
AKSHA=/home/jwestrob/.local/bin/aksha
A=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu
V=$A/validation
echo "host $(hostname)  start $(date -u +%FT%TZ)  CPUs $SLURM_CPUS_ON_NODE  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version --format=csv
$AKSHA --version
for DB in PFAM KOFAM; do
    out=$V/gpu_${DB,,}
    test ! -e "$out"
    echo "== $DB GPU start $(date -u +%FT%TZ)"
    /usr/bin/time -v "$AKSHA" search --installed_hmms "$DB" --cut_ga --prot_in "$V/sample4000.faa" --outdir "$out" \
        --threads "$SLURM_CPUS_ON_NODE" --gpu-manifest "$DB=$A/manifests/$DB.manifest.json"
    echo "== $DB GPU end $(date -u +%FT%TZ)"
done
echo "VAL_GPU_DONE $(date -u +%FT%TZ)"
