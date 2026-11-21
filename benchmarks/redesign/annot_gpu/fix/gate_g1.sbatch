#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu_h200
#SBATCH --qos=gpu_h200_4
#SBATCH --gres=gpu:1
#SBATCH -c 28
#SBATCH -J gpufix_gate_g1
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/aksha_check/gpufix/gates/gate_g1_%j.log

# GPU job 1 of 3: native attestation tests; readiness in the untrusted state (gate f, first half);
# gate (e): an untrusted environment (self-attestation disabled, no record) completes through the CPU path,
# on the default SSV-only route and on the native per-stage fallback route; rows against production CPU.
set -uo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/aksha_check/gpufix/gate_common.sh
echo "host $(hostname)  start $(date -u +%FT%TZ)  CPUs $SLURM_CPUS_ON_NODE  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv
aksha_dev --version
test ! -e "$AKSHA_GPU_ATTESTATION_DIR" || { echo "attestation cache already exists"; exit 1; }

echo "== native attestation tests"
(cd "$W/aksha" && env PYTHONPATH="$NEW/lib/python3.12/site-packages" "$W/testvenv/bin/python" -m pytest \
    -p no:cacheprovider -v tests/test_gpu_attestation_native.py 2>&1 | tail -12)

MANIFESTS=(--gpu-manifest "PFAM=$PFAM_MANIFEST" --gpu-manifest "KOFAM=$KOFAM_MANIFEST")
echo "== readiness, no record (expect PENDING, exit 2)"
aksha_dev gpu-check "${MANIFESTS[@]}" > "$G/f1_pending.txt" 2>&1; echo "f1_pending exit=$?"; cat "$G/f1_pending.txt"
echo "== readiness, AKSHA_GPU_SELF_ATTEST=never (expect FALLBACK / SSV only, exit 1)"
export AKSHA_GPU_SELF_ATTEST=never
aksha_dev gpu-check "${MANIFESTS[@]}" > "$G/f2_never.txt" 2>&1; echo "f2_never exit=$?"; cat "$G/f2_never.txt"

echo "== gate (e): untrusted, default route (GPU SSV filter, every later stage on CPU)"
run_search e_pfam PFAM "$G/e_pfam" --gpu-manifest "PFAM=$PFAM_MANIFEST"
run_search e_kofam KOFAM "$G/e_kofam" --gpu-manifest "KOFAM=$KOFAM_MANIFEST"
echo "== gate (e'): untrusted, native per-stage fallback on the session route"
export ASTRA_GPU_UNTRUSTED_NATIVE_FALLBACK=1
run_search e2_pfam PFAM "$G/e2_pfam" --gpu-manifest "PFAM=$PFAM_MANIFEST"
run_search e2_kofam KOFAM "$G/e2_kofam" --gpu-manifest "KOFAM=$KOFAM_MANIFEST"
unset ASTRA_GPU_UNTRUSTED_NATIVE_FALLBACK AKSHA_GPU_SELF_ATTEST
echo "attestation cache after the untrusted runs: $(ls "$AKSHA_GPU_ATTESTATION_DIR" 2>&1)"

for label in e_pfam e2_pfam; do compare_to "$label" "$REF_PFAM" "$G/$label/PFAM_hits_df.tsv"; done
for label in e_kofam e2_kofam; do compare_to "$label" "$REF_KOFAM" "$G/$label/KOFAM_hits_df.tsv"; done
echo "GATE_G1_DONE $(date -u +%FT%TZ)"
