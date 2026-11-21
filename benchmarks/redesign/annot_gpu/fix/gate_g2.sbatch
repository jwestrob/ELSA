#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu_h200
#SBATCH --qos=gpu_h200_4
#SBATCH --gres=gpu:1
#SBATCH -c 28
#SBATCH -J gpufix_gate_g2
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/aksha_check/gpufix/gates/gate_g2_%j.log

# GPU job 2 of 3: readiness --attest (one-time self-attestation, record cached) and readiness from the record
# (gate f, second half); gate (c): attested GPU searches (Pfam, KOfam, --cut_ga) against production CPU rows.
set -uo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/aksha_check/gpufix/gate_common.sh
echo "host $(hostname)  start $(date -u +%FT%TZ)  CPUs $SLURM_CPUS_ON_NODE  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv
aksha_dev --version
MANIFESTS=(--gpu-manifest "PFAM=$PFAM_MANIFEST" --gpu-manifest "KOFAM=$KOFAM_MANIFEST")
echo "== readiness --attest (expect READY, exit 0)"
aksha_dev gpu-check "${MANIFESTS[@]}" --attest --threads "$SLURM_CPUS_ON_NODE" > "$G/f3_attest.txt" 2>&1; echo "f3_attest exit=$?"; cat "$G/f3_attest.txt"
echo "== readiness from the cached record (expect READY, exit 0)"
aksha_dev gpu-check "${MANIFESTS[@]}" --json > "$G/f4_ready.json" 2> "$G/f4_ready.stderr"; echo "f4_ready exit=$?"
"$NEW/bin/python" -I -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['overall'], d['bias_math']['state'], d['bias_math']['source']); [print(v) for v in d['verdicts']]" "$G/f4_ready.json"
ls -la "$AKSHA_GPU_ATTESTATION_DIR"
"$NEW/bin/python" -I -c "import json,glob,sys; r=json.load(open(glob.glob(sys.argv[1]+'/*.json')[0])); a=r['audit']; print({k: a[k] for k in ('attested','inputs_checked','float_mismatches','double_mismatches','maximum_double_ulp','seconds','host_threads','libm_build_id','target')}); print(r['key']['glibc'], r['key']['cuda'])" "$AKSHA_GPU_ATTESTATION_DIR"

echo "== gate (c): attested GPU searches"
run_search c_pfam PFAM "$G/c_pfam" --gpu-manifest "PFAM=$PFAM_MANIFEST"
run_search c_kofam KOFAM "$G/c_kofam" --gpu-manifest "KOFAM=$KOFAM_MANIFEST"
compare_to c_pfam "$REF_PFAM" "$G/c_pfam/PFAM_hits_df.tsv"
compare_to c_kofam "$REF_KOFAM" "$G/c_kofam/KOFAM_hits_df.tsv"
echo "GATE_G2_DONE $(date -u +%FT%TZ)"
