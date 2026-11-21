#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu_h200
#SBATCH --qos=gpu_h200_4
#SBATCH --gres=gpu:1
#SBATCH -c 28
#SBATCH -J gpufix_gate_g3
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/aksha_check/gpufix/gates/gate_g3_%j.log

# GPU job 3 of 3, gate (d): the mandatory H200 first-1000 semantic audit (H200_ATTESTATION.md) on the build under
# test: all 27,481 Pfam models x PLM2_5.first1000, CPU versus GPU query identity, target/domain table bytes and
# Plan7 accounting bytes; production-namespace port of hmmer_gpu tests/audit_h200_pfam_first1000.py.
set -uo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/aksha_check/gpufix/gate_common.sh
echo "host $(hostname)  start $(date -u +%FT%TZ)  CPUs $SLURM_CPUS_ON_NODE  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
FASTA=$J/hmmer_gpu/results/datasets/PLM2_5.first1000.faa
rc=0
/usr/bin/time -f "d_first1000 wall_s=%e maxrss_kb=%M" "$NEW/bin/python" -I "$W/audit_h200_pfam_first1000_aksha.py" \
    --fasta "$FASTA" --pressed-base /groups/banfield/users/jwestrob/.config/Astra/PFAM/PFAM \
    --manifest "$PFAM_MANIFEST" --output "$G/d_first1000.json" --source-root "$W/aksha" \
    > "$G/d_first1000.stdout" 2> "$G/d_first1000.stderr" || rc=$?
echo "d_first1000 exit=$rc"; tail -n 3 "$G/d_first1000.stderr"; head -n 2 "$G/d_first1000.stdout"
"$NEW/bin/python" -I -c "
import json, sys
d = json.load(open(sys.argv[1])); c = d['comparison']
print('verdict', d['verdict'], 'models', c['models_compared'], 'of', c['expected_models'], 'mismatches', c['mismatch_count'])
print('cpu_sha256', c['cpu_payload_sha256']); print('gpu_sha256', c['gpu_payload_sha256'])
print('nonvacuous', c['nonvacuous_gpu_work'], c['nonvacuous_gpu_work_failures'])
print('trust', d['cuda_attestation']['aksha_trust']['state'], d['cuda_attestation']['aksha_trust']['source'])
print('wall_seconds', d['wall_seconds'])
" "$G/d_first1000.json" 2>&1
echo "GATE_G3_DONE $(date -u +%FT%TZ)"
