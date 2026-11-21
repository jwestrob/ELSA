#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu_h200
#SBATCH --qos=gpu_h200_4
#SBATCH --gres=gpu:1
#SBATCH -c 28
#SBATCH -J annot_diag_gpu
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/annot_gpu/validation/diag_gpu_%j.log

# Diagnostic only (no HMM search, no kernel launch): the installed plan7_gpu runtime's GPU bias-math attestation
# record on the allocated H200, plus the node facts it checks (glibc version, libm GNU build ID, CPU signature).
set -euo pipefail
echo "host $(hostname)  $(date -u +%FT%TZ)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
uname -a
grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
ldd --version | head -1
readelf -n /lib/x86_64-linux-gnu/libm.so.6 | grep 'Build ID'
grep -m4 -E '^(cpu family|model|model name|stepping)' /proc/cpuinfo
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version --format=csv
cd /tmp
/groups/banfield/users/jwestrob/envs/aksha-0.2.3/bin/python -I -c '
import json
from plan7_gpu import _native
print(json.dumps(_native.bias_environment_provenance(), indent=1, default=str))
print("bias_host_environment_attested", _native.bias_host_environment_attested())
'
echo "DIAG_DONE $(date -u +%FT%TZ)"
