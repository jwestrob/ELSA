#!/bin/bash
#SBATCH -A banfield
#SBATCH -p gpu_h200
#SBATCH --qos=gpu_h200_4
#SBATCH --gres=gpu:1
#SBATCH -c 28
#SBATCH -J gpufix_probe
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/aksha_check/gpufix/probe/probe_%j.log

# Diagnostic: exhaustive host-libm versus CUDA log census on the upgraded H200 node (no HMM search).
set -euo pipefail
echo "host $(hostname)  $(date -u +%FT%TZ)  CPUs $SLURM_CPUS_ON_NODE"
ldd --version | head -1
readelf -n /lib/x86_64-linux-gnu/libm.so.6 | grep 'Build ID'
grep -m2 -E '^(model name|flags)' /proc/cpuinfo | cut -c1-200
nvidia-smi --query-gpu=index,name,driver_version --format=csv
/usr/bin/time -v /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/aksha_check/gpufix/probe/log_census_probe "$SLURM_CPUS_ON_NODE" || echo "probe exit $?"
echo "PROBE_DONE $(date -u +%FT%TZ)"
