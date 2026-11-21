#!/bin/bash
# Login-node build (single-threaded, a few minutes) of the dev aksha-cuda12 wheel inside the maintainer
# manylinux2014 container used for the v4/0.1.1 wheels (devtoolset-10, staged CUDA 12.5).
# Usage: build_dev_cuda.sh OUTPUT_DIR
set -euo pipefail
OUT=${1:?output directory}
J=/groups/banfield/projects/environmental/sr/srvp2020/Jacob
W=$J/ELSA/redesign_20261111/aksha_check/gpufix
HG=$J/hmmer_gpu/build
NEW=/groups/banfield/users/jwestrob/envs/aksha-0.2.4.dev0
SCRATCH=$W/container_tmp
mkdir -p "$SCRATCH"
apptainer exec --cleanenv --containall \
    --bind "$J:$J" --bind /groups/banfield/users/jwestrob/envs:/groups/banfield/users/jwestrob/envs \
    --bind "$SCRATCH:/tmp" --bind "$HG/pip-release-cuda12.5-toolchain:/opt/cuda:ro" --pwd /tmp \
    "$HG/pip-release-manylinux2014-rootfs" \
    /bin/bash -c "export PATH=$HG/pip-release-candidate-v4/build-venv/bin:/opt/rh/devtoolset-10/root/usr/bin:/opt/python/cp312-cp312/bin:/usr/local/bin:/usr/bin:/bin LC_ALL=C PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1; g++ --version | head -1; $HG/pip-release-candidate-v4/build-venv/bin/python $W/build_dev_cuda.py $W/aksha $NEW/lib/python3.12/site-packages $OUT 0.1.2.dev0"
