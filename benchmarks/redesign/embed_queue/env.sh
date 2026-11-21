# Sourced by every embed_queue script. Absolute paths, so jobs run from anywhere.
EQ=${EQ:-/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue}
# Python 3.11.3, torch 2.7.1+cu126, transformers 4.57.6, einops: the environment the Talea v0.3.5 lanes run ESM++ in.
PYBIN=${PYBIN:-/groups/banfield/scratch/projects/human/nano/int/kinsey/tools/Talea/.venv/bin/python}
# Python 3.10, torch 2.10.0, transformers 5.5.0 (native ESM-C classes); used only for the cross-implementation check.
ELSA_PY=${ELSA_PY:-/home/jwestrob/miniconda3/envs/elsa/bin/python}
HUB=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/hf_hub
ESMPP_SMALL_REV=46c5f7d562e47d4c14165b424c71ab7db008e6fb     # 2026-06-17, same code generation as Talea's ESM++ large pin
ESMPP_SMALL=$HUB/models--Synthyra--ESMplusplus_small/snapshots/$ESMPP_SMALL_REV
BAD_GPUS=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/Talea/talea_v035_pipeline/common/bad_gpus.txt
export HF_HOME=$EQ/hf_home HF_HUB_CACHE=$HUB HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_ENABLE_XET=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE:-8} MKL_NUM_THREADS=${SLURM_CPUS_ON_NODE:-8}
unset PYENV_VERSION
# sbatch from inside a job: drop the parent job's SLURM_*/SBATCH_* variables
clean_sbatch() { env $(env | grep -oE '^(SLURM|SBATCH|SALLOC)_[A-Za-z0-9_]*' | sed 's/^/-u /') "$@"; }
