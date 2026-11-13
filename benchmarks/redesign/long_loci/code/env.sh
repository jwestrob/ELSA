# sourced by every long_loci wrapper
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
LL=$RD/long_loci; C=$LL/code; TS=$RD/transfer_spacedust
SCALE=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/model_scaling_20260913
PYE=/home/jwestrob/miniconda3/envs/elsa/bin/python          # DPANN scorer environment (vocab workstream)
PYT=$RD/ws_tokens/env/bin/python                             # Spacedust scorer environment (transfer_spacedust)
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=$T NUMBA_THREADING_LAYER=workqueue TMPDIR=$LL/tmp NUMBA_CACHE_DIR=$LL/tmp/numba_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR" "$LL/results" "$LL/logs"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
# step NAME CMD...: run once (marker $LL/logs/done_NAME); peak memory via /usr/bin/time -v into $LL/logs/time_NAME.txt
step() { local n=$1; shift; if [[ -f "$LL/logs/done_$n" ]]; then stamp "skip $n (done)"; return 0; fi
  stamp "START $n"; /usr/bin/time -v -o "$LL/logs/time_$n.txt" "$@"; touch "$LL/logs/done_$n"; stamp "DONE $n"; }
