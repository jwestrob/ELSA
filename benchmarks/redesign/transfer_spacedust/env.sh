# sourced by every transfer_spacedust wrapper
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
TS=$RD/transfer_spacedust; C=$TS/code
ROOT=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust
SCALE=$ROOT/model_scaling_20260913
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SCALE/production:$SCALE/code"
export NUMBA_NUM_THREADS=$T NUMBA_THREADING_LAYER=workqueue TMPDIR=$TS/tmp NUMBA_CACHE_DIR=$TS/tmp/numba_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR" "$TS/results" "$TS/engine"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
# step NAME CMD...: run once (marker in $TS/logs/done_NAME), with /usr/bin/time -v peak memory into $TS/logs/time_NAME.txt
step() { local n=$1; shift; if [[ -f "$TS/logs/done_$n" ]]; then stamp "skip $n (done)"; return 0; fi
  stamp "START $n"; /usr/bin/time -v -o "$TS/logs/time_$n.txt" "$@"; touch "$TS/logs/done_$n"; stamp "DONE $n"; }
