# sourced by every long_loci frozen-run wrapper (copy of long_loci/code/env.sh with LL = $RD/long_loci/frozen)
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
FZ=$RD/long_loci/frozen; LL=$FZ; C=$FZ/code; LC=$RD/long_loci/code; TS=$RD/transfer_spacedust
PKG=$RD/slice_lut/testset/package_k8
PYE=/home/jwestrob/miniconda3/envs/elsa/bin/python          # DPANN scorer environment
PYT=$RD/ws_tokens/env/bin/python                             # token / duckdb environment (slice_lut recipe)
GPY=/groups/banfield/scratch/projects/human/nano/int/kinsey/tools/Talea/.venv/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
T="${SLURM_CPUS_ON_NODE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=$T NUMBA_THREADING_LAYER=workqueue TMPDIR=$FZ/tmp NUMBA_CACHE_DIR=$FZ/tmp/numba_${SLURM_JOB_ID}
mkdir -p "$NUMBA_CACHE_DIR" "$FZ/results" "$FZ/logs"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
# step NAME CMD...: run once (marker $FZ/logs/done_NAME); wall time + peak memory via /usr/bin/time -v -> $FZ/logs/time_NAME.txt
step() { local n=$1; shift; if [[ -f "$FZ/logs/done_$n" ]]; then stamp "skip $n (done)"; return 0; fi
  stamp "START $n"; /usr/bin/time -v -o "$FZ/logs/time_$n.txt" "$@"; touch "$FZ/logs/done_$n"; stamp "DONE $n"; }
