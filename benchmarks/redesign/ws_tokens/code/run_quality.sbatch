#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J tok_quality
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens/logs/quality_%j.log

# usage: sbatch run_quality.sbatch TAG ARMS(comma) TM(0|1) [SKIP_PREP(0|1)] [TM_SAMPLER(pair|token)]
# Statistics, V-measure, pair recall, sampled pair precision (MMseqs SW on the gid-keyed
# database, fake prefilter), and optionally TM-align of sequence-invisible same-token pairs.
# SKIP_PREP=1 reuses the prep/precision outputs already in quality/<arm>/ and runs TM only.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
WS=$RD/ws_tokens
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
FS=/shared/software/foldseek/10-941cd33/bin/foldseek
N=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/native_run_20260910
SEQDB=$WS/work/mmseqs/seqdb
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
TAG=$1; ARMS=$2; TM=${3:-0}; SKIP=${4:-0}; TMS=${5:-pair}
QW=$WS/work/quality/$TAG
stamp() { printf '%s %s\n' "$(date -Is)" "$*"; }
cd "$WS/code"
stamp "host=$(hostname) threads=$T tag=$TAG arms=$ARMS tm=$TM skip_prep=$SKIP tm_sampler=$TMS"

if [ "$SKIP" != "1" ]; then
    rm -rf "$QW"; mkdir -p "$QW/db"
    stamp "START prep"
    "$PY" -u quality.py prep --arms "$ARMS" --pref "$QW/pref.tsv" --threads "$T"
    wc -l "$QW/pref.tsv"
    stamp "START align"
    "$MM" tsv2db "$QW/pref.tsv" "$QW/db/pref" --output-dbtype 7 -v 2
    "$MM" align "$SEQDB" "$SEQDB" "$QW/db/pref" "$QW/db/aln" -e inf --alignment-mode 3 --threads "$T" -v 2
    "$MM" prefixid "$QW/db/aln" "$QW/aln.tsv" --tsv --threads "$T" -v 2
    wc -l "$QW/aln.tsv"
    stamp "START precision"
    "$PY" -u quality.py precision --arms "$ARMS" --aln "$QW/aln.tsv" --threads "$T"
fi

if [ "$TM" = "1" ]; then
    F=$WS/work/fsdb   # key-headed foldseek aliases, built by run_labels.sbatch
    [ -s "$F/rdb_h.index" ] && [ -s "$F/qdb_h.index" ]
    mkdir -p "$QW/db"
    stamp "START tmprep"
    "$PY" -u quality.py tmprep --arms "$ARMS" --tmdir "$QW" --tm-sampler "$TMS"
    # foldseek tmalign parallelizes over the targets of one query entry; with 1-4 targets per
    # query it kept 1-4 of 24 cores busy. Split the pairs into $T chunks and run one
    # single-threaded foldseek per chunk. The structure DB is read once sequentially into
    # page cache first (random reads from network storage are slow).
    stamp "START warm page cache"
    cat "$N"/reference/keggclusterdb_foldseek "$N"/reference/keggclusterdb_foldseek_ca \
        "$N"/reference/keggclusterdb_foldseek_ss "$N"/query/querydb_foldseek "$N"/query/querydb_foldseek_ca \
        "$N"/query/querydb_foldseek_ss > /dev/null
    tm_chunk() {  # A B chunk_tsv
        "$MM" tsv2db "$3" "$3.pref" --output-dbtype 7 -v 1 &&
        "$FS" tmalign "$1" "$2" "$3.pref" "$3.tm" --tmscore-threshold 0.0 -a 1 --db-load-mode 2 --threads 1 -v 1 &&
        "$FS" convertalis "$1" "$2" "$3.tm" "$3.out" \
            --format-output query,target,qtmscore,ttmscore,alntmscore,lddt,fident,qcov,tcov,alnlen,qlen,tlen \
            --exact-tmscore 1 --db-load-mode 2 --threads 1 -v 1
    }
    for c in rr qr qq; do
        [ -s "$QW/pref_$c.tsv" ] || continue
        case $c in rr) A=$F/rdb; B=$F/rdb;; qr) A=$F/qdb; B=$F/rdb;; qq) A=$F/qdb; B=$F/qdb;; esac
        rm -rf "$QW/db/tm_$c"
        mkdir -p "$QW/db/tm_$c"
        n=$(wc -l < "$QW/pref_$c.tsv"); nc=$(( n < T ? n : T ))
        split -n l/$nc -d -a 3 "$QW/pref_$c.tsv" "$QW/db/tm_$c/chunk_"
        stamp "START tmalign $c pairs=$n chunks=$nc"
        pids=()
        for f in "$QW"/db/tm_$c/chunk_???; do tm_chunk "$A" "$B" "$f" & pids+=($!); done
        fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
        [ "$fail" = "0" ] || { stamp "FAILED tmalign chunk in $c"; exit 1; }
        cat "$QW"/db/tm_$c/chunk_???.out > "$QW/aln_$c.tsv"
        stamp "DONE tmalign $c $(wc -l < "$QW/aln_$c.tsv") rows"
    done
    "$PY" -u quality.py tmcollect --arms "$ARMS" --tmdir "$QW" --tm-sampler "$TMS"
fi
"$PY" -u quality.py collect
rm -rf "$QW/db"
stamp "END"
