#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sb_build_a
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_srvp_bac/logs/build_a_%j.log
# Stage A of the SR-VP bacteria package: copy check, universe, Prodigal transcription check, taxonomy, FASTA,
# gold tables (curated callers), genome halves, ribosomal runs, scorer copy and its synthetic test, unique
# sequences for rescue3, embed_queue prep (sharding only, no model), SEAL.txt.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
W=$RD/ws_srvp_bac; P=$W/package
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
SRC_SHA=971795c67fc30093e9fb09c7dd531f4c5158c6539a75df25752e107455494d6b
SRC_SIZE=15172382720
SRC_FILE="mac:/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/srvp_bacteria_pb/sharur.duckdb"
SRC_MTIME="2026-11-12T19:43:44-0700 = 2026-11-13T02:43:44Z (epoch 1794537824)"
RUN=$W/embed/srvp_bac__esmpp_small__20261113
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host $(hostname) cpus $T"
$PY -u $W/code/build_srvp_bac_package.py --db $W/input/sharur.duckdb --seal $W/input/dataset.seal.json \
    --pkg $P --work $W/work --threads "$T" --expect-sha256 $SRC_SHA --expect-size $SRC_SIZE \
    --source-file "$SRC_FILE" --source-mtime "$SRC_MTIME" \
    verify universe prodigal strandqc taxonomy fasta gold split
ln -f $W/work/proteins_gid.faa $P/universe/proteins.faa
stamp "ribosomal runs"
$PY -u $W/code/ribosomal_check_srvp_bac.py gold --threads "$T"
stamp "scorer copy and synthetic test (no real gold read)"
mkdir -p $P/scorer
cp $W/code/score_srvp_bac_loci.py $W/code/synthetic_scorer_test.py $P/scorer/
( cd $W/work && $PY $P/scorer/synthetic_scorer_test.py ) > $P/scorer/synthetic_test.txt 2>&1
tail -2 $P/scorer/synthetic_test.txt
stamp "unique sequences"
mkdir -p $W/work/uniq
$PY -u $W/code/prep_unique.py --faa $W/work/proteins_gid.faa --genes $P/universe/genes.parquet --out $W/work/uniq
stamp "embed_queue prep (no model)"
mkdir -p $W/embed
set +u; source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/code/env.sh; set -u
$PYBIN -u $EQ/code/embed_queue.py prep --fasta $W/work/proteins_gid.faa --run $RUN --backend esmpp --model-dir $ESMPP_SMALL \
    --model-id Synthyra/ESMplusplus_small --model-revision $ESMPP_SMALL_REV --dtype bf16
stamp "seal"
$PY -u $W/code/write_seal.py --pkg $P --source-file "$SRC_FILE" --source-size $SRC_SIZE --source-mtime "$SRC_MTIME"
mkdir -p $P/code; cp $W/code/*.py $W/code/*.sbatch $P/code/
stamp "done"
