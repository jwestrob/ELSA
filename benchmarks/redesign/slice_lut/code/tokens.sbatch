#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand_2
#SBATCH -c 64
#SBATCH -J lut_tokens
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/logs/tokens_%j.log
# Workstream slice_lut, job 2: MMseqs2 family tokens with the DPANN recipe for each slice package (env SCOPES, default
# "beijerinckiaceae rhizobiales"):
#   base    mmseqs cluster -s 7.5 -e 1e-3 --min-seq-id 0 -c 0.5 --cov-mode 0 --cluster-mode 0   (engine/run_seq_families_dpann.sbatch)
#           -> ws_dpann/package/code/mmseqs_tokens.py (copied unchanged)
#   rescue3 unique sequences; genes in base tokens of < 3 genes searched against all unique sequences
#           (prefilter -s 7.5 --max-seqs 1000; align -e 1e-5 -a; repair/search_shard.sbatch) -> 06b_rescue3.py
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
S=$RD/slice_lut; C=$S/code; W=$S/work; P=$S/package
PY=$RD/ws_tokens/env/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T mmseqs=$($MM version)"
for SC in ${SCOPES:-beijerinckiaceae rhizobiales}; do
  PK=${PKROOT:-$P}/$SC; WD=$W/tokens_$SC; ARM=mmseqs__seq__e3c50s75__$SC
  mkdir -p $WD $PK/tokens
  if [[ ! -f $PK/tokens/$ARM.parquet ]]; then
    rm -rf $WD/seqdb* $WD/clu* $WD/tmp
    $MM createdb $PK/proteins_gid.faa $WD/seqdb -v 2
    PARAMS="-s 7.5 -e 1e-3 --min-seq-id 0 -c 0.5 --cov-mode 0 --cluster-mode 0"
    stamp "START cluster $SC"; t0=$(date +%s)
    $MM cluster $WD/seqdb $WD/clu $WD/tmp $PARAMS --threads $T -v 2
    t1=$(date +%s); stamp "DONE cluster $SC $((t1 - t0)) s"
    $MM createtsv $WD/seqdb $WD/seqdb $WD/clu $WD/clu.tsv --threads $T -v 2
    J="{\"tool\": \"mmseqs cluster\", \"mmseqs_version\": \"$($MM version)\", \"params\": \"$PARAMS\", \"input_fasta\": \"$PK/proteins_gid.faa\", \"threads\": $T}"
    $PY -u $C/mmseqs_tokens.py --clusters $WD/clu.tsv --universe $PK/universe/genes.parquet --out $PK/tokens/$ARM \
        --threads $T --params "$J" --mmseqs-seconds $((t1 - t0))
    rm -rf $WD/tmp
  fi
  if [[ ! -f $PK/tokens/mmseqs__seq__e3c50s75_rescue3__$SC.parquet ]]; then
    U=$WD/uniq; mkdir -p $U
    stamp "START unique + rescue queries $SC"
    $PY -u $C/06a_unique.py $PK $PK/tokens/$ARM.parquet $U
    for db in uniqdb Qdb pref aln; do $MM rmdb $U/$db -v 1 2>/dev/null || true; done
    $MM createdb $U/uniq.faa $U/uniqdb -v 2
    $MM createsubdb --id-mode 1 $U/Q.uid $U/uniqdb $U/Qdb -v 2
    stamp "START rescue prefilter $SC ($(wc -l < $U/Q.uid) query uids)"; t0=$(date +%s)
    $MM prefilter $U/Qdb $U/uniqdb $U/pref -s 7.5 --max-seqs 1000 --threads $T -v 2
    t1=$(date +%s); stamp "START align (prefilter $((t1 - t0)) s)"
    $MM align $U/Qdb $U/uniqdb $U/pref $U/aln -e 1e-5 -a --threads $T -v 2
    t2=$(date +%s); stamp "DONE align ($((t2 - t1)) s)"
    $MM convertalis $U/Qdb $U/uniqdb $U/aln $U/conv.tsv \
        --format-output query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen --threads $T -v 2
    $PY -u $C/tsv2parquet.py $U/conv.tsv $U/hits.parquet
    rm -f $U/conv.tsv; $MM rmdb $U/pref -v 1
    $PY -u $C/06b_rescue3.py $U $U/hits.parquet $PK/tokens/$ARM.parquet $PK/tokens/mmseqs__seq__e3c50s75_rescue3__$SC $T
    rm -rf $U/duck_tmp
  fi
done
stamp "DONE"
