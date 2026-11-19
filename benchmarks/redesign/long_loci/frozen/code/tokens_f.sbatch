#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J llf_tok
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/logs/tok_%j.log
# long_loci frozen run, J1: rescue3 family tokens of the lut test set with the DPANN recipe (package code/tokens.sbatch,
# copied; input = the read-only package, output = $FZ/tok):
#   base    mmseqs cluster -s 7.5 -e 1e-3 --min-seq-id 0 -c 0.5 --cov-mode 0 --cluster-mode 0 -> mmseqs_tokens.py
#   rescue3 unique sequences; genes in base tokens of < 3 genes searched against all unique sequences
#           (prefilter -s 7.5 --max-seqs 1000; align -e 1e-5 -a) -> 06b_rescue3.py
# Compute-limit guard (FROZEN.json compute_plan): once the rescue queries are known, the prefilter time is projected
# from the Beijerinckiaceae package (1,855 s for 269,580 query uids against 1.306 G residues on 64 threads, per query
# and scaled by target residues and by 64 / allocated threads) plus align at 0.17x; if the job would pass ~115 min it
# stops and reports. Resubmitted at 24 CPUs on 2026-11-14 at the coordinator's request (64-CPU job held the queue).
set -euo pipefail
source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci/frozen/code/env_f.sh
WD=$FZ/tok/work; OUTT=$FZ/tok; ARM=mmseqs__seq__e3c50s75__lut
mkdir -p $WD $OUTT
stamp "host=$(hostname) threads=$T job=$SLURM_JOB_ID mmseqs=$($MM version)"
if [[ ! -f $OUTT/$ARM.parquet ]]; then
  rm -rf $WD/seqdb* $WD/clu* $WD/tmp
  $MM createdb $PKG/proteins_gid.faa $WD/seqdb -v 2
  PARAMS="-s 7.5 -e 1e-3 --min-seq-id 0 -c 0.5 --cov-mode 0 --cluster-mode 0"
  stamp "START cluster"; t0=$(date +%s)
  /usr/bin/time -v -o $FZ/logs/time_tok_cluster.txt $MM cluster $WD/seqdb $WD/clu $WD/tmp $PARAMS --threads $T -v 2
  t1=$(date +%s); stamp "DONE cluster $((t1 - t0)) s"
  $MM createtsv $WD/seqdb $WD/seqdb $WD/clu $WD/clu.tsv --threads $T -v 2
  J="{\"tool\": \"mmseqs cluster\", \"mmseqs_version\": \"$($MM version)\", \"params\": \"$PARAMS\", \"input_fasta\": \"$PKG/proteins_gid.faa\", \"threads\": $T}"
  $PYT -u $C/mmseqs_tokens.py --clusters $WD/clu.tsv --universe $PKG/universe/genes.parquet --out $OUTT/$ARM \
      --threads $T --params "$J" --mmseqs-seconds $((t1 - t0))
  rm -rf $WD/tmp
fi
R3=$OUTT/mmseqs__seq__e3c50s75_rescue3__lut
if [[ ! -f $R3.parquet ]]; then
  U=$WD/uniq; mkdir -p $U
  stamp "START unique + rescue queries"
  $PYT -u $C/06a_unique.py $PKG $OUTT/$ARM.parquet $U
  # guard
  read Q RES <<< "$($PYT -c "import json; d=json.load(open('$U/unique.json')); print(d['query_uids'], d['residues_unique'])")"
  PROJ=$($PYT -c "q=$Q; r=$RES; p=1855.0/269580*q*(r/1306438806.0)*64.0/$T; print(int(p*1.17+300))")
  stamp "GUARD query uids $Q, target residues $RES, projected prefilter+align+rest ${PROJ} s, elapsed ${SECONDS} s"
  if (( SECONDS + PROJ > 115*60 )); then
    stamp "STOP: projected job time $(( (SECONDS + PROJ) / 60 )) min exceeds the ~2 h limit; rescue not started"
    exit 3
  fi
  for db in uniqdb Qdb pref aln; do $MM rmdb $U/$db -v 1 2>/dev/null || true; done
  $MM createdb $U/uniq.faa $U/uniqdb -v 2
  $MM createsubdb --id-mode 1 $U/Q.uid $U/uniqdb $U/Qdb -v 2
  stamp "START rescue prefilter ($(wc -l < $U/Q.uid) query uids)"; t0=$(date +%s)
  /usr/bin/time -v -o $FZ/logs/time_tok_prefilter.txt $MM prefilter $U/Qdb $U/uniqdb $U/pref -s 7.5 --max-seqs 1000 --threads $T -v 2
  t1=$(date +%s); stamp "START align (prefilter $((t1 - t0)) s)"
  /usr/bin/time -v -o $FZ/logs/time_tok_align.txt $MM align $U/Qdb $U/uniqdb $U/pref $U/aln -e 1e-5 -a --threads $T -v 2
  t2=$(date +%s); stamp "DONE align ($((t2 - t1)) s)"
  $MM convertalis $U/Qdb $U/uniqdb $U/aln $U/conv.tsv \
      --format-output query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen --threads $T -v 2
  $PYT -u $C/tsv2parquet.py $U/conv.tsv $U/hits.parquet
  rm -f $U/conv.tsv; $MM rmdb $U/pref -v 1
  /usr/bin/time -v -o $FZ/logs/time_tok_rescue3.txt $PYT -u $C/06b_rescue3.py $U $U/hits.parquet $OUTT/$ARM.parquet $R3 $T
  rm -rf $U/duck_tmp
fi
stamp "DONE"
