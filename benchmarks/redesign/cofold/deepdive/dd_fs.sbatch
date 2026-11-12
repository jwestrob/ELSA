#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J dd_fs
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/deepdive/logs/fs_%j.log
# (1) Foldseek 10 of candidate-1 predicted chains (A, B, C, AB as one chain, fused homologs) against PDB and AFDB
#     Swiss-Prot (foldseek DBs 'latest', 02_04_2026), Foldseek 9 multimer search of AB / ABC complexes against PDB;
# (2) US-align of every fused-homolog model against every AB-as-one-chain model;
# (3) third gene C: HHblits 3 iterations vs UniRef30_2020_06 from the 258-row C instance alignment, HHsearch vs
#     Pfam 39 (HH) and PDB100; (4) candidate-2 family HMMs (mafft of token members, hmmbuild) vs all DPANN proteins.
set -euo pipefail
export PATH=/shared/software/bin:$PATH
DD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/deepdive
FAA=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/work/proteins_gid.faa
FS=/shared/software/foldseek/10-941cd33/bin/foldseek
DB=/shared/db/foldseek/latest/db
HH=/shared/software/hhsuite/latest/bin
UNIREF=/shared/db/uniclust/30_2020_06/UniRef30_2020_06
PDB100=/shared/db/rosettafold/rosettafold2/pdb100_2021Mar03/pdb100_2021Mar03
PFAM=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/s2/db/pfam39/pfam39
T=$SLURM_CPUS_ON_NODE
H=$((T/2))
echo "threads $T host $(hostname) $(date)"
# --- (3) and (4) in the background while Foldseek runs
(
  cd $DD/hh
  $HH/hhblits -i $DD/fs_c1/C_inst.aln.fa -M first -d $UNIREF -n 3 -e 0.001 -cpu 4 -oa3m c1C.u.a3m -o c1C.u.hhr -v 1 \
     -maxfilt 100000 -realign_max 100000 -B 100000 -Z 100000
  $HH/hhsearch -i c1C.u.a3m -d $PFAM -o c1C.u.pfam.hhr -cpu 4 -v 1 -B 500 -Z 500
  $HH/hhsearch -i c1C.u.a3m -d $PDB100 -o c1C.u.pdb.hhr -cpu 4 -v 1 -B 500 -Z 500
  $HH/hhsearch -i $DD/fs_c1/C_inst.aln.fa -M first -d $PFAM -o c1C.dpann.pfam.hhr -cpu 4 -v 1 -B 500 -Z 500
  $HH/hhsearch -i $DD/fs_c1/C_inst.aln.fa -M first -d $PDB100 -o c1C.dpann.pdb.hhr -cpu 4 -v 1 -B 500 -Z 500
  echo "C family searches done $(date)"
) &
(
  mkdir -p $DD/hmm/c2 && cd $DD/hmm/c2
  for f in $DD/work/c2_tok/*.fa; do b=$(basename $f .fa); mafft --auto --thread 2 --quiet $f > $b.aln.fa; \
    hmmbuild --cpu 2 --informat afa -n $b $b.hmm $b.aln.fa > /dev/null; done
  cat tok*.hmm > c2_tokens.hmm
  hmmsearch --cpu 4 -E 1e-3 --domE 1e-3 --noali --domtblout c2_tokens_vs_dpann.domtbl -o /dev/null c2_tokens.hmm $FAA
  echo "c2 HMM search done $(date)"
) &
cd $DD/fs_c1
FMT=query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,prob,alntmscore,qtmscore,ttmscore,lddt,theader
for db in pdb alphafold_swissprot; do
  if [ ! -s ${db}.m8 ]; then
    $FS easy-search chains $DB/$db ${db}.m8.tmp tmp_$db --threads $H -e 10 --exhaustive-search 1 --format-output $FMT > log_$db.txt 2>&1
    mv ${db}.m8.tmp ${db}.m8; rm -rf tmp_$db
  fi
done
/shared/software/foldseek/9-427df8a/bin/foldseek easy-multimersearch complex $DB/pdb pdb_multimer.m8.tmp tmp_mm --threads $H -e 10 > log_mm.txt 2>&1 \
  && mv pdb_multimer.m8.tmp pdb_multimer.m8 || echo "multimer search failed (non-fatal)"
rm -rf tmp_mm
# --- (2) US-align, all pairs among fused models and AB-as-one-chain models
cd usalign
ls *.pdb > list.txt
python3 - <<'PY' > pairs.txt
f=[l.strip() for l in open('list.txt')]
for i in range(len(f)):
    for j in range(len(f)):
        if i!=j: print(f[i], f[j])
PY
cat pairs.txt | xargs -P $H -n 2 sh -c 'USalign "$0" "$1" -outfmt 2 | grep -v "^#"' > usalign_pairs.tsv
wait
echo "done $(date)"
