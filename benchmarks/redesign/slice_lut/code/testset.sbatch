#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J lut_testset
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/logs/testset_%j.log
# Workstream slice_lut part 2: counts (proteins, unique sequences) for the three size options, the package for the middle
# size (env SEL, PKG), and its ribosomal truth runs (hmmsearch of the 121 Pfam 38.1 ^ribosom(al)?_ models, --cut_ga,
# $SLURM_CPUS_ON_NODE single-threaded processes over FASTA parts). No ELSA method, no embeddings.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
S=$RD/slice_lut; C=$S/code; TS=$S/testset
PY=$RD/ws_tokens/env/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host=$(hostname) threads=$T"
if [[ -z "${SKIP_COUNT:-}" ]]; then $PY -u $C/14_build_testset.py count $TS/selection/selection_k1.tsv $TS/selection/selection_k2.tsv $TS/selection/selection_k8.tsv \
    --out $TS/selection/counts.json --threads $T; fi
stamp "START build ${SEL:?} -> ${PKG:?}"
rm -rf $PKG; $PY -u $C/14_build_testset.py build $SEL --pkg $PKG --threads $T
rm -rf $PKG/.duck_tmp
stamp "START ribosomal markers"
W=$TS/work/ribo_$(basename $PKG); rm -rf $W; mkdir -p $W/parts $W/out
awk -v n=$T -v d=$W/parts 'BEGIN{k=-1} /^>/{k=(k+1)%n} {print > (d "/p_" k ".faa")}' $PKG/proteins_gid.faa
ls $W/parts/*.faa | xargs -P $T -I{} sh -c 'b=$(basename {} .faa); hmmsearch --cut_ga --cpu 1 --noali -o /dev/null --domtblout '$W'/out/$b.domtbl '$TS'/work/ribo.hmm {}'
$PY -u $C/15_ribosomal_truth.py $PKG "$W/out/*.domtbl"
rm -rf $W/parts
if [[ -n "${EMBED_RUN:-}" ]]; then
  stamp "embed_queue prep (sharding only, no model) -> $EMBED_RUN"
  mkdir -p $(dirname $EMBED_RUN)
  set +u; source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/code/env.sh; set -u
  $PYBIN -u $EQ/code/embed_queue.py prep --fasta $PKG/proteins_gid.faa --run $EMBED_RUN --backend esmpp --model-dir $ESMPP_SMALL \
      --model-id Synthyra/ESMplusplus_small --model-revision $ESMPP_SMALL_REV --dtype bf16
fi
stamp "DONE"
