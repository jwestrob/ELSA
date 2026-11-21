#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J lut_final
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/logs/final_%j.log
# After the GPU worker drained the queue: embed_queue finalize (assembly and audits; no model), zca10 whitening
# fitted on this data set (zca10.py), products linked into the package, descriptive receipt, manifest.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
W=$RD/slice_lut; P=$W/testset/package_k8
RUN=$W/testset/embed/luttest__esmpp_small__20261113
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host $(hostname) cpus $T"
set +u; source /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/code/env.sh; set -u
if [ ! -e $RUN/final/MANIFEST.json ]; then
  $PYBIN -u $EQ/code/embed_queue.py finalize --run $RUN --threads "$T"
fi
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
stamp "zca10"
$PY -u $W/code/embed/zca10.py --run $RUN --out $P/embeddings
for f in final/mean_raw_f32__uid.npy final/l2_f32__row.npy final/status__uid.npy uid_of_row.npy length_of_uid.npy; do
  ln -f $RUN/$f $P/embeddings/esmpp_small__$(basename $f)
done
cp $RUN/final/MANIFEST.json $P/embeddings/esmpp_small__MANIFEST.json
cp $RUN/run.json $P/embeddings/esmpp_small__run.json
mkdir -p $P/code; cp $W/code/*.py $W/code/*.sbatch $P/code/; cp -r $W/code/embed $P/code/
stamp "manifest"
$PY -u $W/code/embed/make_manifest.py --pkg $P --threads "$T" > /dev/null
stamp "done"
