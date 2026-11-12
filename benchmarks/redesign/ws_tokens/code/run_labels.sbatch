#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J tok_labels
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_tokens/logs/labels_%j.log

# Label pair sets in gid space, plus the key-headed foldseek alias databases used by run_quality.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
WS=$RD/ws_tokens
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
N=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust/native_run_20260910
T="${SLURM_CPUS_ON_NODE:?}"
cd "$WS/code"
echo "$(date -Is) host=$(hostname) threads=$T"
"$PY" -u labels.py --threads "$T"
F=$WS/work/fsdb
if [ ! -s "$F/qdb_h.index" ]; then
    rm -rf "$F"; mkdir -p "$F"
    for pair in "rdb:$N/reference/keggclusterdb_foldseek:K" "qdb:$N/query/querydb_foldseek:Q"; do
        IFS=: read -r name src pre <<< "$pair"
        ln -sf "$src" "$F/$name"; ln -sf "$src.index" "$F/$name.index"; cp "$src.dbtype" "$F/$name.dbtype"
        for s in ca ss; do
            ln -sf "${src}_$s" "$F/${name}_$s"; ln -sf "${src}_$s.index" "$F/${name}_$s.index"
            cp "${src}_$s.dbtype" "$F/${name}_$s.dbtype"
        done
        awk -v p="$pre" '{print $1"\t"p$1}' "$src.index" > "$F/${name}_h.tsv"
        "$MM" tsv2db "$F/${name}_h.tsv" "$F/${name}_h" --output-dbtype 12 -v 2
    done
fi
ls -la "$F"
echo "$(date -Is) END"
