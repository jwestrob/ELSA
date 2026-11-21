#!/bin/bash
#SBATCH -A banfield
#SBATCH -p standard
#SBATCH --qos=standard_expand
#SBATCH -c 24
#SBATCH -J sb_tokens
#SBATCH --exclude=node-48-256g-2,node-48-256g-3,node-48-256g-8,node-48-256g-9,node-48-256g-12,node-48-256g-13,node-48-256g-14
#SBATCH -o /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_srvp_bac/logs/tokens_%j.log
# MMseqs2 family tokens with the DPANN recipe, then rescue3 (workstream transfer_srvp_bac).
#  1. base families: engine/run_seq_families_dpann.sbatch with COV=0.5 MINID=0
#     (cluster -s 7.5 -e 1e-3 --min-seq-id 0 -c 0.5 --cov-mode 0 --cluster-mode 0 on every row), mmseqs_tokens.py
#  2. rescue search: repair/code/search_shard.sbatch (unique sequences; prefilter -s 7.5 --max-seqs 1000;
#     align -e 1e-5 -a), queries = unique sequences of genes in base tokens with < 3 members (the rescue queries;
#     per-query results do not depend on the other queries, so the DPANN split-repair queries are not needed)
#  3. rescue3 arm (rescue3_tokens.py), then the same code replayed on the DPANN inputs, which must reproduce
#     engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet exactly.
set -euo pipefail
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111
W=$RD/ws_srvp_bac; P=$W/package
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
T="${SLURM_CPUS_ON_NODE:?}"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
stamp "host $(hostname) threads $T"
ARM=mmseqs__seq__e3c50s75__srvp_bac
F=$W/work/seqfam/$ARM
rm -rf "$F"; mkdir -p "$F" "$P/tokens"
"$MM" createdb "$W/work/proteins_gid.faa" "$F/seqdb" -v 2
PARAMS="-s 7.5 -e 1e-3 --min-seq-id 0 -c 0.5 --cov-mode 0 --cluster-mode 0"
stamp "START cluster $PARAMS"; t0=$(date +%s)
"$MM" cluster "$F/seqdb" "$F/clu" "$F/tmp" $PARAMS --threads "$T" -v 2
t1=$(date +%s); stamp "DONE cluster $((t1 - t0)) s"
"$MM" createtsv "$F/seqdb" "$F/seqdb" "$F/clu" "$F/clu.tsv" --threads "$T" -v 2
J="{\"tool\": \"mmseqs cluster\", \"mmseqs_version\": \"$($MM version)\", \"params\": \"$PARAMS\", \"input_fasta\": \"$W/work/proteins_gid.faa\", \"threads\": $T}"
"$PY" -u "$W/code/mmseqs_tokens.py" --clusters "$F/clu.tsv" --universe "$P/universe/genes.parquet" \
    --out "$P/tokens/$ARM" --threads "$T" --params "$J" --mmseqs-seconds $((t1 - t0))
rm -rf "$F/tmp"

U=$W/work/uniq; R=$W/work/rescue; D=$R/db; H=$R/hits; X=$R/run
rm -rf "$R"; mkdir -p "$D" "$H" "$X"
"$MM" createdb "$U/uniq.faa" "$D/uniqdb" --shuffle 0 -v 2
awk -F'\t' '$1!=$2{bad++} END{print "lookup key!=header rows:", bad+0, "of", NR; exit (bad>0)}' "$D/uniqdb.lookup"
"$PY" - <<PYEOF
import numpy as np, pandas as pd, json
t = pd.read_parquet("$P/tokens/$ARM.parquet").sort_values("gid")
m = pd.read_parquet("$U/gid2uid.parquet").sort_values("gid")
assert np.array_equal(t.gid.values, m.gid.values)
size = t.groupby("token").token.transform("size").values
q = np.unique(m.uid.values[size < 3])
np.savetxt("$R/Q.uid", q, fmt="%d")
L = pd.read_parquet("$U/uniq.parquet").sort_values("uid").length.values
json.dump(dict(query_genes=int((size < 3).sum()), query_uids=int(len(q)), query_residues=int(L[q].sum()),
               target_uids=int(len(L)), target_residues=int(L.sum())), open("$R/queries.json", "w"), indent=1)
print(open("$R/queries.json").read())
PYEOF
"$MM" createsubdb "$R/Q.uid" "$D/uniqdb" "$D/Q" -v 2
stamp "START prefilter"; t0=$(date +%s)
"$MM" prefilter "$D/Q" "$D/uniqdb" "$X/Q_pref" -s 7.5 --max-seqs 1000 --threads "$T" -v 3 > "$X/prefilter.log" 2>&1
t1=$(date +%s); stamp "START align (prefilter $((t1 - t0)) s)"
"$MM" align "$D/Q" "$D/uniqdb" "$X/Q_pref" "$X/Q_aln" -e 1e-5 -a --threads "$T" -v 3 > "$X/align.log" 2>&1
t2=$(date +%s); stamp "DONE align ($((t2 - t1)) s)"
"$MM" rmdb "$X/Q_pref" -v 2
"$MM" convertalis "$D/Q" "$D/uniqdb" "$X/Q_aln" "$X/conv_Q.tsv" \
   --format-output query,target,evalue,bits,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen --threads "$T" -v 2
"$PY" -u "$W/code/tsv2parquet.py" "$X/conv_Q.tsv" "$H/Q_0.parquet"
rm -f "$X/conv_Q.tsv"
echo -e "prefilter\t$((t1 - t0))\talign\t$((t2 - t1))" > "$R/timing.tsv"
stamp "rescue3 arm"
"$PY" -u "$W/code/rescue3_tokens.py" --gid2uid "$U/gid2uid.parquet" --uniq "$U/uniq.parquet" \
    --base "$P/tokens/$ARM.parquet" --hits "$H/Q_*.parquet" \
    --out "$P/tokens/mmseqs__seq__e3c50s75_rescue3__srvp_bac" --tmp "$R/tmp"
stamp "replay of the same code on the DPANN inputs"
mkdir -p "$W/work/replay_dpann"
"$PY" -u "$W/code/rescue3_tokens.py" --gid2uid "$RD/subgenic/gid2uid.parquet" --uniq "$RD/subgenic/uniq.parquet" \
    --base "$RD/engine/dpann_tokens/mmseqs__seq__e3c50s75__dpann.parquet" --hits "$RD/repair/hits/Q_*.parquet" \
    --out "$W/work/replay_dpann/mmseqs__seq__e3c50s75_rescue3__dpann_replay" --tmp "$W/work/replay_dpann/tmp" \
    --expect "$RD/engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
stamp "done"
