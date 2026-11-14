#!/bin/bash
# Faithful families for the synton pipeline: MMseqs2 linclust cascade 99% -> 95% -> 80% -> 50% identity, each stage
# clustering the representatives of the previous one (paper: "clustered iteratively at various sequence identity
# thresholds (99%, 95%, 80%, and 50%) using MMseqs2 linclust"). Coverage: linclust default -c 0.8 --cov-mode 0 written
# out explicitly (the paper states none). Input FASTA headers = gid; one terminal '*' is removed (RefSeq proteins carry
# none).
# usage: fam_cascade.sh IN.faa OUTDIR THREADS
set -euo pipefail
FAA=$1; OUT=$2; T=$3
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs
mkdir -p "$OUT/tmp"
stamp() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*"; }
"$MM" version > "$OUT/mmseqs_version.txt"
awk '/^>/{print; next} {sub(/\*$/, ""); print}' "$FAA" > "$OUT/in.faa"
"$MM" createdb "$OUT/in.faa" "$OUT/db" -v 2
prev="$OUT/db"
for id in 0.99 0.95 0.80 0.50; do
  tag=${id#0.}
  stamp "linclust --min-seq-id $id on $(wc -l < "$prev.index") sequences"
  "$MM" linclust "$prev" "$OUT/clu$tag" "$OUT/tmp" --min-seq-id "$id" -c 0.8 --cov-mode 0 --threads "$T" -v 2
  "$MM" createtsv "$prev" "$prev" "$OUT/clu$tag" "$OUT/clu$tag.tsv" --threads "$T" -v 2
  "$MM" createsubdb "$OUT/clu$tag" "$prev" "$OUT/rep$tag" -v 2
  prev="$OUT/rep$tag"
  stamp "stage $tag: $(cut -f1 "$OUT/clu$tag.tsv" | sort -u | wc -l) clusters"
done
rm -rf "$OUT/tmp" "$OUT/in.faa"
stamp "cascade done"
