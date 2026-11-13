# usage (via run_job.sbatch): LIST=<file of lines "tag mode groupfile">  mode = map | weights | genome
U=$RD/ws_dpann/package/universe
TOKF=$RD/engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet
OUT=$RD/engine/results_dpann_lineagefree
while read -r tag mode gfile; do
  [[ -z "$tag" || "$tag" == \#* ]] && continue
  case $mode in
    map) LIN=(--lineage-map $RD/lineagefree/groups/$gfile);;
    weights) LIN=(--lineage-weights $RD/lineagefree/groups/$gfile);;
    genome) LIN=(--lineage-genome);;
  esac
  arm=dpann__mmseqs__seq__e3c50s75_rescue3__LF$tag
  echo "=== $(date -u +%T) $tag $mode $gfile"
  $PY -u $C/recurrence_lf.py --arm $arm --universe $U --tokens $TOKF --w 3 --null genome --benchmark none \
      --out $OUT --threads $T "${LIN[@]}" > $RD/lineagefree/results/engine_$tag.log 2>&1 || { echo "ENGINE FAILED $tag"; tail -5 $RD/lineagefree/results/engine_$tag.log; continue; }
  grep -E "rule|done" $RD/lineagefree/results/engine_$tag.log | cut -c1-200
  $PY -u $C/readings_lf.py score --rdir $OUT/${arm}__w3 --tag $tag --threads $T || echo "READINGS FAILED $tag"
done < "$LIST"
