# genus map through the lineage-map code: must reproduce the deployed engine result and the DH / R1 loci and scores
U=$RD/ws_dpann/package/universe
TOKF=$RD/engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet
OUT=$RD/engine/results_dpann_lineagefree
$PY -u $C/eval_taxonomy.py genusmap --out $RD/lineagefree/groups/genus_map.parquet
$PY -u $C/recurrence_lf.py --arm dpann__mmseqs__seq__e3c50s75_rescue3__LFgenus --universe $U --tokens $TOKF --w 3 --null genome \
   --benchmark none --all-genome-loci --out $OUT --threads $T --lineage-map $RD/lineagefree/groups/genus_map.parquet --report-taxonomy
$PY -u $C/compare_rdir.py $OUT/dpann__mmseqs__seq__e3c50s75_rescue3__LFgenus__w3 $RD/engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3 | tee $RD/lineagefree/results/refcheck_engine_genus.json
$PY -u $C/readings_lf.py score --rdir $OUT/dpann__mmseqs__seq__e3c50s75_rescue3__LFgenus__w3 --tag genus --threads $T --refcheck
