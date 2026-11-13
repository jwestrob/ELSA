U=$RD/ws_dpann/package/universe
S=$RD/lineagefree/sim
OMP_NUM_THREADS=$T $PY -u $C/genome_groups.py famsim --tokens $RD/engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet --universe $U --out $S
$PY -u $C/genome_groups.py embsim --exact $RD/softrec/r3/exact --uid /groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111/uid_of_row.npy --universe $U --out $S --threads $T
for x in fam_jac:mean fam_con:mean fam_con:max emb_mean:mean emb_hom:mean emb_id95:mean emb_id95:max; do
  n=${x%%:*}; s=${x##*:}
  $PY -u $C/genome_groups.py describe --sim $S/$n.npy --sym $s --out $S/describe_${n}_${s}.json &
done
wait
