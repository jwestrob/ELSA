# Test 3 on the selected N_eff landscape (predeclared), then POST HOC on two lineagefree token landscapes
PY2=$RD/ws_tokens/env/bin/python
LFE=$RD/engine/results_dpann_lineagefree
$PY2 -u $C/cuts.py run --threads $T --reuse-kappa --out-tag neff_and_posthoc --landscapes "neff_raw_W50_t2=$RD/context_units/rdir/raw_W50_t2:raw_W50_t2,lf_emb_hom_g0.75=$LFE/dpann__mmseqs__seq__e3c50s75_rescue3__LFemb_hom_g0.75__w3:lf_emb_hom_g0.75,lf_genome=$LFE/dpann__mmseqs__seq__e3c50s75_rescue3__LFgenome__w3:lf_genome"
