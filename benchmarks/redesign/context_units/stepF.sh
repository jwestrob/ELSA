# POST HOC (amendment_3): exterior windows from the taxonomy-free genome-unit landscape, z = 2
P1=$RD/engine/results_dpann_lineagefree/dpann__mmseqs__seq__e3c50s75_rescue3__LFgenome__w3
$PY -u $C/ctxrec.py run --variants raw,gc --threads $T --W 25,50 --t 1.5,2,3 --z 2 --exterior-from $P1
