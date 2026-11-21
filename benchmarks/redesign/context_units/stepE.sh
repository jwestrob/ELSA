# evaluation: context-similarity decay by taxonomic relation (diagnosis), then the per-pair taxonomy relations
$PY -u $C/evaluate.py decay --threads $T
$PY -u $C/evaluate.py taxa --threads $T --variants raw,gc,member,raw_z2,gc_z2
