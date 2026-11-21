#!/bin/bash
# Mirror small result files from biotite into results/ (code stays as is; large arrays stay on biotite).
S=/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad
RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/grammar
L=/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/grammar/results
mkdir -p $L
$S/bt "cd $RD && tar cf - prep/splits.json \$(ls runs/cv?/summary.json runs/phylum/summary.json runs/cv?/train_log.tsv runs/phylum/train_log.tsv 2>/dev/null) \$(ls eval/*.tsv eval/*.json 2>/dev/null) \$(ls loci/loci_info.json loci/*/compare_vs_engine.txt loci/*/ribosomal_check.txt loci/*/ribosomal_check.json loci/*/score_min2/summary.json 2>/dev/null)" | tar xf - -C $L
find $L -type f | sort
