#!/bin/bash
# Census of ggBin (DAS Tool) result trees: one line per sample.
# Uses only ls on known directories (no recursive traversal).
# Usage: 01_census_ggbin.sh LABEL RESULTS_DIR [RESULTS_DIR ...] > out.tsv
label=$1; shift
printf "collection\tresults_dir\tsample\tn_dastool_bins\thas_summary\tsummary_path\tassembly\tbin_dates\n"
for R in "$@"; do
  for s in $(ls "$R"); do
    D="$R/$s/bins/dastool/results/$s.asm.fa.dastool"
    if [ ! -d "$D" ]; then
      D2=$(ls -d "$R/$s"/bins/dastool/results/*.dastool 2>/dev/null | head -1)
      [ -n "$D2" ] && D=$D2
    fi
    nb=0; [ -d "$D/bins" ] && nb=$(ls "$D/bins" | wc -l)
    Iq=$(ls "$R/$s"/bins/dastool/intermediate/*/*_DASTool_summary.tsv 2>/dev/null | head -1)
    hs=0; [ -n "$Iq" ] && hs=1
    asm=$(ls "$R/$s"/assemblies 2>/dev/null | head -1)
    dt=$(ls -ld --time-style=+%Y-%m-%d "$D/bins" 2>/dev/null | awk '{print $6}')
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$label" "$R" "$s" "$nb" "$hs" "$Iq" "$asm" "$dt"
  done
done
