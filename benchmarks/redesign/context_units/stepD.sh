# Test 3: label-free cuts on the genus token landscape (vs its module-cut DH), then on the round-2 embedding landscape
PY2=$RD/ws_tokens/env/bin/python
$PY2 -u $C/cuts.py run --threads $T --landscapes "genus=$RD/context_units/rdir/refcheck_genus" --out-tag genus
$PY2 -u $C/cuts.py emb --threads $T --kh 50 --cap-h 200
