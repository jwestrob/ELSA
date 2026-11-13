# genus reference through this driver, then the raw variant, its shuffled control and the member control
$PY -u $C/ctxrec.py refcheck --threads $T
$PY -u $C/ctxrec.py run --variants raw,shuf_raw,member --threads $T --W 10,25,50,100 --t 1.5,2,3 --t-shuf 2
