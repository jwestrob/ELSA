# amendment_1: noise-aware rho (z = 2) for raw and gc
$PY -u $C/ctxrec.py run --variants raw,gc --threads $T --W 10,25,50,100 --t 1.5,2,3 --z 2
