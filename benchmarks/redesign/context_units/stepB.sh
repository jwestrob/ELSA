# genome-centred variant and its shuffled control
$PY -u $C/ctxrec.py run --variants gc,shuf_gc --threads $T --W 10,25,50,100 --t 1.5,2,3 --t-shuf 2
