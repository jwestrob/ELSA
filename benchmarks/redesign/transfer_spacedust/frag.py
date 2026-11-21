"""Fragmentation of gold regions per output: union vs best single interval coverage (per-region tables). usage: frag.py NAME..."""
import pandas as pd, numpy as np, sys
R = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust/results/"
for n in sys.argv[1:]:
    d = pd.read_parquet(R + "pr_" + n + ".parquet")
    big = d["size"] >= 11
    r = (d.block_span / d["size"]).to_numpy()
    print("%-22s union cov q50 all %.3f (>=11 genes %.3f); best-interval cov q50 %.3f (>=11 %.3f); best span/size q50 %.3f"
          % (n, d.union_coverage.median(), d.union_coverage[big].median(), d.coverage.median(), d.coverage[big].median(), np.median(r)))
