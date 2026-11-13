#!/usr/bin/env python
"""L4b+ locus tables: every node of the L4b tree (no recurrence test). usage: mkplus.py TAG [TAG ...]"""
import sys
from pathlib import Path
import pandas as pd
LL = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/long_loci")
for tag in sys.argv[1:]:
    for f in sorted((LL / "nodes" / tag).glob("obs4b_t*.parquet")):
        th = f.stem.split("_t")[1]
        t = pd.read_parquet(f, columns=["lo", "hi"])
        t = t[t.hi > t.lo].rename(columns={"lo": "s", "hi": "e"}).drop_duplicates().sort_values(["s", "e"])
        t.to_parquet(LL / "loci" / tag / f"L4bp_t{th}.parquet", index=False)
        print(tag, th, len(t), flush=True)
