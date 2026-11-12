#!/usr/bin/env python
"""Token purity of gold members per arm: for member genes of a caller, the share that are singletons,
the mean token size, and the mean share of token-mates that are members of the same caller and system
type. usage: purity.py out.tsv <token parquet> [...]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/package")
mem = pd.read_parquet(PKG / "gold/members.parquet", columns=["instance_id", "caller", "gid"])
ins = pd.read_parquet(PKG / "gold/instances.parquet", columns=["instance_id", "system_type"])
mem = mem.merge(ins, on="instance_id")[["gid", "caller", "system_type"]].drop_duplicates(["gid", "caller"])
rows = []
for f in sys.argv[2:]:
    tok = pd.read_parquet(f, columns=["token"]).token.to_numpy()
    _, t = np.unique(tok, return_inverse=True)
    size = np.bincount(t)
    for caller in ("TXSScan", "DefenseFinder"):
        m = mem[mem.caller == caller]
        g = m.gid.to_numpy()
        key = pd.Series(list(zip(t[g], m.system_type.to_numpy())))
        same = key.map(key.value_counts()).to_numpy() - 1
        mates = size[t[g]] - 1
        has = mates > 0
        rows.append(dict(arm=Path(f).stem, caller=caller, members=len(g), singleton_frac=round(float((~has).mean()), 4),
                         mean_token_size=round(float(size[t[g]].mean()), 1),
                         mean_same_type_share_of_mates=round(float((same[has] / mates[has]).mean()), 4)))
df = pd.DataFrame(rows)
df.to_csv(sys.argv[1], sep="\t", index=False)
print(df.to_string(index=False))
