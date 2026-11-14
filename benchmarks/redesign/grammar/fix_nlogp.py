#!/usr/bin/env python
"""Patch for prep.py (2026-11-12): training-only -log P (nlogP_train) was looked up with keys encoded on the
global token base T while colocalization() encodes on (max token present + 1). Recompute nlogP_train for every
split with the corrected encoding, report agreement with the stored values, and overwrite the array."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/engine/code")
from recurrence import Universe, colocalization, pair_statistics  # noqa: E402
from pathlib import Path
import numba
numba.set_num_threads(int(sys.argv[1]))
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PREP = RD / "grammar/prep"
u = Universe(RD / "ws_dpann/package/universe")
meta = pd.read_parquet(PREP / "genes_meta.parquet")
tok = meta.token.to_numpy(np.int64)
gc = meta.genome_code.to_numpy(np.int64)
T = int(tok.max()) + 1
adj = pd.read_parquet(PREP / "adj1.parquet", columns=["i", "j"])
i, j = adj.i.to_numpy(), adj.j.to_numpy()
k1 = np.minimum(tok[i], tok[j]) * T + np.maximum(tok[i], tok[j])
for s in ["cv0", "cv1", "cv2", "cv3", "cv4", "phylum"]:
    role = meta[f"role_{s}"].to_numpy()
    tok_s = np.where(role != 2, tok, -1)
    ks, ls, ds, Ts = colocalization(u, tok_s, 3)
    sts = pair_statistics(u, tok_s, ks, ls, 3, Ts)
    kg = (sts["key"] // Ts) * T + sts["key"] % Ts
    o = np.argsort(kg)
    kg, lp = kg[o], sts["logP"][o]
    p = np.minimum(np.searchsorted(kg, k1), len(kg) - 1)
    okk = (kg[p] == k1) & (tok[i] != tok[j])
    new = np.where(okk, -lp[p], 0.0).astype(np.float32)
    f = PREP / f"base_{s}.npz"
    b = dict(np.load(f))
    old = b["nlogP_train"]
    print(s, "T", T, "Ts", Ts, "nonzero old/new", int((old != 0).sum()), int((new != 0).sum()),
          "max |old-new|", float(np.abs(old - new).max()), flush=True)
    b["nlogP_train"] = new
    tmp = PREP / f".base_{s}.tmp.npz"
    np.savez(tmp, **b)
    tmp.rename(f)
print("done")
