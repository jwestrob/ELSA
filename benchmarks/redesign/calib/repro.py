#!/usr/bin/env python
"""Reproduce the stored round-3 calibration line (family-absent per-genome maxima) from $RD/softrec/r3/calib.
Login-node sized (< 10 s, one thread). Writes $RD/calib/results/repro.json."""
import json
import numpy as np
from pathlib import Path
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
C = RD / "softrec/r3/calib"
omax = np.load(C / "omax.npy"); famp = np.load(C / "famp.npy"); qids = np.load(C / "qids.npy")
meta = np.load(RD / "softrec/r3/vec/meta.npz")
uq = meta["uq"]; lin_g = meta["lin_of_genome"]
gs = meta["g_start"]
gc = np.repeat(np.arange(len(gs)), np.diff(np.r_[gs, len(uq)]))
_, first = np.unique(uq, return_index=True)
qlin = lin_g[gc[first[qids]]]
qgen = gc[first[qids]]
other = lin_g[None, :] != qlin[:, None]
notown = np.arange(len(gs))[None, :] != qgen[:, None]
res = {}
for name, m in [("all_genomes", np.ones_like(famp)), ("outside_own_genus", other), ("outside_own_genome", notown)]:
    a = m & ~famp
    p = m & famp
    res[name] = {f"{c:g}": dict(absent=float((omax[a] >= c).mean()), present=float((omax[p] >= c).mean()))
                 for c in (0.30, 0.32, 0.40, 0.45, 0.50, 0.60, 0.70)}
res["n_queries"] = int(len(qids))
print(json.dumps(res, indent=1))
(RD / "calib/results/repro.json").write_text(json.dumps(res, indent=1) + "\n")
