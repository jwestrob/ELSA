#!/usr/bin/env python
"""Check the handover table against the query-sample features (gpu_feat outputs) for 200 R queries. Login-node sized."""
import json
from pathlib import Path

import numpy as np

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
T = RD / "calib/table"
GP = RD / "calib/work/gpu"
cnt = np.load(T / "counts.npy")
off = np.r_[0, np.cumsum(cnt, dtype=np.int64)]
G = np.memmap(T / "genome.bin", np.int16, "r")
Lc = np.memmap(T / "local.bin", np.int16, "r")
c = np.memmap(T / "cos.bin", np.float16, "r")
z = np.memmap(T / "z.bin", np.float16, "r")
Q = np.load(RD / "calib/work/prep/queries.npz")
quid = Q["uid"][:200]
top8 = np.load(GP / "top8.npy", mmap_mode="r")
mean = np.load(GP / "mean.npy", mmap_mode="r")
sd = np.load(GP / "sd.npy", mmap_mode="r")
arg1 = np.load(GP / "arg1.npy", mmap_mode="r")
agree_set, agree_arg, dz, n = 0, 0, [], 0
for j, u in enumerate(quid):
    m1 = top8[j, :, 0].astype(np.float32)
    zz = (m1 - mean[j].astype(np.float32)) / np.maximum(sd[j].astype(np.float32), 1e-4)
    ref = set(np.flatnonzero(zz >= 10.5 + 0.02))
    s, e = off[u], off[u + 1]
    got = G[s:e].astype(int)
    agree_set += len(ref - set(got)) == 0
    sel = np.isin(got, list(ref))
    agree_arg += int((Lc[s:e][sel] == arg1[j, got[sel]]).sum())
    n += int(sel.sum())
    dz.append(np.abs(z[s:e][sel].astype(np.float32) - zz[got[sel]]))
dz = np.concatenate(dz)
out = dict(entries=int(off[-1]), entries_per_uid_q50_mean=[float(np.median(cnt)), float(cnt.mean())],
           share_of_cells=float(off[-1] / (len(cnt) * 3564)), queries_checked=len(quid),
           queries_with_all_ref_cells_present=int(agree_set), same_best_gene=float(agree_arg / max(n, 1)),
           z_abs_diff_q99_max=[float(np.quantile(dz, 0.99)), float(dz.max())])
print(json.dumps(out, indent=1))
(RD / "calib/results/table_check.json").write_text(json.dumps(out, indent=1) + "\n")
