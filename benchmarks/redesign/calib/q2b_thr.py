#!/usr/bin/env python
"""POST HOC: thresholds of raw cosine and within-genome z for the handover table, from the Q2b labels (R queries).
Odd and even genomes separately; recall of HOM / alignable best matches at each threshold. Writes results/q2b_thr.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q2b  # noqa: E402
from q1 import evidence  # noqa: E402


def main():
    import numba
    numba.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]))
    meta, Q, jR, quid, t1, QG, NG = q2b.base()
    A = dict(np.load(q2b.P / "ann.npz"))
    aln = pd.read_parquet(q2b.W / "aln.parquet", columns=["query", "target", "evalue"])
    akey = aln["query"].to_numpy(np.int64) * 4_000_000 + aln["target"].to_numpy(np.int64)
    o = np.argsort(akey)
    akey, aev = akey[o], aln.evalue.to_numpy()[o]
    ckey = quid[:, None].astype(np.int64) * 4_000_000 + t1
    pos = np.clip(np.searchsorted(akey, ckey), 0, len(akey) - 1)
    E = np.where(akey[pos] == ckey, aev[pos], np.inf)
    qf = np.broadcast_to(quid[:, None], t1.shape).ravel().astype(np.int64)
    ev = np.zeros((len(qf), 7), np.bool_)
    evidence(qf, t1.ravel().astype(np.int64), A["pfam_ptr"], A["pfam_val"], A["grp_ptr"], A["grp_val"], A["prof_ptr"],
             A["prof_val"], A["pdb_ptr"], A["pdb_val"], A["cdd_ptr"], A["cdd_val"], A["role_ptr"], A["role_val"], ev)
    ev = ev.reshape(len(jR), NG, 7)
    annP = ev[:, :, :4].any(2)
    ALN = E <= 1e-3
    HOM = ALN | annP
    NEGC = ev[:, :, 5] & ev[:, :, 6] & ~annP & (E > 1)
    valid = ~QG
    top = np.load(q2b.GP / "top8.npy", mmap_mode="r")[jR][:, :, 0].astype(np.float32)
    mean = np.load(q2b.GP / "mean.npy", mmap_mode="r")[jR].astype(np.float32)
    sd = np.load(q2b.GP / "sd.npy", mmap_mode="r")[jR].astype(np.float32)
    z = (top - mean) / np.maximum(sd, 1e-4)
    out = {}
    for par, pm in (("odd", np.arange(NG) % 2 == 1), ("even", np.arange(NG) % 2 == 0)):
        v = valid & pm[None, :]
        for name, s in (("cos", top), ("z", z)):
            neg = s[v & NEGC]
            for x in (1e-2, 1e-3, 1e-4):
                th = float(np.quantile(neg, 1 - x, method="higher"))
                out[f"{par}_{name}_{x:g}"] = dict(threshold=th, recall_HOM=float((s[v & HOM] >= th).mean()),
                                                  recall_ALN=float((s[v & ALN] >= th).mean()),
                                                  share_all_cells=float((s[v] >= th).mean()),
                                                  precision_HOM_vs_NEGC=float((s[v & HOM] >= th).sum() / max(1, (s[v & (HOM | NEGC)] >= th).sum())))
    for zf in (5, 8, 10):
        out[f"share_cells_z_ge_{zf}"] = float((z[valid] >= zf).mean())
    Path(q2b.RES / "q2b_thr.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
