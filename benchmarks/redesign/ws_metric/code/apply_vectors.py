#!/usr/bin/env python
"""Apply one arm's transform to all 4,257,531 genes in gid order.

Output: $RD/ws_metric/vectors/<model>__<arm>.npy (float32, L2-normalized rows; a row of NaN
where the model has no vector) and <model>__<arm>.json (manifest). Written under a
temporary name and renamed.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common import (LEARNED, N_GENES, N_QUERY, N_REF, W, Transforms, code_hash, readers, log,
                    ref_tables, vector_paths)

CH = 262_144


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    if a.arm == "csls2":
        raise SystemExit("csls2 is a pair score, not a vector transform")
    base = W / a.tag
    sdir, hdir = base / "stats" / a.model, base / "heads" / a.model
    tf = Transforms(sdir, hdir, arms={a.arm})
    qv, rv = readers(a.model)
    T = ref_tables()
    probe = tf.apply(a.arm, torch.zeros(2, rv.shape[1]) + 0.01, torch.zeros(2, dtype=torch.long), "t")
    D = probe.shape[1]
    outdir = W / "vectors"
    outdir.mkdir(exist_ok=True)
    name = f"{a.model}__{a.arm}"
    tmp = outdir / f".{name}.tmp.npy"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(N_GENES, D))
    t0 = time.time()
    n_nan = 0
    for side, mat, codes, off in (("ref", rv, T["ref_codes"], 0), ("query", qv, T["query_codes"], N_REF)):
        for c0 in range(0, len(mat), CH):
            c1 = min(c0 + CH, len(mat))
            X = mat.read(c0, c1)
            fin = np.isfinite(X).all(axis=1)
            Y = np.full((c1 - c0, D), np.nan, np.float32)
            if fin.any():
                Xt = torch.from_numpy(X[fin])
                ct = torch.from_numpy(codes[c0:c1][fin])
                Y[fin] = tf.apply(a.arm, Xt, ct, "t").numpy()
            n_nan += int((~fin).sum())
            out[off + c0:off + c1] = Y
        log(side, "done", f"{time.time()-t0:.0f}s")
    out.flush()
    del out
    os.replace(tmp, outdir / f"{name}.npy")
    qp, rp = vector_paths(a.model)
    man = dict(name=name, model=a.model, arm=a.arm, arm_config=LEARNED.get(a.arm), shape=[N_GENES, D], dtype="float32",
               order="gid (reference row_index 0..4,190,382, then 4,190,383 + query row_index)",
               l2_normalized=True, rows_nan=n_nan, input_query=qp, input_reference=rp,
               stats=str(sdir / "stats.npz"), head=str(hdir / f"{a.arm}.pt") if a.arm in LEARNED else None,
               code=code_hash(__file__, Path(__file__).with_name("common.py")),
               seconds=round(time.time() - t0, 1), date=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmpj = outdir / f".{name}.json.tmp"
    json.dump(man, open(tmpj, "w"), indent=1)
    os.replace(tmpj, outdir / f"{name}.json")
    log("wrote", outdir / f"{name}.npy", man["shape"], "nan rows", n_nan)


if __name__ == "__main__":
    main()
