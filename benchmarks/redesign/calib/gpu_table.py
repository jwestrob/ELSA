#!/usr/bin/env python
"""ws_calib handover (POST HOC): per-genome best matches with a within-genome z-score, every unique sequence.

For each unique sequence q (2,547,528) and genome G (3,564), on whitened ESM++ small vectors (zca10, float16, unit
norm; one A5000; linear algebra on stored vectors only):
  m1 = max cosine over G's genes, a1 = local index of that gene in G (gid = g_start[G] + a1),
  z1 = (m1 - mean_qG) / sd_qG, mean / sd over all of G's genes.
Stored sparse, in uid order, for cells with z1 >= Z_STORE (environment; default 8):
  $RD/calib/table/counts.npy  int32 per uid (entries of that uid; offsets = cumsum)
  genome.bin int16, local.bin int16, cos.bin float16, z.bin float16 (same order)
Cells of genomes holding q's own sequence are included (self matches, cos 1); drop them by genome when needed.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
R3 = RD / "softrec/r3"
OUT = RD / "calib/table"


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ZS = float(os.environ.get("Z_STORE", "8"))
    QC = int(os.environ.get("QC", "32768"))
    dev = torch.device("cuda")
    vec = np.load(R3 / "vec/zca10_f16.npy", mmap_mode="r")
    meta = np.load(R3 / "vec/meta.npz")
    uq = meta["uq"].astype(np.int64)
    gs, ge = meta["g_start"].astype(np.int64), meta["g_end"].astype(np.int64)
    NG = len(gs)
    U, dim = vec.shape
    info = dict(gpu=torch.cuda.get_device_name(0), U=int(U), NG=int(NG), z_store=ZS, qc=QC,
                slurm_job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename)
    print(json.dumps(info), flush=True)
    Xu = torch.empty((U, dim), dtype=torch.float16, device=dev)
    for s in range(0, U, 200_000):
        Xu[s:s + 200_000] = torch.from_numpy(np.array(vec[s:s + 200_000])).to(dev)
    uq_t = torch.from_numpy(uq).to(dev)
    counts = np.zeros(U, np.int32)
    fh = {k: open(OUT / f"{k}.bin", "wb") for k in ("genome", "local", "cos", "z")}
    total = 0
    t1 = time.time()
    for q0 in range(0, U, QC):
        q1 = min(U, q0 + QC)
        nq = q1 - q0
        Q = Xu[q0:q1]
        M1 = torch.empty((nq, NG), dtype=torch.float16, device=dev)
        Z1 = torch.empty((nq, NG), dtype=torch.float16, device=dev)
        A1 = torch.empty((nq, NG), dtype=torch.int16, device=dev)
        for G in range(NG):
            S = (Q @ Xu[uq_t[gs[G]:ge[G]]].T).float()
            v, i = S.max(1)
            mu = S.mean(1)
            sd = S.std(1).clamp_min(1e-4)
            M1[:, G] = v.half()
            A1[:, G] = i.to(torch.int16)
            Z1[:, G] = ((v - mu) / sd).half()
            del S
        keep = Z1.float() >= ZS
        r, c = keep.nonzero(as_tuple=True)
        fh["genome"].write(c.to(torch.int16).cpu().numpy().tobytes())
        fh["local"].write(A1[r, c].cpu().numpy().tobytes())
        fh["cos"].write(M1[r, c].cpu().numpy().tobytes())
        fh["z"].write(Z1[r, c].cpu().numpy().tobytes())
        counts[q0:q1] = keep.sum(1).int().cpu().numpy()
        total += int(len(r))
        del M1, Z1, A1, keep, r, c
        if (q0 // QC) % 5 == 0 or q1 == U:
            el = time.time() - t1
            print(f"{q1:,}/{U:,} {total:,} entries {el:.0f}s eta {el / q1 * (U - q1):.0f}s", flush=True)
    for f in fh.values():
        f.close()
    np.save(OUT / "counts.npy", counts)
    info.update(entries=int(total), seconds=round(time.time() - t0, 1), seconds_search=round(time.time() - t1, 1),
                entries_per_query_q50_q90_q99=np.quantile(counts, [.5, .9, .99]).tolist(),
                entries_per_query_mean=float(counts.mean()), peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    (OUT / "table.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
