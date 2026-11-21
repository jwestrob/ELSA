#!/usr/bin/env python
"""softrec3: exact best match per genome for every unique sequence (zca10 vectors, float16 matrix products, one GPU).

For each unique query U (2,547,528) and genome G (3,564): the maximum cosine over G's genes, the gene that attains it
(local index within G) and the minimum cosine (decoy: -min is a per-genome maximum under sign reversal). Entries with
max >= the storage floor (cosine 0.32) are written sparse, in query order:
  exact/counts.npy (int32 per query), exact/genome.bin (int16), exact/local.bin (int16), exact/value.bin (float16)
plus per-genome histograms (1,000 bins on [0, 1)) of the maximum and of the decoy maximum over all queries.
Nearest-neighbour search on stored vectors; no language-model inference.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

R3 = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/softrec/r3")


def main():
    t0 = time.time()
    QC = int(os.environ.get("QC", "32768"))
    out = R3 / "exact"
    out.mkdir(parents=True, exist_ok=True)
    vec = np.load(R3 / "vec/zca10_f16.npy", mmap_mode="r")
    meta = np.load(R3 / "vec/meta.npz")
    uq = meta["uq"].astype(np.int64)
    gs, ge = meta["g_start"].astype(np.int64), meta["g_end"].astype(np.int64)
    # storage floor: one global cosine (0.32, the brief's start value and the loosest variant floor)
    STORE = float(os.environ.get("STORE_FLOOR", "0.32"))
    NGq = len(meta["g_start"])
    floors = np.full(NGq, STORE, np.float32)
    U, dim = vec.shape
    NG = len(gs)
    dev = torch.device("cuda")
    info = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, U=int(U), dim=int(dim), genomes=int(NG),
                genes=int(len(uq)), query_chunk=QC, store_floor=STORE,
                slurm_job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename)
    print(json.dumps(info), flush=True)
    Xu = torch.empty((U, dim), dtype=torch.float16, device=dev)
    for s in range(0, U, 200_000):
        Xu[s:s + 200_000] = torch.from_numpy(np.ascontiguousarray(vec[s:s + 200_000])).to(dev)
    Xg = Xu[torch.from_numpy(uq).to(dev)]
    fl = torch.from_numpy(floors).to(dev)
    torch.cuda.synchronize()
    info["seconds_load"] = round(time.time() - t0, 1)
    print("loaded", info["seconds_load"], "s; GPU memory GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
    hmax = torch.zeros(NG * 1000, dtype=torch.int64, device=dev)
    hdec = torch.zeros(NG * 1000, dtype=torch.int64, device=dev)
    gcol = torch.arange(NG, device=dev, dtype=torch.int64)[None, :] * 1000
    counts = np.zeros(U, np.int32)
    fg = open(out / "genome.bin", "wb")
    fi = open(out / "local.bin", "wb")
    fv = open(out / "value.bin", "wb")
    total = 0
    t1 = time.time()
    for q0 in range(0, U, QC):
        q1 = min(U, q0 + QC)
        nq = q1 - q0
        Q = Xu[q0:q1]
        V = torch.empty((nq, NG), dtype=torch.float16, device=dev)
        IX = torch.empty((nq, NG), dtype=torch.int16, device=dev)
        MN = torch.empty((nq, NG), dtype=torch.float16, device=dev)
        for G in range(NG):
            S = Q @ Xg[gs[G]:ge[G]].T
            v, i = S.max(dim=1)
            V[:, G] = v
            IX[:, G] = i.to(torch.int16)
            MN[:, G] = S.amin(dim=1)
        Vf = V.float()
        mask = Vf >= fl[None, :]
        cnt = mask.sum(1)
        r, c = mask.nonzero(as_tuple=True)
        g16 = c.to(torch.int16).cpu().numpy()
        i16 = IX[r, c].cpu().numpy()
        v16 = V[r, c].cpu().numpy()
        g16.tofile(fg)
        i16.tofile(fi)
        v16.tofile(fv)
        counts[q0:q1] = cnt.int().cpu().numpy()
        total += len(g16)
        bm = (Vf * 1000).long().clamp_(0, 999) + gcol
        hmax += torch.bincount(bm.ravel(), minlength=NG * 1000)
        bd = (-MN.float() * 1000).long().clamp_(0, 999) + gcol
        hdec += torch.bincount(bd.ravel(), minlength=NG * 1000)
        del V, IX, MN, Vf, mask, bm, bd
        if (q0 // QC) % 5 == 0 or q1 == U:
            el = time.time() - t1
            print(f"{q1:,}/{U:,} queries, {total:,} entries, {el:.0f}s, eta {el / q1 * (U - q1):.0f}s", flush=True)
    fg.close()
    fi.close()
    fv.close()
    np.save(out / "counts.npy", counts)
    np.save(out / "hist_max.npy", hmax.view(NG, 1000).cpu().numpy())
    np.save(out / "hist_decoy.npy", hdec.view(NG, 1000).cpu().numpy())
    info.update(seconds_search=round(time.time() - t1, 1), seconds=round(time.time() - t0, 1), entries=int(total),
                entries_per_query_q50_q90_q99=np.quantile(counts, [.5, .9, .99]).tolist(), entries_per_query_mean=float(counts.mean()),
                peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    (out / "exact.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
