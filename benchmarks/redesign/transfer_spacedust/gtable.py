#!/usr/bin/env python
"""Per-genome best matches for every unique sequence of the Spacedust collection (calib/gpu_table.py, copied), plus the
genome x genome sums behind lineagefree's embedding similarities (genome_groups.py embsim definitions).

For each unique sequence q and genome G, on this collection's zca10 vectors (float16, unit norm; linear algebra on stored
vectors only, no language model):
  m1 = max cosine over G's genes, a1 = local index of that gene, z1 = (m1 - mean_qG) / sd_qG over all of G's genes.
Stored sparse in uid order where z1 >= Z_STORE (default 10.5): gtable/{counts.npy, genome.bin, local.bin, cos.bin, z.bin}
(format of calib/table_note/FORMAT.txt). Genome-pair sums over A's genes (gene multiplicity of q in A), with v = m1(q, B):
  s_all / n_all  v >= 0.32 (the storage floor of the DPANN exact table; emb_mean counts missing entries as 0.32)
  s_hom / n_hom  v >= 0.5   (emb_hom = s_hom / n_hom)
  n_id           v >= 0.95
written to gtable/sums.npz. Runs on CUDA when available (gpu partition), else on CPU torch (float32).
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
TS = RD / "transfer_spacedust"
E = TS / "emb"
OUT = TS / "gtable"


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ZS = float(os.environ.get("Z_STORE", "10.5"))
    QC = int(os.environ.get("QC", "32768"))
    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    dt = torch.float16 if cuda else torch.float32
    if not cuda:
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_ON_NODE", "4")))
    vec = np.load(E / ("zca10_f16__uid.npy" if cuda else "zca10_f32__uid.npy"), mmap_mode="r")
    meta = np.load(E / "meta.npz")
    uq = meta["uq"].astype(np.int64)
    gs, ge = meta["g_start"].astype(np.int64), meta["g_end"].astype(np.int64)
    NG = len(gs)
    U, dim = vec.shape
    gen_of_gene = np.repeat(np.arange(NG), ge - gs)
    # gene multiplicity of each uid in each genome (CSR over uids)
    key = np.unique(uq * NG + gen_of_gene, return_counts=True)
    k_uid, k_gen, k_cnt = key[0] // NG, key[0] % NG, key[1]
    kptr = np.searchsorted(k_uid, np.arange(U + 1))
    info = dict(device=(torch.cuda.get_device_name(0) if cuda else "cpu"), U=int(U), NG=int(NG), z_store=ZS, qc=QC,
                slurm_job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename)
    print(json.dumps(info), flush=True)
    Xu = torch.empty((U, dim), dtype=dt, device=dev)
    for s in range(0, U, 200_000):
        Xu[s:s + 200_000] = torch.from_numpy(np.array(vec[s:s + 200_000])).to(dev)
    uq_t = torch.from_numpy(uq).to(dev)
    counts = np.zeros(U, np.int32)
    fh = {k: open(OUT / f"{k}.bin", "wb") for k in ("genome", "local", "cos", "z")}
    acc = {k: torch.zeros((NG, NG), dtype=torch.float64, device=dev) for k in ("s_all", "n_all", "s_hom", "n_hom", "n_id")}
    total = 0
    t1 = time.time()
    for q0 in range(0, U, QC):
        q1 = min(U, q0 + QC)
        nq = q1 - q0
        Q = Xu[q0:q1]
        M1 = torch.empty((nq, NG), dtype=torch.float32, device=dev)
        Z1 = torch.empty((nq, NG), dtype=torch.float16, device=dev)
        A1 = torch.empty((nq, NG), dtype=torch.int16, device=dev)
        for G in range(NG):
            S = (Q @ Xu[uq_t[gs[G]:ge[G]]].T).float()
            v, i = S.max(1)
            mu = S.mean(1)
            sd = S.std(1).clamp_min(1e-4)
            M1[:, G] = v.half().float()
            A1[:, G] = i.to(torch.int16)
            Z1[:, G] = ((v - mu) / sd).half()
            del S
        keep = Z1.float() >= ZS
        r, c = keep.nonzero(as_tuple=True)
        fh["genome"].write(c.to(torch.int16).cpu().numpy().tobytes())
        fh["local"].write(A1[r, c].cpu().numpy().tobytes())
        fh["cos"].write(M1[r, c].half().cpu().numpy().tobytes())
        fh["z"].write(Z1[r, c].cpu().numpy().tobytes())
        counts[q0:q1] = keep.sum(1).int().cpu().numpy()
        total += int(len(r))
        # genome-pair sums: C (nq x NG) gene multiplicities of each query uid in each genome A
        a, b = kptr[q0], kptr[q1]
        C = torch.zeros((nq, NG), dtype=torch.float32, device=dev)
        C[torch.from_numpy(k_uid[a:b] - q0).to(dev), torch.from_numpy(k_gen[a:b]).to(dev)] = \
            torch.from_numpy(k_cnt[a:b].astype(np.float32)).to(dev)
        Ct = C.T.contiguous()
        for name, mask in (("all", M1 >= 0.32), ("hom", M1 >= 0.5)):
            mf = mask.float()
            acc[f"s_{name}"] += (Ct @ (M1 * mf)).double()
            acc[f"n_{name}"] += (Ct @ mf).double()
        acc["n_id"] += (Ct @ (M1 >= 0.95).float()).double()
        del M1, Z1, A1, keep, r, c, C, Ct
        if (q0 // QC) % 10 == 0 or q1 == U:
            el = time.time() - t1
            print(f"{q1:,}/{U:,} {total:,} entries {el:.0f}s eta {el / q1 * (U - q1):.0f}s", flush=True)
    for f in fh.values():
        f.close()
    np.save(OUT / "counts.npy", counts)
    np.savez(OUT / "sums.npz", **{k: v.cpu().numpy() for k, v in acc.items()})
    info.update(entries=int(total), seconds=round(time.time() - t0, 1), seconds_search=round(time.time() - t1, 1),
                entries_per_query_q50_q90_q99=np.quantile(counts, [.5, .9, .99]).tolist(),
                entries_per_query_mean=float(counts.mean()),
                peak_gpu_gb=(round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None))
    (OUT / "table.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
