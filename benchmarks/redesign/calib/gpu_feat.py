#!/usr/bin/env python
"""ws_calib GPU features for the query sample (zca10 float16 vectors; linear algebra on stored vectors only).

For every query (unique sequence) x genome (3,564):
  top8      the genome's 8 largest cosines (float16), arg1 = local index of the best gene (int16)
  mean, sd  mean and standard deviation of the query's cosines to all genes of the genome
  cmax/carg best genome-centred cosine (both vectors minus their genome's mean whitened vector, renormalised; the
            query is centred with the mean of its own genome)
  cs10/cs10a, cs50  max_t [2 cos(q, t) - r_k(t)] (CSLS before subtracting r_k(q)), k = 10, 50
  recip     the best gene's own best match in the query's genome carries the query's sequence (reciprocal best)
  rrank     rank of the query's gene among the query genome's genes for the best gene (0 = reciprocal)
Per query: histograms of raw cosines over all 3,585,303 genes (1,500 bins on [-0.5, 1.0)) and of within-genome
z-scores (2,400 bins on [-8, 40)); exact top-2,001 unique sequences (Q1 neighbour lists).
Checks: per-genome maximum of the first 10,000 queries (the softrec3 calibration queries) against softrec3 omax.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
R3 = RD / "softrec/r3"
P = RD / "calib/work/prep"
OUT = RD / "calib/work/gpu"
NB_RAW, LO_RAW, W_RAW = 1500, -0.5, 1000.0
NB_Z, LO_Z, W_Z = 2400, -8.0, 50.0
KTOP = 2001


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    vec = np.load(R3 / "vec/zca10_f16.npy", mmap_mode="r")
    meta = np.load(R3 / "vec/meta.npz")
    uq = meta["uq"].astype(np.int64)
    gs, ge = meta["g_start"].astype(np.int64), meta["g_end"].astype(np.int64)
    NG = len(gs)
    Q = np.load(P / "queries.npz")
    quid, qgen, qgid = Q["uid"], Q["qgenome"].astype(np.int64), Q["qgid"].astype(np.int64)
    NQ = len(quid)
    hub = np.load(P / "hub.npz")
    U, dim = vec.shape
    info = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, NQ=int(NQ), U=int(U), NG=int(NG),
                slurm_job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename)
    print(json.dumps(info), flush=True)
    Xu = torch.empty((U, dim), dtype=torch.float16, device=dev)
    for s in range(0, U, 200_000):
        Xu[s:s + 200_000] = torch.from_numpy(np.ascontiguousarray(vec[s:s + 200_000])).to(dev)
    uq_t = torch.from_numpy(uq).to(dev)
    r10g = torch.from_numpy(hub["r10"][uq]).to(dev)
    r50g = torch.from_numpy(hub["r50"][uq]).to(dev)
    M = torch.zeros((NG, dim), dtype=torch.float32, device=dev)
    for G in range(NG):
        M[G] = Xu[uq_t[gs[G]:ge[G]]].float().mean(0)
    qu_t = torch.from_numpy(quid).to(dev)
    Xq = Xu[qu_t]
    Xqc = torch.nn.functional.normalize(Xq.float() - M[torch.from_numpy(qgen).to(dev)], dim=1).half()
    torch.cuda.synchronize()
    info["seconds_load"] = round(time.time() - t0, 1)
    print("loaded", info["seconds_load"], "s; GPU GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)

    top8 = torch.empty((NQ, NG, 8), dtype=torch.float16, device=dev)
    arg1 = torch.empty((NQ, NG), dtype=torch.int16, device=dev)
    mean = torch.empty((NQ, NG), dtype=torch.float16, device=dev)
    sdv = torch.empty((NQ, NG), dtype=torch.float16, device=dev)
    cmax = torch.empty((NQ, NG), dtype=torch.float16, device=dev)
    carg = torch.empty((NQ, NG), dtype=torch.int16, device=dev)
    cs10 = torch.empty((NQ, NG), dtype=torch.float16, device=dev)
    cs10a = torch.empty((NQ, NG), dtype=torch.int16, device=dev)
    cs50 = torch.empty((NQ, NG), dtype=torch.float16, device=dev)
    hraw = torch.zeros(NQ * NB_RAW, dtype=torch.int64, device=dev)
    hz = torch.zeros(NQ * NB_Z, dtype=torch.int64, device=dev)
    off_raw = (torch.arange(NQ, device=dev, dtype=torch.int64) * NB_RAW)[:, None]
    off_z = (torch.arange(NQ, device=dev, dtype=torch.int64) * NB_Z)[:, None]
    QC = 6000                                     # query chunk inside a genome (bounds temporary memory)
    t1 = time.time()
    for G in range(NG):
        a, b = int(gs[G]), int(ge[G])
        Xg = Xu[uq_t[a:b]]
        Xgc = torch.nn.functional.normalize(Xg.float() - M[G], dim=1).half()
        rr10 = r10g[a:b].float()[None, :]
        rr50 = r50g[a:b].float()[None, :]
        for q0 in range(0, NQ, QC):
            q1 = min(NQ, q0 + QC)
            S = (Xq[q0:q1] @ Xg.T).float()
            v, i = torch.topk(S, min(8, b - a), dim=1)
            if b - a < 8:
                v = torch.cat([v, torch.full((q1 - q0, 8 - (b - a)), -1.0, device=dev)], 1)
            top8[q0:q1, G] = v.half()
            arg1[q0:q1, G] = i[:, 0].to(torch.int16)
            mu = S.mean(1)
            sd = S.std(1).clamp_min(1e-4)
            mean[q0:q1, G] = mu.half()
            sdv[q0:q1, G] = sd.half()
            nq = q1 - q0
            bi = ((S - LO_RAW) * W_RAW).long().clamp_(0, NB_RAW - 1) + off_raw[:nq]
            hraw[q0 * NB_RAW:q1 * NB_RAW] += torch.bincount(bi.view(-1), minlength=nq * NB_RAW)
            del bi
            Z = (S - mu[:, None]) / sd[:, None]
            bz = ((Z - LO_Z) * W_Z).long().clamp_(0, NB_Z - 1) + off_z[:nq]
            del Z
            hz[q0 * NB_Z:q1 * NB_Z] += torch.bincount(bz.view(-1), minlength=nq * NB_Z)
            del bz
            cv, ci = (2 * S - rr10).max(1)
            cs10[q0:q1, G] = cv.half()
            cs10a[q0:q1, G] = ci.to(torch.int16)
            cs50[q0:q1, G] = (2 * S - rr50).max(1).values.half()
            Sc = Xqc[q0:q1] @ Xgc.T
            cv, ci = Sc.float().max(1)
            cmax[q0:q1, G] = cv.half()
            carg[q0:q1, G] = ci.to(torch.int16)
            del S, Sc
        if G % 500 == 0:
            torch.cuda.synchronize()
            print(f"genome {G}/{NG} {time.time() - t1:.0f}s", flush=True)
    torch.cuda.synchronize()
    info["seconds_genome_loop"] = round(time.time() - t1, 1)
    print("genome loop", info["seconds_genome_loop"], flush=True)

    # reciprocal best match in the query's genome
    t2 = time.time()
    recip = torch.zeros((NQ, NG), dtype=torch.bool, device=dev)
    rrank = torch.zeros((NQ, NG), dtype=torch.int16, device=dev)
    gs_t = torch.from_numpy(gs).to(dev)
    for j in range(NQ):
        g = int(qgen[j])
        a, b = int(gs[g]), int(ge[g])
        tg = gs_t + arg1[j].long()
        T = Xu[uq_t[tg]]
        S = (T @ Xu[uq_t[a:b]].T).float()
        best = S.argmax(1)
        recip[j] = uq_t[a + best] == int(quid[j])
        ql = int(qgid[j]) - a
        rrank[j] = (S > S[:, ql:ql + 1]).sum(1).clamp_max(32000).to(torch.int16)
    torch.cuda.synchronize()
    info["seconds_recip"] = round(time.time() - t2, 1)

    # exact top-2,001 unique sequences per query
    t3 = time.time()
    topv = np.empty((NQ, KTOP), np.float16)
    topi = np.empty((NQ, KTOP), np.int32)
    CH = 512
    for q0 in range(0, NQ, CH):
        q1 = min(NQ, q0 + CH)
        S = Xq[q0:q1] @ Xu.T
        v, i = torch.topk(S, KTOP, dim=1)
        topv[q0:q1] = v.cpu().numpy()
        topi[q0:q1] = i.int().cpu().numpy()
        del S
    torch.cuda.synchronize()
    info["seconds_topk"] = round(time.time() - t3, 1)

    # write
    np.save(OUT / "top8.npy", top8.cpu().numpy())
    for name, t in dict(arg1=arg1, mean=mean, sd=sdv, cmax=cmax, carg=carg, cs10=cs10, cs10a=cs10a, cs50=cs50,
                        recip=recip, rrank=rrank).items():
        np.save(OUT / f"{name}.npy", t.cpu().numpy())
    np.save(OUT / "hist_raw.npy", hraw.view(NQ, NB_RAW).int().cpu().numpy())
    np.save(OUT / "hist_z.npy", hz.view(NQ, NB_Z).int().cpu().numpy())
    np.save(OUT / "topv.npy", topv)
    np.save(OUT / "topi.npy", topi)
    # check against softrec3 calibration (first 10,000 queries are its qids, same order)
    qids = np.load(R3 / "calib/qids.npy")
    assert np.array_equal(qids, quid[:len(qids)])
    omax = np.load(R3 / "calib/omax.npy")
    m1 = top8[:len(qids), :, 0].float().cpu().numpy()
    d = np.abs(m1 - omax)
    info["check_omax"] = dict(share_within_1e3=float((d <= 1e-3).mean()), max_abs=float(d.max()),
                              q999_abs=float(np.quantile(d, 0.999)))
    oarg = np.load(R3 / "calib/oarg.npy")
    info["check_oarg_same"] = float((arg1[:len(qids)].cpu().numpy() == oarg).mean())
    info["hist_raw_total_per_query"] = int(hraw.view(NQ, NB_RAW)[0].sum().item())
    info["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    info["seconds"] = round(time.time() - t0, 1)
    (OUT / "gpu.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
