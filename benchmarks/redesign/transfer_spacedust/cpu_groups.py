#!/usr/bin/env python
"""CPU fallback for the genome-pair embedding similarity behind the P1 groups (used only while the gpu partition is held).

emb_hom(A, B) = mean over A's genes whose best zca10 match in B is >= 0.5 of that best cosine (genome_groups.py embsim).
Exact needs every gene x every gene (~3.5e16 flop, ~9 h on one Broadwell node). Here:
  stage 1  estimate for every directed pair from S genes per genome (uniform sample, seed 17; all genes if fewer), exact
           per-genome maxima over ALL of B's genes for each sampled query (float32);
  stage 2  exact emb_hom (all of A's genes against all of B's genes, both directions) for every genome pair whose
           stage-1 symmetrised estimate is >= LO (default 0.65, ten points below the cut);
  output   groups/sums_cpu.npz-equivalent matrices: emb_hom.npy etc. as emb.py groups expects (gtable/sums.npz with the
           exact sums where computed and the scaled sample sums elsewhere), flagged in gtable/table.json.
usage: cpu_groups.py --threads T [--sample 300] [--lo 0.65]
"""
import argparse
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


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def maxes(Q, X, gs, ge, genes_uid):
    """per-genome max cosine of each query row against all genes of every genome (float32)."""
    NG = len(gs)
    M = torch.empty((Q.shape[0], NG), dtype=torch.float32)
    for G in range(NG):
        S = Q @ X[genes_uid[gs[G]:ge[G]]].T
        M[:, G] = S.max(1).values
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--lo", type=float, default=0.65)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    X = torch.from_numpy(np.load(E / "zca10_f32__uid.npy"))
    meta = np.load(E / "meta.npz")
    uq = meta["uq"].astype(np.int64)
    gs, ge = meta["g_start"].astype(np.int64), meta["g_end"].astype(np.int64)
    NG = len(gs)
    uq_t = torch.from_numpy(uq)
    rng = np.random.default_rng(17)
    # ---- stage 1: sampled genes of every genome
    samp = [np.sort(rng.choice(np.arange(s, e), min(a.sample, e - s), replace=False)) for s, e in zip(gs, ge)]
    sg = np.concatenate(samp)
    owner = np.repeat(np.arange(NG), [len(x) for x in samp])
    s_hom = np.zeros((NG, NG))
    n_hom = np.zeros((NG, NG))
    s_all = np.zeros((NG, NG))
    n_all = np.zeros((NG, NG))
    n_id = np.zeros((NG, NG))
    QC = 4096
    t1 = time.time()
    for q0 in range(0, len(sg), QC):
        q = sg[q0:q0 + QC]
        M = maxes(X[uq_t[torch.from_numpy(q)]], X, gs, ge, uq_t).numpy()
        o = owner[q0:q0 + QC]
        for th, sm, nm in ((0.5, s_hom, n_hom), (0.32, s_all, n_all)):
            mk = M >= th
            np.add.at(sm, o, M * mk)
            np.add.at(nm, o, mk)
        np.add.at(n_id, o, M >= 0.95)
        if (q0 // QC) % 10 == 0:
            el = time.time() - t1
            log(f"stage 1 {q0 + len(q):,}/{len(sg):,} {el:.0f}s eta {el / (q0 + len(q)) * (len(sg) - q0 - len(q)):.0f}s")
    ns = np.array([len(x) for x in samp], np.float64)
    n_g = (ge - gs).astype(np.float64)
    scale = (n_g / ns)[:, None]
    est = np.where(n_hom > 0, s_hom / np.maximum(n_hom, 1), 0.32)
    Z = 0.5 * (est + est.T)
    np.fill_diagonal(Z, 0)
    iu = np.argwhere(np.triu(Z >= a.lo, 1))
    log("stage 1 done", f"{time.time()-t1:.0f}s", "pairs for exact stage 2:", len(iu))
    s_all, n_all, s_id = s_all * scale, n_all * scale, n_id * scale
    s_hom_x, n_hom_x = s_hom * scale, n_hom * scale
    exact = np.zeros((NG, NG), bool)
    t2 = time.time()
    for k, (A, B) in enumerate(iu):
        for P, R in ((A, B), (B, A)):
            Q = X[uq_t[gs[P]:ge[P]]]
            m = (Q @ X[uq_t[gs[R]:ge[R]]].T).max(1).values.numpy()
            s_hom_x[P, R] = float(m[m >= 0.5].sum())
            n_hom_x[P, R] = float((m >= 0.5).sum())
            s_all[P, R] = float(m[m >= 0.32].sum())
            n_all[P, R] = float((m >= 0.32).sum())
            s_id[P, R] = float((m >= 0.95).sum())
            exact[P, R] = True
        if k % 2000 == 0:
            log(f"stage 2 {k:,}/{len(iu):,} {time.time()-t2:.0f}s")
    np.savez(OUT / "sums.npz", s_all=s_all, n_all=n_all, s_hom=s_hom_x, n_hom=n_hom_x, n_id=s_id)
    np.save(OUT / "exact_pairs.npy", exact)
    info = dict(method="cpu fallback: sampled stage 1 + exact stage 2", sample_per_genome=a.sample, lo=a.lo,
                sampled_genes=int(len(sg)), exact_pairs=int(len(iu)), threads=a.threads,
                seconds_stage1=round(t2 - t1, 1), seconds_stage2=round(time.time() - t2, 1), seconds=round(time.time() - t0, 1),
                note="z table not produced (P3 needs the GPU pass)")
    (OUT / "table.json").write_text(json.dumps(info, indent=1) + "\n")
    log("cpu groups", info)


if __name__ == "__main__":
    main()
