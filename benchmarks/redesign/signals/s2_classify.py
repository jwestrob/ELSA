#!/usr/bin/env python
"""Signal 2, test 1 of ws_signals: does a family-centred residual carry the genome group of origin?

Families are split by token parity. A linear softmax is trained on residuals r_i = x_i - c_F^(-i) of genes from one
half of the families and scores every eligible gene of the other half (cross-fit: no gene is scored by a model that saw
its family). Label sets: fam_jac 0.2 groups (taxonomy-free, primary), GTDB phylum and order (evaluation), fam_jac groups
permuted across genomes (control), and the uncentred vector x (reference) for fam_jac and phylum.
Writes $RD/signals/work/s2_gene.parquet (per eligible gene: f = -log2 p(host fam_jac group), host rank, correctness
per label set), s2_v_famjac_f16.npy (centred log-probabilities over fam_jac groups, rows = eligible genes in gid
order) and s2_test1.json (accuracy, balanced accuracy, information, genome-bootstrap intervals).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PKG, W, EMB, SEED, BLK, log, load_genes, load_tokens, load_groups, boot_weights, ci  # noqa: E402

NTRAIN = 1_200_000
EPOCHS, BATCH, LR, WD = int(os.environ.get("SIG_EPOCHS", "4")), 4096, 1e-3, 1e-4
TAG = os.environ.get("SIG_TAG", "")          # post hoc sensitivity runs write <name><TAG>.*


def resid(X, fam_row, S, m, tok, uid, ix):
    mm = m[tok[ix]].astype(np.float32)[:, None]
    return X[uid[ix]] * (mm / (mm - 1)) - S[fam_row[tok[ix]]] / (mm - 1)


def train_softmax(Z, y, K, seed):
    torch.manual_seed(seed)
    mu = Z.mean(0)
    sd = Z.std(0) + 1e-6
    Zt = torch.from_numpy((Z - mu) / sd)
    yt = torch.from_numpy(y.astype(np.int64))
    lin = torch.nn.Linear(Z.shape[1], K)
    opt = torch.optim.AdamW(lin.parameters(), lr=LR, weight_decay=WD)
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            loss = torch.nn.functional.cross_entropy(lin(Zt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        log(f"   epoch {ep} train CE {tot / n / np.log(2):.3f} bits")
    Wt = lin.weight.detach().numpy() / sd[None, :]
    bt = lin.bias.detach().numpy() - Wt @ mu
    return Wt.astype(np.float32), bt.astype(np.float32)


def logsoftmax(L):
    L = L - L.max(1, keepdims=True)
    return L - np.log(np.exp(L).sum(1, keepdims=True))


def metrics(y, lp_true, pred, K, gcode, prior):
    """held-out metrics + genome bootstrap. lp_true natural-log prob of the true label."""
    G = 3564
    n_g = np.bincount(gcode, minlength=G).astype(np.float64)
    c_g = np.bincount(gcode, weights=(pred == y).astype(np.float64), minlength=G)
    ce_g = np.bincount(gcode, weights=-lp_true / np.log(2), minlength=G)
    Hp = float(-(prior[prior > 0] * np.log2(prior[prior > 0])).sum())
    acc, ce = c_g.sum() / n_g.sum(), ce_g.sum() / n_g.sum()
    rec = np.bincount(y, weights=(pred == y).astype(np.float64), minlength=K) / np.maximum(np.bincount(y, minlength=K), 1)
    present = np.bincount(y, minlength=K) > 0
    Wb = boot_weights(G)
    accb = (Wb @ c_g) / (Wb @ n_g)
    ceb = (Wb @ ce_g) / (Wb @ n_g)
    return dict(n=int(len(y)), classes=int(present.sum()), acc=float(acc), acc_ci=ci(accb), balanced_acc=float(rec[present].mean()),
                majority=float(prior.max()), prior_entropy_bits=Hp, ce_bits=float(ce), ce_ci=ci(ceb),
                info_bits=Hp - float(ce), info_ci=ci(Hp - ceb), acc_over_majority=float(acc / prior.max()))


def main():
    threads = int(os.environ.get("SIG_THREADS", "1"))
    torch.set_num_threads(threads)
    t0 = time.time()
    g = load_genes()
    gm = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome_code", "phylum", "ord"]).sort_values("genome_code")
    tok = load_tokens()
    uid = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    elig = pd.read_parquet(W / "gene_stats.parquet", columns=["eligible"]).eligible.to_numpy(bool)
    X = np.load(W / "zca10_uid_f32.npy")
    fam_row, S, m = np.load(W / "fam_row.npy"), np.load(W / "fam_sum.npy"), np.load(W / "fam_m.npy")
    gc = g.genome_code.to_numpy(np.int64)
    ge = np.flatnonzero(elig)
    half = (tok[ge] % 2).astype(np.int8)
    log("eligible genes", len(ge), "half sizes", np.bincount(half).tolist(), f"{time.time() - t0:.0f}s")

    rng = np.random.default_rng(SEED)
    lab = {}
    lab["famjac"] = load_groups("fam_jac_average_t0.2")
    lab["phylum"] = pd.factorize(gm.phylum)[0]
    lab["order"] = pd.factorize(gm.ord)[0]
    lab["famjac_genomeperm"] = lab["famjac"][rng.permutation(len(lab["famjac"]))]
    K = {k: int(v.max()) + 1 for k, v in lab.items()}
    runs = [("famjac", "r"), ("phylum", "r"), ("order", "r"), ("famjac_genomeperm", "r"), ("famjac", "x"), ("phylum", "x")]

    n_e = len(ge)
    out_cols = {}
    lp_host = {f"{l}_{s}": np.zeros(n_e, np.float32) for l, s in runs}
    pred = {f"{l}_{s}": np.zeros(n_e, np.int32) for l, s in runs}
    V = np.lib.format.open_memmap(W / f"s2_v_famjac_f16{TAG}.npy", mode="w+", dtype=np.float16, shape=(n_e, K["famjac"]))
    host_rank = np.zeros(n_e, np.int16)
    for h in (0, 1):
        tr_pool = np.flatnonzero(half == h)
        tr = np.sort(rng.choice(tr_pool, min(NTRAIN, len(tr_pool)), replace=False))
        te = np.flatnonzero(half != h)
        Zr = resid(X, fam_row, S, m, tok, uid, ge[tr])
        Zx = X[uid[ge[tr]]]
        models = {}
        for l, s in runs:
            log(f"half {h}: train {l} on {s} ({len(tr):,} genes, {K[l]} classes)")
            models[(l, s)] = train_softmax(Zr if s == "r" else Zx, lab[l][gc[ge[tr]]], K[l], SEED + h)
        del Zr, Zx
        for b0 in range(0, len(te), BLK):
            bi = te[b0:b0 + BLK]
            gi = ge[bi]
            Rb = resid(X, fam_row, S, m, tok, uid, gi)
            Xb = X[uid[gi]]
            for (l, s), (Wt, bt) in models.items():
                LP = logsoftmax((Rb if s == "r" else Xb) @ Wt.T + bt)
                y = lab[l][gc[gi]]
                lp_host[f"{l}_{s}"][bi] = LP[np.arange(len(bi)), y]
                pred[f"{l}_{s}"][bi] = LP.argmax(1)
                if (l, s) == ("famjac", "r"):
                    V[bi] = (LP - LP.mean(1, keepdims=True)).astype(np.float16)
                    host_rank[bi] = (LP > LP[np.arange(len(bi)), y][:, None]).sum(1)
        log(f"half {h} scored {len(te):,}", f"{time.time() - t0:.0f}s")
    V.flush()
    res = {}
    for l, s in runs:
        k = f"{l}_{s}"
        y = lab[l][gc[ge]]
        prior = np.bincount(y, minlength=K[l]) / len(y)
        res[k] = metrics(y, lp_host[k], pred[k], K[l], gc[ge], prior)
        ev = gc[ge] % 2 == 0
        res[k + "_even"] = metrics(y[ev], lp_host[k][ev], pred[k][ev], K[l], gc[ge][ev], np.bincount(y[ev], minlength=K[l]) / ev.sum())
        log(k, json.dumps(res[k]))
    df = pd.DataFrame(dict(gid=ge, half=half, f=-lp_host["famjac_r"] / np.log(2), host_rank=host_rank,
                           f_phylum=-lp_host["phylum_r"] / np.log(2),
                           ok_famjac=pred["famjac_r"] == lab["famjac"][gc[ge]],
                           ok_phylum=pred["phylum_r"] == lab["phylum"][gc[ge]]))
    df.to_parquet(W / f"s2_gene{TAG}.parquet", index=False)
    res["seconds"] = time.time() - t0
    json.dump(res, open(W / f"s2_test1{TAG}.json", "w"), indent=1)
    log("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
