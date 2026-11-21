#!/usr/bin/env python
"""Workstream head, step 2: train the declared student configurations on training-genome sequences, choose one on
validation genomes (predeclared.json), then evaluate homolog retrieval on even (test) genomes and write the chosen
student's vectors and its exact k-nearest-neighbour lists.

  head_train.py --data DIR --out DIR [--device cuda|cpu] [--batch 4096] [--steps 12000] [--eval-every 1500]

Retrieval metric (validation and test): for each query sequence, every other unique sequence is ranked by cosine;
AP@2000 = (1 / min(n_aligned, 2000)) * sum over aligned sequences in the top 2,000 of precision at their rank.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIGS = {
    "A_mlp256_full": dict(arch="mlp", out=256, fam=True, hard=True, weighted=True),
    "B_linear256_full": dict(arch="linear", out=256, fam=True, hard=True, weighted=True),
    "C_mlp960_full": dict(arch="mlp_id", out=960, fam=True, hard=True, weighted=True),
    "D_mlp256_nohard": dict(arch="mlp", out=256, fam=True, hard=False, weighted=True),
    "E_mlp256_nofam": dict(arch="mlp", out=256, fam=False, hard=True, weighted=True),
    "F_mlp256_unweighted": dict(arch="mlp", out=256, fam=True, hard=True, weighted=False),
    "G_mlp256_minimal": dict(arch="mlp", out=256, fam=False, hard=False, weighted=False),
}
SIMPLICITY = {"linear": 0, "mlp": 1, "mlp_id": 2}
TAU = 0.05
P_FAM = 0.3
K_AP = 2000
TIE = 0.002


def stamp(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Head(nn.Module):
    def __init__(self, arch, d_in=960, d_hid=1024, d_out=256):
        super().__init__()
        self.arch = arch
        self.scale = math.sqrt(d_in)                 # zca10 rows are unit vectors; unit-variance coordinates
        if arch == "linear":
            self.lin = nn.Linear(d_in, d_out, bias=False)
        else:
            self.l1 = nn.Linear(d_in, d_hid)
            self.ln = nn.LayerNorm(d_hid)
            self.l2 = nn.Linear(d_hid, d_out)
            if arch == "mlp":
                self.skip = nn.Linear(d_in, d_out, bias=False)
            else:                                    # identity skip: starts at the raw vector
                assert d_out == d_in
                self.skip = None
                nn.init.zeros_(self.l2.weight)
                nn.init.zeros_(self.l2.bias)

    def forward(self, x):
        x = x.float()
        if self.arch == "linear":
            y = self.lin(x)
        else:
            h = self.l2(F.gelu(self.ln(self.l1(x * self.scale))))
            y = h / self.scale + (self.skip(x) if self.skip is not None else x)
        return F.normalize(y.float(), dim=-1)


# =============================================================================================== data
class Data:
    def __init__(self, d: Path, dev, load_l2=False):
        t0 = time.time()
        self.dev = dev
        self.X = self.load_rows(d / "zca10_f16__uid.npy")
        self.U = self.X.shape[0]
        s = np.load(d / "split.npz")
        self.fam = torch.from_numpy(s["fam"].astype(np.int64)).to(dev)
        p = np.load(d / "pos_csr.npz")
        self.ptr = torch.from_numpy(p["indptr"]).to(dev)
        self.tgt = torch.from_numpy(p["tgt"].astype(np.int64)).to(dev)
        self.cw = torch.from_numpy(np.cumsum(p["w"].astype(np.float64))).to(dev)
        self.n_aln = (self.ptr[1:] - self.ptr[:-1])
        f = np.load(d / "fam.npz")
        self.fmem = torch.from_numpy(f["members"].astype(np.int64)).to(dev)
        self.fstart = torch.from_numpy(f["start"]).to(dev)
        self.fn = torch.from_numpy(f["n"].astype(np.int64)).to(dev)
        self.fpos = torch.from_numpy(f["pos"].astype(np.int64)).to(dev)
        hn = np.load(d / "hardneg.npy")
        self.hn = torch.from_numpy(hn.astype(np.int64)).to(dev)
        self.nh = (self.hn >= 0).sum(1)
        anc = torch.from_numpy(np.load(d / "anchors.npy").astype(np.int64)).to(dev)
        self.pool_all = anc
        self.pool_aln = anc[self.n_aln[anc] > 0]
        self.ev = {k: dict(np.load(d / f"eval_{k}.npz")) for k in ("val", "test")}
        self.L2 = self.load_rows(d / "l2_f16__uid.npy") if load_l2 else None
        stamp(f"data: {self.U:,} sequences, {len(self.tgt):,} positive slots, anchors {len(anc):,} "
              f"(with alignment {len(self.pool_aln):,}), hard-negative anchors {(self.nh[anc] > 0).sum().item():,}, "
              f"{time.time() - t0:.0f}s")

    def load_rows(self, path, chunk=200_000):
        A = np.load(path, mmap_mode="r")
        dt_ = torch.float16 if self.dev.type == "cuda" else torch.float32     # CPU: no half-precision conversions
        out = torch.empty(A.shape, dtype=dt_, device=self.dev)
        for s in range(0, A.shape[0], chunk):
            out[s:s + chunk] = torch.from_numpy(np.ascontiguousarray(A[s:s + chunk])).to(self.dev)
        return out

    def batch(self, cfg, B, gen):
        dev = self.dev
        pool = self.pool_all if cfg["fam"] else self.pool_aln
        a = pool[torch.randint(0, len(pool), (B,), device=dev, generator=gen)]
        s, e = self.ptr[a], self.ptr[a + 1]
        n = e - s
        has_aln = n > 0
        u1 = torch.rand(B, device=dev, generator=gen, dtype=torch.float64)
        if cfg["weighted"]:
            lo = torch.where(s > 0, self.cw[(s - 1).clamp(min=0)], torch.zeros_like(u1))
            hi = self.cw[(e - 1).clamp(min=0)]
            idx = torch.searchsorted(self.cw, lo + u1 * (hi - lo), right=True)
            idx = torch.minimum(torch.maximum(idx, s), (e - 1).clamp(min=0))
        else:
            idx = torch.minimum(s + (u1 * n).long(), (e - 1).clamp(min=0))
        p = self.tgt[idx.clamp(0, len(self.tgt) - 1)]
        if cfg["fam"]:
            fn = self.fn[a]
            has_fam = fn >= 2
            u2 = torch.rand(B, device=dev, generator=gen)
            r = (u2 * (fn - 1).clamp(min=1)).long()
            r = r + (r >= self.fpos[a]).long()
            pf = self.fmem[(self.fstart[a] + r).clamp(0, len(self.fmem) - 1)]
            use = has_fam & (~has_aln | (torch.rand(B, device=dev, generator=gen) < P_FAM))
            p = torch.where(use, pf, p)
        rnd = pool[torch.randint(0, len(pool), (B,), device=dev, generator=gen)]
        if cfg["hard"]:
            nh = self.nh[a]
            j = (torch.rand(B, device=dev, generator=gen) * nh.clamp(min=1)).long().clamp(max=self.hn.shape[1] - 1)
            h = torch.where(nh > 0, self.hn[a, j], rnd)
        else:
            h = rnd
        return a, p, h


def loss_fn(model, D, a, p, h):
    za, zp, zh = model(D.X[a]), model(D.X[p]), model(D.X[h])
    B = len(a)
    ar = torch.arange(B, device=a.device)
    fa, fp = D.fam[a], D.fam[p]
    cu = torch.cat([p, h])
    cf = D.fam[cu]
    L1 = za @ torch.cat([zp, zh]).T / TAU
    m1 = ((cf[None, :] == fa[:, None]) & (fa[:, None] >= 0)) | ((cf[None, :] == fp[:, None]) & (fp[:, None] >= 0)) \
        | (cu[None, :] == p[:, None]) | (cu[None, :] == a[:, None])
    m1[ar, ar] = False
    l1 = F.cross_entropy(L1.masked_fill(m1, float("-inf")), ar)
    L2 = zp @ za.T / TAU
    m2 = ((fa[None, :] == fp[:, None]) & (fp[:, None] >= 0)) | ((fa[None, :] == fa[:, None]) & (fa[:, None] >= 0)) \
        | (a[None, :] == a[:, None]) | (a[None, :] == p[:, None])
    m2[ar, ar] = False
    l2 = F.cross_entropy(L2.masked_fill(m2, float("-inf")), ar)
    return 0.5 * (l1 + l2)


# =============================================================================================== evaluation
@torch.no_grad()
def embed_all(model, D, chunk=131072, dtype=torch.float32):
    Z = torch.empty((D.U, model_out(model)), dtype=dtype, device=D.dev)
    for s in range(0, D.U, chunk):
        Z[s:s + chunk] = model(D.X[s:s + chunk]).to(dtype)
    return Z


def model_out(model):
    return model.lin.out_features if model.arch == "linear" else model.l2.out_features


@torch.no_grad()
def topk_exact(Z, qidx, k, qchunk=256, dbchunk=400_000, exclude_self=True):
    """Exact top-k by inner product (float32 arithmetic) for rows qidx of Z against all rows of Z."""
    dev = Z.device
    U = Z.shape[0]
    outI = torch.empty((len(qidx), k), dtype=torch.int64, device=dev)
    outD = torch.empty((len(qidx), k), dtype=torch.float32, device=dev)
    for qs in range(0, len(qidx), qchunk):
        qi = qidx[qs:qs + qchunk]
        Q = Z[qi].float()
        bestD, bestI = None, None
        for ds in range(0, U, dbchunk):
            S = Q @ Z[ds:ds + dbchunk].float().T
            if exclude_self:
                inr = (qi >= ds) & (qi < ds + S.shape[1])
                if inr.any():
                    rr = torch.nonzero(inr).squeeze(1)
                    S[rr, qi[rr] - ds] = float("-inf")
            kk = min(k, S.shape[1])
            d, i = torch.topk(S, kk, dim=1)
            i = i + ds
            if bestD is None:
                bestD, bestI = d, i
            else:
                cd = torch.cat([bestD, d], 1)
                ci = torch.cat([bestI, i], 1)
                d2, j = torch.topk(cd, k, dim=1)
                bestD, bestI = d2, torch.gather(ci, 1, j)
        outI[qs:qs + len(qi)] = bestI
        outD[qs:qs + len(qi)] = bestD
    return outI, outD


@torch.no_grad()
def retrieval(Z, ev, dev, K=K_AP):
    """Per-query AP@K, recall@K and Spearman(cosine, bits) over aligned targets."""
    q = torch.from_numpy(ev["q"].astype(np.int64)).to(dev)
    ip = ev["indptr"]
    tgt = ev["tgt"].astype(np.int64)
    bits = ev["bits"]
    U = Z.shape[0]
    I, _ = topk_exact(Z, q, K)
    nq = len(q)
    row = np.repeat(np.arange(nq), np.diff(ip))
    keys = torch.from_numpy(np.sort(row * U + tgt)).to(dev)
    k2 = torch.arange(nq, device=dev)[:, None] * U + I
    pos = torch.searchsorted(keys, k2).clamp(max=len(keys) - 1)
    rel = (keys[pos] == k2).float()
    npos = torch.from_numpy(np.diff(ip)).to(dev).float()
    prec = torch.cumsum(rel, 1) / torch.arange(1, K + 1, device=dev).float()
    ap = (prec * rel).sum(1) / torch.clamp(npos, max=K)
    rec = rel.sum(1) / npos
    # Spearman between cosine and bit score among each query's aligned targets (queries with >= 5)
    cos = np.empty(len(tgt), np.float32)
    for s in range(0, len(tgt), 500_000):
        qq = torch.from_numpy(row[s:s + 500_000]).to(dev)
        tt = torch.from_numpy(tgt[s:s + 500_000]).to(dev)
        cos[s:s + 500_000] = (Z[q[qq]].float() * Z[tt].float()).sum(1).cpu().numpy()
    rho = np.full(nq, np.nan)
    for i in range(nq):
        a, b = ip[i], ip[i + 1]
        if b - a >= 5:
            rho[i] = spearman(cos[a:b], bits[a:b])
    return dict(ap=ap.cpu().numpy(), recall=rec.cpu().numpy(), rho=rho)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def rankdata(y):
    o = np.argsort(y, kind="mergesort")
    r = np.empty(len(y))
    ys = y[o]
    i = 0
    while i < len(y):
        j = i
        while j + 1 < len(y) and ys[j + 1] == ys[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def boot(x, draws=2000, seed=17):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (draws, len(x)))].mean(1)
    return [float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def summarize(r):
    return dict(ap=boot(r["ap"]), recall=boot(r["recall"]), spearman_bits=boot(r["rho"]))


@torch.no_grad()
def mine_hard(D, d: Path, M, keep=8, achunk=1024, dbchunk=300_000, limit=0):
    """Hard negatives: the M nearest training sequences of each anchor by zca10 cosine (exact), kept when they are
    of another rescue3 family, have no training alignment with the anchor in either direction, and at least one
    of the two has a complete (unsaturated) alignment list, so that the missing alignment is verified."""
    t0 = time.time()
    dev = D.dev
    s = np.load(d / "split.npz")
    tr = torch.from_numpy(np.flatnonzero(s["is_train"])).to(dev)
    comp = torch.from_numpy(s["complete"]).to(dev)
    keys = torch.from_numpy(np.load(d / "pair_keys_sorted.npy")).to(dev)
    U = D.U
    Xt = D.X[tr]
    anc = D.pool_all[:limit] if limit else D.pool_all
    out = torch.full((U, keep), -1, dtype=torch.int64, device=dev)
    cos_kept = []
    n_cand = n_aln = n_fam = n_unver = 0
    for s0 in range(0, len(anc), achunk):
        a = anc[s0:s0 + achunk]
        Q = D.X[a].float()
        bestD = bestI = None
        for ds in range(0, len(tr), dbchunk):
            S = Q @ Xt[ds:ds + dbchunk].float().T
            d_, i_ = torch.topk(S, min(M + 1, S.shape[1]), dim=1)
            i_ = i_ + ds
            if bestD is None:
                bestD, bestI = d_, i_
            else:
                cd, ci = torch.cat([bestD, d_], 1), torch.cat([bestI, i_], 1)
                d2, j = torch.topk(cd, M + 1, dim=1)
                bestD, bestI = d2, torch.gather(ci, 1, j)
        C = tr[bestI]
        ok = C != a[:, None]
        n_cand += int(ok.sum())
        fa = D.fam[a][:, None]
        same = (fa >= 0) & (D.fam[C] == fa)
        kk = a[:, None] * U + C
        p = torch.searchsorted(keys, kk).clamp(max=len(keys) - 1)
        aligned = keys[p] == kk
        ver = comp[a][:, None] | comp[C]
        n_fam += int((ok & same).sum())
        n_aln += int((ok & ~same & aligned).sum())
        n_unver += int((ok & ~same & ~aligned & ~ver).sum())
        ok = ok & ~same & ~aligned & ver
        rk = torch.cumsum(ok.int(), 1)
        sel = ok & (rk <= keep)
        r, c = torch.nonzero(sel, as_tuple=True)
        out[a[r], rk[r, c] - 1] = C[r, c]
        cos_kept.append(bestD[r, c].cpu())
    D.hn = out
    D.nh = (out >= 0).sum(1)
    ck = torch.cat(cos_kept).numpy() if cos_kept else np.zeros(1)
    nh = D.nh[anc]
    info = dict(M=M, keep=keep, anchors=int(len(anc)), anchors_with_any=int((nh > 0).sum().item()),
                mean_per_anchor=float(nh.float().mean().item()), candidates=n_cand, same_family=n_fam,
                aligned_other_family=n_aln, unverifiable=n_unver,
                hard_negative_cos_q10_q50_q90=np.quantile(ck, [.1, .5, .9]).tolist(), seconds=round(time.time() - t0, 1))
    del keys, Xt
    stamp("hard negatives mined", json.dumps(info))
    return info


# =============================================================================================== training
def train(name, cfg, D, a):
    dev = D.dev
    torch.manual_seed(17)
    model = Head(cfg["arch"], d_out=cfg["out"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    warm = max(1, a.steps // 40)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    gen = torch.Generator(device=dev)
    gen.manual_seed(17)
    hist, best = [], dict(ap=-1.0)
    t0 = time.time()
    run = 0.0
    nrun = 0
    for step in range(1, a.steps + 1):
        A_, P_, H_ = D.batch(cfg, a.batch, gen)
        loss = loss_fn(model, D, A_, P_, H_)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run += loss.item() if step % 50 == 0 else 0.0
        nrun += step % 50 == 0
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            Z = embed_all(model, D, dtype=torch.float32 if cfg["out"] <= 256 else torch.float16)
            r = retrieval(Z, D.ev["val"], dev)
            del Z
            model.train()
            row = dict(step=step, loss=run / max(nrun, 1), val_ap=float(np.mean(r["ap"])),
                       val_recall=float(np.mean(r["recall"])), val_spearman=float(np.nanmean(r["rho"])),
                       seconds=round(time.time() - t0, 1))
            run, nrun = 0.0, 0
            hist.append(row)
            stamp(name, json.dumps(row))
            if row["val_ap"] > best["ap"]:
                best = dict(ap=row["val_ap"], step=step, state=copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                            val=r)
    return hist, best


def load_model(cfg, state, dev):
    m = Head(cfg["arch"], d_out=cfg["out"]).to(dev)
    m.load_state_dict(state)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--knn-k", type=int, default=50)
    ap.add_argument("--test-limit", type=int, default=0, help="code test only: mine and kNN for this many rows")
    ap.add_argument("--mine-m", type=int, default=64, help="zca10 neighbours among training sequences searched for "
                                                            "hard negatives (0 = keep the prep 50-NN set)")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_ON_NODE", "2")))
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    man = dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               device=str(dev), gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None, torch=torch.__version__,
               args={k: str(v) for k, v in vars(a).items()}, tau=TAU, p_fam=P_FAM, k_ap=K_AP, configs=CONFIGS)
    stamp("start", json.dumps({k: man[k] for k in ("device", "gpu", "torch")}))
    D = Data(a.data, dev)
    man["hard_negatives_prep_50nn"] = dict(anchors_with_any=int((D.nh[D.pool_all] > 0).sum().item()))
    if a.mine_m > 0:
        man["hard_negatives_mined"] = mine_hard(D, a.data, a.mine_m, limit=a.test_limit)
    names = a.configs.split(",")
    res, states = {}, {}
    for name in names:
        cfg = CONFIGS[name]
        hist, best = train(name, cfg, D, a)
        states[name] = best["state"]
        torch.save(best["state"], a.out / f"{name}.pt")
        res[name] = dict(cfg=cfg, history=hist, best_step=best["step"], val=summarize(best["val"]),
                         params=int(sum(v.numel() for v in best["state"].values())))
        np.savez(a.out / f"{name}_val_perquery.npz", **best["val"])
        stamp(name, "best", best["step"], json.dumps(res[name]["val"]))
    # raw baseline on validation
    raw_val = retrieval(D.X, D.ev["val"], dev)
    res["raw_zca10"] = dict(val=summarize(raw_val))
    # ---------------------------------------------------------------- selection (validation only)
    vals = {n: res[n]["val"]["ap"][0] for n in names}
    top = max(vals.values())
    near = [n for n in names if vals[n] >= top - TIE]
    pick = sorted(near, key=lambda n: (SIMPLICITY[CONFIGS[n]["arch"]], -vals[n]))[0]
    contrib = {}
    v = lambda n: vals.get(n)
    for lab_, x, y in (("hard_negatives", "A_mlp256_full", "D_mlp256_nohard"), ("family_positives", "A_mlp256_full", "E_mlp256_nofam"),
                       ("weighting", "A_mlp256_full", "F_mlp256_unweighted"), ("all_three", "A_mlp256_full", "G_mlp256_minimal"),
                       ("mlp_vs_linear", "A_mlp256_full", "B_linear256_full"), ("out960_vs_out256", "C_mlp960_full", "A_mlp256_full")):
        if v(x) is not None and v(y) is not None:
            d = np.load(a.out / f"{x}_val_perquery.npz")["ap"] - np.load(a.out / f"{y}_val_perquery.npz")["ap"]
            contrib[lab_] = dict(contrast=f"{x} - {y}", val_ap_diff=boot(d))
    sel = dict(rule="highest mean validation AP@2000; within 0.002 the simpler model (linear < mlp256 < mlp960)",
               val_ap=vals, raw_zca10_val_ap=res["raw_zca10"]["val"]["ap"][0], selected=pick,
               contributions=contrib, written=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (a.out / "selection.json").write_text(json.dumps(sel, indent=1, default=float) + "\n")
    stamp("selection", json.dumps(sel, default=float))
    # ---------------------------------------------------------------- test retrieval (after the choice)
    del D.tgt, D.cw, D.hn
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    test = {}
    raw_t = retrieval(D.X, D.ev["test"], dev)
    test["raw_zca10"] = summarize(raw_t)
    np.savez(a.out / "raw_zca10_test_perquery.npz", **raw_t)
    for name in names:
        m = load_model(CONFIGS[name], states[name], dev)
        Z = embed_all(m, D, dtype=torch.float32 if CONFIGS[name]["out"] <= 256 else torch.float16)
        r = retrieval(Z, D.ev["test"], dev)
        del Z
        np.savez(a.out / f"{name}_test_perquery.npz", **r)
        test[name] = summarize(r)
        test[name]["minus_raw_zca10"] = dict(ap=boot(r["ap"] - raw_t["ap"]), recall=boot(r["recall"] - raw_t["recall"]))
        stamp("test", name, json.dumps(test[name]))
    D.L2 = D.load_rows(a.data / "l2_f16__uid.npy")
    raw_l2 = retrieval(D.L2, D.ev["test"], dev)
    test["raw_l2"] = summarize(raw_l2)
    np.savez(a.out / "raw_l2_test_perquery.npz", **raw_l2)
    raw_l2v = retrieval(D.L2, D.ev["val"], dev)
    res["raw_l2"] = dict(val=summarize(raw_l2v))
    D.L2 = None
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    man.update(results=res, test=test, selection=sel)
    (a.out / "train.json").write_text(json.dumps(man, indent=1, default=float) + "\n")
    # ---------------------------------------------------------------- chosen student: vectors and exact kNN
    cfg = CONFIGS[pick]
    m = load_model(cfg, states[pick], dev)
    od = a.out / pick
    od.mkdir(exist_ok=True)
    t1 = time.time()
    Z = embed_all(m, D, dtype=torch.float32 if cfg["out"] <= 256 else torch.float16)
    del D.X
    torch.cuda.empty_cache() if dev.type == "cuda" else None
    mm = np.lib.format.open_memmap(od / "student_f32__uid.npy", mode="w+", dtype=np.float32, shape=tuple(Z.shape))
    for s in range(0, Z.shape[0], 200_000):
        mm[s:s + 200_000] = Z[s:s + 200_000].float().cpu().numpy()
    mm.flush()
    del mm
    stamp("student vectors", tuple(Z.shape), f"{time.time() - t1:.0f}s")
    t1 = time.time()
    k = a.knn_k
    nk = a.test_limit if a.test_limit else D.U
    Iall = np.empty((nk, k), np.int32)
    Dall = np.empty((nk, k), np.float32)
    step = 8192
    for s in range(0, nk, step):
        qi = torch.arange(s, min(nk, s + step), device=dev)
        I, Dd = topk_exact(Z, qi, k, qchunk=1024, dbchunk=600_000)
        Iall[s:s + len(qi)] = I.cpu().numpy().astype(np.int32)
        Dall[s:s + len(qi)] = Dd.cpu().numpy()
        if (s // step) % 50 == 0:
            stamp(f"knn {s + len(qi):,}/{D.U:,} {time.time() - t1:.0f}s")
    np.savez(od / f"knn_k{k}.npz", I=Iall, D=Dall)
    knn = dict(k=k, exact=True, self_excluded=True, seconds=round(time.time() - t1, 1),
               cos_rank1_q10_q50_q90=np.quantile(Dall[:, 0], [.1, .5, .9]).tolist(),
               cos_rank50_q10_q50_q90=np.quantile(Dall[:, -1], [.1, .5, .9]).tolist())
    man["knn"] = knn
    man["seconds"] = round(time.time() - t0, 1)
    (a.out / "train.json").write_text(json.dumps(man, indent=1, default=float) + "\n")
    stamp("done", json.dumps(knn), man["seconds"], "s")


if __name__ == "__main__":
    main()
