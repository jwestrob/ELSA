#!/usr/bin/env python
"""Neighbourhood-grammar workstream, step 2: train the critics of one split and score its held-out genomes.

Split roles come from prep/genes_meta.parquet (role_<split>: 0 train, 1 validation, 2 test).
Variants (predeclared.json):
  sel_c1..c6  offset-1 critic, genome-centred zca10, the six configurations; the best by validation AUROC = 'primary'
  w3          offsets 1-3 critic, selected configuration
  shuf        offset-1 critic on gene order permuted within each genome, selected configuration
  glob        offset-1 critic on global-mean zca10 (genome accent left in), selected configuration
Critic: s(a, b) = u(x_a) . v(x_b) / sqrt(d) + bias, class-balanced logistic loss; negatives are in-chunk pairs of
the same genome > 50 genes apart or on different contigs (chunks of 128 positive pairs from one genome).
Outputs under $RD/grammar/runs/<split>/: summary.json, models/*.pt, scores_adj1.npz (test adjacencies),
scores_folded.parquet, null_scores.npz (random same-genome pairs), uv_primary.npz (test genes), train_log.tsv.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PREP = RD / "grammar/prep"
L2 = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111/final/l2_f32__row.npy")
CONFIGS = {
    "c1": dict(h=[512], d=128, drop=0.1, wd=1e-4),
    "c2": dict(h=[1024], d=256, drop=0.1, wd=1e-4),
    "c3": dict(h=[256], d=64, drop=0.1, wd=1e-4),
    "c4": dict(h=[], d=128, drop=0.0, wd=1e-4),
    "c5": dict(h=[512], d=128, drop=0.3, wd=1e-3),
    "c6": dict(h=[512, 512], d=128, drop=0.1, wd=1e-4),
}
M = 128            # positive pairs per chunk (one genome per chunk)
BC = 32            # chunks per batch
MAX_EPOCHS = 8
PATIENCE = 2
FAR = 50
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"{time.time() - T0:7.0f}s", *a, flush=True)


# ------------------------------------------------------------------------------------- features
def features(gc, fit_mask, mode, blk=262_144):
    """L2 rows -> minus genome mean (mode 'genome') or minus training-side global mean (mode 'global')
    -> soft-ZCA alpha 0.1 fitted on fit_mask rows -> unit length -> float16."""
    X = np.load(L2)
    n, D = X.shape
    info = dict(mode=mode)
    if mode == "genome":
        starts = np.r_[0, np.flatnonzero(np.diff(gc)) + 1]
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            X[s:e] -= X[s:e].mean(0, dtype=np.float64).astype(np.float32)
        info["genome_mean_removed"] = int(len(starts))
    else:
        mu = np.zeros(D)
        for s in range(0, n, blk):
            m = fit_mask[s:s + blk]
            mu += X[s:s + blk][m].sum(0, dtype=np.float64)
        mu /= fit_mask.sum()
        info["global_mean_norm"] = float(np.linalg.norm(mu))
        X -= mu.astype(np.float32)
    C = torch.zeros(D, D, dtype=torch.float64)
    for s in range(0, n, blk):
        xb = torch.from_numpy(X[s:s + blk][fit_mask[s:s + blk]])
        C += (xb.T @ xb).double()
    C /= float(fit_mask.sum())
    ev, U = torch.linalg.eigh(C)
    ev = ev.clamp(min=0)
    lam = 0.9 * ev + 0.1 * ev.mean()
    W = ((U / lam.sqrt()) @ U.T).float()
    info["participation_ratio_before"] = float(ev.sum() ** 2 / (ev ** 2).sum())
    Xh = np.empty((n, D), dtype=np.float16)
    for s in range(0, n, blk):
        y = torch.from_numpy(X[s:s + blk]) @ W
        y = y / y.norm(dim=1, keepdim=True).clamp(min=1e-12)
        Xh[s:s + blk] = y.numpy().astype(np.float16)
    del X
    return Xh, info


# ------------------------------------------------------------------------------------- pairs
def make_pairs(cc, strand, offsets, gene_at):
    """Pairs of the genes sitting at slots (s, s + d) of one contig. Same strand: a = upstream in transcription;
    opposite strands: sym = True (random order in training, mean of both orders in scoring)."""
    A, B, S = [], [], []
    for d in offsets:
        s = np.flatnonzero(cc[:-d] == cc[d:])
        gi, gj = gene_at[s], gene_at[s + d]
        si, sj = strand[gi], strand[gj]
        same = si == sj
        A.append(np.where(same & (si < 0), gj, gi))
        B.append(np.where(same & (si < 0), gi, gj))
        S.append(~same)
    return np.concatenate(A), np.concatenate(B), np.concatenate(S)


def make_chunks(genome_of_pair, idx, rng, m=M):
    order = idx[np.lexsort((rng.random(len(idx)), genome_of_pair[idx]))]
    g = genome_of_pair[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    lens = np.diff(np.r_[starts, len(order)])
    nch = (lens + m - 1) // m
    out = np.full((int(nch.sum()), m), -1, dtype=np.int64)
    within = np.arange(len(order)) - np.repeat(starts, lens)
    cid = np.repeat(np.cumsum(np.r_[0, nch[:-1]]), lens) + within // m
    out[cid, within % m] = order
    return out


# ------------------------------------------------------------------------------------- model
class Tower(nn.Module):
    def __init__(self, din, h, d, drop):
        super().__init__()
        layers, prev = [], din
        for k in h:
            layers += [nn.Linear(prev, k), nn.GELU(), nn.Dropout(drop)]
            prev = k
        layers.append(nn.Linear(prev, d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, din, cfg):
        super().__init__()
        self.u = Tower(din, cfg["h"], cfg["d"], cfg["drop"])
        self.v = Tower(din, cfg["h"], cfg["d"], cfg["drop"])
        self.bias = nn.Parameter(torch.zeros(()))
        self.scale = 1.0 / np.sqrt(cfg["d"])


class Data:
    def __init__(self, Xh, A, B, SYM, pos_key, contig_key, genome_of_pair):
        self.Xh, self.A, self.B, self.SYM = Xh, A, B, SYM
        self.pos_key, self.contig_key, self.gp = pos_key, contig_key, genome_of_pair

    def batch(self, ch, rng, swap):
        valid = ch >= 0
        p = np.where(valid, ch, 0)
        a, b = self.A[p], self.B[p]
        if swap:
            sw = self.SYM[p] & (rng.random(p.shape) < 0.5)
            a, b = np.where(sw, b, a), np.where(sw, a, b)
        D = self.Xh.shape[1]
        xa = torch.from_numpy(self.Xh[a.ravel()].astype(np.float32)).view(*ch.shape, D)
        xb = torch.from_numpy(self.Xh[b.ravel()].astype(np.float32)).view(*ch.shape, D)
        pa, pb = self.pos_key[a], self.pos_key[b]
        ca, cb = self.contig_key[a], self.contig_key[b]
        far = (ca[:, :, None] != cb[:, None, :]) | (np.abs(pa[:, :, None] - pb[:, None, :]) > FAR)
        neg = far & valid[:, :, None] & valid[:, None, :]
        return xa, xb, torch.from_numpy(valid), torch.from_numpy(neg)


def scores_matrix(model, xa, xb):
    U, V = model.u(xa), model.v(xb)
    return torch.bmm(U, V.transpose(1, 2)) * model.scale + model.bias


def evaluate_val(model, data, val_chunks, seed):
    from sklearn.metrics import roc_auc_score
    model.eval()
    rng = np.random.default_rng(seed)
    pos, neg, loss_p, loss_n = [], [], 0.0, 0.0
    with torch.no_grad():
        for s in range(0, len(val_chunks), BC):
            xa, xb, valid, nm = data.batch(val_chunks[s:s + BC], rng, swap=False)
            S = scores_matrix(model, xa, xb)
            d = torch.diagonal(S, dim1=1, dim2=2)[valid].numpy()
            nv = S[nm].numpy()
            pos.append(d)
            k = min(len(nv), 4 * len(d))
            neg.append(nv[rng.choice(len(nv), k, replace=False)] if k < len(nv) else nv)
    pos, neg = np.concatenate(pos), np.concatenate(neg)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    auc = float(roc_auc_score(y, np.r_[pos, neg]))
    bce = float(np.mean(np.logaddexp(0, -pos)) + np.mean(np.logaddexp(0, neg)))
    return auc, bce


def train_one(name, cfg, data, tr_idx, va_idx, seed, logf):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Critic(data.Xh.shape[1], cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=cfg["wd"])
    val_chunks = make_chunks(data.gp, va_idx, np.random.default_rng(seed + 1))
    best, best_state, best_ep, bad, hist = -1.0, None, -1, 0, []
    for ep in range(MAX_EPOCHS):
        model.train()
        t = time.time()
        chunks = make_chunks(data.gp, tr_idx, rng)
        chunks = chunks[rng.permutation(len(chunks))]
        tot, nb = 0.0, 0
        for s in range(0, len(chunks), BC):
            xa, xb, valid, nm = data.batch(chunks[s:s + BC], rng, swap=True)
            S = scores_matrix(model, xa, xb)
            d = torch.diagonal(S, dim1=1, dim2=2)[valid]
            nv = S[nm]
            loss = F.binary_cross_entropy_with_logits(d, torch.ones_like(d)) + \
                F.binary_cross_entropy_with_logits(nv, torch.zeros_like(nv))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        auc, bce = evaluate_val(model, data, val_chunks, seed + 2)
        hist.append(dict(epoch=ep, train_loss=tot / nb, val_auc=auc, val_bce=bce, seconds=round(time.time() - t, 1)))
        logf.write(f"{name}\t{ep}\t{tot / nb:.5f}\t{auc:.5f}\t{bce:.5f}\t{time.time() - t:.1f}\n")
        logf.flush()
        log(name, "epoch", ep, f"train {tot / nb:.4f} val AUROC {auc:.4f} val BCE {bce:.4f} ({time.time() - t:.0f}s, {nb} batches)")
        if auc > best + 1e-4:
            best, best_ep, bad = auc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(best_val_auc=best, best_epoch=best_ep, history=hist)


def towers(model, Xh, genes, blk=131_072):
    d = model.u.net[-1].out_features
    U = np.empty((len(genes), d), dtype=np.float32)
    V = np.empty((len(genes), d), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(genes), blk):
            x = torch.from_numpy(Xh[genes[s:s + blk]].astype(np.float32))
            U[s:s + blk] = model.u(x).numpy()
            V[s:s + blk] = model.v(x).numpy()
    return U, V


def pair_scores(model, U, V, row, a, b, sym):
    sc, bias = model.scale, float(model.bias)
    ra, rb = row[a], row[b]
    assert (ra >= 0).all() and (rb >= 0).all()
    s_ab = np.einsum("ij,ij->i", U[ra], V[rb]) * sc + bias
    s_ba = np.einsum("ij,ij->i", U[rb], V[ra]) * sc + bias
    return np.where(sym, 0.5 * (s_ab + s_ba), s_ab).astype(np.float32)


def orient(g0, g1, strand):
    """Order a gene pair by the scoring rule: same strand -> upstream first; else symmetric."""
    lo, hi = np.minimum(g0, g1), np.maximum(g0, g1)
    same = strand[lo] == strand[hi]
    a = np.where(same & (strand[lo] < 0), hi, lo)
    b = np.where(same & (strand[lo] < 0), lo, hi)
    return a, b, ~same


# ------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--configs", default="c1,c2,c3,c4,c5,c6")
    ap.add_argument("--variants", default="w3,shuf,glob")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--limit-genomes", type=int, default=0, help="smoke test: keep this many genomes per role")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    globals()["MAX_EPOCHS"] = a.max_epochs
    torch.set_num_threads(a.threads)
    out = a.out or RD / "grammar/runs" / a.split
    (out / "models").mkdir(parents=True, exist_ok=True)
    k = int(a.split[2:]) if a.split.startswith("cv") else 9
    meta = pd.read_parquet(PREP / "genes_meta.parquet", columns=["gid", "genome_code", "contig_code", "strand", f"role_{a.split}"])
    gc = meta.genome_code.to_numpy(np.int64)
    cc = meta.contig_code.to_numpy(np.int64)
    strand = meta.strand.to_numpy(np.int64)
    role_g = meta[f"role_{a.split}"].to_numpy()
    if a.limit_genomes:
        keep = np.zeros(len(gc), dtype=bool)
        for r in (0, 1, 2):
            gs = np.unique(gc[role_g == r])[: a.limit_genomes]
            keep |= np.isin(gc, gs)
        role_g = np.where(keep, role_g, 3)
    n = len(gc)
    summary = dict(split=a.split, threads=a.threads, genes={int(r): int((role_g == r).sum()) for r in (0, 1, 2)})
    log("split", a.split, summary["genes"])

    fit = role_g <= 1
    Xg, finfo = features(gc, fit, "genome")
    summary["features_genome"] = finfo
    log("features (genome-centred zca10)", finfo)
    logf = open(out / "train_log.tsv", "a")
    logf.write("model\tepoch\ttrain_loss\tval_auc\tval_bce\tseconds\n")

    ident = np.arange(n)
    A1, B1, S1 = make_pairs(cc, strand, (1,), ident)
    gp1 = gc[A1]
    tr1, va1 = np.flatnonzero(role_g[A1] == 0), np.flatnonzero(role_g[A1] == 1)
    data1 = Data(Xg, A1, B1, S1, ident.astype(np.int64), cc, gp1)
    summary["pairs_offset1"] = dict(train=int(len(tr1)), val=int(len(va1)))
    log("offset-1 pairs", summary["pairs_offset1"])

    models, res = {}, {}
    for name in a.configs.split(","):
        m, r = train_one(f"sel_{name}", CONFIGS[name], data1, tr1, va1, 17 + k, logf)
        models[f"sel_{name}"], res[f"sel_{name}"] = m, r
        torch.save(m.state_dict(), out / "models" / f"sel_{name}.pt")
    best = max(a.configs.split(","), key=lambda c: res[f"sel_{c}"]["best_val_auc"])
    summary["selected_config"] = best
    summary["selection"] = {c: res[f"sel_{c}"]["best_val_auc"] for c in a.configs.split(",")}
    log("selected", best, summary["selection"])
    cfg = CONFIGS[best]
    models["primary"] = models[f"sel_{best}"]

    variants = [v for v in a.variants.split(",") if v]
    if "w3" in variants:
        A3, B3, S3 = make_pairs(cc, strand, (1, 2, 3), ident)
        data3 = Data(Xg, A3, B3, S3, ident.astype(np.int64), cc, gc[A3])
        tr3, va3 = np.flatnonzero(role_g[A3] == 0), np.flatnonzero(role_g[A3] == 1)
        m, r = train_one("w3", cfg, data3, tr3, va3, 117 + k, logf)
        models["w3"], res["w3"] = m, r
        torch.save(m.state_dict(), out / "models" / "w3.pt")
        del data3, A3, B3, S3
    if "shuf" in variants:
        rng = np.random.default_rng(1000 + k)
        gene_at = ident.copy()
        starts = np.r_[0, np.flatnonzero(np.diff(gc)) + 1]
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            gene_at[s:e] = s + rng.permutation(e - s)
        slot_of = np.empty(n, dtype=np.int64)
        slot_of[gene_at] = ident
        As, Bs, Ss = make_pairs(cc, strand, (1,), gene_at)
        datas = Data(Xg, As, Bs, Ss, slot_of, cc[slot_of], gc[As])
        trs, vas = np.flatnonzero(role_g[As] == 0), np.flatnonzero(role_g[As] == 1)
        m, r = train_one("shuf", cfg, datas, trs, vas, 217 + k, logf)
        models["shuf"], res["shuf"] = m, r
        torch.save(m.state_dict(), out / "models" / "shuf.pt")
        del datas
    if "glob" in variants:
        Xb, binfo = features(gc, fit, "global")
        summary["features_global"] = binfo
        datab = Data(Xb, A1, B1, S1, ident.astype(np.int64), cc, gp1)
        m, r = train_one("glob", cfg, datab, tr1, va1, 317 + k, logf)
        models["glob"], res["glob"] = m, r
        torch.save(m.state_dict(), out / "models" / "glob.pt")
    summary["training"] = res

    # ------------------------------------------------------------------ scoring of held-out genomes
    test_genes = np.flatnonzero(role_g == 2)
    row = np.full(n, -1, dtype=np.int64)
    row[test_genes] = np.arange(len(test_genes))
    adj = pd.read_parquet(PREP / "adj1.parquet", columns=["a", "b", "config", "genome_code"])
    te = np.flatnonzero(role_g[adj.a.to_numpy()] == 2)
    aa, bb, sym = adj.a.to_numpy()[te], adj.b.to_numpy()[te], adj.config.to_numpy()[te] != 0
    fold = pd.read_parquet(PREP / "folded.parquet")
    fold = fold[role_g[fold.g0.to_numpy()] == 2].reset_index(drop=True)
    fa, fb, fs = orient(fold.g0.to_numpy(), fold.g1.to_numpy(), strand)
    rng = np.random.default_rng(4242 + k)
    x = rng.choice(test_genes, 1_000_000)
    gstart = np.r_[0, np.flatnonzero(np.diff(gc)) + 1]
    gsize = np.diff(np.r_[gstart, n])
    off = (rng.random(len(x)) * (gsize[gc[x]] - 1)).astype(np.int64)
    y = gstart[gc[x]] + off
    y = np.where(y >= x, y + 1, y)                 # uniform over the genome's other genes
    na, nb_, ns = orient(x, y, strand)
    adj_scores, null_scores = dict(idx=te), {}
    for name, m in models.items():
        Xuse = Xb if name == "glob" else Xg
        U, V = towers(m, Xuse, test_genes)
        adj_scores[name] = pair_scores(m, U, V, row, aa, bb, sym)
        fold[f"s_{name}"] = pair_scores(m, U, V, row, fa, fb, fs)
        null_scores[name] = pair_scores(m, U, V, row, na, nb_, ns)
        if name == "primary":
            np.savez(out / "uv_primary.npz", genes=test_genes, U=U.astype(np.float16), V=V.astype(np.float16),
                     scale=m.scale, bias=float(m.bias))
        log("scored", name, "adjacencies", len(te), "null q99", float(np.quantile(null_scores[name], 0.99)))
    np.savez(out / "scores_adj1.npz", **adj_scores)
    np.savez(out / "null_scores.npz", **null_scores)
    fold.to_parquet(out / "scores_folded.parquet", index=False)
    summary["n_test_adjacencies"] = int(len(te))
    summary["n_folded_test"] = int(len(fold))
    summary["seconds"] = round(time.time() - T0, 1)
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    log("done", a.split)


if __name__ == "__main__":
    main()
