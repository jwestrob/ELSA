#!/usr/bin/env python
"""Neighbourhood-grammar workstream, step 1: folds, pair tables, labels and discrete baselines.

Outputs under $RD/grammar/prep/:
  genes_meta.parquet   per gid: genome_code, contig_code, position_index, strand, start, end, length,
                       token, token_size, unannotated, gold flags, role_<split> (0 train, 1 val, 2 test)
  splits.json          orders per fold / validation orders per split / counts
  adj1.parquet         every offset-1 adjacency (i, j = i + 1, same contig): orientation, intergenic nt,
                       tokens, gold labels per caller, engine all-genome statistics (window 3), raw cosine
  base_<split>.npz     per adjacency (row of adj1): training-side baselines for that split
  folded.parquet       screen-1 folded pairs (cognate, uncalled) with engine statistics
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
TOK = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
OUT = RD / "grammar/prep"
sys.path.insert(0, str(RD / "engine/code"))
from recurrence import Universe, colocalization, pair_statistics  # noqa: E402

CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
NFOLD = 5
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"{time.time() - T0:7.0f}s", *a, flush=True)


@njit(parallel=True, cache=True)
def expected_instances(lo, hi, indptr, gen, cnt, A_g, n_g):
    """Expected offset-d instances of the unordered token pair (lo, hi) over the genomes in the
    token-genome index, under a uniform permutation of each genome's genes (A_g slots)."""
    out = np.zeros(len(lo), dtype=np.float64)
    for p in prange(len(lo)):
        i, ie = indptr[lo[p]], indptr[lo[p] + 1]
        j, je = indptr[hi[p]], indptr[hi[p] + 1]
        s = 0.0
        while i < ie and j < je:
            if gen[i] == gen[j]:
                g = gen[i]
                n = n_g[g]
                if n > 1:
                    if lo[p] == hi[p]:
                        s += A_g[g] * cnt[i] * (cnt[i] - 1.0) / (n * (n - 1.0))
                    else:
                        s += A_g[g] * 2.0 * cnt[i] * cnt[j] / (n * (n - 1.0))
                i += 1
                j += 1
            elif gen[i] < gen[j]:
                i += 1
            else:
                j += 1
        out[p] = s
    return out


def token_genome_index(tok, gc, T):
    valid = tok >= 0
    ng = int(gc.max()) + 1
    tg, cnt = np.unique(tok[valid] * ng + gc[valid], return_counts=True)
    t_of, gen = tg // ng, tg % ng
    indptr = np.searchsorted(t_of, np.arange(T + 1)).astype(np.int64)
    return indptr, gen.astype(np.int64), cnt.astype(np.float64)


def assign_folds(gm):
    by = gm.groupby("ord").n_genes.sum().sort_values(ascending=False, kind="stable")
    load = np.zeros(NFOLD)
    fold_of = {}
    for o, n in sorted(by.items(), key=lambda kv: (-kv[1], kv[0])):
        f = int(np.argmin(load))
        fold_of[o] = f
        load[f] += n
    return fold_of


def pick_validation(orders, genes_of, seed):
    """Order-grouped validation subset: seeded order shuffle, add orders until >= 18% of the
    training-side genes, skipping any order that would push the subset past 30%."""
    tot = sum(genes_of[o] for o in orders)
    rng = np.random.default_rng(seed)
    perm = [orders[k] for k in rng.permutation(len(orders))]
    val, s = [], 0
    for o in perm:
        if s >= 0.18 * tot:
            break
        if s + genes_of[o] > 0.30 * tot:
            continue
        val.append(o)
        s += genes_of[o]
    return sorted(val), s / tot


def main():
    import numba
    threads = int(sys.argv[1])
    numba.set_num_threads(threads)
    OUT.mkdir(parents=True, exist_ok=True)
    u = Universe(PKG / "universe")
    assert np.array_equal(u.gid, np.arange(u.n)), "gid order must equal the engine Universe order"
    g = pd.read_parquet(PKG / "universe/genes.parquet").sort_values("gid").reset_index(drop=True)
    gm = pd.read_parquet(PKG / "universe/genomes.parquet").sort_values("genome_code").reset_index(drop=True)
    n = len(g)
    gc = g.genome_code.to_numpy(np.int64)
    cc = g.contig_code.to_numpy(np.int64)
    strand = g.strand.to_numpy(np.int64)
    start, end = g.start.to_numpy(np.int64), g.end.to_numpy(np.int64)
    log("genes", n, "genomes", len(gm), "start<=end", bool((start <= end).all()))

    # ---------------------------------------------------------------- folds and splits
    fold_of = assign_folds(gm)
    gm["fold"] = gm.ord.map(fold_of).astype(int)
    genes_of = gm.groupby("ord").n_genes.sum().to_dict()
    splits = {}
    role = {}
    for k in range(NFOLD):
        tr_orders = sorted(o for o, f in fold_of.items() if f != k)
        val, frac = pick_validation(tr_orders, genes_of, 17 + k)
        r = np.where(gm.fold == k, 2, np.where(gm.ord.isin(val), 1, 0))
        role[f"cv{k}"] = r
        splits[f"cv{k}"] = dict(test_orders=sorted(o for o, f in fold_of.items() if f == k), val_orders=val,
                                val_frac_of_training_genes=round(frac, 4),
                                genomes=dict(train=int((r == 0).sum()), val=int((r == 1).sum()), test=int((r == 2).sum())),
                                genes=dict(train=int(gm.n_genes[r == 0].sum()), val=int(gm.n_genes[r == 1].sum()),
                                           test=int(gm.n_genes[r == 2].sum())))
    nano = sorted(gm.ord[gm.phylum == "Nanobdellota"].unique())
    val, frac = pick_validation(nano, genes_of, 117)
    r = np.where(gm.phylum != "Nanobdellota", 2, np.where(gm.ord.isin(val), 1, 0))
    role["phylum"] = r
    splits["phylum"] = dict(train_phylum="Nanobdellota", val_orders=val, val_frac_of_training_genes=round(frac, 4),
                            genomes=dict(train=int((r == 0).sum()), val=int((r == 1).sum()), test=int((r == 2).sum())),
                            genes=dict(train=int(gm.n_genes[r == 0].sum()), val=int(gm.n_genes[r == 1].sum()),
                                       test=int(gm.n_genes[r == 2].sum())))
    splits["fold_of_order"] = fold_of
    for s, v in splits.items():
        if s != "fold_of_order":
            log(s, v["genomes"], v["genes"], "val orders", v["val_orders"])
    json.dump(splits, open(OUT / "splits.json", "w"), indent=1)
    gm[["genome", "genome_code", "phylum", "class", "ord", "family", "genus", "n_genes", "fold"]].to_parquet(
        OUT / "genomes_meta.parquet", index=False)

    # ---------------------------------------------------------------- tokens, flags, gold
    tk = pd.read_parquet(TOK)
    tok = np.full(n, -1, dtype=np.int64)
    tok[tk.gid.to_numpy()] = tk.token.to_numpy()
    assert (tok >= 0).all()
    T = int(tok.max()) + 1
    tsize = np.bincount(tok, minlength=T)
    flags = pd.read_parquet(PKG / "annotation_flags.parquet").sort_values("gid")
    assert np.array_equal(flags.gid.to_numpy(), np.arange(n))
    mem = pd.read_parquet(PKG / "gold/members.parquet")
    meta = pd.DataFrame(dict(gid=np.arange(n), genome_code=gc, contig_code=cc,
                             position_index=g.position_index.to_numpy(), strand=strand.astype(np.int8),
                             start=start, end=end, length=g.length.to_numpy(), token=tok, token_size=tsize[tok],
                             unannotated=flags.unannotated.to_numpy()))
    any_gold = np.zeros(n, dtype=bool)
    for c in CALLERS:
        m = np.zeros(n, dtype=bool)
        m[mem.gid[mem.caller == c].to_numpy()] = True
        meta[f"in_{c}"] = m
        any_gold |= m
    meta["in_gold"] = any_gold
    for s, r in role.items():
        meta[f"role_{s}"] = r[gc].astype(np.int8)
    meta.to_parquet(OUT / "genes_meta.parquet", index=False)
    log("tokens", T, "singletons", int((tsize == 1).sum()), "genes in singleton tokens", int((tsize[tok] == 1).sum()))

    # ---------------------------------------------------------------- offset-1 adjacencies
    i = np.flatnonzero(cc[:-1] == cc[1:])
    j = i + 1
    same = strand[i] == strand[j]
    # config: 0 co-directional, 1 divergent (- +), 2 convergent (+ -)
    config = np.where(same, 0, np.where(strand[i] < 0, 1, 2)).astype(np.int8)
    a = np.where(same & (strand[i] < 0), j, i)
    b = np.where(same & (strand[i] < 0), i, j)
    adj = pd.DataFrame(dict(i=i, j=j, a=a, b=b, genome_code=gc[i], config=config,
                            intergenic=(start[j] - end[i] - 1), ta=tok[i], tb=tok[j],
                            singleton_either=(tsize[tok[i]] == 1) | (tsize[tok[j]] == 1),
                            same_token=tok[i] == tok[j],
                            unannotated_both=meta.unannotated.to_numpy()[i] & meta.unannotated.to_numpy()[j],
                            gold_none=~any_gold[i] & ~any_gold[j]))
    ai = pd.DataFrame(dict(k=np.arange(len(i)), gi=i, gj=j))
    for c in CALLERS:
        mc = mem[mem.caller == c][["gid", "instance_id"]]
        x = ai.merge(mc.rename(columns=dict(gid="gi")), on="gi").merge(mc.rename(columns=dict(gid="gj")), on=["gj", "instance_id"])
        pos = np.zeros(len(i), dtype=bool)
        pos[x.k.to_numpy()] = True
        inc = meta[f"in_{c}"].to_numpy()
        adj[f"pos_{c}"] = pos
        adj[f"straddle_{c}"] = inc[i] ^ inc[j]
        log(c, "within-instance adjacencies", int(pos.sum()), "straddle", int((inc[i] ^ inc[j]).sum()))
    log("offset-1 adjacencies", len(adj), "no-gold", int(adj.gold_none.sum()))

    # ---------------------------------------------------------------- engine statistics, all genomes, window 3
    key3, left3, dist3, T_ = colocalization(u, tok, 3)
    assert T_ == T
    st = pair_statistics(u, tok, key3, left3, 3, T)
    sig = (st["P"] <= 1e-4) & (st["L"] >= 2)
    log("engine token pairs", len(st["key"]), "significant (P<=1e-4 & L>=2)", int(sig.sum()), "(engine summary: 162,786)")
    k1 = np.minimum(tok[i], tok[j]) * T + np.maximum(tok[i], tok[j])
    pidx = np.searchsorted(st["key"], k1)
    pidx = np.minimum(pidx, len(st["key"]) - 1)
    hit = (st["key"][pidx] == k1) & (tok[i] != tok[j])
    adj["eng_nlogP"] = np.where(hit, -st["logP"][pidx], 0.0)
    adj["eng_G"] = np.where(hit, st["G"][pidx], 0)
    adj["eng_L"] = np.where(hit, st["L"][pidx], 0)
    adj["eng_called"] = hit & sig[pidx]
    log("adjacencies in engine-called pairs", int(adj.eng_called.sum()))

    # ---------------------------------------------------------------- raw cosine
    X = np.load(EMB / "final/l2_f32__row.npy", mmap_mode="r")
    Xu = np.load(EMB / "final/mean_raw_f32__uid.npy", mmap_mode="r")
    uid = np.load(EMB / "uid_of_row.npy")
    chk = np.random.default_rng(3).choice(n, 200, replace=False)
    xr = np.asarray(Xu[uid[chk]], dtype=np.float64)
    xr /= np.linalg.norm(xr, axis=1, keepdims=True)
    log("l2 rows equal normalised mean_raw rows: max |diff|", float(np.abs(xr - np.asarray(X[chk])).max()))
    X = np.load(EMB / "final/l2_f32__row.npy")
    cos = np.empty(len(i), dtype=np.float32)
    B = 500_000
    for s in range(0, len(i), B):
        cos[s:s + B] = np.einsum("ij,ij->i", X[i[s:s + B]], X[j[s:s + B]])
    adj["raw_cos"] = cos
    del X
    log("raw cosine done; q10/50/90", np.quantile(cos, [.1, .5, .9]).round(3))
    adj.to_parquet(OUT / "adj1.parquet", index=False)

    # ---------------------------------------------------------------- training-side baselines per split
    contig_n = np.bincount(cc)
    n_contig_genes = contig_n[contig_n > 0]
    genome_of_contig = gc[np.r_[0, np.flatnonzero(np.diff(cc)) + 1]]
    A1 = np.bincount(genome_of_contig, weights=np.maximum(n_contig_genes - 1, 0), minlength=len(gm)).astype(np.float64)
    n_g = gm.n_genes.to_numpy(np.float64)
    lo1, hi1 = np.minimum(tok[i], tok[j]), np.maximum(tok[i], tok[j])
    for s, r in role.items():
        trg = np.flatnonzero(r != 2)                      # training side = train + validation genomes
        tr_gene = np.isin(gc, trg)
        tok_s = np.where(tr_gene, tok, -1)
        # offset-1 instance counts incl. identical tokens, training genomes
        m1 = tr_gene[i]
        kk, obs = np.unique(k1[m1], return_counts=True)
        p = np.minimum(np.searchsorted(kk, k1), len(kk) - 1)
        obs1 = np.where(kk[p] == k1, obs[p], 0).astype(np.float64)
        # genome counts within window 1 and 3 (identical tokens included)
        G = {}
        for w in (1, 3):
            keys, gens = [], []
            for d in range(1, w + 1):
                l = np.flatnonzero((cc[:-d] == cc[d:]) & tr_gene[:-d])
                keys.append(np.minimum(tok[l], tok[l + d]) * T + np.maximum(tok[l], tok[l + d]))
                gens.append(gc[l])
            kg = np.unique(np.concatenate(keys) * len(gm) + np.concatenate(gens))
            kk2, cnt2 = np.unique(kg // len(gm), return_counts=True)
            p = np.minimum(np.searchsorted(kk2, k1), len(kk2) - 1)
            G[w] = np.where(kk2[p] == k1, cnt2[p], 0)
        indptr, gen, cnt = token_genome_index(tok_s, gc, T)
        E1 = expected_instances(lo1, hi1, indptr, gen, cnt, A1, n_g)
        # training-only engine statistics at window 3 (distinct tokens only)
        ks, ls, ds, Ts = colocalization(u, tok_s, 3)      # Ts = max token present + 1 (may be < T)
        sts = pair_statistics(u, tok_s, ks, ls, 3, Ts)
        kg = (sts["key"] // Ts) * T + sts["key"] % Ts      # re-encode with the global base
        o = np.argsort(kg)
        kg, lp = kg[o], sts["logP"][o]
        p = np.minimum(np.searchsorted(kg, k1), len(kg) - 1)
        okk = (kg[p] == k1) & (tok[i] != tok[j])
        nlogp_tr = np.where(okk, -lp[p], 0.0)
        out = dict(obs1=obs1.astype(np.float32), E1=E1.astype(np.float32), G1_train=G[1].astype(np.int32),
                   G3_train=G[3].astype(np.int32), nlogP_train=nlogp_tr.astype(np.float32))
        for alpha, nm in ((1.0, "pmi_a1"), (0.1, "pmi_a01"), (10.0, "pmi_a10")):
            out[nm] = np.log((obs1 + alpha) / (E1 + alpha)).astype(np.float32)
        np.savez(OUT / f"base_{s}.npz", **out)
        te = r[gc[i]] == 2
        log(s, "test adjacencies", int(te.sum()), "token pair never within 3 in training:",
            int((G[3][te] == 0).sum()), "1-4:", int(((G[3][te] >= 1) & (G[3][te] <= 4)).sum()), ">=5:", int((G[3][te] >= 5).sum()))

    # ---------------------------------------------------------------- folded pairs (screen 1)
    f = pd.read_csv(RD / "grammar/inputs/folds.tsv", sep="\t")
    f = f[f.cls.isin(["cognate", "uncalled"])].copy()
    ab = f.gids.str.split(";", expand=True).astype(np.int64)
    f["g0"], f["g1"] = ab[0], ab[1]
    f["offset"] = (f.g1 - f.g0).abs()
    f["same_contig"] = cc[f.g0] == cc[f.g1]
    kf = np.minimum(tok[f.g0], tok[f.g1]) * T + np.maximum(tok[f.g0], tok[f.g1])
    p = np.minimum(np.searchsorted(st["key"], kf), len(st["key"]) - 1)
    hit = (st["key"][p] == kf) & (tok[f.g0] != tok[f.g1])
    f["eng_nlogP"] = np.where(hit, -st["logP"][p], 0.0)
    f["eng_called"] = hit & sig[p]
    f["genome_code"] = gc[f.g0]
    f["len_sum"] = g.length.to_numpy()[f.g0] + g.length.to_numpy()[f.g1]
    for s, r in role.items():
        f[f"role_{s}"] = r[gc[f.g0]]
    keep = ["fold_id", "cls", "cand_rank", "rep", "g0", "g1", "offset", "same_contig", "iptm", "pass_c123",
            "eng_nlogP", "eng_called", "genome_code", "len_sum"] + [f"role_{s}" for s in role]
    f[keep].to_parquet(OUT / "folded.parquet", index=False)
    log("folded pairs", len(f), f.cls.value_counts().to_dict(), "offsets", f.offset.value_counts().to_dict(),
        "engine-called by class", f.groupby("cls").eng_called.sum().to_dict())
    log("done")


if __name__ == "__main__":
    main()
