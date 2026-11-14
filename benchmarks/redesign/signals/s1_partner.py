#!/usr/bin/env python
"""Signal 1 of ws_signals: partner covariation in vector space (a mirror-tree on genome-centred residuals).

For a co-located family pair (F, G) (rescue3 tokens; offset 1..3; one instance per genome; distinct uid pairs; capped at
400), each instance carries e = x - c_F^(-i) - b_g^(-i,j) for both genes. M(F, G) = partial Spearman correlation over
instance pairs between cos(e_F,i, e_F,j) and cos(e_G,i, e_G,j), partialling a cubic in rank genome similarity
(lineagefree emb_mean) and two same-group indicators. E(F, G) = mean over far background families H (same genomes,
>= 20 genes away or on another contig, matched to the replaced partner on tree-likeness) of
M(F, G | I_H) - M(F, H | I_H), both sides. Permutation null: G rows permuted within emb_hom 0.75 groups (199), and
within fam_jac 0.2 groups (199). Secondary: out-of-sample first canonical correlation (genome-parity split).
See predeclared.json. Writes $RD/signals/work/s1_pairs.parquet and s1_instances_meta.parquet.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RD, W, ENG, LF, EMB, SEED, log, load_genes, load_tokens, load_groups  # noqa: E402

CAP = 400
NPERM = 199
NCAND = 60
NBG = 10
FAR = 20
MIN_SUB = 10

G_ = {}


# --------------------------------------------------------------------------------------------- instances
def build_instances(g, tok, uid, elig):
    gc = g.genome_code.to_numpy(np.int64)
    cc = g.contig_code.to_numpy(np.int64)
    st = g.strand.to_numpy(np.int64)
    s0, e0 = g.start.to_numpy(np.int64), g.end.to_numpy(np.int64)
    n = len(g)
    T = int(tok.max()) + 1
    rows = []
    for off in (1, 2, 3):
        i = np.arange(n - off)
        j = i + off
        m = (cc[i] == cc[j]) & (tok[i] != tok[j]) & elig[i] & elig[j]
        i, j = i[m], j[m]
        swap = tok[i] > tok[j]
        a = np.where(swap, j, i)
        b = np.where(swap, i, j)
        rows.append(pd.DataFrame(dict(key=np.minimum(tok[i], tok[j]) * T + np.maximum(tok[i], tok[j]), a=a, b=b,
                                      off=np.int8(off), gc=gc[i], same=(st[i] == st[j]), gap=(s0[j] - e0[i] - 1))))
    d = pd.concat(rows, ignore_index=True)
    d = d.sort_values(["key", "gc", "off", "a"], kind="mergesort").drop_duplicates(["key", "gc"])
    d["ua"], d["ub"] = uid[d.a.to_numpy()], uid[d.b.to_numpy()]
    d = d.drop_duplicates(["key", "ua", "ub"]).reset_index(drop=True)
    return d, T


# --------------------------------------------------------------------------------------------- residuals
def resid(ix):
    X, fr, S, m, tok, uid = G_["X"], G_["fam_row"], G_["S"], G_["m"], G_["tok"], G_["uid"]
    mm = m[tok[ix]].astype(np.float64)[:, None]
    return np.asarray(X[uid[ix]], dtype=np.float64) * (mm / (mm - 1)) - S[fr[tok[ix]]].astype(np.float64) / (mm - 1)


def e_pair(a, b):
    """genome-centred residuals of an instance pair (leave-both-out background)."""
    gc, B, ng = G_["gc"], G_["B"], G_["ng"]
    ra, rb = resid(a), resid(b)
    bg = (B[gc[a]] - ra - rb) / np.maximum(ng[gc[a]] - 2, 1)[:, None]
    return ra - bg, rb - bg


def e_one(h):
    gc, B, ng = G_["gc"], G_["B"], G_["ng"]
    rh = resid(h)
    return rh - (B[gc[h]] - rh) / np.maximum(ng[gc[h]] - 1, 1)[:, None]


def cosmat(E):
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    return E @ E.T


# --------------------------------------------------------------------------------------------- statistics
class Design:
    """Partialling basis over the upper triangle of an instance set."""

    def __init__(self, gcs):
        n = len(gcs)
        self.iu = np.triu_indices(n, 1)
        gi, gj = gcs[self.iu[0]], gcs[self.iu[1]]
        s = G_["sim"][gi, gj].astype(np.float64)
        self.s_rank = rankdata(s)
        rs = self.s_rank / len(s)
        cols = [np.ones(len(s)), rs, rs ** 2, rs ** 3,
                (G_["g_sp"][gi] == G_["g_sp"][gj]).astype(np.float64),
                (G_["g_fam"][gi] == G_["g_fam"][gj]).astype(np.float64)]
        D = np.column_stack(cols)
        U, sv, _ = np.linalg.svd(D, full_matrices=False)
        self.Q = U[:, sv > 1e-8 * sv[0]]

    def res(self, v):
        return v - self.Q @ (self.Q.T @ v)

    def rank_res(self, C):
        r = rankdata(C[self.iu])
        return self.res(r), r


def corr(x, y):
    d = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / d) if d > 0 else np.nan


def spearman_S(C, des):
    r = rankdata(C[des.iu])
    a, b = r - r.mean(), des.s_rank - des.s_rank.mean()
    return corr(a, b)


def M_stat(CF, CG, des):
    rf, _ = des.rank_res(CF)
    rg, _ = des.rank_res(CG)
    return corr(rf, rg)


def perm_p(CF, CG, des, groups, rng):
    rf, _ = des.rank_res(CF)
    n = CG.shape[0]
    RG = np.zeros((n, n))
    RG[des.iu] = rankdata(CG[des.iu])
    RG = RG + RG.T
    obs = corr(rf, des.res(RG[des.iu]))
    order = np.argsort(groups, kind="mergesort")
    gs = groups[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], len(gs)]
    movable = int(sum(e - s for s, e in zip(starts, ends) if e - s > 1))
    ge = 0
    for _ in range(NPERM):
        perm = np.arange(n)
        for s, e in zip(starts, ends):
            if e - s > 1:
                seg = order[s:e]
                perm[seg] = seg[rng.permutation(e - s)]
        v = des.res(RG[perm[des.iu[0]], perm[des.iu[1]]])
        ge += corr(rf, v) >= obs
    return (1 + ge) / (NPERM + 1), movable


def cca_oos(EF, EG, gcs):
    out = []
    par = gcs % 2
    for tr_bit in (0, 1):
        tr, te = np.flatnonzero(par == tr_bit), np.flatnonzero(par != tr_bit)
        k = min(10, len(tr) // 5)
        if k < 2 or len(te) < 5:
            continue
        zs = []
        for E in (EF, EG):
            mu = E[tr].mean(0)
            _, _, Vt = np.linalg.svd(E[tr] - mu, full_matrices=False)
            a = (E[tr] - mu) @ Vt[:k].T
            sd = a.std(0)
            sd[sd == 0] = 1
            zs.append((a / sd, ((E[te] - mu) @ Vt[:k].T) / sd))
        (a, at), (b, bt) = zs
        nn = len(tr)
        Caa = a.T @ a / nn + 0.1 * np.eye(k)
        Cbb = b.T @ b / nn + 0.1 * np.eye(k)
        Cab = a.T @ b / nn

        def isqrt(C):
            w, V = np.linalg.eigh(C)
            return (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T
        Ia, Ib = isqrt(Caa), isqrt(Cbb)
        u, _, vt = np.linalg.svd(Ia @ Cab @ Ib)
        wa, wb = Ia @ u[:, 0], Ib @ vt[0]
        x, y = at @ wa, bt @ wb
        out.append(corr(x - x.mean(), y - y.mean()))
    return float(np.mean(out)) if out else np.nan


# --------------------------------------------------------------------------------------------- per pair
def background(side, a, b, gcs, keep_tok, target_rho, rng):
    """side 'F': H replaces G (pairs with F); side 'G': H replaces F. Returns list of (subset idx, CH)."""
    pres = G_["pres"]
    cnt = np.asarray(pres[gcs].sum(0)).ravel()
    cnt[keep_tok] = 0
    cover = cnt / len(gcs)
    pool = np.flatnonzero(cover >= 0.8)
    if len(pool) < 10:
        pool = np.flatnonzero(cover >= 0.5)
    if len(pool) == 0:
        return []
    cand = rng.choice(pool, size=min(NCAND, len(pool)), replace=False)
    lk_key, lk_gid, T = G_["lk_key"], G_["lk_gid"], G_["T"]
    cc, pos = G_["cc"], G_["pos"]
    res = []
    for H in cand:
        q = gcs * T + H
        p = np.searchsorted(lk_key, q)
        p = np.minimum(p, len(lk_key) - 1)
        ok = lk_key[p] == q
        h = lk_gid[p]
        far = ok & ((cc[h] != cc[a]) | (np.abs(pos[h] - pos[a]) >= FAR)) & ((cc[h] != cc[b]) | (np.abs(pos[h] - pos[b]) >= FAR))
        sub = np.flatnonzero(far)
        if len(sub) < MIN_SUB:
            continue
        CH = cosmat(e_one(h[sub]))
        des = Design(gcs[sub])
        res.append((abs(spearman_S(CH, des) - target_rho), int(H), sub, CH, des))
    res.sort(key=lambda t: t[0])
    return res[:NBG]


def one_pair(args):
    key, lo, hi, s, e = args
    t0 = time.time()
    rng = np.random.default_rng([SEED, int(key) % (2 ** 32)])
    I = G_["inst"]
    idx = np.arange(s, e)
    n_all = len(idx)
    if n_all > CAP:
        idx = np.sort(rng.choice(idx, CAP, replace=False))
    a, b, gcs = I["a"][idx], I["b"][idx], I["gc"][idx]
    EF, EG = e_pair(a, b)
    CF, CG = cosmat(EF), cosmat(EG)
    des = Design(gcs)
    out = dict(key=int(key), lo=int(lo), hi=int(hi), n=int(n_all), n_used=int(len(idx)))
    out["M"] = M_stat(CF, CG, des)
    out["rhoS_F"], out["rhoS_G"] = spearman_S(CF, des), spearman_S(CG, des)
    # plain Spearman of the two similarity matrices (no partialling), for reference
    out["mantel_raw"] = corr(rankdata(CF[des.iu]) - (len(des.iu[0]) + 1) / 2, rankdata(CG[des.iu]) - (len(des.iu[0]) + 1) / 2)
    out["p_hom"], out["movable_hom"] = perm_p(CF, CG, des, G_["g_hom"][gcs], rng)
    out["p_fam"], out["movable_fam"] = perm_p(CF, CG, des, G_["g_fam"][gcs], rng)
    out["cca"] = cca_oos(EF, EG, gcs)
    keep = np.array([lo, hi])
    dF, dG, ccF, ccG = [], [], [], []
    for _, H, sub, CH, dsub in background("F", a, b, gcs, keep, out["rhoS_G"], rng):
        mfg = M_stat(CF[np.ix_(sub, sub)], CG[np.ix_(sub, sub)], dsub)
        dF.append(mfg - M_stat(CF[np.ix_(sub, sub)], CH, dsub))
    for _, H, sub, CH, dsub in background("G", a, b, gcs, keep, out["rhoS_F"], rng):
        mfg = M_stat(CF[np.ix_(sub, sub)], CG[np.ix_(sub, sub)], dsub)
        dG.append(mfg - M_stat(CH, CG[np.ix_(sub, sub)], dsub))
    out["n_bgF"], out["n_bgG"] = len(dF), len(dG)
    out["E_F"] = float(np.mean(dF)) if dF else np.nan
    out["E_G"] = float(np.mean(dG)) if dG else np.nan
    allD = dF + dG
    out["E"] = float(np.mean(allD)) if (len(dF) >= 3 and len(dG) >= 3) else np.nan
    out["E_sd"] = float(np.std(allD)) if allD else np.nan
    out["sec"] = time.time() - t0
    return out


# --------------------------------------------------------------------------------------------- main
def main():
    threads = int(os.environ.get("SIG_THREADS", "1"))
    t0 = time.time()
    g = load_genes()
    tok = load_tokens()
    uid = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    gs = pd.read_parquet(W / "gene_stats.parquet", columns=["eligible"])
    elig = gs.eligible.to_numpy(bool)
    inst, T = build_instances(g, tok, uid, elig)
    log("instances (distinct)", len(inst), "pairs", inst.key.nunique(), f"{time.time() - t0:.0f}s")

    # pair-level attributes
    inst["o1s"] = (inst.off == 1) & inst.same
    inst["o1o"] = (inst.off == 1) & ~inst.same
    inst["o2"], inst["o3"] = inst.off == 2, inst.off == 3
    grp = inst.groupby("key", sort=True)
    pa = pd.DataFrame(dict(n=grp.size(), f_o1s=grp.o1s.mean(), f_o1o=grp.o1o.mean(), f_o2=grp.o2.mean(), f_o3=grp.o3.mean(),
                           med_gap=inst[inst.o1s].groupby("key").gap.median()))
    pa["lo"], pa["hi"] = pa.index // T, pa.index % T
    sig = pd.read_parquet(ENG / "significant_pairs.parquet", columns=["key", "lo", "hi", "L", "P"])
    skey = sig.lo.to_numpy(np.int64) * T + sig.hi.to_numpy(np.int64)
    sig.index = skey
    pa["called"] = pa.index.isin(skey)
    pa["L"] = sig.L.reindex(pa.index).to_numpy()
    pa["P"] = sig.P.reindex(pa.index).to_numpy()
    inst_sorted = inst.sort_values("key", kind="mergesort").reset_index(drop=True)
    kk = inst_sorted.key.to_numpy()
    first = np.searchsorted(kk, pa.index.to_numpy(), "left")
    last = np.searchsorted(kk, pa.index.to_numpy(), "right")
    pa["s"], pa["e"] = first, last

    # strata and evaluation keys
    rng = np.random.default_rng(SEED)
    big = pa[pa.n >= 30]
    strata = {
        "S_op": big[big.called & (big.f_o1s > 0.5) & (big.med_gap <= 50)],
        "S_opp": big[big.f_o1o > 0.5],
        "S_unc": big[~big.called & (big.f_o1s > 0.5)],
    }
    extra = os.environ.get("SIG_EXTRA", "")
    if extra == "offsets":          # exploratory, after the main run: called pairs whose instances sit mostly 2 or 3 apart
        strata = {"S_o2": big[big.called & (big.f_o2 > 0.5)], "S_o3": big[big.called & (big.f_o3 > 0.5)]}
    sel = {}
    for nm, d in strata.items():
        ks = d.index.to_numpy()
        pick = np.sort(rng.choice(ks, min(1000, len(ks)), replace=False)) if len(ks) else ks
        sel[nm] = pick
        log(nm, "available", len(ks), "sampled", len(pick))
    f = pd.read_parquet(RD / "grammar/prep/folded.parquet")
    pos1 = pd.read_csv(RD / "signals/code/screen1_positive.tsv", sep="\t")
    ab = pos1.gids.str.split(";", expand=True).astype(np.int64)
    fk = np.r_[np.minimum(tok[f.g0], tok[f.g1]) * T + np.maximum(tok[f.g0], tok[f.g1]),
               np.minimum(tok[ab[0]], tok[ab[1]]) * T + np.maximum(tok[ab[0]], tok[ab[1]])]
    cur = pd.read_parquet(W / "curated_pairs.parquet")
    abc = np.array([min(x, y) * T + max(x, y) for x, y in ((81718, 393054), (393054, 130016), (81718, 130016))])
    ev = np.unique(np.r_[fk, cur.key.to_numpy(), abc])
    ev = ev[np.isin(ev, pa.index[pa.n >= 10])]
    log("evaluation keys with n >= 10:", len(ev))
    todo = np.unique(np.r_[np.concatenate(list(sel.values())), ev]) if not extra else np.unique(np.concatenate(list(sel.values())))
    smoke = int(os.environ.get("SIG_SMOKE", "0"))
    if smoke:
        todo = np.unique(np.r_[abc[np.isin(abc, pa.index)], rng.choice(todo, smoke, replace=False)])
    log("pairs to score", len(todo))

    # globals for workers
    X = np.load(W / "zca10_uid_f32.npy", mmap_mode="r")
    G_.update(X=X, fam_row=np.load(W / "fam_row.npy"), S=np.load(W / "fam_sum.npy"), m=np.load(W / "fam_m.npy"),
              tok=tok, uid=uid, gc=g.genome_code.to_numpy(np.int64), cc=g.contig_code.to_numpy(np.int64),
              pos=g.position_index.to_numpy(np.int64), B=np.load(W / "genome_B.npy"), ng=np.load(W / "genome_n.npy"),
              sim=np.load(LF / "sim/emb_mean.npy", mmap_mode="r"), T=T,
              g_sp=load_groups("emb_mean_average_t0.753"), g_fam=load_groups("fam_jac_average_t0.2"),
              g_hom=load_groups("emb_hom_average_t0.75"),
              inst=dict(a=inst_sorted.a.to_numpy(np.int64), b=inst_sorted.b.to_numpy(np.int64),
                        gc=inst_sorted.gc.to_numpy(np.int64)))
    sim = G_["sim"]
    assert sim.shape == (3564, 3564)
    G_["sim"] = np.asarray(sim, dtype=np.float32)
    ge = np.flatnonzero(elig)
    gk = G_["gc"][ge] * T + tok[ge]
    o = np.argsort(gk, kind="mergesort")
    gk, gg = gk[o], ge[o]
    firstk = np.r_[True, gk[1:] != gk[:-1]]
    G_["lk_key"], G_["lk_gid"] = gk[firstk], gg[firstk]
    G_["pres"] = sp.csr_matrix((np.ones(firstk.sum(), dtype=np.float32), (G_["gc"][gg[firstk]], tok[gg[firstk]])),
                               shape=(3564, T))
    log("lookup", len(G_["lk_key"]), f"{time.time() - t0:.0f}s")

    jobs = [(k, int(pa.at[k, "lo"]), int(pa.at[k, "hi"]), int(pa.at[k, "s"]), int(pa.at[k, "e"])) for k in todo]
    jobs.sort(key=lambda j: -(j[4] - j[3]))            # largest first for load balance
    res = []
    with Pool(threads) as pool:
        for i, r in enumerate(pool.imap_unordered(one_pair, jobs, chunksize=1)):
            res.append(r)
            if (i + 1) % 200 == 0:
                log(f"{i + 1}/{len(jobs)}", f"{time.time() - t0:.0f}s")
    R = pd.DataFrame(res).set_index("key")
    R = R.join(pa[["f_o1s", "f_o1o", "f_o2", "f_o3", "med_gap", "called", "L", "P"]])
    for nm, ks in sel.items():
        R[nm] = R.index.isin(ks)
    R["eval"] = R.index.isin(ev)
    if smoke:
        R.reset_index().to_parquet(W / "s1_pairs_smoke.parquet", index=False)
        print(R.drop(columns=["lo", "hi"]).round(3).to_string())
        return
    if extra:
        R.reset_index().to_parquet(W / f"s1_pairs_{extra}.parquet", index=False)
        return
    R.reset_index().to_parquet(W / "s1_pairs.parquet", index=False)
    pa.reset_index().rename(columns={"index": "key"}).drop(columns=["s", "e"]).to_parquet(W / "s1_pair_attrs.parquet", index=False)
    json.dump(dict(instances=len(inst), pairs=int(len(pa)), scored=len(R), seconds=time.time() - t0,
                   strata={k: dict(available=int(len(strata[k])), sampled=int(len(v))) for k, v in sel.items()}),
              open(W / "s1_summary.json", "w"), indent=1)
    log("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
