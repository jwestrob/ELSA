#!/usr/bin/env python
"""ws_calib Q1: what the similarity beyond the MMseqs2 family is (evaluation only).

usage
  q1.py pairs --threads T      sample neighbour pairs (rank bands, family-absent best matches by cosine band) and
                               baselines (uniform, length-matched); write work/q1/pairs.parquet and work/q1/pref.tsv
                               (explicit pairs for `mmseqs align`)
  q1.py classify --threads T   evidence lines per pair (alignment, Pfam family, clan, CDD superfamily, family-profile
                               link, curated role, physical kind), predeclared partition, shares by band with query
                               bootstrap; Cas-pattern generalisation (H4). Writes results/q1_*.json / .tsv
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
C = RD / "calib"
P = C / "work/prep"
GP = C / "work/gpu"
W = C / "work/q1"
RES = C / "results"
R3 = RD / "softrec/r3"
TOK = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
COS_EDGES = np.array([0.20, 0.32, 0.45, 0.60, 1.01])
COS_FINE = np.array([0.20, 0.25, 0.32, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 1.01])
KIND_NAMES = ["rank1_10", "rank11_100", "rank101_1000", "cos20_32", "cos32_45", "cos45_60", "cos60_100"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def jdump(path, obj):
    path.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def load_common():
    meta = np.load(R3 / "vec/meta.npz")
    uq = meta["uq"].astype(np.int64)
    gs = meta["g_start"].astype(np.int64)
    n = len(uq)
    gc = np.repeat(np.arange(len(gs)), np.diff(np.r_[gs, n])).astype(np.int64)
    tokdf = pd.read_parquet(TOK)
    tok = np.full(n, -1, np.int64)
    tok[tokdf.gid.to_numpy()] = tokdf.token.to_numpy()
    return meta, uq, gs, gc, tok


# ============================================================================================ pairs
def cmd_pairs(a):
    W.mkdir(parents=True, exist_ok=True)
    meta, uq, gs, gc, tok = load_common()
    NG = len(gs)
    uf = np.load(P / "uidfeat.npz")
    tok_u = uf["tok"]
    Q = np.load(P / "queries.npz")
    quid, strat = Q["uid"], Q["stratum"]
    NQ = len(quid)
    topi = np.load(GP / "topi.npy")
    topv = np.load(GP / "topv.npy")
    top8 = np.load(GP / "top8.npy", mmap_mode="r")
    arg1 = np.load(GP / "arg1.npy", mmap_mode="r")
    # family presence and query genomes
    fam_key = np.unique(tok * NG + gc)
    uid_key = np.unique(uq * NG + gc)
    glen = np.load(P / "uidfeat.npz")["length"][uq]          # gene lengths via their sequence
    order_len = np.argsort(glen, kind="stable")
    slen = glen[order_len]
    rng = np.random.default_rng([17, 11])
    rows = []
    trunc = 0
    for j in range(NQ):
        q = int(quid[j])
        tq = int(tok_u[q])
        ti = topi[j]
        tv = topv[j].astype(np.float32)
        keep = (ti != q) & (tok_u[ti] != tq)
        ti, tv = ti[keep][:1000], tv[keep][:1000]
        if len(ti) < 1000:
            trunc += 1
        rk = np.arange(1, len(ti) + 1)
        for kind, lo, hi, take in ((0, 1, 10, 10), (1, 11, 100, 10), (2, 101, 1000, 10)):
            idx = np.flatnonzero((rk >= lo) & (rk <= hi))
            if len(idx) > take:
                idx = np.sort(rng.choice(idx, take, replace=False))
            for i in idx:
                rows.append((j, q, int(ti[i]), kind, int(rk[i]), float(tv[i]), -1))
        # family-absent best matches by cosine band
        fpres = np.zeros(NG, bool)
        lo_, hi_ = np.searchsorted(fam_key, [tq * NG, (tq + 1) * NG])
        fpres[fam_key[lo_:hi_] - tq * NG] = True
        qg = np.zeros(NG, bool)
        lo_, hi_ = np.searchsorted(uid_key, [q * NG, (q + 1) * NG])
        qg[uid_key[lo_:hi_] - q * NG] = True
        m1 = np.asarray(top8[j, :, 0], dtype=np.float32)
        ok = ~fpres & ~qg
        band = np.digitize(m1, COS_EDGES) - 1
        for bnd in range(4):
            cells = np.flatnonzero(ok & (band == bnd))
            if len(cells) > 5:
                cells = np.sort(rng.choice(cells, 5, replace=False))
            for G in cells:
                t_gid = gs[G] + int(arg1[j, G])
                rows.append((j, q, int(uq[t_gid]), 3 + bnd, int(G), float(m1[G]), -1))
    pr = pd.DataFrame(rows, columns=["j", "q", "t", "kind", "rank_or_genome", "cos", "match_of"])
    log("neighbour pairs", len(pr), "queries with < 1,000 non-family in top-2,000:", trunc)
    # baselines: uniform (20 per query) and length-matched (one per neighbour pair)
    n = len(uq)
    base = []
    for j in range(NQ):
        q = int(quid[j])
        tq = int(tok_u[q])
        got = 0
        while got < 20:
            g = rng.integers(n, size=40)
            g = g[tok[g] != tq][:20 - got]
            for x in g:
                base.append((j, q, int(uq[x]), 7, -1, np.nan, -1))
            got += len(g)
    t_len = uf["length"][pr.t.to_numpy()]
    lo = np.searchsorted(slen, np.floor(0.9 * t_len), side="left")
    hi = np.searchsorted(slen, np.ceil(1.1 * t_len), side="right")
    qtok = tok_u[pr.q.to_numpy()]
    for i in range(len(pr)):
        for _ in range(20):
            x = order_len[rng.integers(lo[i], max(hi[i], lo[i] + 1))]
            if tok[x] != qtok[i]:
                break
        base.append((int(pr.j.iat[i]), int(pr.q.iat[i]), int(uq[x]), 8, int(pr.kind.iat[i]), np.nan, i))
    pr = pd.concat([pr, pd.DataFrame(base, columns=pr.columns)], ignore_index=True)
    pr.to_parquet(W / "pairs.parquet")
    u = pr.loc[pr.q != pr.t, ["q", "t"]].drop_duplicates().sort_values(["q", "t"])
    with open(W / "pref.tsv", "w") as fh:
        for q, t in zip(u.q.to_numpy(), u.t.to_numpy()):
            fh.write(f"{q}\t{t}\t0\t0\n")
    jdump(W / "pairs.json", dict(pairs=len(pr), by_kind=pr.kind.value_counts().sort_index().to_dict(),
                                 align_pairs=len(u), truncated_lists=trunc, seconds=round(time.time() - T0, 1)))
    log("pairs total", len(pr), "unique align pairs", len(u))


# ============================================================================================ classify
@njit(cache=True)
def inter(ptr, val, a, ptr2, val2, b):
    i, ie = ptr[a], ptr[a + 1]
    j, je = ptr2[b], ptr2[b + 1]
    while i < ie and j < je:
        if val[i] == val2[j]:
            return True
        if val[i] < val2[j]:
            i += 1
        else:
            j += 1
    return False


@njit(parallel=True, cache=True)
def evidence(q, t, pfp, pfv, grp, grv, prp, prv, pdp, pdv, cdp, cdv, rop, rov, out):
    """out columns: 0 P pfam family, 1 clan (grp, not P), 2 CDD superfamily, 3 profile link, 4 role,
    5 q annotated (pfam or cdd), 6 t annotated."""
    for k in prange(len(q)):
        a, b = q[k], t[k]
        P_ = inter(pfp, pfv, a, pfp, pfv, b)
        G_ = inter(grp, grv, a, grp, grv, b)
        D_ = inter(cdp, cdv, a, cdp, cdv, b)
        F_ = (inter(prp, prv, a, grp, grv, b) or inter(prp, prv, a, prp, prv, b) or inter(prp, prv, b, grp, grv, a)
              or inter(pdp, pdv, a, pdp, pdv, b))
        R_ = inter(rop, rov, a, rop, rov, b)
        out[k, 0] = P_
        out[k, 1] = G_ and not P_
        out[k, 2] = D_
        out[k, 3] = F_
        out[k, 4] = R_
        out[k, 5] = (pfp[a + 1] > pfp[a]) or (cdp[a + 1] > cdp[a])
        out[k, 6] = (pfp[b + 1] > pfp[b]) or (cdp[b + 1] > cdp[b])


def tm_class(n):
    return np.digitize(n, [1, 2, 5])


def boot_shares(jidx, groups, ngroups, flags, nq, draws=2000, seed=17):
    """Share of each flag within each group, pair-weighted, with a query bootstrap. jidx: query index per pair (0..nq-1).
    Returns point (ngroups x F), lo, hi."""
    F = flags.shape[1]
    num = np.zeros((nq, ngroups, F))
    den = np.zeros((nq, ngroups))
    np.add.at(num, (jidx, groups), flags.astype(np.float64))
    np.add.at(den, (jidx, groups), 1.0)
    point = num.sum(0) / np.maximum(den.sum(0), 1)[:, None]
    rng = np.random.default_rng(seed)
    Wt = rng.multinomial(nq, np.full(nq, 1.0 / nq), size=draws).astype(np.float64)
    bn = np.einsum("dq,qgf->dgf", Wt, num)
    bd = Wt @ den
    bs = bn / np.maximum(bd, 1)[:, :, None]
    return point, np.quantile(bs, 0.025, axis=0), np.quantile(bs, 0.975, axis=0), den.sum(0)


def cmd_classify(a):
    import numba
    numba.set_num_threads(a.threads)
    RES.mkdir(parents=True, exist_ok=True)
    pr = pd.read_parquet(W / "pairs.parquet")
    uf = np.load(P / "uidfeat.npz")
    A = np.load(P / "ann.npz")
    Q = np.load(P / "queries.npz")
    quid, strat = Q["uid"], Q["stratum"]
    NQ = len(quid)
    # alignment
    if not (W / "aln.parquet").exists():
        cols = ["query", "target", "evalue", "bits", "fident", "alnlen", "qstart", "qend", "qlen", "tstart", "tend", "tlen"]
        pd.read_csv(W / "aln.tsv", sep="\t", header=None, names=cols).to_parquet(W / "aln.parquet")
    aln = pd.read_parquet(W / "aln.parquet")
    aln = aln.sort_values("evalue").drop_duplicates(["query", "target"])
    key = pd.Series(aln.evalue.to_numpy(), index=pd.MultiIndex.from_arrays([aln["query"].to_numpy(), aln["target"].to_numpy()]))
    ev = key.reindex(pd.MultiIndex.from_arrays([pr.q.to_numpy(), pr.t.to_numpy()])).to_numpy()
    ev = np.where(pr.q.to_numpy() == pr.t.to_numpy(), 0.0, ev)
    fid = pd.Series(aln.fident.to_numpy(), index=key.index).reindex(pd.MultiIndex.from_arrays([pr.q.to_numpy(), pr.t.to_numpy()])).to_numpy()
    A_al = np.nan_to_num(ev, nan=np.inf) <= 1e-3
    q = pr.q.to_numpy(np.int64)
    t = pr.t.to_numpy(np.int64)
    out = np.zeros((len(pr), 7), np.bool_)
    evidence(q, t, A["pfam_ptr"], A["pfam_val"], A["grp_ptr"], A["grp_val"], A["prof_ptr"], A["prof_val"],
             A["pdb_ptr"], A["pdb_val"], A["cdd_ptr"], A["cdd_val"], A["role_ptr"], A["role_val"], out)
    L = uf["length"].astype(np.float64)
    ntm = uf["ntm"]
    comp = uf["comp"].astype(np.float32)
    lr = np.minimum(L[q], L[t]) / np.maximum(L[q], L[t])
    cd = np.abs(comp[q] - comp[t]).sum(1)
    same_tm = tm_class(ntm[q]) == tm_class(ntm[t])
    kind = pr.kind.to_numpy()
    isR = strat[pr.j.to_numpy()] == 0
    cd25 = float(np.quantile(cd[(kind == 7) & isR], 0.25))
    K = same_tm & (lr >= 0.67) & (cd <= cd25)
    P_, C_, D_, F_, R_, aq, at = (out[:, i] for i in range(7))
    HOM = A_al | P_ | C_ | D_ | F_
    ROLE = R_ & ~HOM
    KIND = K & ~HOM & ~ROLE
    REST = ~HOM & ~ROLE & ~KIND
    contra = aq & at & ~(P_ | C_ | D_)
    log("pairs", len(pr), "comp L1 p25 uniform", round(cd25, 4))
    flags = np.column_stack([HOM, A_al, P_, C_, D_, F_, A_al & ~(P_ | C_ | D_ | F_), (P_ | C_ | D_ | F_) & ~A_al,
                             ROLE, KIND & contra, KIND & ~contra, REST & contra, REST & ~contra,
                             aq & at, contra, same_tm, lr >= 0.67, cd <= cd25, K, (ntm[q] >= 1) & (ntm[t] >= 1),
                             R_])
    fnames = ["HOM", "A_align", "P_pfam", "C_clan_only", "D_cdd", "F_profile", "align_only", "annot_only_no_align",
              "ROLE", "KIND_contradicted", "KIND_unknown", "REST_contradicted", "REST_unknown",
              "both_annotated", "contradicted_any", "same_TM_class", "len_ratio_ge_067", "comp_le_p25", "K_kind",
              "both_TM", "same_role_any"]
    # group: neighbour kind 0-6; uniform 7; length-matched baseline by the band of its neighbour: 8 + band
    grp = np.where(kind == 8, 8 + pr.rank_or_genome.to_numpy(), kind)
    names = KIND_NAMES + ["uniform"] + ["lenmatch_" + k for k in KIND_NAMES]
    jj = pr.j.to_numpy()
    res = {}
    for sname, smask in [("R", strat == 0), ("all_strata", np.ones(NQ, bool))]:
        qsel = np.flatnonzero(smask)
        remap = np.full(NQ, -1)
        remap[qsel] = np.arange(len(qsel))
        m = remap[jj] >= 0
        pt, lo, hi, nn = boot_shares(remap[jj[m]], grp[m], len(names), flags[m], len(qsel))
        res[sname] = {names[g]: dict(n=int(nn[g]), **{fnames[f]: [round(pt[g, f], 4), round(lo[g, f], 4), round(hi[g, f], 4)]
                                                        for f in range(len(fnames))}) for g in range(len(names))}
        # evaluable pairs only (both annotated)
        m2 = m & (aq & at)
        pt, lo, hi, nn = boot_shares(remap[jj[m2]], grp[m2], len(names), flags[m2], len(qsel))
        res[sname + "_evaluable"] = {names[g]: dict(n=int(nn[g]), **{fnames[f]: [round(pt[g, f], 4), round(lo[g, f], 4), round(hi[g, f], 4)]
                                                                     for f in range(len(fnames))}) for g in range(len(names))}
        log(sname, "done")
    # fine cosine table (R queries, family-absent best matches), and excess over the length-matched baseline
    m = isR & (kind >= 3) & (kind <= 6)
    fb = np.digitize(pr.cos.to_numpy()[m], COS_FINE) - 1
    qsel = np.flatnonzero(strat == 0)
    remap = np.full(NQ, -1)
    remap[qsel] = np.arange(len(qsel))
    pt, lo, hi, nn = boot_shares(remap[jj[m]], fb, len(COS_FINE) - 1, flags[m], len(qsel))
    res["R_cos_fine"] = {f"{COS_FINE[b]:.2f}-{COS_FINE[b + 1]:.2f}": dict(n=int(nn[b]), **{fnames[f]: [round(pt[b, f], 4), round(lo[b, f], 4), round(hi[b, f], 4)]
                                                                                   for f in range(len(fnames))}) for b in range(len(COS_FINE) - 1)}
    m = isR & (kind == 8)
    fbl = np.digitize(pr.cos.to_numpy()[pr.match_of.to_numpy()[m]], COS_FINE) - 1
    okb = pr.kind.to_numpy()[pr.match_of.to_numpy()[m]] >= 3
    mm = np.flatnonzero(m)[okb]
    pt, lo, hi, nn = boot_shares(remap[jj[mm]], fbl[okb], len(COS_FINE) - 1, flags[mm], len(qsel))
    res["R_cos_fine_lenmatch"] = {f"{COS_FINE[b]:.2f}-{COS_FINE[b + 1]:.2f}": dict(n=int(nn[b]), **{fnames[f]: [round(pt[b, f], 4), round(lo[b, f], 4), round(hi[b, f], 4)]
                                                                                            for f in range(len(fnames))}) for b in range(len(COS_FINE) - 1)}
    # strata of the query (all strata pooled; descriptive): HOM share by kind for family size, length, annotation, TM
    fs = uf["fam_genes"][q]
    strata = {
        "famsize": np.digitize(fs, [2, 5, 20, 100, 500]),
        "length": np.digitize(L[q], [100, 200, 400]),
        "q_annot": (np.diff(A["pfam_ptr"])[q] > 0).astype(int) * 2 + (np.diff(A["cdd_ptr"])[q] > 0).astype(int),
        "q_TM": tm_class(ntm[q]),
        "q_curated": A["callbit"][q].astype(int),
    }
    sres = {}
    for sn, sv in strata.items():
        sres[sn] = {}
        for v in np.unique(sv):
            mm = sv == v
            d = {}
            for g in range(len(names)):
                g_m = mm & (grp == g)
                if g_m.sum() >= 50:
                    d[names[g]] = dict(n=int(g_m.sum()), HOM=round(float(HOM[g_m].mean()), 4), A=round(float(A_al[g_m].mean()), 4),
                                       KIND=round(float(KIND[g_m].mean()), 4), ROLE=round(float(ROLE[g_m].mean()), 4),
                                       evaluable=round(float((aq & at)[g_m].mean()), 4),
                                       HOM_evaluable=round(float(HOM[g_m & aq & at].mean()) if (g_m & aq & at).sum() else float("nan"), 4))
            sres[sn][int(v)] = d
    res["by_query_stratum"] = sres
    res["alignment"] = dict(pairs_with_alignment_reported=int(np.isfinite(ev).sum()), share_E_le_1e3=float(A_al.mean()),
                            fident_q50_when_A=float(np.nanmedian(fid[A_al & (q != t)])) if (A_al & (q != t)).any() else None)
    res["kind_threshold_comp_l1_p25_uniform"] = cd25
    jdump(RES / "q1_classes.json", res)
    pr2 = pr[["j", "q", "t", "kind", "rank_or_genome", "cos", "match_of"]].copy()
    pr2["evalue"] = ev
    for i, f in enumerate(fnames):
        pr2[f] = flags[:, i]
    pr2.to_parquet(W / "pairs_classified.parquet")
    log("classes written")
    cas_pattern(quid, strat, Q, uf, A)


def cas_pattern(quid, strat, Q, uf, A):
    """H4: same-role / same-caller / same-clan genes of other families in the top-1,000 non-family list."""
    topi = np.load(GP / "topi.npy")
    tok_u = uf["tok"]
    NQ = len(quid)
    voc = json.loads((P / "ann_vocab.json").read_text())
    rp, rv = A["role_ptr"], A["role_val"]
    gp, gv = A["grp_ptr"], A["grp_val"]
    NPf = int(A["n_pfam"])
    callbit = A["callbit"]
    nlist = np.zeros(NQ, np.int64)
    lists = []
    for j in range(NQ):
        q = quid[j]
        ti = topi[j]
        keep = (ti != q) & (tok_u[ti] != tok_u[q])
        lists.append(ti[keep][:1000])
        nlist[j] = len(lists[-1])
    role_of = lambda u: rv[rp[u]:rp[u + 1]]
    U = len(uf["length"])
    cache = {}

    def member_mask(ptr, val, code, tag):
        k = (tag, int(code))
        if k not in cache:
            hit = np.flatnonzero(val == code)
            u = np.searchsorted(ptr, hit, side="right") - 1
            m = np.zeros(U, bool)
            m[u] = True
            cache[k] = m
        return cache[k]

    out = {}
    # roles: Y queries (stratum 3) vs their matched controls (stratum 4, ctrl_of -> Y index)
    ctrl_of = Q["ctrl_of"]
    ctrl_for = {int(ctrl_of[j]): j for j in range(NQ) if ctrl_of[j] >= 0}
    rows = []
    for j in np.flatnonzero(strat == 3):
        roles = role_of(quid[j])
        cb = int(callbit[quid[j]])
        lst = lists[j]
        mask = np.zeros(U, bool)
        for r in roles:
            mask |= member_mask(rp, rv, r, "role")
        n_role = int(mask[lst].sum())
        n_call = int(((callbit[lst] & cb) > 0).sum())
        c = ctrl_for.get(j)
        if c is not None:
            lc = lists[c]
            n_role_c = int(mask[lc].sum())
            n_call_c = int(((callbit[lc] & cb) > 0).sum())
        else:
            n_role_c = n_call_c = np.nan
        for r in roles:
            rows.append((j, voc["roles"][r], cb, n_role, n_call, n_role_c, n_call_c, len(lst)))
    df = pd.DataFrame(rows, columns=["j", "role", "caller", "same_role", "same_caller", "ctrl_same_role", "ctrl_same_caller", "list_len"])
    df.to_csv(RES / "q1_role_counts.tsv", sep="\t", index=False)
    agg = df.groupby("role").agg(n=("j", "size"), same_role=("same_role", "mean"), ctrl=("ctrl_same_role", "mean"),
                                 same_caller=("same_caller", "mean"), ctrl_caller=("ctrl_same_caller", "mean")).reset_index()
    agg["enrich_role"] = agg.same_role / agg.ctrl.clip(lower=1e-3)
    agg["enrich_role_floor0.1"] = agg.same_role / agg.ctrl.clip(lower=0.1)
    agg.sort_values("n", ascending=False).to_csv(RES / "q1_role_enrichment.tsv", sep="\t", index=False)
    big = agg[agg.n >= 30]
    noncas = big[~big.role.str.startswith("cct:Cas") & ~big.role.str.startswith("cct:Cs")]
    out["roles_n_ge_30"] = int(len(big))
    out["noncas_roles_n_ge_30"] = int(len(noncas))
    out["noncas_roles_enrich_ge_10_floor0.1"] = int((noncas["enrich_role_floor0.1"] >= 10).sum())
    dj = df.drop_duplicates("j")
    out["by_caller"] = {int(b): dict(queries=int((dj.caller == b).sum()),
                                     same_role_mean=float(dj[dj.caller == b].same_role.mean()),
                                     ctrl_same_role_mean=float(dj[dj.caller == b].ctrl_same_role.mean()),
                                     same_caller_mean=float(dj[dj.caller == b].same_caller.mean()),
                                     ctrl_same_caller_mean=float(dj[dj.caller == b].ctrl_same_caller.mean()))
                        for b in (1, 2, 4)}
    # clans: R queries; 20 largest clans among R queries' direct Pfam clans
    R = np.flatnonzero(strat == 0)
    qcl = {j: [c - NPf for c in gv[gp[quid[j]]:gp[quid[j] + 1]] if c >= NPf] for j in R}
    cnt = pd.Series([c for j in R for c in qcl[j]]).value_counts()
    top = cnt.index[:20].tolist()
    fbin = np.digitize(uf["fam_genes"][quid], [2, 5, 20, 100, 500])
    validu = np.flatnonzero(uf["valid"])
    crow = []
    for c in top:
        has = member_mask(gp, gv, NPf + c, "grp")
        inn = np.array([j for j in R if c in qcl[j]])
        cnt_in = np.array([has[lists[j]].sum() for j in inn])
        bins_in = fbin[inn]
        ctl = np.array([j for j in R if c not in qcl[j]])
        rng = np.random.default_rng([17, 21, int(c)])
        cnt_c = []
        for bnd in np.unique(bins_in):
            pool = ctl[fbin[ctl] == bnd]
            k = int((bins_in == bnd).sum())
            if len(pool):
                pick = rng.choice(pool, min(len(pool), 3 * k), replace=False)
                cnt_c += [int(has[lists[j]].sum()) for j in pick]
        base_share = float(has[rng.choice(validu, 200000, replace=False)].mean())
        crow.append(dict(clan=voc["clans"][c], queries=len(inn), same_clan_mean=float(cnt_in.mean()),
                         ctrl_mean=float(np.mean(cnt_c)) if cnt_c else float("nan"), base_rate_x1000=base_share * 1000))
    cdf = pd.DataFrame(crow)
    cdf["enrich"] = cdf.same_clan_mean / cdf.ctrl_mean.clip(lower=0.1)
    cdf.to_csv(RES / "q1_clan_enrichment.tsv", sep="\t", index=False)
    out["clans_enrich_ge_3"] = int((cdf.enrich >= 3).sum())
    out["list_len_q50"] = float(np.median(nlist))
    jdump(RES / "q1_h4.json", out)
    log("H4 done", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    dict(pairs=cmd_pairs, classify=cmd_classify)[a.cmd](a)


if __name__ == "__main__":
    main()
