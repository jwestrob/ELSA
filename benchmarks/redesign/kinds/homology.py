#!/usr/bin/env python
"""Coordinator-requested addition (after ws_calib): how much of kind recurrence is homology that MMseqs2 split.

Direct homology evidence between two MMseqs2 (rescue3) families x and f (evaluation only; ws_calib's lines, copied):
  - alignment: a representative of one family (<= 2 members, genes of this workstream's sets first, then seed-17
    random members) hits a member of the other at Smith-Waterman E <= 1e-3 (MMseqs2 18 search, -s 7.5, against the
    2,547,528-sequence uniqdb, --max-seqs 4000);
  - shared Pfam family or Pfam 39 clan, or shared CDD superfamily, on any member of each family (ws_calib ann.npz,
    grp codes = Pfam family + clan, cdd = CDD superfamily);
  - same family.

usage
  homology.py reps     --threads T    family sets, representatives (work/hom/reps_uid.txt), family codes
  homology.py post     --threads T    hits -> symmetric family homology CSR (work/hom/famhom.npz)
  homology.py arch     --threads T    1A homology-independent groups per architecture, class a, passers;
                                      1B support units of every class-a pair by homology to the query
  homology.py withheld --threads T    passes at L1h (query's families + homologs) and L2h (type families + homologs)
  homology.py weval    --threads T    AUROC at L1h / L2h next to full / L1 / L2
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kinds as KD  # noqa: E402

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
K = RD / "kinds"
HW = K / "hom"
ANN = RD / "calib/work/prep/ann.npz"
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"{time.time() - T0:7.0f}s", *a, flush=True)


def csr_from_pairs(keys, vals, n):
    o = np.lexsort((vals, keys))
    k, v = keys[o], vals[o]
    keep = np.r_[True, (k[1:] != k[:-1]) | (v[1:] != v[:-1])]
    k, v = k[keep], v[keep]
    ptr = np.r_[0, np.cumsum(np.bincount(k, minlength=n))].astype(np.int64)
    return ptr, v.astype(np.int64)


# ============================================================================== family sets and codes
def cmd_reps(a):
    HW.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    tokM = np.load(K / "prep/tokM.npy").astype(np.int64)
    uq = np.load(K / "prep/uq.npy")
    TM = int(tokM.max()) + 1
    H = pd.read_parquet(K / "arch/H_pairs.parquet", columns=["i", "j"])
    W = np.load(K / "withheld/design.npz")
    types = pd.read_csv(K / "withheld/types.tsv", sep="\t")
    keep_t = (types.type != "CRISPRCasTyper:False").to_numpy()
    fam_T = np.flatnonzero(W["mask_l2"][keep_t].any(0))
    pri_genes = np.unique(np.r_[H.i, H.j, W["qa"], W["qb"]])
    F = np.unique(np.r_[tokM[pri_genes], fam_T])
    log("families: H+withheld genes", len(np.unique(tokM[pri_genes])), "type families (no 'False')", len(fam_T),
        "union", len(F))
    # members per family (uids), priority genes first
    order = np.argsort(tokM, kind="stable")
    fptr = np.searchsorted(tokM[order], np.arange(TM + 1))
    is_pri = np.zeros(len(tokM), bool)
    is_pri[pri_genes] = True
    reps = []
    for f in F:
        mem = order[fptr[f]:fptr[f + 1]]
        u_pri = np.unique(uq[mem[is_pri[mem]]])
        u_all = np.unique(uq[mem])
        pick = list(rng.permutation(u_pri)[:2])
        if len(pick) < 2:
            rest = np.setdiff1d(u_all, pick)
            pick += list(rng.permutation(rest)[:2 - len(pick)])
        reps += [(int(f), int(u)) for u in pick]
    R = pd.DataFrame(reps, columns=["fam", "uid"])
    R.to_parquet(HW / "reps.parquet", index=False)
    np.savetxt(HW / "reps_uid.txt", np.unique(R.uid.to_numpy()), fmt="%d")
    # family annotation codes: union over all members' uids of grp (Pfam family + clan) and CDD superfamily
    A = np.load(ANN)
    gptr, gval, cptr, cval = A["grp_ptr"], A["grp_val"], A["cdd_ptr"], A["cdd_val"]
    off = int(gval.max()) + 1
    fam_u = pd.DataFrame(dict(f=tokM, u=uq)).drop_duplicates().to_numpy()
    kk, vv = [], []
    for ptr, val, o in ((gptr, gval, 0), (cptr, cval, off)):
        starts = ptr[fam_u[:, 1]]
        n_per = ptr[fam_u[:, 1] + 1] - starts
        rep_f = np.repeat(fam_u[:, 0], n_per)
        idx = np.repeat(starts - np.r_[0, np.cumsum(n_per)[:-1]], n_per) + np.arange(int(n_per.sum()))
        kk.append(rep_f)
        vv.append(val[idx].astype(np.int64) + o)
    cp, cv = csr_from_pairs(np.concatenate(kk), np.concatenate(vv), TM)
    np.savez(HW / "famcodes.npz", ptr=cp, val=cv)
    log("reps", len(R), "unique uids", R.uid.nunique(), "families with codes", int((np.diff(cp) > 0).sum()), "of", TM)


def cmd_post(a):
    tokM = np.load(K / "prep/tokM.npy").astype(np.int64)
    uq = np.load(K / "prep/uq.npy")
    TM = int(tokM.max()) + 1
    R = pd.read_parquet(HW / "reps.parquet")
    h = pd.read_csv(HW / "hits.tsv", sep="\t", header=None, names=["q", "t", "evalue"])
    h = h[h.evalue <= 1e-3]
    # uid -> families (a uid may sit in more than one family)
    uf = pd.DataFrame(dict(u=uq, f=tokM)).drop_duplicates()
    qf = R.rename(columns=dict(uid="q", fam="fq"))
    e = h[["q", "t"]].merge(qf, on="q").merge(uf.rename(columns=dict(u="t", f="ft")), on="t")[["fq", "ft"]]
    e = e[e.fq != e.ft].drop_duplicates()
    k = np.r_[e.fq.to_numpy(), e.ft.to_numpy()]
    v = np.r_[e.ft.to_numpy(), e.fq.to_numpy()]
    hp, hv = csr_from_pairs(k, v, TM)
    np.savez(HW / "famhom.npz", ptr=hp, val=hv)
    nf = np.diff(hp)[R.fam.unique()]
    log("hits", len(h), "family edges (symmetric)", len(hv), "homolog families per searched family q50/q90/max",
        np.quantile(nf, [.5, .9]).tolist(), int(nf.max()))


# ============================================================================== homology test
@njit(cache=True)
def hom(x, f, hp, hv, cp, cv):
    if x == f:
        return True
    lo, hi = hp[f], hp[f + 1]
    while lo < hi:
        mid = (lo + hi) // 2
        if hv[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    if lo < hp[f + 1] and hv[lo] == x:
        return True
    i, ie, j, je = cp[x], cp[x + 1], cp[f], cp[f + 1]
    while i < ie and j < je:
        if cv[i] == cv[j]:
            return True
        if cv[i] < cv[j]:
            i += 1
        else:
            j += 1
    return False


class Hom:
    def __init__(self):
        z = np.load(HW / "famhom.npz")
        c = np.load(HW / "famcodes.npz")
        self.hp, self.hv, self.cp, self.cv = z["ptr"], z["val"], c["ptr"], c["val"]
        self.cache = {}

    def __call__(self, x, f):
        key = (x, f) if x < f else (f, x)
        r = self.cache.get(key)
        if r is None:
            r = bool(hom(int(x), int(f), self.hp, self.hv, self.cp, self.cv))
            self.cache[key] = r
        return r


def groups(fps, h):
    """fps: list of (x, y) family pairs. Returns (n_both_linked, groups_both, groups_any)."""
    n = len(fps)
    pb, pa = list(range(n)), list(range(n))

    def find(p, i):
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i
    linked = np.zeros(n, bool)
    for i in range(n):
        x1, y1 = fps[i]
        for j in range(i + 1, n):
            x2, y2 = fps[j]
            hxx, hyy, hxy, hyx = h(x1, x2), h(y1, y2), h(x1, y2), h(y1, x2)
            if (hxx and hyy) or (hxy and hyx):
                linked[i] = linked[j] = True
                ri, rj = find(pb, i), find(pb, j)
                if ri != rj:
                    pb[ri] = rj
            if hxx or hyy or hxy or hyx:
                ri, rj = find(pa, i), find(pa, j)
                if ri != rj:
                    pa[ri] = rj
    gb = len({find(pb, i) for i in range(n)})
    ga = len({find(pa, i) for i in range(n)})
    return int(linked.sum()), gb, ga


def cmd_arch(a):
    h = Hom()
    tokM = np.load(K / "prep/tokM.npy").astype(np.int64)
    Hp = pd.read_parquet(K / "arch/H_pairs.parquet")
    A = pd.read_csv(K / "arch/architectures.tsv", sep="\t")
    rows = []
    for arch in A.arch:
        g = Hp[Hp.arch == arch]
        fps = sorted({(min(x, y), max(x, y)) for x, y in zip(g.ta, g.tb)})
        nb, gb, ga = groups(fps, h)
        rows.append(dict(arch=int(arch), pairs=len(g), family_pairs=len(fps), fp_both_side_homologous_to_another=nb,
                         groups_homology_independent=gb, groups_no_shared_homolog_either_side=ga))
    T = A.merge(pd.DataFrame(rows), on=["arch", "pairs", "family_pairs"])
    T.to_csv(HW / "arch_homology_groups.tsv", sep="\t", index=False)
    log("architectures", len(T), "family pairs", int(T.family_pairs.sum()), "both-side linked",
        int(T.fp_both_side_homologous_to_another.sum()), "groups", int(T.groups_homology_independent.sum()),
        "groups (no shared homolog either side)", int(T.groups_no_shared_homolog_either_side.sum()))
    # folded set (class a) and passers
    f = pd.read_csv(K / "cofold/folds_kinds.tsv", sep="\t")
    out = {}
    for nm, sel in (("class_a", f.cls == "a_high"), ("class_a_passers", (f.cls == "a_high") & f.pass_c123),
                    ("class_b", f.cls == "b_low")):
        g = f[sel]
        fps = sorted({(min(x, y), max(x, y)) for x, y in zip(tokM[g.gid_A], tokM[g.gid_B])})
        nb, gb, ga = groups(fps, h)
        out[nm] = dict(pairs=int(len(g)), family_pairs=len(fps), fp_both_side_homologous_to_another=nb,
                       groups_homology_independent=gb, groups_no_shared_homolog_either_side=ga)
    log(out)
    # 1B: support units of each class-a pair by homology to the query's families (depth 1,000, observed order)
    d = KD.Data()
    a_ = f[f.cls == "a_high"]
    ptr, left = KD.support_lists(a_.gid_A.to_numpy(np.int64), a_.gid_B.to_numpy(np.int64), d.uq, d.I, 1000, d.mptr,
                                 d.members, d.gc, d.cid, d.U, d.n, d.bufcap)
    rows = []
    for k, r in enumerate(a_.itertuples()):
        fa, fb = int(tokM[r.gid_A]), int(tokM[r.gid_B])
        units = {}
        for l in left[ptr[k]:ptr[k + 1]]:
            x, y = int(tokM[l]), int(tokM[l + 1])
            key = (min(x, y), max(x, y))
            if key in units:
                continue
            both = (h(x, fa) and h(y, fb)) or (h(x, fb) and h(y, fa))
            anyh = h(x, fa) or h(x, fb) or h(y, fa) or h(y, fb)
            units[key] = 2 if both else (1 if anyh else 0)
        c = np.bincount(np.array(list(units.values()), int), minlength=3) if units else np.zeros(3, int)
        rows.append(dict(fold_id=r.fold_id, passed=bool(r.pass_c123), units=len(units), units_both_homologous=int(c[2]),
                         units_one_side=int(c[1]), units_no_homology=int(c[0]), kappa=r.kappa))
    S = pd.DataFrame(rows)
    S.to_csv(HW / "classa_support_homology.tsv", sep="\t", index=False)
    for nm, m in (("class_a", np.ones(len(S), bool)), ("passers", S.passed.to_numpy())):
        s = S[m]
        tot = s[["units", "units_both_homologous", "units_one_side", "units_no_homology"]].sum()
        out[f"support_{nm}"] = dict(queries=int(len(s)), units=int(tot.units),
                                   share_both=float(tot.units_both_homologous / tot.units),
                                   share_one=float(tot.units_one_side / tot.units),
                                   share_none=float(tot.units_no_homology / tot.units),
                                   median_units_no_homology=float(s.units_no_homology.median()),
                                   median_units_one_side=float(s.units_one_side.median()),
                                   median_units_both=float(s.units_both_homologous.median()),
                                   queries_with_ge3_no_homology_units=int((s.units_no_homology >= 3).sum()))
    json.dump(out, open(HW / "folded_homology.json", "w"), indent=1)
    log(json.dumps(out, indent=1))


# ============================================================================== withheld with homologs
def cmd_withheld(a):
    import hom_kernel as HK
    d = KD.Data()
    W = np.load(K / "withheld/design.npz")
    types = pd.read_csv(K / "withheld/types.tsv", sep="\t")
    z = np.load(HW / "famhom.npz")
    c = np.load(HW / "famcodes.npz")
    hp, hv, cp, cv = z["ptr"], z["val"], c["ptr"], c["val"]
    TM = d.TM
    # L2h masks: type families + every family homologous to one of them
    inv_ord = np.argsort(cv, kind="stable")
    code_fam = np.repeat(np.arange(TM), np.diff(cp))[inv_ord]
    code_sorted = cv[inv_ord]
    mask = np.zeros((len(types), TM), np.bool_)
    for t in range(len(types)):
        fams = np.flatnonzero(W["mask_l2"][t])
        m = mask[t]
        m[fams] = True
        for f in fams:
            m[hv[hp[f]:hp[f + 1]]] = True
        codes = np.unique(np.concatenate([cv[cp[f]:cp[f + 1]] for f in fams])) if len(fams) else np.zeros(0, int)
        lo = np.searchsorted(code_sorted, codes, "left")
        hi = np.searchsorted(code_sorted, codes, "right")
        for x, y in zip(lo, hi):
            m[code_fam[x:y]] = True
        types.loc[t, "families_withheld_L2h"] = int(m.sum())
    types.to_csv(HW / "types_L2h.tsv", sep="\t", index=False)
    log(types[["type", "n_families_withheld", "families_withheld_L2h"]].to_string(index=False))
    qa, qb = W["qa"], W["qb"]
    # reproduction checks of the copied kernel on 3,000 queries (observed order): no withholding = 'full'; empty
    # homology tables with excl_hom = L1 (own families only); original type masks = L2
    gene_at, slot_of = d.order(-1)
    nchk = 3000
    e1 = np.zeros(TM + 1, np.int64)
    e0 = np.zeros(0, np.int64)
    for lvl, eh, mk, mid, hpx, hvx, cpx, cvx in (
            ("full", False, np.zeros((1, 1), np.bool_), np.full(nchk, -1, np.int32), hp, hv, cp, cv),
            ("L1", True, np.zeros((1, 1), np.bool_), np.full(nchk, -1, np.int32), e1, e0, e1, e0),
            ("L2", False, W["mask_l2"], W["mid_l2"][:nchk].astype(np.int32), hp, hv, cp, cv)):
        o = np.zeros((nchk, len(KD.DEPTHS), 8), np.int32)
        HK.kind_pass_hom(qa[:nchk], qb[:nchk], d.uq, d.I, int(KD.DEPTHS.max()), KD.DEPTHS, d.mptr, d.members, d.gc,
                         slot_of, gene_at, d.cid, d.tokM, d.tokF, d.TF, eh, mk, mid, hpx, hvx, cpx, cvx, d.U, o,
                         np.zeros(nchk, np.int8), a.threads * 16, d.bufcap)
        ref_o = np.load(K / f"withheld/pass_{lvl}_obs.npz")["out"][:nchk, :, :4]
        ok = np.array_equal(o[:, :, :4], ref_o)
        log("kernel reproduction", lvl, "equal" if ok else "MISMATCH", int((o[:, :, :4] != ref_o).sum()))
        assert ok, lvl
    # hom() against a direct set computation on random family pairs
    rng = np.random.default_rng(5)
    fs = rng.choice(np.flatnonzero(np.diff(hp) > 0), 300)
    for f in fs[:300]:
        cand = np.r_[hv[hp[f]:hp[f + 1]][:3], rng.integers(0, TM, 3)]
        for x in cand:
            want = (x == f) or (x in set(hv[hp[f]:hp[f + 1]])) or bool(
                set(cv[cp[x]:cp[x + 1]]) & set(cv[cp[f]:cp[f + 1]]))
            assert hom(int(x), int(f), hp, hv, cp, cv) == want
    log("hom() agrees with set computation on 1,800 family pairs")
    for lvl in ("L1h", "L2h"):
        for r in [-1, 0, 1, 2]:
            fo = HW / f"pass_{lvl}_{'obs' if r < 0 else 'null%d' % r}.npz"
            if fo.exists():
                continue
            t0 = time.time()
            gene_at, slot_of = d.order(r)
            nq = len(qa)
            out = np.zeros((nq, len(KD.DEPTHS), 8), np.int32)
            ovf = np.zeros(nq, np.int8)
            if lvl == "L1h":
                HK.kind_pass_hom(qa, qb, d.uq, d.I, int(KD.DEPTHS.max()), KD.DEPTHS, d.mptr, d.members, d.gc,
                                 slot_of, gene_at, d.cid, d.tokM, d.tokF, d.TF, True, np.zeros((1, 1), np.bool_),
                                 np.full(nq, -1, np.int32), hp, hv, cp, cv, d.U, out, ovf, a.threads * 16, d.bufcap)
            else:
                HK.kind_pass_hom(qa, qb, d.uq, d.I, int(KD.DEPTHS.max()), KD.DEPTHS, d.mptr, d.members, d.gc,
                                 slot_of, gene_at, d.cid, d.tokM, d.tokF, d.TF, False, mask,
                                 W["qtype"].astype(np.int32), hp, hv, cp, cv, d.U, out, ovf, a.threads * 16, d.bufcap)
            np.savez(fo, out=out, ovf=ovf)
            log(f"pass {lvl} r={r} {time.time() - t0:.0f}s overflow {int(ovf.sum())}")


def cmd_weval(a):
    from bench_eval import Boot
    import numba
    numba.set_num_threads(a.threads)
    W = np.load(K / "withheld/design.npz")
    qt, pos = W["qtype"], W["is_pos"]
    gc = np.load(K / "prep/gc.npy")[W["qa"]]
    types = pd.read_csv(HW / "types_L2h.tsv", sep="\t")
    v, s = 3, 2                                    # F_D1000_excess (selected)
    Kx = {}
    for lvl, d in (("full", K / "withheld"), ("L1", K / "withheld"), ("L2", K / "withheld"), ("L1h", HW), ("L2h", HW)):
        O = np.load(d / f"pass_{lvl}_obs.npz")["out"][:, v, s].astype(np.float64)
        R0 = np.mean([np.load(d / f"pass_{lvl}_null{r}.npz")["out"][:, v, s] for r in range(3)], axis=0)
        Kx[lvl] = O - R0
        Kx[lvl + "_R"] = O
    rows = []
    for t, ty in enumerate(types.type):
        m = qt == t
        y = pos[m]
        r = dict(type=ty, n_pos=int(y.sum()))
        for lvl in ("full", "L1", "L1h", "L2", "L2h"):
            r[f"auc_{lvl}"] = Boot(gc[m], draws=0).auc(Kx[lvl][m], y, draws=False)[0]
            r[f"pos_R_med_{lvl}"] = float(np.median(Kx[lvl + "_R"][m][y]))
        rows.append(r)
    T = types.merge(pd.DataFrame(rows), on=["type", "n_pos"])
    T.to_csv(HW / "withheld_hom_by_type.tsv", sep="\t", index=False)
    pooled = {}
    for name, cm in (("all_but_False", (types.type != "CRISPRCasTyper:False").to_numpy()),
                     ("DefenseFinder", types.type.str.startswith("DefenseFinder").to_numpy()),
                     ("TXSScan", types.type.str.startswith("TXSScan").to_numpy()),
                     ("CRISPRCasTyper_subtypes", (types.type.str.startswith("CRISPRCasTyper")
                                                  & (types.type != "CRISPRCasTyper:False")).to_numpy())):
        m = np.isin(qt, np.flatnonzero(cm))
        bt = Boot(gc[m], draws=2000, seed=17)
        res = {lvl: bt.auc(Kx[lvl][m], pos[m]) for lvl in ("full", "L1", "L1h", "L2", "L2h")}
        pooled[name] = {lvl: [round(float(p), 4), round(float(np.quantile(dr, .025)), 4), round(float(np.quantile(dr, .975)), 4)]
                        for lvl, (p, dr) in res.items()}
        dd = res["L1"][1] - res["L1h"][1]
        pooled[name]["L1_minus_L1h"] = [round(float(res["L1"][0] - res["L1h"][0]), 4), round(float(np.quantile(dd, .025)), 4),
                                        round(float(np.quantile(dd, .975)), 4)]
        pooled[name]["pos_R_median"] = {lvl: float(np.median(Kx[lvl + "_R"][m & pos])) for lvl in ("full", "L1", "L1h", "L2", "L2h")}
    json.dump(pooled, open(HW / "withheld_hom_summary.json", "w"), indent=1)
    log(json.dumps(pooled, indent=1))
    log(T[["type", "n_pos", "auc_full", "auc_L1", "auc_L1h", "auc_L2", "auc_L2h", "pos_R_med_L1", "pos_R_med_L1h"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("reps", "post", "arch", "withheld", "weval"))
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    dict(reps=cmd_reps, post=cmd_post, arch=cmd_arch, withheld=cmd_withheld, weval=cmd_weval)[a.cmd](a)
    log("end")


if __name__ == "__main__":
    main()
