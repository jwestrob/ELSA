#!/usr/bin/env python
"""long_loci part 3: Spacedust adapter for the part-2 list-bond pipeline (predeclared3.json).

  SDU / SDUniv   the vl.py / softrec universe interface over $RD/universe (gene order, contigs, genomes, lineage unit =
                 transfer_spacedust's 618 groups or one unit per genome), prio / sig as vl.Univ
  prep           after the kNN (knn_sd.py): round-0 homolog exclusion masks exactly as softrec.exclusion mode 1 (K200,
                 K1000 variants; capcos from softrec.expand_knn on the kNN outside the lineage unit), for lineage = group
                 and genome, observed order and shuffles 0-2; kNN cohesion kappa_var_knn50 (context_units cuts.py
                 gene_cohesion_knn, copied; kh 50, cap_h 200, floor from 1,000,000 random cross-genome gene pairs, seed 17)
  arm            list bonds (vl.Learner round 0 via lb.Lrn), tau by vl's rule on this collection's shuffles, L1 / L4b /
                 L4b+ at theta .02 (part-2 settings) with kNN and token cohesion, family-free node filter (lb.count_lists)
  smoke          adapter checks on the login node (gene order equals ll.load('sd'); instance layout)
usage: sdu.py prep --threads T | sdu.py arm --arm A|As|Z|Zs --threads T | sdu.py smoke
"""
import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
LL = RD / "long_loci"
TS = RD / "transfer_spacedust"
SDL = LL / "sd_lists"
KNN = LL / "sd_knn"
MMSD = LL / "mm_sd"
ZTAB = LL / "sd_calib/table"
sys.path.insert(0, str(RD / "softrec/code"))
sys.path.insert(0, str(RD / "vocab_lists/code"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vl  # noqa: E402  (read-only import)
import ll  # noqa: E402
import lb  # noqa: E402

log = ll.log
ARMS = {"A": ("grp", "aln", "group"), "As": ("syn", "aln", "genome"), "Z": ("grp", "z10.6", "group"),
        "Zs": ("syn", "z10.6", "genome")}
KAPPA = {"knn": SDL / "kappa_var_knn50.npy", "tok": TS / "emb/kappa_var_tokens.npy"}
THETA = 0.02


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1)


# =========================================================================== universe
class SDU:
    """softrec.Universe fields over the Spacedust universe; lineage unit from a genome -> unit array."""

    def __init__(self, lineage):
        import softrec as SR
        g = pd.read_parquet(RD / "universe/genes.parquet",
                            columns=["gid", "genome", "genome_code", "contig", "contig_code", "position_index"])
        assert (g.gid.to_numpy() == np.arange(len(g))).all()
        self.n = n = len(g)
        self.genome_name = g.genome.to_numpy()
        self.contig_name = g.contig.to_numpy()
        self.pos = g.position_index.to_numpy(np.int64)
        self.gc = g.genome_code.to_numpy(np.int32)
        cc = g.contig_code.to_numpy(np.int64)
        assert (np.diff(self.gc) >= 0).all() and (np.diff(cc) >= 0).all()
        brk = np.flatnonzero(np.diff(cc)) + 1
        self.c_start = np.r_[0, brk].astype(np.int64)
        self.c_end = np.r_[brk, n].astype(np.int64)
        self.cid = np.repeat(np.arange(len(self.c_start)), self.c_end - self.c_start).astype(np.int32)
        self.ng = int(self.gc.max()) + 1
        self.n_g = np.bincount(self.gc, minlength=self.ng).astype(np.float64)
        self.lin_of_genome = np.asarray(lineage, np.int32)
        assert len(self.lin_of_genome) == self.ng
        self.nl = int(self.lin_of_genome.max()) + 1
        self.lin = self.lin_of_genome[self.gc]
        gchange = np.flatnonzero(np.diff(self.gc)) + 1
        self.g_start = np.r_[0, gchange].astype(np.int64)
        self.g_end = np.r_[gchange, n].astype(np.int64)
        self.genes_in_lin = np.bincount(self.lin, minlength=self.nl).astype(np.int64)
        self._inst = SR.Universe.instances

    def instances(self, w):
        return self._inst(self, w)


def groups():
    ng = int(pd.read_parquet(RD / "universe/genes.parquet", columns=["genome_code"]).genome_code.max()) + 1
    m = pd.read_parquet(TS / "groups/emb_hom_average_t0.75.parquet", columns=["genome_code", "group"]).sort_values("genome_code")
    assert np.array_equal(m.genome_code.to_numpy(), np.arange(ng))
    return pd.factorize(m.group.to_numpy())[0].astype(np.int64)


class SDUniv:
    """vl.Univ over SDU: grp (618 groups, the counting unit for grp arms and for the node test), mg, prio, sig."""

    def __init__(self, excl):
        grp = groups()
        lineage = grp if excl == "group" else np.arange(len(grp))
        self.u = SDU(lineage)
        self.grp = grp
        self.ngroup = int(grp.max()) + 1
        self.mg = self.u.n_g.astype(np.float64)
        self.prio = np.random.default_rng(17).permutation(self.u.ng).astype(np.int64)
        self.sig = np.random.default_rng(17).permutation(self.u.n).astype(np.int64)


def uid_of_gene():
    return np.load(TS / "fam/gid2uid.npy").astype(np.int64)


# =========================================================================== lists (vl.aln_lists / vl.z_lists, paths)
def aln_lists_sd(U):
    """vl.aln_lists with this collection's MMseqs2 arrays (logic copied)."""
    import softrec as SR
    t0 = time.time()
    I = np.load(MMSD / "aln__k1000__I.npy")
    D = np.load(MMSD / "aln__k1000__B.npy").astype(np.float32)
    u = U.u
    uq = uid_of_gene()
    ok = uq >= 0
    order = np.flatnonzero(ok)[np.argsort(uq[ok], kind="stable")]
    mptr = np.r_[0, np.cumsum(np.bincount(uq[ok], minlength=I.shape[0]))].astype(np.int64)
    K = 1000
    n = u.n
    nb = np.full((n, K), -1, np.int32)
    tail = np.zeros((n, K), np.uint16)
    llen = np.zeros(n, np.int64)
    ntie = np.zeros(n, np.int64)
    rf = np.zeros((n, 1), np.int64)
    cos_at = np.zeros((n, 1), np.float64)
    SR.expand_knn(uq, u.lin.astype(np.int64), I, D, mptr, order.astype(np.int64), K, np.array([-2.0]),
                  np.array([K], np.int64), nb, tail, llen, ntie, rf, cos_at)
    del I, D
    nbf = nb.ravel()
    nbg = np.where(nbf >= 0, u.gc[np.maximum(nbf, 0)], 0).astype(np.int16)
    info = dict(source="aln", genes_with_list=int((llen > 0).sum()),
                list_len_q01_q10_q50_q90=np.quantile(llen, [.01, .1, .5, .9]).tolist(),
                genes_list_shorter_than_1000=int((llen < K).sum()), seconds=round(time.time() - t0, 1))
    log("aln lists", info)
    return vl.ListSet("aln", nbf, tail.ravel(), nbg, np.arange(n, dtype=np.int64) * K, {1000: llen}, info), 1000


def z_lists_sd(U, cut=10.6):
    """vl.z_lists with this collection's z table (logic copied; path changed)."""
    t0 = time.time()
    cnt = np.load(ZTAB / "counts.npy").astype(np.int64)
    off = np.r_[0, np.cumsum(cnt)]
    N = len(cnt)
    zG_all = np.memmap(ZTAB / "genome.bin", np.int16, "r")
    zL_all = np.memmap(ZTAB / "local.bin", np.int16, "r")
    zz_all = np.memmap(ZTAB / "z.bin", np.float16, "r")
    keep_cnt = np.zeros(N, np.int64)
    parts = []
    for a in range(0, N, 100_000):
        b = min(N, a + 100_000)
        s, e = off[a], off[b]
        z = np.asarray(zz_all[s:e]).astype(np.float32)
        k = z >= cut
        uid_of = np.repeat(np.arange(a, b), cnt[a:b])
        keep_cnt[a:b] = np.bincount(uid_of[k] - a, minlength=b - a)
        parts.append((np.asarray(zG_all[s:e])[k], np.asarray(zL_all[s:e])[k], z[k]))
    zG = np.concatenate([p[0] for p in parts]).astype(np.int16)
    zL = np.concatenate([p[1] for p in parts]).astype(np.int16)
    zz = np.concatenate([p[2] for p in parts]).astype(np.float32)
    del parts
    zoff = np.r_[0, np.cumsum(keep_cnt)].astype(np.int64)
    u = U.u
    uq = uid_of_gene()
    gcount = np.zeros(u.n, np.int64)
    lin = u.lin_of_genome.astype(np.int64)
    vl.z_count(uq, u.gc.astype(np.int64), lin, zoff, zG, gcount)
    lptr = np.r_[0, np.cumsum(gcount)].astype(np.int64)
    nb = np.full(lptr[-1], -1, np.int32)
    tail = np.zeros(lptr[-1], np.uint16)
    nbg = np.zeros(lptr[-1], np.int16)
    vl.z_fill(uq, u.gc.astype(np.int64), lin, zoff, zG, zL, zz, u.g_start.astype(np.int64), lptr, nb, tail, nbg)
    assert (nb >= 0).all()
    info = dict(source=f"z{cut:g}", entries=int(lptr[-1]), uid_entries=int(zoff[-1]),
                list_len_q01_q10_q50_q90_max=np.quantile(gcount, [.01, .1, .5, .9, 1]).tolist(),
                genes_empty=int((gcount == 0).sum()), seconds=round(time.time() - t0, 1))
    log("z lists", info)
    return vl.ListSet(f"z{cut:g}", nb, tail, nbg, lptr[:-1].copy(), {0: gcount}, info), 0


class LOrderSD(lb.LOrder):
    def __init__(self, U, LS, r, excl_tag):
        import softrec as SR
        u = U.u
        self.r = r
        self.cache = {}
        if r < 0:
            self.gene_at = np.arange(u.n, dtype=np.int64)
            self.slot_of = self.gene_at.copy()
        else:
            self.gene_at, self.slot_of = SR.null_order(u, r)
        x = np.empty_like(LS.nb)
        SR.map_slots(LS.nb, self.slot_of.astype(np.int32), x)
        self.nbs = {LS.name: x}
        nd, off, left, d = u.instances(vl.WGAP)
        self.off = off
        z = np.load(SDL / "excl" / f"{excl_tag}_order{r}.npz")
        self.excl = {200: z["K200"], 1000: z["K1000"]}
        self.floor = {False: dict(c0=0.0, sd0=0.0), True: dict(c0=0.0, sd0=0.0)}
        self.P = None


# =========================================================================== prep: exclusion masks, kNN cohesion
@njit(parallel=True, cache=True)
def gene_cohesion_knn(uq, I, Kh, mptr, members, uid, X, strand, cid, c0, cap_h, out_right, out_left):
    """context_units cuts.py gene_cohesion_knn, copied unchanged."""
    n = len(uq)
    d = X.shape[1]
    for i in prange(n):
        out_right[i] = np.nan
        out_left[i] = np.nan
        U = uq[i]
        if U < 0:
            continue
        for which in range(2):
            nb = i + 1 if which == 0 else i - 1
            if nb < 0 or nb >= n or cid[nb] != cid[i]:
                continue
            side = 0 if (strand[i] == 1) == (which == 0) else 1
            A = np.zeros(d)
            cnt = 0
            j = -1
            while j < Kh and cnt < cap_h:
                V = U if j < 0 else I[U, j]
                if V < 0:
                    break
                if j >= 0 and V == U:
                    j += 1
                    continue
                for t in range(mptr[V], mptr[V + 1]):
                    m = members[t]
                    if m == i or cnt >= cap_h:
                        continue
                    step = 1 if (strand[m] == 1) == (side == 0) else -1
                    nm = m + step
                    if nm < 0 or nm >= n or cid[nm] != cid[m]:
                        continue
                    q = uid[nm]
                    for k in range(d):
                        A[k] += X[q, k]
                    cnt += 1
                j += 1
            if cnt < 2:
                continue
            s2 = 0.0
            for k in range(d):
                s2 += A[k] * A[k]
            rho = (s2 - cnt) / (cnt * (cnt - 1.0))
            rho = (rho - c0) / (1.0 - c0)
            if which == 0:
                out_right[i] = rho
            else:
                out_left[i] = rho


@njit(parallel=True, cache=True)
def gene_pair_cos(ga, gb, X, uid, out):
    d = X.shape[1]
    for s in prange(len(ga)):
        qa, qb = uid[ga[s]], uid[gb[s]]
        x1 = 0.0
        x2 = 0.0
        x12 = 0.0
        for q in range(d):
            a = X[qa, q]
            b = X[qb, q]
            x1 += a * a
            x2 += b * b
            x12 += a * b
        out[s] = x12 / math.sqrt(x1 * x2) if x1 > 0 and x2 > 0 else np.nan


def cmd_prep(a):
    import numba
    import softrec as SR
    numba.set_num_threads(a.threads)
    t0 = time.time()
    (SDL / "excl").mkdir(parents=True, exist_ok=True)
    X = np.load(TS / "emb/zca10_f32__uid.npy")
    uq = uid_of_gene()
    I = np.load(KNN / "zca10__k2000__I.npy", mmap_mode="r")
    D = np.load(KNN / "zca10__k2000__D.npy", mmap_mode="r")
    res = dict(knn=json.load(open(KNN / "zca10__k2000.json")), excl={})
    # kNN cohesion (landscape independent)
    g = pd.read_parquet(RD / "universe/genes.parquet", columns=["gid", "genome_code", "contig_code", "strand"])
    strand = g.strand.to_numpy(np.int64)
    gc = g.genome_code.to_numpy(np.int64)
    cc = g.contig_code.to_numpy(np.int64)
    n = len(g)
    if not KAPPA["knn"].exists():
        t1 = time.time()
        I50 = np.ascontiguousarray(I[:, :50])
        ok = uq >= 0
        order = np.flatnonzero(ok)[np.argsort(uq[ok], kind="stable")]
        mptr = np.r_[0, np.cumsum(np.bincount(uq[ok], minlength=X.shape[0]))].astype(np.int64)
        rng = np.random.default_rng(17)
        ga = rng.integers(0, n, 1_000_000)
        gb = rng.integers(0, n, 1_000_000)
        okp = gc[ga] != gc[gb]
        o = np.empty(int(okp.sum()))
        gene_pair_cos(ga[okp], gb[okp], X, uq, o)
        c0 = float(np.nanmean(o))
        kr = np.empty(n)
        kl = np.empty(n)
        gene_cohesion_knn(uq, I50, 50, mptr, order.astype(np.int64), uq, X, strand, cc, c0, 200, kr, kl)
        kappa = np.fmin(kr[:-1], kl[1:])
        same = cc[:-1] == cc[1:]
        kappa = np.r_[np.where(same, kappa, np.nan), np.nan]
        np.save(KAPPA["knn"], kappa.astype(np.float32))
        tok = np.load(KAPPA["tok"]).astype(np.float64)
        res["kappa_knn50"] = dict(gene_pair_floor=c0, seconds=round(time.time() - t1, 1),
                                  q05_q25_q50_q75_q95=np.nanquantile(kappa, [.05, .25, .5, .75, .95]).tolist(),
                                  nan_within_contig=float(np.isnan(kappa[:-1][same]).mean()),
                                  share_below_02_05_1_2=[float(np.nanmean(kappa[:-1][same] < t)) for t in (.02, .05, .1, .2)],
                                  spearman_with_token_kappa=float(pd.Series(kappa).corr(pd.Series(tok), method="spearman")))
        log("kappa knn50", res["kappa_knn50"])
        del I50
    # exclusion masks per lineage unit and order (softrec Lists._knn + exclusion, variants K200 / K1000, floor none)
    Ia = np.ascontiguousarray(I)
    Da = np.ascontiguousarray(D).astype(np.float32)
    ok = uq >= 0
    order = np.flatnonzero(ok)[np.argsort(uq[ok], kind="stable")]
    mptr = np.r_[0, np.cumsum(np.bincount(uq[ok], minlength=Ia.shape[0]))].astype(np.int64)
    for excl in ("group", "genome"):
        t1 = time.time()
        U = SDUniv(excl)
        u = U.u
        depths = np.array([200, 1000], np.int64)
        floors = np.array([-2.0])
        nb = np.full((n, 1000), -1, np.int32)
        tail = np.zeros((n, 1000), np.uint16)
        llen = np.zeros(n, np.int64)
        ntie = np.zeros(n, np.int64)
        rf = np.zeros((n, 1), np.int64)
        cos_at = np.zeros((n, 2), np.float64)
        SR.expand_knn(uq, u.lin.astype(np.int64), Ia, Da, mptr, order.astype(np.int64), 1000, floors, depths, nb, tail,
                      llen, ntie, rf, cos_at)
        del nb, tail
        capcos = np.stack([np.maximum(cos_at[:, 0], -2.0), np.maximum(cos_at[:, 1], -2.0)]).astype(np.float32)
        nd, off, left, d = u.instances(vl.WGAP)
        info = dict(lists_len_q01_q50=np.quantile(llen, [.01, .5]).tolist(), shorter_than_1000=int((llen < 1000).sum()),
                    capcos_K1000_q10_q50_q90=np.quantile(capcos[1], [.1, .5, .9]).tolist(), orders={})
        for r in (-1, 0, 1, 2):
            gene_at = np.arange(n, dtype=np.int64) if r < 0 else SR.null_order(u, r)[0]
            ex = np.zeros((2, len(left)), np.bool_)
            SR.exclusion(n, 2, gene_at, nd, off, 1, np.zeros(1, np.int64), X, uq, capcos, ex)
            np.savez(SDL / "excl" / f"{excl}_order{r}.npz", K200=ex[0], K1000=ex[1])
            info["orders"][str(r)] = dict(instances=int(len(left)), excluded_K1000=int(ex[1].sum()),
                                          excluded_K200=int(ex[0].sum()))
        info["seconds"] = round(time.time() - t1, 1)
        res["excl"][excl] = info
        log("exclusion", excl, info)
    res["seconds"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = rss_gb()
    ll.jdump(res, LL / "results" / "sd_prep.json")


# =========================================================================== arm
def cmd_arm(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    kind, src, excl = ARMS[a.arm]
    U = SDUniv(excl)
    u = U.u
    n = u.n
    LS, K = aln_lists_sd(U) if src == "aln" else z_lists_sd(U, float(src[1:]))
    res = dict(arm=a.arm, kind=kind, lists=LS.info, excl=excl, K=K, ladder={}, timing={}, outputs={})
    res["timing"]["lists"] = round(time.time() - t0, 1)
    bonds = None
    for r in (-1, 0, 1, 2):
        tr = time.time()
        O = LOrderSD(U, LS, r, excl)
        Lr = lb.Lrn(U, LS, O, K, kind, a.threads, excl_K=1000)
        st, aptr, ak, ev = Lr.round0(min(vl.LADDER))
        res["ladder"][str(r)] = dict(counts=vl.ladder_counts(ev), adjacencies=int(len(ak)), sec_A=ev["sec_A"],
                                     sec_B=ev["sec_B"], seconds=round(time.time() - tr, 1),
                                     S=O.cache.get(("Sinfo", LS.name, K)))
        log("LADDER", a.arm, r, res["ladder"][str(r)], "rss", rss_gb())
        if r < 0:
            ok = ev["ok"]
            bonds = dict(j=ev["rj"][ok].astype(np.int64), d=(ak - ev["rj"])[ok].astype(np.int64),
                         lam=ev["lam"][ok].astype(np.float64))
        del O, Lr, ev, st, aptr, ak
    rows = []
    for t in vl.LADDER:
        obs = res["ladder"]["-1"]["counts"][str(t)]
        nul = float(np.mean([res["ladder"][str(r)]["counts"][str(t)] for r in (0, 1, 2)]))
        rows.append(dict(tau=t, observed=obs, null_mean=nul, fdr=(nul / obs if obs else None)))
    okt = [x["tau"] for x in rows if x["fdr"] is not None and x["fdr"] <= vl.FDR]
    tau = min(okt) if okt else None
    res["tau_table"], res["tau_star"] = rows, tau
    log("TAU", a.arm, tau, rows)
    keep = bonds["lam"] >= tau
    bk, bd, bl = bonds["j"][keep], bonds["d"][keep], bonds["lam"][keep]
    tag = f"sd_{a.arm}"
    od, nd = LL / "loci" / tag, LL / "nodes" / tag
    od.mkdir(parents=True, exist_ok=True)
    nd.mkdir(parents=True, exist_ok=True)
    np.savez(nd / "bonds_obs.npz", k=bk, d=bd, lam=bl)
    s = ll.boundary_score(bk, bd, bl, n)
    res["bonds"] = dict(tau=tau, bonds=int(keep.sum()), boundaries_bonded=int((s > 0).sum()))
    st_, en_ = ll.runs(s > 0)
    res["outputs"]["L1"] = ll.loci_info(ll.save_loci(od, "L1", st_, en_), n)
    cid = u.cid.astype(np.int64)
    gc = u.gc.astype(np.int64)
    grp = U.grp.astype(np.int64)
    trees = {}
    for kn, path in KAPPA.items():
        kv = np.load(path).astype(np.float64)
        low = (kv < THETA) & (s > 0)
        w = np.where(low, 1e-6 + np.maximum(np.nan_to_num(kv, nan=0.0), 0.0), s)
        trees[kn] = ll.tree(w)
    alln = pd.concat([t[["lo", "hi"]] for t in trees.values()]).drop_duplicates().reset_index(drop=True)
    lo, hi = alln.lo.to_numpy(np.int64), alln.hi.to_numpy(np.int64)
    tr = time.time()
    nO, eO, nI = lb.count_lists(lo, hi, grp[gc[lo]], LS.lstart.astype(np.int64), LS.tcnt[K].astype(np.int64), LS.nb, cid,
                                gc, grp, int(grp.max()) + 1, U.mg.astype(np.float64), 8192)
    alln["n"], alln["E"], alln["nI"] = nO, np.maximum(eO, 0.0), nI
    alln["lam"] = ll.lam_of(alln.n.to_numpy(), alln.E.to_numpy())
    res["node_test"] = dict(unique_intervals=int(len(alln)), seconds=round(time.time() - tr, 1))
    for kn, t in trees.items():
        t = t.merge(alln, on=["lo", "hi"], how="left")
        t.to_parquet(nd / f"obs4b_{kn}_t{THETA:g}.parquet", index=False)
        km = (t.n.to_numpy() >= 2) & (t.lam.to_numpy() >= tau) & (t.nI.to_numpy() >= 2)
        sfx = "" if kn == "knn" else "tok"
        d1 = ll.save_loci(od, f"L4b{sfx}_t{THETA:g}", t.lo.to_numpy()[km], t.hi.to_numpy()[km])
        pm = t.hi.to_numpy() > t.lo.to_numpy()
        d2 = ll.save_loci(od, f"L4bp{sfx}_t{THETA:g}", t.lo.to_numpy()[pm], t.hi.to_numpy()[pm])
        res["outputs"][f"L4b{sfx}_t{THETA:g}"] = ll.loci_info(d1, n)
        res["outputs"][f"L4bp{sfx}_t{THETA:g}"] = ll.loci_info(d2, n)
        log(a.arm, kn, res["outputs"][f"L4b{sfx}_t{THETA:g}"], res["outputs"][f"L4bp{sfx}_t{THETA:g}"])
    res["timing"]["total"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = rss_gb()
    ll.jdump(res, LL / "results" / f"sdbuild_{a.arm}.json")


# =========================================================================== smoke
def cmd_smoke(a):
    """Adapter checks: gene order, contigs and genomes equal ll.load('sd'); groups equal; instance layout as softrec."""
    U = SDUniv("group")
    u = U.u
    tok, cc, gc, grp, kappa = ll.load("sd")
    assert np.array_equal(u.gc.astype(np.int64), gc)
    cid_ll = pd.factorize(cc)[0]
    assert np.array_equal(u.cid.astype(np.int64), cid_ll)
    assert np.array_equal(U.grp, grp)
    nd, off, left, d = u.instances(vl.WGAP)
    assert int(nd.sum()) == len(left) and (d >= 1).all() and (d <= 3).all()
    assert (u.cid[left] == u.cid[left + d]).all()
    uq = uid_of_gene()
    meta = np.load(TS / "emb/meta.npz")["uq"].astype(np.int64)
    assert np.array_equal(uq, meta)
    Ug = SDUniv("genome")
    assert Ug.u.nl == u.ng and u.nl == U.ngroup
    log("smoke OK", dict(genes=u.n, genomes=u.ng, contigs=len(u.c_start), groups=U.ngroup, instances=len(left),
                         uids=int(uq.max()) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "arm", "smoke"])
    ap.add_argument("--arm", default="A")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    {"prep": cmd_prep, "arm": cmd_arm, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()
