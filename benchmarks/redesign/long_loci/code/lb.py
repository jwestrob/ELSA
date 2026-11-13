#!/usr/bin/env python
"""long_loci part 2 (DPANN): L1 / L4b / L4b+ over list-based first-round bonds (predeclared2.json).

Bonds = vl.py round 0 (Learner.round0 with pass_count / pass_synneff, imported read-only from $RD/vocab_lists/code) on
ws_calib z >= 10.6 lists or MMseqs2 alignment lists, group or synteny counting; stored round-0 homolog exclusion masks.
Tree = part-1 code (ll.boundary_score / ll.tree, unchanged). Family-free node filter: list partners in place of families.

usage (Slurm):  lb.py arm --arm Z|A|Zs|As [--fz] --threads T
Writes $RD/long_loci/loci/dpann_<arm>/, nodes/dpann_<arm>/, results/lbuild_<arm>.json. Nothing is written under
$RD/vocab_lists (the synteny-matrix save of vl.Learner.syn_S is replaced by a copy without it).
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
VL = RD / "vocab_lists"
sys.path.insert(0, str(RD / "softrec/code"))
sys.path.insert(0, str(VL / "code"))
import vl  # noqa: E402  (read-only import)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ll  # noqa: E402

ARMS = {"Z": ("grp", "z10.6", "group", 6.0), "A": ("grp", "aln", "group", 6.0),
        "Zs": ("syn", "z10.6", "genome", None), "As": ("syn", "aln", "genome", None)}
STORED = {"Z": (VL / "results/job_z.json", "z10.6"), "A": (VL / "results/job_aln.json", "aln")}
KAPPA = {"knn": RD / "context_units/ctx/kappa_var_knn50.npy", "tok": RD / "context_units/ctx/kappa_var_tokens.npy"}
log = ll.log


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1)


class Vq:
    def __init__(self, uq):
        self.uq = uq


def make_lists(U, src):
    import softrec as SR
    V = Vq(np.load(SR.EMB / "uid_of_row.npy").astype(np.int64))
    if src == "aln":
        return vl.aln_lists(U, V), 1000
    return vl.z_lists(U, V, float(src[1:])), 0


class LOrder:
    """Slot space of one gene order for the grp / syn arms (no contexts; exclusion masks read from vocab_lists)."""

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
        z = np.load(VL / "excl" / f"{excl_tag}_order{r}.npz")
        self.excl = {200: z["K200"], 1000: z["K1000"]}
        self.floor = {False: dict(c0=0.0, sd0=0.0), True: dict(c0=0.0, sd0=0.0)}
        self.P = None


class Lrn(vl.Learner):
    def syn_S(self):
        """vl.Learner.syn_S copied without its np.save into $RD/vocab_lists/results."""
        key = ("S", self.LS.name, self.K)
        if key in self.O.cache:
            return self.O.cache[key]
        t0 = time.time()
        st = self.init_state()
        aptr, ak = self.adj(st)
        u = self.U.u
        ng = u.ng
        cnt = np.zeros((ng, ng), np.int64)
        tot = np.zeros(ng, np.int64)
        vl.syn_matrix(ng, u.g_start.astype(np.int64), u.g_end.astype(np.int64), aptr, ak, st["ns"], st["nterm"],
                      st["lslot"], self.O.gene_at, self.LS.lstart, self.tcnt, self.nbs, self.LS.tail, self.LS.nbg,
                      st["wptr"], st["ws"], st["we"], st["wr"], st["wg"], self.cid, cnt, tot, self.threads * 4)
        Sd = cnt / np.maximum(tot, 1)[:, None]
        S = (Sd + Sd.T) / 2.0
        np.fill_diagonal(S, 1.0)
        off = S[~np.eye(ng, dtype=bool)]
        info = dict(seconds=round(time.time() - t0, 1), offdiag_mean=float(off.mean()),
                    offdiag_q50_q90_q99=np.quantile(off, [.5, .9, .99]).tolist())
        log("synteny matrix", f"order {self.O.r}", info)
        self.O.cache[key] = S
        self.O.cache[("Sinfo", self.LS.name, self.K)] = info
        return S


# =========================================================================== family-free node filter
@njit(parallel=True, cache=True)
def count_lists(lo, hi, own, fstart, fcnt, fgen, cc, gc, grp, n_groups, mg, nblocks):
    """Per node: |I| = genes with a non-empty list; q = max(2, ceil(|I|/2)); W = 2|I|; n = groups (other than own) with a
    genome holding list partners of >= q distinct node genes within <= W genes on one contig (one node gene per partner
    position: the lowest-indexed); closed-form E over other groups (part-1 formula with c = partners per genome)."""
    nn = len(lo)
    n_out = np.zeros(nn, dtype=np.int64)
    e_out = np.zeros(nn, dtype=np.float64)
    nI_out = np.zeros(nn, dtype=np.int64)
    bs = (nn + nblocks - 1) // nblocks
    for blk in prange(nblocks):
        u0 = blk * bs
        u1 = min(nn, u0 + bs)
        if u0 >= u1:
            continue
        stamp = np.full(n_groups, -1, dtype=np.int64)
        acc = np.zeros(n_groups, dtype=np.float64)
        touched = np.empty(n_groups, dtype=np.int64)
        for u in range(u0, u1):
            kf = 0
            tot = 0
            for g in range(lo[u], hi[u] + 1):
                c = 0
                for x in range(fstart[g], fstart[g] + fcnt[g]):
                    if fgen[x] >= 0:
                        c += 1
                if c > 0:
                    kf += 1
                    tot += c
            nI_out[u] = kf
            if kf < 2:
                continue
            q = max(2, int(math.ceil(0.5 * kf)))
            Wn = 2 * kf
            buf = np.empty(tot, dtype=np.int64)
            p = 0
            loc = 0
            for g in range(lo[u], hi[u] + 1):
                c = 0
                for x in range(fstart[g], fstart[g] + fcnt[g]):
                    if fgen[x] >= 0:
                        buf[p] = np.int64(fgen[x]) * kf + loc
                        p += 1
                        c += 1
                if c > 0:
                    loc += 1
            buf.sort()
            m = 0
            last = -1
            for x in range(tot):
                gx = buf[x] // kf
                if gx != last:
                    buf[m] = buf[x]
                    m += 1
                    last = gx
            tot = m
            cnt = np.zeros(kf, dtype=np.int64)
            nh = 0
            nt = 0
            j0 = 0
            while j0 < tot:
                g0 = gc[buf[j0] // kf]
                j1 = j0 + 1
                while j1 < tot and gc[buf[j1] // kf] == g0:
                    j1 += 1
                G = grp[g0]
                pres = 0
                for x in range(j0, j1):
                    lx = buf[x] - (buf[x] // kf) * kf
                    if cnt[lx] == 0:
                        pres += 1
                    cnt[lx] += 1
                if pres >= q:
                    e = np.zeros(q + 1)
                    e[0] = 1.0
                    lsc = 0.0
                    cur = 0
                    for x in range(j0, j1):
                        lx = buf[x] - (buf[x] // kf) * kf
                        c = cnt[lx]
                        if c > 0:
                            top = cur + 1 if cur + 1 < q else q
                            for tt in range(top, 0, -1):
                                e[tt] += e[tt - 1] * c
                            cur += 1
                            cnt[lx] = 0
                            mx = 0.0
                            for tt in range(q + 1):
                                if e[tt] > mx:
                                    mx = e[tt]
                            if mx > 1e250:
                                for tt in range(q + 1):
                                    e[tt] *= 1e-250
                                lsc += 250.0 * math.log(10.0)
                    if e[q] > 0:
                        mgv = mg[g0]
                        lmu = (math.log(mgv) + math.lgamma(Wn) - math.lgamma(q) - math.lgamma(Wn - q + 1)
                               + math.lgamma(q + 1) + math.log(e[q]) + lsc - q * math.log(mgv))
                        mu = math.exp(min(lmu, 50.0))
                        if acc[G] == 0.0:
                            touched[nt] = G
                            nt += 1
                        acc[G] += mu
                else:
                    for x in range(j0, j1):
                        cnt[buf[x] - (buf[x] // kf) * kf] = 0
                if stamp[G] != u and pres >= q:
                    i = j0
                    dist = 0
                    hit = False
                    jl = j1 - 1
                    for j in range(j0, j1):
                        gj = buf[j] // kf
                        lj = buf[j] - gj * kf
                        while i < j:
                            gi = buf[i] // kf
                            if cc[gi] != cc[gj] or gj - gi >= Wn:
                                li = buf[i] - gi * kf
                                cnt[li] -= 1
                                if cnt[li] == 0:
                                    dist -= 1
                                i += 1
                            else:
                                break
                        cnt[lj] += 1
                        if cnt[lj] == 1:
                            dist += 1
                        if dist >= q:
                            hit = True
                            jl = j
                            break
                    for x in range(i, jl + 1):
                        cnt[buf[x] - (buf[x] // kf) * kf] = 0
                    if hit:
                        stamp[G] = u
                        nh += 1
                j0 = j1
            Eall = 0.0
            for t in range(nt):
                Eall += 1.0 - math.exp(-acc[touched[t]])
            ow = own[u]
            n_out[u] = nh - (1 if stamp[ow] == u else 0)
            e_out[u] = Eall - (1.0 - math.exp(-acc[ow]))
            for t in range(nt):
                acc[touched[t]] = 0.0
    return n_out, e_out, nI_out


# =========================================================================== command
def cmd_arm(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    kind, src, excl, tau_stored = ARMS[a.arm]
    U = vl.Univ(vl.GROUPS if excl == "group" else vl.GENOME_MAP)
    u = U.u
    n = u.n
    LS, K = make_lists(U, src)
    res = dict(arm=a.arm, kind=kind, lists=LS.info, excl=excl, K=K, ladder={}, timing={}, outputs={})
    res["timing"]["lists"] = round(time.time() - t0, 1)
    bonds = None
    for r in (-1, 0, 1, 2):
        tr = time.time()
        O = LOrder(U, LS, r, excl)
        Lr = Lrn(U, LS, O, K, kind, a.threads, excl_K=1000)
        st, aptr, ak, ev = Lr.round0(min(vl.LADDER))
        res["ladder"][str(r)] = dict(counts=vl.ladder_counts(ev), adjacencies=int(len(ak)), sec_A=ev["sec_A"],
                                     sec_B=ev["sec_B"], seconds=round(time.time() - tr, 1),
                                     S=O.cache.get(("Sinfo", LS.name, K)))
        log("LADDER", a.arm, r, res["ladder"][str(r)], "rss", rss_gb())
        if r < 0:
            ok = ev["ok"]
            bonds = dict(j=ev["rj"][ok].astype(np.int64), d=(ak - ev["rj"])[ok].astype(np.int64),
                         lam=ev["lam"][ok].astype(np.float64), n=ev["n"][ok], ng=ev["ng"][ok], neff=ev["neff"][ok])
        del O, Lr, ev, st, aptr, ak
    rows = []
    for t in vl.LADDER:
        obs = res["ladder"]["-1"]["counts"][str(t)]
        nul = float(np.mean([res["ladder"][str(r)]["counts"][str(t)] for r in (0, 1, 2)]))
        rows.append(dict(tau=t, observed=obs, null_mean=nul, fdr=(nul / obs if obs else None)))
    okt = [x["tau"] for x in rows if x["fdr"] is not None and x["fdr"] <= vl.FDR]
    tau_star = min(okt) if okt else None
    res["tau_table"] = rows
    res["tau_star"] = tau_star
    if a.arm in STORED:
        fn, key = STORED[a.arm]
        st_ = json.load(open(fn))
        same = {r: st_["ladder"][key][r]["counts"] == res["ladder"][r]["counts"] for r in ("-1", "0", "1", "2")}
        res["reproduces_stored_ladder"] = dict(per_order=same, all=bool(all(same.values())),
                                              stored_tau=st_["tau"][key]["tau_star"])
        log("REPRODUCTION", a.arm, res["reproduces_stored_ladder"])
        assert res["reproduces_stored_ladder"]["all"] and st_["tau"][key]["tau_star"] == tau_star
    tau = tau_star
    keep = bonds["lam"] >= tau
    bk, bd, bl = bonds["j"][keep], bonds["d"][keep], bonds["lam"][keep]
    nd = LL / "nodes" / f"dpann_{a.arm}"
    nd.mkdir(parents=True, exist_ok=True)
    np.savez(nd / "bonds_obs.npz", k=bk, d=bd, lam=bl, n=bonds["n"][keep], ng=bonds["ng"][keep], neff=bonds["neff"][keep])
    res["bonds"] = dict(tau=tau, bonds=int(keep.sum()), lam_q10_q50_q90_q99=np.quantile(bl, [.1, .5, .9, .99]).tolist())
    s = ll.boundary_score(bk, bd, bl, n)
    variants = [(a.arm, s, None)]
    if a.fz:
        zf = np.load(LL / "nodes/dpann_group/bonds_obs.npz")
        sF = ll.boundary_score(zf["k"].astype(np.int64), zf["d"].astype(np.int64), zf["lam"].astype(np.float64), n)
        variants.append(("FZ", np.maximum(s, sF), dict(F_only=int(((sF > 0) & (s == 0)).sum()),
                                                        Z_only=int(((s > 0) & (sF == 0)).sum()),
                                                        both=int(((s > 0) & (sF > 0)).sum()))))
    # family-free node filter inputs (observed order: list entries are gene ids)
    fstart = LS.lstart.astype(np.int64)
    fcnt = LS.tcnt[K].astype(np.int64)
    fgen = LS.nb
    cid = u.cid.astype(np.int64)
    gc = u.gc.astype(np.int64)
    grp = U.grp.astype(np.int64)
    mg = U.mg.astype(np.float64)
    kap = {k: np.load(v).astype(np.float64) for k, v in KAPPA.items()}
    for name, sv, extra in variants:
        tb = time.time()
        od = LL / "loci" / f"dpann_{name}"
        od.mkdir(parents=True, exist_ok=True)
        ndn = LL / "nodes" / f"dpann_{name}"
        ndn.mkdir(parents=True, exist_ok=True)
        st_, en_ = ll.runs(sv > 0)
        L1 = ll.save_loci(od, "L1", st_, en_)
        item = dict(boundaries_bonded=int((sv > 0).sum()), extra=extra, outputs={"L1": ll.loci_info(L1, n)}, nodes={})
        trees = {}
        for kn, kv in kap.items():
            for th in ll.THETAS:
                low = (kv < th) & (sv > 0)
                w = np.where(low, 1e-6 + np.maximum(np.nan_to_num(kv, nan=0.0), 0.0), sv)
                trees[(kn, th)] = ll.tree(w)
        alln = pd.concat([t[["lo", "hi"]] for t in trees.values()]).drop_duplicates().reset_index(drop=True)
        lo, hi = alln.lo.to_numpy(np.int64), alln.hi.to_numpy(np.int64)
        tr = time.time()
        nO, eO, nI = count_lists(lo, hi, grp[gc[lo]], fstart, fcnt, fgen, cid, gc, grp, int(grp.max()) + 1, mg, 8192)
        alln["n"], alln["E"], alln["nI"] = nO, np.maximum(eO, 0.0), nI
        alln["lam"] = ll.lam_of(alln.n.to_numpy(), alln.E.to_numpy())
        item["node_test"] = dict(unique_intervals=int(len(alln)), seconds=round(time.time() - tr, 1),
                                 E_q50_q90=np.quantile(alln.E, [.5, .9]).tolist(), n_ge2=int((alln.n >= 2).sum()))
        for (kn, th), t in trees.items():
            t = t.merge(alln, on=["lo", "hi"], how="left")
            t.to_parquet(ndn / f"obs4b_{kn}_t{th:g}.parquet", index=False)
            km = (t.n.to_numpy() >= 2) & (t.lam.to_numpy() >= tau) & (t.nI.to_numpy() >= 2)
            sfx = "" if kn == "knn" else "tok"
            d1 = ll.save_loci(od, f"L4b{sfx}_t{th:g}", t.lo.to_numpy()[km], t.hi.to_numpy()[km])
            pm = t.hi.to_numpy() > t.lo.to_numpy()
            d2 = ll.save_loci(od, f"L4bp{sfx}_t{th:g}", t.lo.to_numpy()[pm], t.hi.to_numpy()[pm])
            item["outputs"][f"L4b{sfx}_t{th:g}"] = ll.loci_info(d1, n)
            item["outputs"][f"L4bp{sfx}_t{th:g}"] = ll.loci_info(d2, n)
            item["nodes"][f"{kn}_{th:g}"] = dict(nodes=int(len(t)), kept=int(km.sum()), roots=int((t.par < 0).sum()))
            log(name, kn, th, item["outputs"][f"L4b{sfx}_t{th:g}"], item["nodes"][f"{kn}_{th:g}"])
        item["seconds"] = round(time.time() - tb, 1)
        res["outputs"][name] = item
    res["timing"]["total"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = rss_gb()
    ll.jdump(res, LL / "results" / f"lbuild_{a.arm}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["arm"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--fz", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    cmd_arm(a)


if __name__ == "__main__":
    main()
