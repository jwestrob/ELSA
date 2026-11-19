#!/usr/bin/env python
"""long_loci part 6 (predeclared6.json): strand-aware splitting (S), bridging (B) and both (SB) on the two frozen bases.

Bases (bonds, cohesion and node rule unchanged; stored bonds of parts 2 and 4):
  P  family bonds at the adaptive distance (nodes/<data>_WA), token cohesion, L4b with the closed-form family node test
  Z  embedding-list bonds (nodes/<data>_Z), kNN cohesion, L4b+ (every node kept)
Tree weights: w = s; low cohesion (kappa < theta, bonded) 1e-6 + max(kappa, 0); S: bonded strand changes 1e-9;
B: bridged junctions 1e-8 (bridges computed once from the frozen base output, rule (i) AND (iii) of ribo_audit).

usage (Slurm, 24 CPUs)
  p6.py bridges --data dpann|sd     candidate junctions, rules, gates (DPANN P vs the audit), list rule for Z
  p6.py build   --data dpann|sd --base P|Z [--thetas ...]   variant loci (gate: base == stored loci)
  p6.py score   --base P|Z          DPANN scoring, odd selection, even report, ribosomal readings, merge check
  p6.py sd                          Spacedust variants at the selected and frozen theta, source207 (DEVELOPMENT)
"""
import argparse
import json
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
SPD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
P6 = LL / "p6"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ll  # noqa: E402
import words_ts as W  # noqa: E402

log = ll.log
THETAS = (0.02, 0.05, 0.1, 0.2)
FROZEN_TH = {"P": 0.2, "Z": 0.02}
NREF = 4190383
W_STRAND, W_BRIDGE = 1e-9, 1e-8
CFG = {
    "dpann": dict(genes=RD / "ws_dpann/package/universe/genes.parquet",
                  kappa={"P": RD / "context_units/ctx/kappa_var_tokens.npy", "Z": RD / "context_units/ctx/kappa_var_knn50.npy"},
                  bonds={"P": LL / "nodes/dpann_WA/bonds_obs.npz", "Z": LL / "nodes/dpann_Z/bonds_obs.npz"},
                  stored={"P": LL / "loci/dpann_WA/L4b_t{th}.parquet", "Z": LL / "loci/dpann_Z/L4bp_t{th}.parquet"},
                  tau_json={"P": (LL / "results/wbuild_dpann.json", ("arms", "WA", "tau_star"))}),
    "sd": dict(genes=RD / "universe/genes.parquet",
               kappa={"P": TS / "emb/kappa_var_tokens.npy", "Z": LL / "sd_lists/kappa_var_knn50.npy"},
               bonds={"P": LL / "nodes/sd_WA/bonds_obs.npz", "Z": LL / "nodes/sd_Z/bonds_obs.npz"},
               stored={"P": LL / "loci/sd_WA/L4b_t{th}.parquet", "Z": LL / "loci/sd_Z/L4bp_t{th}.parquet"},
               tau_json={"P": (LL / "results/wbuild_sd.json", ("arms", "WA", "tau_star"))}),
}


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1)


def jdump(obj, fn):
    Path(fn).write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    log("wrote", fn)


# =========================================================================== data
class Data:
    def __init__(self, data):
        self.data = data
        tok, cc, gc, grp, _ = ll.load(data)
        self.tok, self.cc, self.gc, self.grp = tok, cc, gc, grp
        self.n = n = len(tok)
        if data == "dpann":
            g = pd.read_parquet(CFG[data]["genes"], columns=["gid", "strand", "start", "end"]).sort_values("gid")
            self.strand = g.strand.to_numpy(np.int64)
            start, end = g.start.to_numpy(np.int64), g.end.to_numpy(np.int64)
        else:
            g = pd.read_parquet(CFG[data]["genes"], columns=["gid", "side", "row_index", "strand"]).sort_values("gid")
            self.strand = g.strand.to_numpy(np.int64)
            m = pd.read_parquet(SPD / "results/esm2_t6/shard_000/export/metadata.parquet", columns=["row_index", "start", "end"])
            q = pd.read_parquet(SPD / "native_run_20260910/query/native_lookup_crosswalk.parquet", columns=["row_index", "start", "end"])
            start = np.full(n, -1, np.int64)
            end = np.full(n, -1, np.int64)
            start[m.row_index.to_numpy()] = m.start.to_numpy()
            end[m.row_index.to_numpy()] = m.end.to_numpy()
            start[NREF + q.row_index.to_numpy()] = q.start.to_numpy()
            end[NREF + q.row_index.to_numpy()] = q.end.to_numpy()
            assert (start >= 0).all() and (end >= 0).all()
        lo, hi = np.minimum(start, end), np.maximum(start, end)
        same = cc[:-1] == cc[1:]
        self.same = np.r_[same, False]
        self.dist = np.full(n, 10 ** 9, np.int64)
        self.dist[:-1] = np.where(same, lo[1:] - hi[:-1] - 1, 10 ** 9)
        self.strand_change = np.r_[(self.strand[:-1] != self.strand[1:]) & same, False]
        self.coord_order_ok = float(np.mean((lo[1:] >= lo[:-1])[same]))

    def s_of(self, base):
        z = np.load(CFG[self.data]["bonds"][base])
        return ll.boundary_score(z["k"].astype(np.int64), z["d"].astype(np.int64), z["lam"].astype(np.float64), self.n)

    def kappa(self, base):
        k = np.load(CFG[self.data]["kappa"][base]).astype(np.float64)
        assert len(k) == self.n
        return k


def read_loci(path):
    df = pd.read_parquet(path)
    sc, ec = ("gid_s", "gid_e") if "gid_s" in df else ("s", "e")
    s, e = df[sc].to_numpy(np.int64), df[ec].to_numpy(np.int64)
    o = np.lexsort((e, s))
    return s[o], e[o]


def top_level(a, b, n):
    """gene -> index of the maximal locus containing it (-1 none), ribo_audit's rule (partial overlaps extend)."""
    order = np.lexsort((-(b - a), a))
    top = np.full(n, -1, np.int64)
    t, reach = -1, -1
    for j in order:
        if a[j] > reach:
            t += 1
            top[a[j]:b[j] + 1] = t
            reach = b[j]
        elif b[j] > reach:
            top[reach + 1:b[j] + 1] = t
            reach = b[j]
    return top


def candidates(D, a, b, s):
    span = np.zeros(D.n + 1, np.int64)
    np.add.at(span, a, 1)
    np.add.at(span, b, -1)
    span = np.cumsum(span)[:D.n]
    ends = np.zeros(D.n, bool)
    ends[b] = True
    starts = np.zeros(D.n + 1, bool)
    starts[a] = True
    k = np.flatnonzero(ends[:-1] & starts[1:D.n] & D.same[:-1] & (span[:-1] == 0) & (s[:-1] == 0))
    return k


def family_rule(D, k, top):
    """number of groups other than the junction's own with a top-level locus holding the token families of genes k
    and k+1 (ribo_audit rule (i)); computed per distinct token pair, then the own group is removed per junction."""
    tx, ty = D.tok[k], D.tok[k + 1]
    need = np.unique(np.r_[tx, ty])
    intop = (top >= 0) & np.isin(D.tok, need)
    tg = np.full(int(top.max()) + 1, -1, np.int64)
    tg[top[top >= 0]] = D.grp[D.gc[top >= 0]]
    pairs = pd.DataFrame(dict(tok=D.tok[intop], top=top[intop])).drop_duplicates()
    U = pd.DataFrame(dict(ta=tx, tb=ty)).drop_duplicates().reset_index(drop=True)
    U["pid"] = np.arange(len(U))
    A = U.merge(pairs.rename(columns={"tok": "ta"}), on="ta")[["pid", "top"]]
    Bm = U.merge(pairs.rename(columns={"tok": "tb"}), on="tb")[["pid", "top"]]
    co = A.merge(Bm, on=["pid", "top"])
    co["grp"] = tg[co.top.to_numpy()]
    pg = co[["pid", "grp"]].drop_duplicates()
    ng = pg.groupby("pid").size()
    J = pd.DataFrame(dict(ta=tx, tb=ty, own=D.grp[D.gc[k]])).merge(U, on=["ta", "tb"], how="left")
    n_all = ng.reindex(J.pid.to_numpy()).fillna(0).to_numpy(np.int64)
    own_in = J.merge(pg.rename(columns={"grp": "own"}).assign(hit=1), on=["pid", "own"], how="left").hit.fillna(0).to_numpy(np.int64)
    return n_all - own_in


@njit(parallel=True, cache=True)
def list_rule(kx, lstart, lcnt, nb, nbg, top, grp_of_genome, own, n_genomes, n_groups, out):
    for j in prange(len(kx)):
        x = kx[j]
        y = x + 1
        part = np.full(n_genomes, -1, np.int64)
        for t in range(lstart[y], lstart[y] + lcnt[y]):
            part[nbg[t]] = nb[t]
        seen = np.zeros(n_groups, np.bool_)
        c = 0
        for t in range(lstart[x], lstart[x] + lcnt[x]):
            g = nbg[t]
            py = part[g]
            if py >= 0:
                tx = top[nb[t]]
                if tx >= 0 and tx == top[py]:
                    q = grp_of_genome[g]
                    if q != own[j] and not seen[q]:
                        seen[q] = True
                        c += 1
        out[j] = c


def z_lists(data):
    """the Z arm's z >= 10.6 lists (own group excluded), as built by lb.make_lists (DPANN) / sdu.z_lists_sd (Spacedust)."""
    if data == "dpann":
        import lb
        import vl
        U = vl.Univ(vl.GROUPS)
        LS, K = lb.make_lists(U, "z10.6")
    else:
        import sdu
        U = sdu.SDUniv("group")
        LS, K = sdu.z_lists_sd(U)
    return LS.lstart.astype(np.int64), LS.tcnt[0].astype(np.int64), LS.nb.astype(np.int64), LS.nbg.astype(np.int64), LS.info


def curated(D):
    PKG = RD / "ws_dpann/package"
    inst = pd.read_parquet(PKG / "gold/instances.parquet", columns=["instance_id", "caller", "resolved"])
    mem = pd.read_parquet(PKG / "gold/members.parquet")
    nm = mem.groupby("instance_id").size()
    mem = mem[mem.instance_id.isin(nm[nm >= 2].index)]
    span = mem.groupby("instance_id").gid.agg(["min", "max"]).reset_index()
    span = span[D.cc[span["min"].to_numpy()] == D.cc[span["max"].to_numpy()]]
    return mem, span.sort_values("min").reset_index(drop=True)


# =========================================================================== bridges
def cmd_bridges(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    D = Data(a.data)
    P6.mkdir(parents=True, exist_ok=True)
    res = dict(data=a.data, coord_order_ok=D.coord_order_ok, bases={})
    for base in ("P", "Z"):
        s = D.s_of(base)
        fa, fb = read_loci(str(CFG[a.data]["stored"][base]).format(th=f"{FROZEN_TH[base]:g}"))
        top = top_level(fa, fb, D.n)
        k = candidates(D, fa, fb, s)
        J = pd.DataFrame(dict(k=k, same_strand=D.strand[k] == D.strand[k + 1], dist=D.dist[k]))
        J["rule_iii"] = J.same_strand & (J.dist <= 150)
        J["fam_groups"] = family_rule(D, k, top)
        J["rule_i"] = J.fam_groups >= 3
        if base == "Z":
            lstart, lcnt, nb, nbg, info = z_lists(a.data)
            out = np.zeros(len(k), np.int64)
            list_rule(k, lstart, lcnt, nb, nbg, top, D.grp, D.grp[D.gc[k]], int(D.gc.max()) + 1, int(D.grp.max()) + 1, out)
            J["list_groups"] = out
            J["rule_i_list"] = J.list_groups >= 3
            del lstart, lcnt, nb, nbg
        J["bridge"] = (J.rule_i_list if base == "Z" else J.rule_i) & J.rule_iii
        J["bridge_fam"] = J.rule_i & J.rule_iii
        item = dict(candidates=int(len(J)), rule_i=int(J.rule_i.sum()), rule_iii=int(J.rule_iii.sum()),
                    bridge=int(J.bridge.sum()), bridge_share=float(J.bridge.mean()), bridge_fam=int(J.bridge_fam.sum()))
        if base == "Z":
            bl, bf = J.bridge.to_numpy(), J.bridge_fam.to_numpy()
            item.update(rule_i_list=int(J.rule_i_list.sum()), list_vs_family=dict(
                both=int((bl & bf).sum()), list_only=int((bl & ~bf).sum()), family_only=int((~bl & bf).sum()),
                jaccard=float((bl & bf).sum() / max((bl | bf).sum(), 1))),
                lists=info)
        if a.data == "dpann":
            item.update(audit_checks(D, base, J, fa, fb, top))
        J.to_parquet(P6 / f"bridges_{a.data}_{base}.parquet", index=False)
        res["bases"][base] = item
        log(a.data, base, item, f"{time.time()-t0:.0f}s", "rss", rss_gb())
    res["seconds"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = rss_gb()
    jdump(res, P6 / f"bridges_{a.data}.json")


def audit_checks(D, base, J, fa, fb, top):
    """ribo_audit comparisons (DPANN): stored audit junction table (P), verdict-1 repairs, curated merges / joins, target
    edges (accurate targets of the audit's runinfo)."""
    W2 = RD / "ribo_audit/work2"
    out = {}
    name = {"P": "WA_L4b", "Z": "Z_L4bp"}[base]
    if base == "P":
        st = pd.read_parquet(W2 / "bridge_junctions_WA_L4b.parquet")
        out["audit_candidates_equal"] = bool(np.array_equal(np.sort(st.k.to_numpy()), J.k.to_numpy()))
        m = J.merge(st[["k", "co_groups", "rule_i", "rule_iii", "rule_i_iii"]], on="k", suffixes=("", "_audit"))
        out["audit_rule_i_equal"] = bool((m.rule_i == m.rule_i_audit).all())
        out["audit_co_groups_equal"] = bool((m.fam_groups == m.co_groups).all())
        out["audit_bridge_equal"] = bool((m.bridge == m.rule_i_iii).all())
        out["audit_fires"] = int(st.rule_i_iii.sum())
    mem, _ = curated(D)
    gene_inst = mem.groupby("gid").instance_id.apply(set).to_dict()
    cd = cs_ = 0
    for kk in J.k[J.bridge].to_numpy():
        lt, rt = top[kk], top[kk + 1]
        li, ri = set(), set()
        for x in np.flatnonzero(top[max(0, kk - 300):kk + 1] == lt) + max(0, kk - 300):
            li |= gene_inst.get(int(x), set())
        for x in np.flatnonzero(top[kk + 1:kk + 302] == rt) + kk + 1:
            ri |= gene_inst.get(int(x), set())
        cd += int(bool(li and ri and not (li & ri)))
        cs_ += int(bool(li & ri))
    out["curated_merges_junction_level"] = cd
    out["curated_joins_junction_level"] = cs_
    ri = pd.read_parquet(W2 / "runinfo.parquet")
    edges = set()
    for t in ri.targets:
        if not t:
            continue
        for x in t.split(";"):
            a0, b0 = map(int, x.split(".."))
            edges.update((a0 - 1, b0))
    out["fires_at_target_edges"] = int(sum(1 for x in J.k[J.bridge] if x in edges))
    cases = pd.read_csv(W2 / "cases2.tsv", sep="\t")
    v1 = cases[(cases.output == name) & (cases.verdict == 1)]
    fires = set(J.k[J.bridge].tolist())
    rep = 0
    for c in v1.itertuples():
        s0 = int(c.run_s)
        ts, te = s0 + int(c.C_s), s0 + int(c.C_e)
        idx = np.flatnonzero((fa <= te) & (fb >= ts))
        ivs = sorted(set(zip(fa[idx].tolist(), fb[idx].tolist())))
        grown = set(ivs)
        changed = True
        while changed:
            changed = False
            cur = sorted(grown)
            for (x0, x1) in cur:
                for (y0, y1) in cur:
                    if y0 == x1 + 1 and x1 in fires and (x0, y1) not in grown:
                        grown.add((x0, y1))
                        changed = True
        best = max(grown, key=lambda iv: (max(0, min(iv[1], te) - max(iv[0], ts) + 1), -(iv[1] - iv[0])))
        ov = max(0, min(best[1], te) - max(best[0], ts) + 1)
        rep += int(ov / (te - ts + 1) >= 0.8 and ov / (best[1] - best[0] + 1) >= 0.8)
    out["verdict1_cases"] = int(len(v1))
    out["verdict1_repaired"] = rep
    return out


# =========================================================================== build
def weights(s, kappa, th, strand_change=None, bridges=None):
    low = (kappa < th) & (s > 0)
    w = np.where(low, 1e-6 + np.maximum(np.nan_to_num(kappa, nan=0.0), 0.0), s)
    if strand_change is not None:
        w = np.where((s > 0) & strand_change, W_STRAND, w)
    if bridges is not None:
        w = w.copy()
        w[bridges] = W_BRIDGE
    return w


def variant_list(base):
    v = ["base", "S", "B", "SB"]
    return v + (["Bf", "SBf"] if base == "Z" else [])


def cmd_build(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    D = Data(a.data)
    base = a.base
    s = D.s_of(base)
    kap = D.kappa(base)
    J = pd.read_parquet(P6 / f"bridges_{a.data}_{base}.parquet")
    br = J.k[J.bridge].to_numpy()
    brf = J.k[J.bridge_fam].to_numpy()
    assert (s[br] == 0).all()
    thetas = [float(x) for x in a.thetas.split(",")] if a.thetas else list(THETAS)
    res = dict(data=a.data, base=base, thetas=thetas, junctions=dict(
        bonded_boundaries=int((s > 0).sum()), bonded_strand_changes=int(((s > 0) & D.strand_change).sum()),
        bridges=int(len(br)), bridges_family=int(len(brf)), candidates=int(len(J))), outputs={}, gate={})
    spec = {"base": (None, None), "S": (D.strand_change, None), "B": (None, br), "SB": (D.strand_change, br),
            "Bf": (None, brf), "SBf": (D.strand_change, brf)}
    trees = {}
    for v in variant_list(base):
        sc_, b_ = spec[v]
        for th in thetas:
            trees[(v, th)] = ll.tree(weights(s, kap, th, sc_, b_))
    log("trees", len(trees), f"{time.time()-t0:.0f}s")
    keep = {}
    if base == "P":
        tok, gc, cc, grp = D.tok, D.gc, D.cc, D.grp
        T = int(tok.max()) + 1
        fsize = np.bincount(tok, minlength=T)
        mg = np.bincount(gc, minlength=int(gc.max()) + 1).astype(np.float64)
        alln = pd.concat([t[["lo", "hi"]] for t in trees.values()]).drop_duplicates().reset_index(drop=True)
        rec, kinfo = ll.recurrence(alln.lo.to_numpy(), alln.hi.to_numpy(), grp[gc[alln.lo.to_numpy()]], tok, fsize,
                                   {-1: ll.fam_csr(tok, T)}, -1, cc, gc, grp, mg, a.blocks)
        rec = pd.concat([alln, rec], axis=1)
        rec["lam"] = ll.lam_of(rec.n.to_numpy(), rec.E_ana.to_numpy())
        node, key = json.load(open(CFG[a.data]["tau_json"]["P"][0])), CFG[a.data]["tau_json"]["P"][1]
        tau = float(node[key[0]][key[1]][key[2]])
        res["tau"] = tau
        res["recurrence"] = dict(unique_intervals=int(len(alln)), seconds=round(time.time() - t0, 1))
        for kk, t in trees.items():
            t = t.merge(rec[["lo", "hi", "n", "nI", "lam"]], on=["lo", "hi"], how="left")
            km = (t.n.to_numpy() >= 2) & (t.lam.to_numpy() >= tau) & (t.nI.to_numpy() >= 2)
            keep[kk] = (t.lo.to_numpy()[km], t.hi.to_numpy()[km])
    else:
        for kk, t in trees.items():
            pm = t.hi.to_numpy() > t.lo.to_numpy()
            keep[kk] = (t.lo.to_numpy()[pm], t.hi.to_numpy()[pm])
    nm = "L4b" if base == "P" else "L4bp"
    for (v, th), (lo, hi) in keep.items():
        od = LL / "loci" / f"p6_{a.data}_{base}{v}"
        od.mkdir(parents=True, exist_ok=True)
        df = ll.save_loci(od, f"{nm}_t{th:g}", lo, hi)
        res["outputs"][f"{v}_t{th:g}"] = ll.loci_info(df, D.n)
        if v == "base":
            sp = Path(str(CFG[a.data]["stored"][base]).format(th=f"{th:g}"))
            if sp.exists():
                st = pd.read_parquet(sp)
                res["gate"][f"t{th:g}"] = bool(st[["s", "e"]].reset_index(drop=True).equals(df[["s", "e"]].reset_index(drop=True)))
    log("gate", res["gate"])
    assert all(res["gate"].values()) and len(res["gate"]) >= 1, res["gate"]
    res["seconds"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = rss_gb()
    jdump(res, P6 / f"build_{a.data}_{base}.json")


# =========================================================================== any fitting locus
@njit(cache=True)
def _any_fit(a, b, maxlen, qs, qe):
    out = np.zeros(len(qs), np.int8)
    for r in range(len(qs)):
        s, e = qs[r], qe[r]
        j = np.searchsorted(a, s - maxlen + 1)
        n = e - s + 1
        while j < len(a) and a[j] <= e:
            if b[j] >= s:
                ov = min(b[j], e) - max(a[j], s) + 1
                if ov >= 0.8 * n and ov >= 0.8 * (b[j] - a[j] + 1):
                    out[r] = 1
                    break
            j += 1
    return out


def any_fit_count(runs, s, e):
    """runs with some locus covering >= 80% of the run and lying >= 80% inside it (e > s loci)."""
    k = e > s
    s, e = s[k], e[k]
    o = np.argsort(s, kind="stable")
    s, e = s[o], e[o]
    return int(_any_fit(s, e, int((e - s + 1).max()), runs.gid_start.to_numpy(np.int64), runs.gid_end.to_numpy(np.int64)).sum())


# =========================================================================== merge check
@njit(cache=True)
def co_pairs(a, b, ist, ien, M):
    """pairs (i, j), i < j, of instances whose spans [ist, ien] (sorted by ist) both lie inside one locus [a, b]."""
    cnt = 0
    for x in range(len(a)):
        lo = np.searchsorted(ist, a[x])
        hi = np.searchsorted(ist, b[x], side="right")
        m = 0
        for t in range(lo, hi):
            if ien[t] <= b[x]:
                m += 1
        cnt += m * (m - 1) // 2
    out = np.empty(cnt, np.int64)
    c = 0
    buf = np.empty(len(ist), np.int64)
    for x in range(len(a)):
        lo = np.searchsorted(ist, a[x])
        hi = np.searchsorted(ist, b[x], side="right")
        m = 0
        for t in range(lo, hi):
            if ien[t] <= b[x]:
                buf[m] = t
                m += 1
        for i in range(m):
            for j in range(i + 1, m):
                out[c] = buf[i] * M + buf[j]
                c += 1
    return out


# =========================================================================== score (DPANN)
def cmd_score(a):
    import numba
    numba.set_num_threads(a.threads)
    import dscore as DS
    S1, RU, B = DS.S1, DS.RU, DS.B
    t0 = time.time()
    base = a.base
    ctx = S1.Ctx()
    ref = DS.ref_odd()
    assert ref["reproduced"]
    dh_per = pd.read_parquet(S1.DH_SCORES).set_index("instance_id")
    nm = "L4b" if base == "P" else "L4bp"
    frozen_tag = {"P": "dpann_WA", "Z": "dpann_Z"}[base]
    th0 = FROZEN_TH[base]
    frozen_scores = pd.read_parquet(LL / "scores" / frozen_tag / f"{nm}_t{th0:g}.parquet").set_index("instance_id")
    fz_s, fz_e = read_loci(LL / "loci" / frozen_tag / f"{nm}_t{th0:g}.parquet")
    fz_ribo = B.ribo_score(ctx.runs, fz_s, fz_e)
    # audit's accurate reading (bio.py imported read-only; strict then loose)
    sys.path.insert(0, str(RD / "ribo_audit/code"))
    import bio
    import audit as AU
    targets = {}
    for rd, dd in (("strict", (150, 500)), ("loose", (500, 500))):
        bio.D_OP, bio.D_MEM = dd
        Bb = bio.Bio()
        targets[rd] = [Bb.run_units(i)[6] for i in range(len(Bb.D.runs))]
        del Bb

    def bio_count(path):
        L = AU.Loci(str(path))
        return {rd: int(sum(1 for t in tg if t and all(bio.score_target(L, ts, te)["ok"] for ts, te in t))) for rd, tg in targets.items()}

    res = dict(base=base, frozen=dict(name=f"{frozen_tag}/{nm}_t{th0:g}"), outputs={}, selection={}, even={})
    res["frozen"]["bio"] = bio_count(LL / "loci" / frozen_tag / f"{nm}_t{th0:g}.parquet")
    log("frozen bio", res["frozen"]["bio"])
    # curated spans for the merge check
    PKG = RD / "ws_dpann/package"
    memb = pd.read_parquet(PKG / "gold/members.parquet")
    k_ = memb.groupby("instance_id").size()
    memb = memb[memb.instance_id.isin(k_[k_ >= 2].index)]
    sp = memb.groupby("instance_id").gid.agg(["min", "max"]).reset_index().sort_values("min").reset_index(drop=True)
    sp = sp[ctx.genes.contig.to_numpy()[sp["min"].to_numpy()] == ctx.genes.contig.to_numpy()[sp["max"].to_numpy()]]
    sp = sp[ctx.genome_arr[sp["min"].to_numpy()] == ctx.genome_arr[sp["max"].to_numpy()]].reset_index(drop=True)
    ist, ien = sp["min"].to_numpy(np.int64), sp["max"].to_numpy(np.int64)
    M = len(ist) + 1
    base_pairs = np.unique(co_pairs(fz_s, fz_e, ist, ien, M))
    res["frozen"]["curated_pairs_in_one_locus"] = int(len(base_pairs))
    per_all = {}
    for v in variant_list(base):
        for th in THETAS:
            p = LL / "loci" / f"p6_dpann_{base}{v}" / f"{nm}_t{th:g}.parquet"
            if v in ("Bf", "SBf") and not p.exists():
                continue
            df = pd.read_parquet(p)
            per, summ, sc, rib, loc = S1.score(ctx, DS.table(ctx, df), a.threads)
            score, cond = DS.sel_score(summ["odd"], rib["odd"], ref)
            per_all[(v, th)] = (per, summ, sc, rib, loc, df)
            res["outputs"][f"{v}_t{th:g}"] = dict(score_odd=score, conditions_odd=cond, summary_odd=summ["odd"], ribo_odd=rib["odd"])
            log(base, v, th, "ODD", score, cond, f"{time.time()-t0:.0f}s")
    # selection (odd rule; Bf / SBf are checks at the theta selected for B / SB)
    for v in ("base", "S", "B", "SB"):
        c = {th: res["outputs"][f"{v}_t{th:g}"]["score_odd"] for th in THETAS}
        res["selection"][v] = max(c, key=lambda th: tuple(c[th]))
    res["selection"]["Bf"] = res["selection"]["B"] if base == "Z" else None
    res["selection"]["SBf"] = res["selection"]["SB"] if base == "Z" else None
    res["selection_written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jdump(res, P6 / f"score_dpann_{base}.json")
    log("SELECTION", res["selection"])
    # even report for the selected theta of each variant and the frozen theta (like-for-like)
    rows = []
    for v in variant_list(base):
        ths = sorted({res["selection"][v], th0}) if res["selection"].get(v) is not None else []
        for th in ths:
            if (v, th) not in per_all:
                continue
            per, summ, sc, rib, loc, df = per_all[(v, th)]
            P = per.set_index("instance_id")
            s_, e_ = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
            item = dict(selected=bool(th == res["selection"][v]), summary_even=summ["even"], ribo_all=rib["all"], loci=loc,
                        genes_in_loci_frac=loc["genes_covered"] / len(ctx.gc))
            item["vs_DH_even"] = RU.paired(P, dh_per, ctx.gcode, "even")
            item["verdict_vs_DH"] = S1.verdict(item["vs_DH_even"], rib["all"], ref["ribo_all"])
            item["seven_bars"] = dict(summed=item["verdict_vs_DH"]["summed_ok"], f1=item["verdict_vs_DH"]["f1_ok"],
                                      recovered=bool(rib["all"]["recovered"] >= 1650), resolved=bool(rib["all"]["resolved"] > 1512),
                                      parv=bool(S1.parv(rib["all"]["parvarchaeaceae"]) >= 110))
            item["seven_bars"]["all"] = bool(item["seven_bars"]["summed"] and all(item["seven_bars"]["f1"].values())
                                             and item["seven_bars"]["recovered"] and item["seven_bars"]["resolved"]
                                             and item["seven_bars"]["parv"])
            item["vs_frozen_even"] = RU.paired(P, frozen_scores, ctx.gcode, "even")
            item["ribo_resolved_vs_frozen"] = RU.ribo_paired(sc, fz_ribo)
            item["any_fit_resolved"] = any_fit_count(ctx.runs, s_, e_)
            item["bio"] = bio_count(LL / "loci" / f"p6_dpann_{base}{v}" / f"{nm}_t{th:g}.parquet")
            pr = np.unique(co_pairs(s_, e_, ist, ien, M))
            item["curated_pairs_in_one_locus"] = int(len(pr))
            item["new_curated_pairs_vs_frozen"] = int(len(np.setdiff1d(pr, base_pairs)))
            res["even"][f"{v}_t{th:g}"] = item
            log(base, v, th, "EVEN summed", round(summ["even"]["summed_DF_TXSS_excess"], 4), "ribo", rib["all"]["recovered"],
                rib["all"]["resolved"], "anyfit", item["any_fit_resolved"], "bio", item["bio"], "new pairs",
                item["new_curated_pairs_vs_frozen"], "bars", item["seven_bars"]["all"], f"{time.time()-t0:.0f}s")
    res["any_fit_frozen"] = any_fit_count(ctx.runs, fz_s, fz_e)
    res["seconds"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = rss_gb()
    jdump(res, P6 / f"score_dpann_{base}.json")


# =========================================================================== Spacedust (DEVELOPMENT)
def cmd_sd(a):
    import numba
    numba.set_num_threads(a.threads)
    import s207
    t0 = time.time()
    sel = {b: json.load(open(P6 / f"score_dpann_{b}.json"))["selection"] for b in ("P", "Z")}
    g = s207.load_universe_genes()
    Bn = s207.Bench(a.threads)
    refs = {"P": pd.read_parquet(LL / "scores/sd_WA/pr_L4b_t0.2.parquet"),
            "Z": pd.read_parquet(LL / "scores/sd_Z/pr_L4bp_t0.02.parquet")}
    res = dict(label="DEVELOPMENT", selection_from_dpann=sel, outputs={})
    for base in ("P", "Z"):
        ths = sorted({FROZEN_TH[base]} | {th for v, th in sel[base].items() if th is not None})
        a2 = argparse.Namespace(data="sd", base=base, thetas=",".join(f"{t:g}" for t in ths), blocks=a.blocks, threads=a.threads)
        cmd_build(a2)
        nm = "L4b" if base == "P" else "L4bp"
        for v in variant_list(base):
            for th in sorted({FROZEN_TH[base], sel[base].get(v) or FROZEN_TH[base]}):
                df = pd.read_parquet(LL / "loci" / f"p6_sd_{base}{v}" / f"{nm}_t{th:g}.parquet")
                s, e = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
                r, pr = Bn.score(s207.gene_spans(g, s, e), f"p6 sd {base}{v} {th}", np.random.default_rng(17))
                od = LL / "scores" / f"p6_sd_{base}{v}"
                od.mkdir(parents=True, exist_ok=True)
                pr.to_parquet(od / f"pr_{nm}_t{th:g}.parquet", index=False)
                pd_ = s207.paired(pr, refs[base])
                res["outputs"][f"{base}{v}_t{th:g}"] = dict(selected_on_dpann=bool(th == sel[base].get(v)), score=r,
                                                           vs_frozen=pd_, verdict_vs_frozen=s207.verdict(pd_))
                log("SD", base, v, th, {k: r[k] for k in ("f1_observed", "f1_random", "f1_excess", "ci_lo", "ci_hi")},
                    "vs frozen", pd_["excess"])
    res["seconds"] = round(time.time() - t0, 1)
    jdump(res, P6 / "sd_source207.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bridges", "build", "score", "sd"])
    ap.add_argument("--data", default="dpann")
    ap.add_argument("--base", default="P")
    ap.add_argument("--thetas", default="")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    {"bridges": cmd_bridges, "build": cmd_build, "score": cmd_score, "sd": cmd_sd}[a.cmd](a)


if __name__ == "__main__":
    main()
