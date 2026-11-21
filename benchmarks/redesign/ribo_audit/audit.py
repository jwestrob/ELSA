#!/usr/bin/env python
"""ribo_audit: case-by-case audit of ribosomal-locus failures (DPANN; outputs TOK, WCP, EMB).

Rubric, thresholds, sampling and seeds: predeclared.json (written before any case was inspected).

usage (biotite login node, one thread):
  audit.py repro      reproduce recovered / resolved of TOK, WCP, EMB (own scorer and ribosomal_check.score)
  audit.py sample     predeclared stratified samples -> work/sample.json
  audit.py stats      flank association, member shares, junction recurrence, anomalies -> work/stats.pkl
  audit.py classify   rubric on every run x output (census), features, sensitivity -> work/census.parquet
  audit.py cases      per-case table, gene-by-gene layouts, diagrams for the samples (Pfam from duckdb)
  audit.py full       nearby-run statistics over all 1,686 runs -> work/full.json
"""
import os
import sys
import json
import time
import pickle
from pathlib import Path

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
AD = RD / "ribo_audit"
WK = AD / "work"
os.environ.setdefault("NUMBA_CACHE_DIR", str(AD / "numba_cache"))
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

PKG = RD / "ws_dpann/package"
DB = RD / "ws_dpann/input/sharur.duckdb"
TOK = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
GROUPS = RD / "lineagefree/groups/emb_hom_average_t0.75.parquet"
GOLD = RD / "ribosomal/gold"
OUTPUTS = {"TOK": RD / "boundary/loci/rescue3/DH_coreH_res0.3_f0.5.parquet",
           "WCP": RD / "vocab/loci/runs_main_tau10_s1coh_V-coh0.1.parquet",
           "EMB": RD / "vocab_lists/loci/z10.6_V-nest.parquet"}
TARGET = {"TOK": (1686, 1512), "WCP": (1684, 1420), "EMB": (1675, 1172)}
ORDER = ("TOK", "WCP", "EMB")
FLANK = 10          # association window and layout window
TAU = 10.0          # nats
MIN_OTHER_GROUPS = 3
TESTABLE_GROUPS = 10
OTHERKIND_MIN_GROUPS = 5
R5_GAP = 5
ASSOC_BRIDGE = 2
ASSOC_REACH = 10
SEED = 20261113


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def lam(n, e):
    n = np.asarray(n, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = n * np.log(n / np.maximum(e, 1e-300)) - n + e
    return np.where(n > e, v, 0.0)


# ------------------------------------------------------------------------------------------- data
class Data:
    def __init__(self, protein_ids=False):
        cols = ["gid", "genome_code", "contig_code", "position_index", "strand", "length"]
        if protein_ids:
            cols.append("protein_id")
        g = pd.read_parquet(PKG / "universe/genes.parquet", columns=cols).sort_values("gid").reset_index(drop=True)
        assert np.array_equal(g.gid.to_numpy(), np.arange(len(g)))
        self.n = len(g)
        self.cc = g.contig_code.to_numpy(np.int64)
        self.gc = g.genome_code.to_numpy(np.int64)
        self.pos = g.position_index.to_numpy(np.int64)
        self.strand = g.strand.to_numpy(np.int8)
        self.length = g.length.to_numpy(np.int64)
        self.pid = g.protein_id.to_numpy() if protein_ids else None
        c = pd.read_parquet(PKG / "universe/contigs.parquet").set_index("contig_code")
        self.cfirst = c.first_gid.reindex(self.cc).to_numpy(np.int64)
        self.clast = self.cfirst + c.n_genes.reindex(self.cc).to_numpy(np.int64) - 1
        gm = pd.read_parquet(PKG / "universe/genomes.parquet").sort_values("genome_code").reset_index(drop=True)
        assert np.array_equal(gm.genome_code.to_numpy(), np.arange(len(gm)))
        self.genomes = gm
        self.ng_genes = gm.n_genes.to_numpy(np.int64)
        t = pd.read_parquet(TOK)
        self.tok = np.full(self.n, -1, np.int64)
        self.tok[t.gid.to_numpy()] = t.token.to_numpy()
        assert (self.tok >= 0).all()
        self.T = int(self.tok.max()) + 1
        m = pd.read_parquet(GROUPS, columns=["genome_code", "group"]).sort_values("genome_code")
        assert np.array_equal(m.genome_code.to_numpy(), np.arange(len(gm)))
        self.grp, _ = pd.factorize(m.group.to_numpy())
        self.grp = self.grp.astype(np.int64)
        self.G = int(self.grp.max()) + 1
        mk = pd.read_parquet(GOLD / "marked.parquet")
        self.marked = np.zeros(self.n, bool)
        self.marked[mk.gid.to_numpy()] = True
        codes, self.marker_names = pd.factorize(mk.marker.to_numpy())
        self.topm = np.full(self.n, -1, np.int64)
        self.topm[mk.gid.to_numpy()] = codes
        self.markers_all = pd.Series(mk.markers.to_numpy(), index=mk.gid.to_numpy())
        r = pd.read_parquet(GOLD / "runs.parquet")
        self.runs = r[(r.run_set == "g2") & (r.n_marked >= 16)].reset_index(drop=True)
        self.runs["group"] = self.grp[self.runs.genome_code.to_numpy()]
        self.runs["stratum"] = np.where(self.runs.family == "Parvarchaeaceae", "Parvarchaeaceae", self.runs.phylum)
        self.in_run = np.zeros(self.n, bool)
        self.run_of_gid = np.full(self.n, -1, np.int64)
        for i, (s, e) in enumerate(zip(self.runs.gid_start, self.runs.gid_end)):
            self.in_run[s:e + 1] = True
            self.run_of_gid[s:e + 1] = i


class Loci:
    def __init__(self, path):
        import pyarrow.parquet as pq
        cols = pq.read_schema(path).names
        sc, ec = ("gid_s", "gid_e") if "gid_s" in cols else ("s", "e")
        df = pd.read_parquet(path, columns=[sc, ec])
        a, b = df[sc].to_numpy(np.int64), df[ec].to_numpy(np.int64)
        o = np.argsort(a, kind="stable")
        self.a, self.b = a[o], b[o]
        self.maxlen = int((self.b - self.a + 1).max())
        self.n = len(a)

    def overlapping(self, lo, hi):
        j0 = np.searchsorted(self.a, lo - self.maxlen + 1, "left")
        j1 = np.searchsorted(self.a, hi, "right")
        idx = np.arange(j0, j1)
        return idx[self.b[idx] >= lo]

    def best(self, cs, ce):
        """Largest overlap with [cs, ce]; ties shorter; then first in start order (ribosomal_check rule)."""
        idx = self.overlapping(cs, ce)
        if len(idx) == 0:
            return -1, 0
        ov = np.minimum(self.b[idx], ce) - np.maximum(self.a[idx], cs) + 1
        ln = self.b[idx] - self.a[idx] + 1
        k = np.lexsort((idx, ln, -ov))[0]
        return int(idx[k]), int(ov[k])

    def length(self, j):
        return int(self.b[j] - self.a[j] + 1)


def ovl(a1, b1, a2, b2):
    return max(0, min(b1, b2) - max(a1, a2) + 1)


def standard(L, s, e):
    j, ov = L.best(s, e)
    n = e - s + 1
    if j < 0:
        return dict(best=-1, best_frac=0.0, precision=np.nan, recovered=False, resolved=False, coverage=0.0,
                    fragments=0, best_a=-1, best_b=-1)
    bf = ov / n
    prec = ov / max(L.length(j), 1)
    idx = L.overlapping(s, e)
    cov = np.zeros(n, bool)
    for k in idx:
        cov[max(L.a[k], s) - s:min(L.b[k], e) - s + 1] = True
    return dict(best=j, best_frac=bf, precision=prec, recovered=bf >= 0.5, resolved=(bf >= 0.8) and (prec >= 0.8),
                coverage=cov.mean(), fragments=len(idx), best_a=int(L.a[j]), best_b=int(L.b[j]))


# ------------------------------------------------------------------------------------------- repro
def cmd_repro(a):
    D = Data()
    runs = D.runs
    res = {}
    sys.path.insert(0, str(RD / "ribosomal/code"))
    import ribosomal_check as RC  # read-only import; bytecode and numba cache redirected
    for name in ORDER:
        L = Loci(OUTPUTS[name])
        rows = [standard(L, s, e) for s, e in zip(runs.gid_start, runs.gid_end)]
        df = pd.DataFrame(rows)
        locset = dict(a=L.a, b=L.b, mod=np.full(L.n, -1, np.int64), maxlen=L.maxlen, n=L.n)
        sc = RC.score(locset, runs.gid_start.to_numpy(), runs.gid_end.to_numpy())
        same = dict(recovered=bool(np.array_equal(sc.recovered.to_numpy(), df.recovered.to_numpy())),
                    resolved=bool(np.array_equal(sc.resolved.to_numpy(), df.resolved.to_numpy())),
                    best_frac=bool(np.allclose(sc.best_frac.to_numpy(), df.best_frac.to_numpy())),
                    precision=bool(np.allclose(sc.precision.to_numpy(), df.precision.to_numpy(), equal_nan=True)))
        rec, resd = int(df.recovered.sum()), int(df.resolved.sum())
        parv = runs.family.to_numpy() == "Parvarchaeaceae"
        res[name] = dict(loci=L.n, recovered=rec, resolved=resd, rc_recovered=int(sc.recovered.sum()),
                         rc_resolved=int(sc.resolved.sum()), per_run_equal_rc=same,
                         parv_recovered=int(df.recovered[parv].sum()), parv_resolved=int(df.resolved[parv].sum()),
                         target=TARGET[name], reproduced=bool((rec, resd) == TARGET[name] and all(same.values())))
        df.insert(0, "run_id", runs.run_id.to_numpy())
        df.to_parquet(WK / f"std_{name}.parquet", index=False)
        log(name, res[name])
    res["all_reproduced"] = all(res[k]["reproduced"] for k in ORDER)
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(res, open(WK / "repro.json", "w"), indent=1)
    log("ALL REPRODUCED:", res["all_reproduced"])


# ------------------------------------------------------------------------------------------- sample
def largest_remainder(counts, N):
    """counts: dict stratum -> count; returns dict stratum -> allocation summing to min(N, total)."""
    tot = sum(counts.values())
    N = min(N, tot)
    if tot == 0:
        return {k: 0 for k in counts}
    q = {k: N * v / tot for k, v in counts.items()}
    al = {k: int(np.floor(v)) for k, v in q.items()}
    rem = sorted(counts, key=lambda k: (-(q[k] - al[k]), k))
    for k in rem[:N - sum(al.values())]:
        al[k] += 1
    return al


def draw(rng, runs, mask, alloc):
    out = []
    for s in sorted(alloc):
        if alloc[s] == 0:
            continue
        cand = np.sort(runs.run_id.to_numpy()[mask & (runs.stratum.to_numpy() == s)])
        k = min(alloc[s], len(cand))
        out += [int(x) for x in rng.choice(cand, size=k, replace=False)]
    return out


def cmd_sample(a):
    rep = json.load(open(WK / "repro.json"))
    assert rep["all_reproduced"], "reproduction must hold before sampling"
    runs = pd.read_parquet(GOLD / "runs.parquet")
    runs = runs[(runs.run_set == "g2") & (runs.n_marked >= 16)].reset_index(drop=True)
    runs["stratum"] = np.where(runs.family == "Parvarchaeaceae", "Parvarchaeaceae", runs.phylum)
    strata = sorted(runs.stratum.unique())
    res = {n: pd.read_parquet(WK / f"std_{n}.parquet").resolved.to_numpy() for n in ORDER}
    out = dict(seed_base=SEED, strata=strata, unresolved={}, controls={}, alloc={})
    st = runs.stratum.to_numpy()
    for k, n in enumerate(ORDER):
        un = ~res[n]
        cnt = {s: int((un & (st == s)).sum()) for s in strata}
        al = largest_remainder(cnt, 40)
        out["unresolved"][n] = draw(np.random.default_rng(SEED + k), runs, un, al)
        # controls: phylum-matched to the failures, from resolved runs
        alc = largest_remainder(cnt, 20)
        avail = {s: int((res[n] & (st == s)).sum()) for s in strata}
        short = {s: alc[s] - avail[s] for s in strata if alc[s] > avail[s]}
        if short:
            for s in short:
                alc[s] = avail[s]
            extra = sum(short.values())
            room = {s: avail[s] - alc[s] for s in strata if avail[s] > alc[s]}
            add = largest_remainder({s: avail[s] for s in room}, extra)
            for s in add:
                alc[s] += min(add[s], room[s])
        out["controls"][n] = draw(np.random.default_rng(SEED + 4 + k), runs, res[n], alc)
        out["alloc"][n] = dict(unresolved_counts=cnt, unresolved_alloc=al, control_alloc=alc)
    un3 = ~res["TOK"] & ~res["WCP"] & ~res["EMB"]
    cnt3 = {s: int((un3 & (st == s)).sum()) for s in strata}
    al3 = largest_remainder(cnt3, 20)
    out["all_three"] = draw(np.random.default_rng(SEED + 3), runs, un3, al3)
    out["alloc"]["all_three"] = dict(counts=cnt3, alloc=al3, total=int(un3.sum()))
    out["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(out, open(WK / "sample.json", "w"), indent=1)
    log({k: len(v) for k, v in out["unresolved"].items()}, "all3", len(out["all_three"]),
        {k: len(v) for k, v in out["controls"].items()}, "unresolved in all three:", int(un3.sum()))


# ------------------------------------------------------------------------------------------- stats
@njit(cache=True)
def pair_expect(fa, fb, ptr, gen, cnt, m, grp, ng, w, qptr, qg):
    K = len(fa)
    acc = np.zeros(ng)
    mark = np.zeros(ng, np.bool_)
    touched = np.empty(ng, np.int64)
    etot = np.zeros(K)
    eq = np.zeros(len(qg))
    for k in range(K):
        i, i1 = ptr[fa[k]], ptr[fa[k] + 1]
        j, j1 = ptr[fb[k]], ptr[fb[k] + 1]
        same = fa[k] == fb[k]
        nt = 0
        while i < i1 and j < j1:
            if gen[i] < gen[j]:
                i += 1
            elif gen[i] > gen[j]:
                j += 1
            else:
                g = gen[i]
                if same:
                    p = w * cnt[i] * (cnt[i] - 1) / m[g]
                else:
                    p = 2.0 * w * cnt[i] * cnt[j] / m[g]
                q = grp[g]
                if not mark[q]:
                    mark[q] = True
                    touched[nt] = q
                    nt += 1
                    acc[q] = 0.0
                if p >= 1.0:
                    acc[q] = -np.inf
                elif p > 0.0:
                    acc[q] += np.log1p(-p)
                i += 1
                j += 1
        tot = 0.0
        for t in range(nt):
            tot += 1.0 - np.exp(acc[touched[t]])
        etot[k] = tot
        for z in range(qptr[k], qptr[k + 1]):
            q = qg[z]
            eq[z] = (1.0 - np.exp(acc[q])) if mark[q] else 0.0
        for t in range(nt):
            mark[touched[t]] = False
    return etot, eq


def cmd_stats(a):
    t0 = time.time()
    D = Data()
    runs = D.runs
    G, T = D.G, D.T
    S = {}
    # ---- A: flank association ------------------------------------------------------------------------
    fl_g, fl_r = [], []
    W = np.zeros(len(D.genomes), np.int64)
    Lg = np.zeros(len(D.genomes), np.int64)
    for i, r in enumerate(runs.itertuples()):
        cf, cl = D.cfirst[r.gid_start], D.clast[r.gid_start]
        left = np.arange(max(cf, r.gid_start - FLANK), r.gid_start)
        right = np.arange(r.gid_end + 1, min(cl, r.gid_end + FLANK) + 1)
        f = np.r_[left, right]
        fl_g.append(f)
        fl_r.append(np.full(len(f), i))
        W[r.genome_code] = len(f)
        Lg[r.genome_code] = r.gid_end - r.gid_start + 1
    fl_g = np.concatenate(fl_g)
    key = np.unique(D.tok[fl_g] * G + D.grp[D.gc[fl_g]])
    nX = np.bincount(key // G, minlength=T)
    S["flank_presence"] = key                      # sorted keys tok*G + group
    run_genome = np.zeros(len(D.genomes), bool)
    run_genome[runs.genome_code.to_numpy()] = True
    sel = run_genome[D.gc] & ~D.in_run & (nX[D.tok] > 0)
    df = pd.DataFrame(dict(tok=D.tok[sel], gc=D.gc[sel])).groupby(["tok", "gc"]).size().rename("c").reset_index()
    gcv = df.gc.to_numpy()
    p = np.minimum(1.0, df.c.to_numpy() * W[gcv] / (D.ng_genes[gcv] - Lg[gcv]))
    df["lg"] = np.where(p >= 1, -np.inf, np.log1p(-np.minimum(p, 0.999999999999)))
    df["group"] = D.grp[gcv]
    tg = df.groupby(["tok", "group"]).lg.sum().reset_index()
    tg["term"] = 1.0 - np.exp(tg.lg.to_numpy())
    EX = np.bincount(tg.tok.to_numpy(), weights=tg.term.to_numpy(), minlength=T)
    S["nX"], S["EX"] = nX, EX
    S["flank_term_key"] = (tg.tok.to_numpy() * G + tg.group.to_numpy())
    S["flank_term"] = tg.term.to_numpy()
    o = np.argsort(S["flank_term_key"])
    S["flank_term_key"], S["flank_term"] = S["flank_term_key"][o], S["flank_term"][o]
    log("flank association: families with n>=1", int((nX > 0).sum()), "all-group associated",
        int(((nX >= MIN_OTHER_GROUPS) & (lam(nX, EX) >= TAU)).sum()), f"{time.time()-t0:.0f}s")
    # ---- B: member shares, lengths ------------------------------------------------------------------
    rg = run_genome[D.gc]
    occ = np.unique(D.tok[rg] * G + D.grp[D.gc[rg]])
    ins = np.unique(D.tok[D.in_run] * G + D.grp[D.gc[D.in_run]])
    S["occ_run_genomes"], S["occ_inside_runs"] = occ, ins
    S["GX"] = np.bincount(occ // G, minlength=T)
    S["GinX"] = np.bincount(ins // G, minlength=T)
    allk = np.unique(D.tok * G + D.grp[D.gc])
    S["occ_all"] = allk
    S["GallX"] = np.bincount(allk // G, minlength=T)
    fl = pd.DataFrame(dict(tok=D.tok, length=D.length)).groupby("tok").length.agg(["median", "size"])
    S["fam_median_len"] = np.zeros(T)
    S["fam_median_len"][fl.index.to_numpy()] = fl["median"].to_numpy()
    S["fam_size"] = np.zeros(T, np.int64)
    S["fam_size"][fl.index.to_numpy()] = fl["size"].to_numpy()
    mm = pd.DataFrame(dict(m=D.topm[D.marked], length=D.length[D.marked])).groupby("m").length.median()
    S["marker_median_len"] = np.zeros(len(D.marker_names))
    S["marker_median_len"][mm.index.to_numpy()] = mm.to_numpy()
    # ---- D: anomalies -------------------------------------------------------------------------------
    same_ct = D.cc[:-1] == D.cc[1:]
    adj = same_ct & ((D.tok[:-1] == D.tok[1:]) | ((D.topm[:-1] >= 0) & (D.topm[:-1] == D.topm[1:])))
    anom_adj = np.zeros(D.n, bool)
    anom_adj[:-1] |= adj
    anom_adj[1:] |= adj
    short = (S["fam_size"][D.tok] >= 5) & (D.length < 0.5 * S["fam_median_len"][D.tok])
    S["anom_adj"], S["anom_short"] = anom_adj, short
    log("anomalies: adjacent", int(anom_adj.sum()), "short", int(short.sum()))
    # ---- C: junction recurrence for every crossing pair inside every run -----------------------------
    qkeys, qgroups = [], []
    for r in runs.itertuples():
        s, e = r.gid_start, r.gid_end
        for x in range(s, e):
            for y in range(x + 1, min(e, x + 3) + 1):
                tx, ty = D.tok[x], D.tok[y]
                qkeys.append(min(tx, ty) * T + max(tx, ty))
                qgroups.append(r.group)
    qkeys = np.array(qkeys, np.int64)
    qgroups = np.array(qgroups, np.int64)
    K = np.unique(qkeys)
    log("junction pair keys", len(K), "queries", len(qkeys))
    pk, pg = [], []
    for d in (1, 2, 3):
        i = np.arange(D.n - d)
        ok = D.cc[i] == D.cc[i + d]
        i = i[ok]
        tx, ty = D.tok[i], D.tok[i + d]
        k = np.minimum(tx, ty) * T + np.maximum(tx, ty)
        keep = np.isin(k, K)
        pk.append(k[keep])
        pg.append(D.grp[D.gc[i[keep]]])
    pk, pg = np.concatenate(pk), np.concatenate(pg)
    pres = np.unique(np.searchsorted(K, pk) * G + pg)
    S["pair_K"] = K
    S["pair_presence"] = pres                         # kidx*G + group
    S["pair_n"] = np.bincount(pres // G, minlength=len(K))
    fams = np.unique(np.r_[K // T, K % T])
    sel = np.isin(D.tok, fams)
    cdf = pd.DataFrame(dict(tok=D.tok[sel], gc=D.gc[sel])).groupby(["tok", "gc"]).size().rename("c").reset_index()
    fidx = np.searchsorted(fams, cdf.tok.to_numpy())
    ptr = np.r_[0, np.cumsum(np.bincount(fidx, minlength=len(fams)))].astype(np.int64)
    gen = cdf.gc.to_numpy(np.int64)
    cnt = cdf.c.to_numpy(np.float64)
    fa = np.searchsorted(fams, K // T)
    fb = np.searchsorted(fams, K % T)
    uq = np.unique(np.searchsorted(K, qkeys) * G + qgroups)
    qk, qg = uq // G, uq % G
    qptr = np.r_[0, np.cumsum(np.bincount(qk, minlength=len(K)))].astype(np.int64)
    etot, eq = pair_expect(fa.astype(np.int64), fb.astype(np.int64), ptr, gen, cnt, D.ng_genes.astype(np.float64),
                           D.grp, G, 3.0, qptr, qg.astype(np.int64))
    S["pair_E"] = etot
    S["pair_q_key"] = uq
    S["pair_q_term"] = eq
    log("pairs: all-group recurrent", int(((S["pair_n"] >= MIN_OTHER_GROUPS) & (lam(S["pair_n"], etot) >= TAU)).sum()),
        "of", len(K), f"{time.time()-t0:.0f}s")
    S["G"], S["T"] = G, T
    pickle.dump(S, open(WK / "stats.pkl", "wb"), protocol=4)
    log("stats written", f"{time.time()-t0:.0f}s")


# ------------------------------------------------------------------------------------------- evidence
def has(sorted_keys, key):
    i = np.searchsorted(sorted_keys, key)
    return i < len(sorted_keys) and sorted_keys[i] == key


class Ev:
    """Evidence for one run with its own group removed."""
    def __init__(self, D, S, ridx, mode="pre", window=None):
        """mode 'pre' = predeclared OTHER-KIND rule; 'corr' = post hoc rule (marker atypical for the family, or
        length > 2 x marker median). window = (s, e) evaluates an arbitrary interval (random controls; no breaks)."""
        self.D, self.S = D, S
        self.mode = mode
        if window is None:
            r = D.runs.iloc[ridx]
            self.r = r
            self.s, self.e = int(r.gid_start), int(r.gid_end)
            self.q = int(r.group)
            self.is_run = True
        else:
            self.r = None
            self.s, self.e = int(window[0]), int(window[1])
            self.q = int(D.grp[D.gc[self.s]])
            self.is_run = False
        self.cf, self.cl = int(D.cfirst[self.s]), int(D.clast[self.s])
        self._assoc = {}
        self._pair = {}
        self.strict = False      # post hoc strict variant: marked genes admissible only if run-associated
        self.adm_rule = "assoc"  # addendum: assoc | assoc_T | assoc_I | T | I (annotation-only tiers)
        self.no_s2 = False       # random windows: S2 not applied (a window holds 0-2 marked genes)

    # A
    def assoc_stats(self, X):
        S, G = self.S, self.S["G"]
        if X in self._assoc:
            return self._assoc[X]
        n = int(S["nX"][X]) - int(has(S["flank_presence"], X * G + self.q))
        i = np.searchsorted(S["flank_term_key"], X * G + self.q)
        own = S["flank_term"][i] if (i < len(S["flank_term_key"]) and S["flank_term_key"][i] == X * G + self.q) else 0.0
        E = max(float(S["EX"][X]) - own, 0.0)
        L = float(lam(n, E)) if n > 0 else 0.0
        out = (n, E, L, bool(n >= MIN_OTHER_GROUPS and L >= TAU))
        self._assoc[X] = out
        return out

    def assoc_gene(self, k):
        return self.assoc_stats(int(self.D.tok[k]))[3]

    def adm_ok(self, k):
        r = self.adm_rule
        if r == "assoc":
            return self.assoc_gene(k)
        if r == "assoc_T":
            return self.assoc_gene(k) and bool(self.S["annT"][k])
        if r == "assoc_I":
            return self.assoc_gene(k) and bool(self.S["annI"][k])
        if r == "T":
            return bool(self.S["annT"][k])
        if r == "I":
            return bool(self.S["annI"][k])
        raise ValueError(r)

    # B
    def other_kind(self, k):
        D, S, G = self.D, self.S, self.S["G"]
        X = int(D.tok[k])
        gx = int(S["GX"][X]) - int(has(S["occ_run_genomes"], X * G + self.q))
        gin = int(S["GinX"][X]) - int(has(S["occ_inside_runs"], X * G + self.q))
        share = gin / gx if gx > 0 else np.nan
        if self.mode == "pre":
            a = gx >= OTHERKIND_MIN_GROUPS and share < 0.5
        else:
            a = bool(S["fam_marker_share"][X] < 0.5)
        m = D.topm[k]
        b = m >= 0 and D.length[k] > 2 * S["marker_median_len"][m]
        return bool(a or b), gx, gin, share, bool(b)

    # C
    def pair_rec(self, x, y):
        D, S, G, T = self.D, self.S, self.S["G"], self.S["T"]
        tx, ty = int(D.tok[x]), int(D.tok[y])
        key = min(tx, ty) * T + max(tx, ty)
        if key in self._pair:
            return self._pair[key]
        ki = int(np.searchsorted(S["pair_K"], key))
        assert S["pair_K"][ki] == key
        n = int(S["pair_n"][ki]) - int(has(S["pair_presence"], ki * G + self.q))
        qi = int(np.searchsorted(S["pair_q_key"], ki * G + self.q))
        assert S["pair_q_key"][qi] == ki * G + self.q
        E = max(float(S["pair_E"][ki]) - float(S["pair_q_term"][qi]), 0.0)
        L = float(lam(n, E)) if n > 0 else 0.0
        rec = n >= MIN_OTHER_GROUPS and L >= TAU
        ga = int(S["GallX"][tx]) - int(has(S["occ_all"], tx * G + self.q))
        gb = int(S["GallX"][ty]) - int(has(S["occ_all"], ty * G + self.q))
        testable = ga >= TESTABLE_GROUPS and gb >= TESTABLE_GROUPS
        out = (bool(rec), bool(testable), n, E, L)
        self._pair[key] = out
        return out

    def breaks(self, cs, ce):
        out = []
        if not self.is_run:
            return out
        for p in range(cs, ce):
            rec_any, test_any = False, False
            for x in range(max(cs, p - 2), p + 1):
                for y in range(p + 1, min(ce, x + 3) + 1):
                    rec, test, *_ = self.pair_rec(x, y)
                    rec_any |= rec
                    test_any |= test
            if (not rec_any) and test_any:
                out.append(p)
        return out

    # D
    def anomaly(self, k):
        return bool(self.S["anom_adj"][k] or self.S["anom_short"][k])

    # R5
    def r5(self):
        D = self.D
        last = self.s
        k = self.s - 1
        while k >= self.cf:
            if D.marked[k]:
                last = k
            elif last - k > R5_GAP:
                break
            k -= 1
        lo = last
        last = self.e
        k = self.e + 1
        while k <= self.cl:
            if D.marked[k]:
                last = k
            elif k - last > R5_GAP:
                break
            k += 1
        return lo, last

    def s2_core(self):
        D = self.D
        mk = [k for k in range(self.s, self.e + 1) if D.marked[k]]
        if self.no_s2:
            return self.s, self.e, 0
        ok = [k for k in mk if not self.other_kind(k)[0]]
        if len(ok) == len(mk) or len(ok) == 0:
            return self.s, self.e, len(mk) - len(ok)
        pieces = []
        st = ok[0]
        prev = ok[0]
        for k in ok[1:]:
            if k - prev > 3:
                pieces.append((st, prev))
                st = k
            prev = k
        pieces.append((st, prev))
        pieces.sort(key=lambda p: (-(p[1] - p[0]), p[0]))
        return pieces[0][0], pieces[0][1], len(mk) - len(ok)

    def s3_adm(self, lo, hi):
        D = self.D

        def ok(k):
            return (D.marked[k] and not self.strict) or self.adm_ok(k)
        last = lo
        k = lo - 1
        while k >= max(self.cf, lo - ASSOC_REACH):
            if ok(k):
                last = k
            elif last - k > ASSOC_BRIDGE:
                break
            k -= 1
        nlo = last
        last = hi
        k = hi + 1
        while k <= min(self.cl, hi + ASSOC_REACH):
            if ok(k):
                last = k
            elif k - last > ASSOC_BRIDGE:
                break
            k += 1
        return nlo, last


def state(L, cs, ce, as_, ae):
    j, ov = L.best(cs, ce)
    if j < 0:
        return dict(best=-1, bf=0.0, prec=np.nan, ok=False, a=-1, b=-1)
    ln = L.length(j)
    bf = ov / (ce - cs + 1)
    prec = ovl(L.a[j], L.b[j], as_, ae) / ln
    return dict(best=j, bf=bf, prec=prec, ok=bool(bf >= 0.8 and prec >= 0.8), a=int(L.a[j]), b=int(L.b[j]))


def covered_mask(L, lo, hi, exclude=-1):
    m = np.zeros(hi - lo + 1, bool)
    for k in L.overlapping(lo, hi):
        if k == exclude:
            continue
        m[max(L.a[k], lo) - lo:min(L.b[k], hi) - lo + 1] = True
    return m


KIND = {"DEF_SCORING": "definition", "DEF_NEARBY_RIBO": "definition", "DEF_OTHER_KIND": "definition",
        "DEF_ASSOC_FLANK": "definition", "DEF_JUNCTION": "definition", "DATA_CONTIG_EDGE": "data",
        "DATA_GENE_CALL": "data", "METHOD_SPLIT": "method", "METHOD_UNCOVERED": "method",
        "METHOD_EXTENSION": "method", "RESOLVED": "resolved", "OTHER": "other"}


def evaluate(ev, L):
    """Rubric (predeclared.json) for one run under one output. Returns label, states and features."""
    s, e = ev.s, ev.e
    out = {}
    st0 = state(L, s, e, s, e)
    out.update(std_bf=st0["bf"], std_prec=st0["prec"], std_a=st0["a"], std_b=st0["b"], std_ok=st0["ok"])
    out["fail_type"] = ("none" if st0["ok"] else "coverage" if (st0["bf"] < 0.8 and not (st0["prec"] < 0.8))
                        else "precision" if st0["bf"] >= 0.8 else "both")
    idx = L.overlapping(s, e)
    cov = covered_mask(L, s, e)
    out["std_cov"] = float(cov.mean())
    out["n_loci_touch"] = int(len(idx))
    # any-locus
    anyok = False
    for k in idx:
        ov = ovl(L.a[k], L.b[k], s, e)
        if ov / (e - s + 1) >= 0.8 and ov / L.length(k) >= 0.8:
            anyok = True
            break
    out["any_ok"] = anyok
    r5lo, r5hi = ev.r5()
    out["r5_lo"], out["r5_hi"] = r5lo - s, r5hi - s
    r5m = [k for k in list(range(r5lo, s)) + list(range(e + 1, r5hi + 1)) if ev.D.marked[k]]
    out["n_r5_marked_out"] = len(r5m)
    out["n_r5_marked_out_assoc"] = int(sum(ev.assoc_gene(k) for k in r5m))
    if ev.strict:
        r5lo, r5hi = s, e
    st1 = state(L, s, e, r5lo, r5hi)
    cs, ce, n_ok = ev.s2_core()
    out["n_other_kind"] = n_ok
    out["s2_cs"], out["s2_ce"] = cs - s, ce - s
    as2, ae2 = min(r5lo, s), max(r5hi, e)
    st2 = state(L, cs, ce, as2, ae2)
    as3, ae3 = ev.s3_adm(as2, ae2)
    out["s3_lo"], out["s3_hi"] = as3 - s, ae3 - s
    st3 = state(L, cs, ce, as3, ae3)
    for nm, stx in (("s1", st1), ("s2", st2), ("s3", st3)):
        out[f"{nm}_bf"], out[f"{nm}_prec"], out[f"{nm}_ok"] = stx["bf"], stx["prec"], stx["ok"]
    out["s3_best_a"], out["s3_best_b"] = stx["a"], stx["b"]
    # features (output-independent unless stated)
    out["f_nearby"] = bool(r5lo < s or r5hi > e)
    out["f_edge"] = bool(s == ev.cf or e == ev.cl)
    out["n_anom_run"] = int(sum(ev.anomaly(k) for k in range(s, e + 1)))
    brk = ev.breaks(s, e)
    out["n_breaks_run"] = len(brk)
    out["breaks_run"] = ";".join(str(p - s) for p in brk)
    out["f_ext"] = bool(st0["best"] >= 0 and (st0["a"] < s or st0["b"] > e))
    out["f_ext_beyond_s3"] = bool(st0["best"] >= 0 and (st0["a"] < as3 or st0["b"] > ae3))
    out["f_multi"] = bool(any((L.a[k] < st0["a"] or L.b[k] > st0["b"]) for k in idx if k != st0["best"]))
    # label
    data_note = ""
    if st0["ok"]:
        label = "RESOLVED"
    elif anyok:
        label = "DEF_SCORING"
    elif st1["ok"]:
        label = "DEF_NEARBY_RIBO"
    elif st2["ok"]:
        label = "DEF_OTHER_KIND"
    elif st3["ok"]:
        label = "DEF_ASSOC_FLANK"
    else:
        label = None
        j3 = st3["best"]
        if st3["bf"] < 0.8:
            # data check
            if j3 >= 0:
                outside = [k for k in range(cs, ce + 1) if not (L.a[j3] <= k <= L.b[j3])]
            else:
                outside = list(range(cs, ce + 1))
            edge = False
            if cs == ev.cf and j3 >= 0 and all(k < L.a[j3] for k in outside):
                edge = True
            if ce == ev.cl and j3 >= 0 and all(k > L.b[j3] for k in outside):
                edge = True
            if j3 < 0 and (cs == ev.cf or ce == ev.cl):
                edge = False
            covc = covered_mask(L, cs, ce)
            fail_pts = [k for k in outside if (j3 >= 0 and (k == L.a[j3] - 1 or k == L.b[j3] + 1))]
            fail_pts += [k for k in range(cs, ce + 1) if not covc[k - cs]]
            gcall = any(ev.anomaly(x) for k in set(fail_pts) for x in (k - 1, k, k + 1) if ev.cf <= x <= ev.cl)
            if edge:
                label = "DATA_CONTIG_EDGE"
            elif gcall:
                label = "DATA_GENE_CALL"
            if label is None:
                br = ev.breaks(cs, ce)
                if br and j3 >= 0:
                    cuts = [cs] + [p + 1 for p in br]
                    ends = [p for p in br] + [ce]
                    pieces = list(zip(cuts, ends))
                    pieces.sort(key=lambda p: (-ovl(L.a[j3], L.b[j3], p[0], p[1]), p[0]))
                    pc = pieces[0]
                    st4 = state(L, pc[0], pc[1], as3, ae3)
                    out["s4_piece"] = f"{pc[0]-s}..{pc[1]-s}"
                    out["s4_bf"], out["s4_prec"] = st4["bf"], st4["prec"]
                    if st4["ok"]:
                        label = "DEF_JUNCTION"
            if label is None:
                oth = covered_mask(L, cs, ce, exclude=j3)
                k_other = sum(1 for k in outside if oth[k - cs])
                k_none = len(outside) - k_other
                out["k_other"], out["k_none"] = k_other, k_none
                label = "METHOD_SPLIT" if k_other >= k_none else "METHOD_UNCOVERED"
                if not (st3["prec"] >= 0.8):
                    data_note = "+ext"
        else:
            label = "METHOD_EXTENSION"
    out["label"] = label
    out["kind"] = KIND[label]
    out["note"] = data_note
    return out


# ------------------------------------------------------------------------------------------- classify
CORR_KEYS = ("n_other_kind", "s2_cs", "s2_ce", "s2_bf", "s2_prec", "s2_ok", "s3_lo", "s3_hi", "s3_bf", "s3_prec", "s3_ok",
             "s4_piece", "s4_bf", "s4_prec", "k_other", "k_none", "label", "kind", "note")


def add_marker_share(D, S):
    fam_marked = np.bincount(D.tok[D.marked], minlength=S["T"])
    S["fam_marker_share"] = np.where(S["fam_size"] > 0, fam_marked / np.maximum(S["fam_size"], 1), 0.0)


def cmd_classify(a):
    """Census: every run x output under the predeclared rubric (columns as named) and the post hoc corrected
    OTHER-KIND rule (columns prefixed c_). Random windows (5 draws per run) under the corrected chain for chance level."""
    t0 = time.time()
    D = Data()
    S = pickle.load(open(WK / "stats.pkl", "rb"))
    add_marker_share(D, S)
    rows = []
    Ls = {n: Loci(OUTPUTS[n]) for n in ORDER}
    for i in range(len(D.runs)):
        ev = Ev(D, S, i, "pre")
        evc = Ev(D, S, i, "corr")
        evx = Ev(D, S, i, "corr")
        evx.strict = True
        evc._assoc, evc._pair = ev._assoc, ev._pair          # identical caches (own group identical)
        evx._assoc, evx._pair = ev._assoc, ev._pair
        for n in ORDER:
            o = evaluate(ev, Ls[n])
            oc = evaluate(evc, Ls[n])
            ox = evaluate(evx, Ls[n])
            for k in CORR_KEYS:
                o["c_" + k] = oc.get(k)
                o["x_" + k] = ox.get(k)
            o["x_s1_ok"] = ox["s1_ok"]
            o.update(output=n, ridx=i, run_id=int(D.runs.run_id.iat[i]))
            rows.append(o)
        if i % 400 == 0:
            log(i, f"{time.time()-t0:.0f}s")
    df = pd.DataFrame(rows)
    df.to_parquet(WK / "census.parquet", index=False)
    for n in ORDER:
        d = df[df.output == n]
        log(n, "std resolved", int(d.std_ok.sum()), "pre", d.label.value_counts().to_dict())
        log(n, "corr", d.c_label.value_counts().to_dict())
        log(n, "strict", d.x_label.value_counts().to_dict())
    # random windows: chance level of each state
    ctrl = pd.read_parquet(GOLD / "controls_random.parquet")
    ctrl = ctrl[ctrl.run_id.isin(D.runs.run_id) & (ctrl.draw < 5)].reset_index(drop=True)
    rr = []
    for w in ctrl.itertuples():
        ev = Ev(D, S, -1, "corr", window=(w.gid_start, w.gid_end))
        ev.no_s2 = True
        evx = Ev(D, S, -1, "corr", window=(w.gid_start, w.gid_end))
        evx.no_s2, evx.strict = True, True
        evx._assoc = ev._assoc
        for n in ORDER:
            o = evaluate(ev, Ls[n])
            ox = evaluate(evx, Ls[n])
            rr.append(dict(output=n, run_id=w.run_id, draw=w.draw, std_ok=o["std_ok"], any_ok=o["any_ok"],
                           s1_ok=o["s1_ok"], s3_ok=o["s3_ok"], x_s3_ok=ox["s3_ok"], std_bf=o["std_bf"],
                           f_nearby=o["f_nearby"], n_marked=int(D.marked[w.gid_start:w.gid_end + 1].sum())))
    rw = pd.DataFrame(rr)
    rw.to_parquet(WK / "random_windows.parquet", index=False)
    log("random windows", len(ctrl), rw.groupby("output")[["std_ok", "any_ok", "s1_ok", "s3_ok", "x_s3_ok"]].mean().round(4).to_dict())
    log("classify done", f"{time.time()-t0:.0f}s")


# ------------------------------------------------------------------------------------------- full set
def cmd_full(a):
    """Nearby-run statistics over all 1,686 runs (no sample)."""
    D = Data()
    runs = D.runs
    allg2 = pd.read_parquet(GOLD / "runs.parquet")
    allg2 = allg2[allg2.run_set == "g2"]
    out = {}
    # (a) pairs of g2_m16 runs within <= 5 intervening genes on one contig
    r = runs.sort_values("gid_start")
    same = r.contig_code.to_numpy()[1:] == r.contig_code.to_numpy()[:-1]
    gap = r.gid_start.to_numpy()[1:] - r.gid_end.to_numpy()[:-1] - 1
    out["m16_pairs_same_contig"] = int(same.sum())
    out["m16_pairs_within5"] = int((same & (gap <= 5)).sum())
    # (b) nearest marked gene / g2 cluster outside each run
    mk = np.flatnonzero(D.marked)
    new = np.r_[True, (np.diff(mk) > 3) | (D.cc[mk[1:]] != D.cc[mk[:-1]])]
    cid = np.cumsum(new) - 1
    cl_s = mk[new]
    cl_e = mk[np.r_[new[1:], True]]
    cl_n = np.bincount(cid)
    rows = []
    for x in runs.itertuples():
        s, e = x.gid_start, x.gid_end
        cf, cl = D.cfirst[s], D.clast[s]
        best = {}
        for side in ("L", "R"):
            if side == "L":
                j = np.searchsorted(cl_e, s, "left") - 1
                ok = j >= 0 and cl_e[j] >= cf
                dist = (s - cl_e[j] - 1) if ok else 10 ** 6
            else:
                j = np.searchsorted(cl_s, e, "right")
                ok = j < len(cl_s) and cl_s[j] <= cl
                dist = (cl_s[j] - e - 1) if ok else 10 ** 6
            best[side] = (dist, int(cl_n[j]) if ok else 0)
        m8 = allg2[(allg2.contig_code == x.contig_code) & (allg2.n_marked >= 8) & (allg2.n_marked < 16)]
        d8 = [min(abs(int(t.gid_start) - e - 1), abs(s - int(t.gid_end) - 1)) for t in m8.itertuples()]
        rows.append(dict(run_id=x.run_id, stratum=x.stratum, dL=best["L"][0], nL=best["L"][1], dR=best["R"][0],
                         nR=best["R"][1], d_m8=min(d8) if d8 else 10 ** 6, n_m8_contig=len(d8)))
    nb = pd.DataFrame(rows)
    nb["d_any"] = nb[["dL", "dR"]].min(axis=1)
    nb["d_ge2"] = np.minimum(np.where(nb.nL >= 2, nb.dL, 10 ** 6), np.where(nb.nR >= 2, nb.dR, 10 ** 6))
    nb.to_parquet(WK / "nearby.parquet", index=False)
    for k in ("d_any", "d_ge2", "d_m8"):
        out[f"runs_{k}_le5"] = int((nb[k] <= 5).sum())
        out[f"runs_{k}_le10"] = int((nb[k] <= 10).sum())
    out["runs"] = int(len(nb))
    out["by_stratum_d_any_le5"] = nb[nb.d_any <= 5].stratum.value_counts().to_dict()
    out["by_stratum_runs"] = nb.stratum.value_counts().to_dict()
    json.dump(out, open(WK / "full.json", "w"), indent=1)
    log(out)


# ------------------------------------------------------------------------------------------- cases
def pfam_for(pids, threads=2):
    import duckdb
    con = duckdb.connect(str(DB), read_only=True)
    con.execute(f"SET threads={threads}")
    ids = pd.DataFrame(dict(protein_id=pd.unique(np.asarray(pids))))
    con.register("ids", ids)
    df = con.execute("SELECT a.protein_id, a.name, a.start_aa, a.score FROM annotations a JOIN ids USING (protein_id) "
                     "WHERE a.source = 'pfam' ORDER BY a.protein_id, a.start_aa").df()
    con.close()
    return df.groupby("protein_id").name.apply(lambda s: ",".join(s)).to_dict()


def diagram(ev, L, o, lo, hi, chars, brk):
    """One-line diagram over [lo, hi]: '|' contig end, [ ] best locus, ( ) maximal other loci, '/' break junction."""
    s = ev.s
    bj = o["std_a"], o["std_b"]
    idx = [k for k in L.overlapping(lo, hi)]
    others = [(int(L.a[k]), int(L.b[k])) for k in idx if not (L.a[k] == bj[0] and L.b[k] == bj[1])]
    others = [x for x in others if not (bj[0] >= 0 and x[0] >= bj[0] and x[1] <= bj[1])]
    maximal = [x for x in others if not any((y != x) and y[0] <= x[0] and y[1] >= x[1] for y in others)]
    maximal = sorted(set(maximal))
    pre = {k: "" for k in range(lo, hi + 1)}
    suf = {k: "" for k in range(lo, hi + 1)}
    if lo == ev.cf:
        pre[lo] += "|"
    for (x0, x1) in maximal:
        if x0 >= lo:
            pre[x0] += "("
        else:
            pre[lo] += f"<{lo - x0}("
        if x1 <= hi:
            suf[x1] = ")" + suf[x1]
        else:
            suf[hi] += f")>{x1 - hi}"
    if bj[0] >= 0:
        if bj[0] >= lo:
            pre[bj[0]] += "["
        else:
            pre[lo] += f"<{lo - bj[0]}["
        if bj[1] <= hi:
            suf[bj[1]] = "]" + suf[bj[1]]
        else:
            suf[hi] += f"]>{bj[1] - hi}"
    for p in brk:
        if lo <= p <= hi:
            suf[p] += "/"
    if hi == ev.cl:
        suf[hi] += "|"
    line, strand = [], []
    D = ev.D
    for k in range(lo, hi + 1):
        line.append(pre[k] + chars[k] + suf[k])
        strand.append(" " * len(pre[k]) + (">" if D.strand[k] > 0 else "<") + " " * len(suf[k]))
    loci_rel = ";".join(f"{int(L.a[k]) - s}..{int(L.b[k]) - s}" for k in idx)
    return "".join(line), "".join(strand), loci_rel


def cmd_cases(a):
    t0 = time.time()
    D = Data(protein_ids=True)
    S = pickle.load(open(WK / "stats.pkl", "rb"))
    smp = json.load(open(WK / "sample.json"))
    cen = pd.read_parquet(WK / "census.parquet").set_index(["output", "run_id"])
    Ls = {n: Loci(OUTPUTS[n]) for n in ORDER}
    ridx_of = pd.Series(np.arange(len(D.runs)), index=D.runs.run_id.to_numpy())
    plan = []
    for n in ORDER:
        plan += [("U", n, r) for r in smp["unresolved"][n]]
    for r in smp["all_three"]:
        plan += [("A3", n, r) for n in ORDER]
    for n in ORDER:
        plan += [("C", n, r) for r in smp["controls"][n]]
    # Pfam for every window gene and every best-locus gene of the plan
    need = set()
    for (_, n, r) in plan:
        i = ridx_of[r]
        s, e = int(D.runs.gid_start.iat[i]), int(D.runs.gid_end.iat[i])
        cf, cl = D.cfirst[s], D.clast[s]
        need.update(range(max(cf, s - FLANK), min(cl, e + FLANK) + 1))
        c = cen.loc[(n, r)]
        if c.std_a >= 0:
            need.update(range(int(c.std_a), int(c.std_b) + 1))
    need = np.array(sorted(need))
    pf = pfam_for(D.pid[need])
    log("pfam for", len(need), "genes", f"{time.time()-t0:.0f}s")
    case_rows, gene_rows, diag_lines, ext_rows = [], [], [], []
    count = {}
    for (smpl, n, r) in plan:
        count[(smpl, n)] = count.get((smpl, n), 0) + 1
        cid = f"{smpl}-{n}-{count[(smpl, n)]:02d}"
        i = ridx_of[r]
        ev = Ev(D, S, i)
        L = Ls[n]
        o = cen.loc[(n, r)].to_dict()
        s, e = ev.s, ev.e
        lo, hi = max(ev.cf, s - FLANK), min(ev.cl, e + FLANK)
        chars = {}
        okflag = {}
        for k in range(min(lo, max(ev.cf, o["std_a"] if o["std_a"] >= 0 else lo)),
                       max(hi, min(ev.cl, o["std_b"] if o["std_b"] >= 0 else hi)) + 1):
            if s <= k <= e:
                if D.marked[k]:
                    okf = ev.other_kind(k)
                    okflag[k] = okf
                    chars[k] = "K" if okf[0] else "R"
                else:
                    chars[k] = "r"
            else:
                chars[k] = "M" if D.marked[k] else ("a" if ev.assoc_gene(k) else "o")
        brk = ev.breaks(s, e)
        line, strand, loci_rel = diagram(ev, L, o, lo, hi, chars, brk)
        r0 = D.runs.iloc[i]
        row = dict(case_id=cid, sample=smpl, output=n, run_id=int(r), genome=r0.genome, contig=r0.contig,
                   phylum=r0.phylum, family=r0.family, stratum=r0.stratum, group=int(r0.group),
                   run_genes=e - s + 1, n_marked=int(r0.n_marked), contig_genes=int(ev.cl - ev.cf + 1),
                   run_pos=f"{s - ev.cf}..{e - ev.cf}", best_rel=(f"{o['std_a'] - s}..{o['std_b'] - s}" if o["std_a"] >= 0 else "none"))
        for k in ("std_bf", "std_prec", "std_cov", "n_loci_touch", "fail_type", "label", "kind", "note", "any_ok",
                  "r5_lo", "r5_hi", "n_other_kind", "s2_cs", "s2_ce", "s3_lo", "s3_hi", "s1_bf", "s1_prec", "s2_bf",
                  "s2_prec", "s3_bf", "s3_prec", "s4_piece", "s4_bf", "s4_prec", "k_other", "k_none", "n_breaks_run",
                  "breaks_run", "f_nearby", "f_edge", "n_anom_run", "f_ext", "f_ext_beyond_s3", "f_multi",
                  "n_r5_marked_out", "n_r5_marked_out_assoc") + tuple("c_" + k for k in CORR_KEYS) \
                + tuple("x_" + k for k in CORR_KEYS):
            v = o.get(k)
            row[k] = (round(v, 3) if isinstance(v, float) and not np.isnan(v) else v)
        # extension genes of the best locus (outside the run)
        ext = []
        if o["std_a"] >= 0:
            for k in list(range(int(o["std_a"]), s)) + list(range(e + 1, int(o["std_b"]) + 1)):
                X = int(D.tok[k])
                nn, EE, LL, cl = ev.assoc_stats(X)
                kind = "M" if D.marked[k] else ("a" if cl else "o")
                inadm = (s + o["s3_lo"]) <= k <= (s + o["s3_hi"])
                ext.append(kind)
                ext_rows.append(dict(case_id=cid, output=n, sample=smpl, label=row["label"], rel=k - s, gid=k, token=X,
                                     cls=kind, in_s3_adm=bool(inadm), n_other=nn, E_other=round(EE, 3),
                                     Lambda=round(LL, 2), pfam=pf.get(D.pid[k], ""),
                                     marker=(D.marker_names[D.topm[k]] if D.topm[k] >= 0 else "")))
        row["ext_genes"] = len(ext)
        row["ext_M_a_o"] = f"{ext.count('M')}/{ext.count('a')}/{ext.count('o')}"
        row["other_kind_members"] = ";".join(
            f"{k - s}:{D.marker_names[D.topm[k]]}:t{int(D.tok[k])}:G{v[1]}:in{v[2]}{':len' if v[4] else ''}"
            for k, v in okflag.items() if v[0])
        row["diagram"] = line
        row["loci_rel"] = loci_rel
        case_rows.append(row)
        diag_lines.append(f"{cid}  run {r} {r0.genome} {r0.stratum} len {e - s + 1} ({int(r0.n_marked)} marked)  "
                          f"bf {row['std_bf']} prec {row['std_prec']} cov {row['std_cov']}  -> {row['label']}{row['note']}"
                          f" | corrected {row['c_label']}{row['c_note'] or ''} | strict {row['x_label']}{row['x_note'] or ''}"
                          f"  [ext M/a/o {row['ext_M_a_o']}; other-kind {row['n_other_kind']}; breaks {row['n_breaks_run']}]")
        diag_lines.append("    " + line)
        diag_lines.append("    " + strand)
        for k in range(lo, hi + 1):
            X = int(D.tok[k])
            nn, EE, LL, cl = ev.assoc_stats(X)
            covers = [f"{int(L.a[j]) - s}..{int(L.b[j]) - s}" for j in L.overlapping(k, k)]
            gene_rows.append(dict(
                case_id=cid, rel=k - s, gid=k, contig_pos=k - ev.cf, strand=int(D.strand[k]), length_aa=int(D.length[k]),
                token=X, member=("K" if chars[k] == "K" else "R" if chars[k] == "R" else "r" if chars[k] == "r" else ""),
                marked=bool(D.marked[k]), top_marker=(D.marker_names[D.topm[k]] if D.topm[k] >= 0 else ""),
                pfam=pf.get(D.pid[k], ""), assoc_n_other=nn, assoc_E_other=round(EE, 3), assoc_Lambda=round(LL, 2),
                assoc=bool(cl), other_kind=("" if k not in okflag else f"G{okflag[k][1]} in{okflag[k][2]} "
                                            f"share{okflag[k][3]:.2f}{' len' if okflag[k][4] else ''}"),
                anomaly=("adj" if S["anom_adj"][k] else "") + ("short" if S["anom_short"][k] else ""),
                in_best=bool(o["std_a"] <= k <= o["std_b"]), loci_covering=";".join(covers),
                in_R5=bool(s + o["r5_lo"] <= k <= s + o["r5_hi"]), in_S3_adm=bool(s + o["s3_lo"] <= k <= s + o["s3_hi"])))
    pd.DataFrame(case_rows).to_csv(WK / "cases.tsv", sep="\t", index=False)
    pd.DataFrame(gene_rows).to_csv(WK / "layouts.tsv", sep="\t", index=False)
    pd.DataFrame(ext_rows).to_csv(WK / "ext_genes.tsv", sep="\t", index=False)
    with open(WK / "diagrams.txt", "w") as fh:
        fh.write("Legend: R run member with a Ribosomal_* domain; r run member without (gap gene); K run member flagged\n"
                 "OTHER-KIND; M marked gene outside the run; a gene of a run-associated family (leave-own-group-out);\n"
                 "o other gene; [ ] best locus; ( ) maximal other loci not inside the best; <n / >n locus continues n\n"
                 "genes past the window; | contig end; / BREAK junction (non-recurring, testable). Second line: strand.\n\n")
        fh.write("\n".join(diag_lines) + "\n")
    log("cases", len(case_rows), "genes", len(gene_rows), f"{time.time()-t0:.0f}s")


# ------------------------------------------------------------------------------------------- addendum
NEW = {"L4b": RD / "long_loci/loci/dpann_group/L4b_t0.1.parquet",
       "L4b+": RD / "long_loci/loci/dpann_group/L4bp_t0.1.parquet",
       "L1": RD / "long_loci/loci/dpann_group/L1.parquet"}
NEW_ORDER = ("L4b", "L4b+", "L1")
ALL6 = ORDER + NEW_ORDER
TIER_T = ("ribosomal_protein", "translation", "translation_factor")
TIER_I = TIER_T + ("trna_synthetase", "trna_modification", "rna_processing", "transcription", "rna_polymerase",
                   "replication", "sec_pathway", "signal_recognition")
RULES = ("assoc", "assoc_T", "assoc_I", "T", "I")


def all_paths():
    d = dict(OUTPUTS)
    d.update(NEW)
    return d


def cmd_annot(a):
    """KO-derived role flags per gene (annotation only; no gene order)."""
    import duckdb
    D = Data(protein_ids=True)
    con = duckdb.connect(str(DB), read_only=True)
    con.execute("SET threads=2")
    terms = ",".join(f"'{t}'" for t in TIER_I)
    df = con.execute(f"SELECT DISTINCT protein_id, term_id FROM semantic_terms WHERE facet = 'role' AND "
                     f"source_db IN ('kofam', 'kegg') AND term_id IN ({terms})").df()
    con.close()
    pid2gid = pd.Series(np.arange(D.n), index=D.pid)
    df["gid"] = pid2gid.reindex(df.protein_id.to_numpy()).to_numpy()
    df = df.dropna(subset=["gid"])
    annT = np.zeros(D.n, bool)
    annI = np.zeros(D.n, bool)
    annT[df[df.term_id.isin(TIER_T)].gid.astype(np.int64).to_numpy()] = True
    annI[df.gid.astype(np.int64).to_numpy()] = True
    np.savez(WK / "annot.npz", annT=annT, annI=annI)
    per = df.groupby("term_id").gid.nunique().to_dict()
    log("annot: genes T", int(annT.sum()), "I", int(annI.sum()), per)
    json.dump(dict(T=int(annT.sum()), I=int(annI.sum()), per_term=per, marked_with_T=int((annT & D.marked).sum()),
                   marked=int(D.marked.sum())), open(WK / "annot.json", "w"), indent=1)


def sens_states(ev, L, rules):
    """Fast sensitivity states (corrected S2 core, strict admissibility) for one run or window under one output."""
    s, e = ev.s, ev.e
    out = {}
    st0 = state(L, s, e, s, e)
    out["std_ok"] = st0["ok"]
    anyok = False
    for k in L.overlapping(s, e):
        ov = ovl(L.a[k], L.b[k], s, e)
        if ov / (e - s + 1) >= 0.8 and ov / L.length(k) >= 0.8:
            anyok = True
            break
    out["any_ok"] = anyok
    cs, ce, _ = ev.s2_core()
    j, ov = L.best(cs, ce)
    for r in rules:
        ev.adm_rule = r
        as3, ae3 = ev.s3_adm(s, e)
        if j < 0:
            out[f"{r}_ok"] = False
            continue
        bf = ov / (ce - cs + 1)
        prec = ovl(L.a[j], L.b[j], as3, ae3) / L.length(j)
        out[f"{r}_ok"] = bool(bf >= 0.8 and prec >= 0.8)
    ev.adm_rule = "assoc"
    return out


def cmd_addendum(a):
    t0 = time.time()
    D = Data()
    S = pickle.load(open(WK / "stats.pkl", "rb"))
    add_marker_share(D, S)
    z = np.load(WK / "annot.npz")
    S["annT"], S["annI"] = z["annT"], z["annI"]
    runs = D.runs
    paths = all_paths()
    Ls = {n: Loci(paths[n]) for n in ALL6}
    # ---- (1) reproduction ----------------------------------------------------------------------------------
    sys.path.insert(0, str(RD / "ribosomal/code"))
    import ribosomal_check as RC
    rep = {}
    parv = runs.family.to_numpy() == "Parvarchaeaceae"
    for n in NEW_ORDER:
        L = Ls[n]
        df = pd.DataFrame([standard(L, s, e) for s, e in zip(runs.gid_start, runs.gid_end)])
        sc = RC.score(dict(a=L.a, b=L.b, mod=np.full(L.n, -1, np.int64), maxlen=L.maxlen, n=L.n),
                      runs.gid_start.to_numpy(), runs.gid_end.to_numpy())
        multi = (L.b > L.a)
        rep[n] = dict(loci=L.n, single_gene_loci=int((~multi).sum()), recovered=int(df.recovered.sum()),
                      resolved=int(df.resolved.sum()), rc_recovered=int(sc.recovered.sum()),
                      rc_resolved=int(sc.resolved.sum()),
                      per_run_equal_rc=bool(np.array_equal(sc.resolved.to_numpy(), df.resolved.to_numpy())),
                      parv_recovered=int(df.recovered[parv].sum()), parv_resolved=int(df.resolved[parv].sum()),
                      fail_coverage_only=int(((df.best_frac < 0.8) & (df.precision >= 0.8)).sum()),
                      fail_precision_only=int(((df.best_frac >= 0.8) & (df.precision < 0.8)).sum()),
                      fail_both=int(((df.best_frac < 0.8) & (df.precision < 0.8)).sum()))
        log(n, rep[n])
    json.dump(rep, open(WK / "repro_add.json", "w"), indent=1)
    # ---- (2) census of the new outputs under the three rubrics ------------------------------------------------
    rows, srows = [], []
    for i in range(len(runs)):
        ev = Ev(D, S, i, "pre")
        evc = Ev(D, S, i, "corr")
        evx = Ev(D, S, i, "corr")
        evx.strict = True
        evc._assoc, evc._pair = ev._assoc, ev._pair
        evx._assoc, evx._pair = ev._assoc, ev._pair
        for n in NEW_ORDER:
            o = evaluate(ev, Ls[n])
            oc = evaluate(evc, Ls[n])
            ox = evaluate(evx, Ls[n])
            for k in CORR_KEYS:
                o["c_" + k] = oc.get(k)
                o["x_" + k] = ox.get(k)
            o["x_s1_ok"] = ox["s1_ok"]
            o.update(output=n, ridx=i, run_id=int(runs.run_id.iat[i]))
            rows.append(o)
        for n in ALL6:
            st = sens_states(evx, Ls[n], RULES)
            st.update(output=n, run_id=int(runs.run_id.iat[i]))
            srows.append(st)
        if i % 400 == 0:
            log(i, f"{time.time()-t0:.0f}s")
    cen = pd.DataFrame(rows)
    cen.to_parquet(WK / "census_add.parquet", index=False)
    sens = pd.DataFrame(srows)
    sens.to_parquet(WK / "sens_add.parquet", index=False)
    for n in NEW_ORDER:
        d = cen[cen.output == n]
        log(n, "std", int(d.std_ok.sum()), "corr", d.c_label.value_counts().to_dict())
    # ---- random windows, all six outputs, all rules ----------------------------------------------------------
    ctrl = pd.read_parquet(GOLD / "controls_random.parquet")
    ctrl = ctrl[ctrl.run_id.isin(runs.run_id) & (ctrl.draw < 5)].reset_index(drop=True)
    rr = []
    for w in ctrl.itertuples():
        ev = Ev(D, S, -1, "corr", window=(w.gid_start, w.gid_end))
        ev.no_s2, ev.strict = True, True
        for n in ALL6:
            st = sens_states(ev, Ls[n], RULES)
            st.update(output=n, run_id=w.run_id, draw=w.draw)
            rr.append(st)
    rw = pd.DataFrame(rr)
    rw.to_parquet(WK / "random_windows_add.parquet", index=False)
    log("random windows", rw.groupby("output")[["std_ok", "any_ok"] + [f"{r}_ok" for r in RULES]].mean().round(4).to_dict())
    # ---- (4) bond gaps inside under-covered L4b runs ------------------------------------------------------------
    b = np.load(RD / "long_loci/nodes/dpann_group/bonds_obs.npz")
    bk, bd, bl = b["k"], b["d"], b["lam"]
    sk = np.zeros(D.n)
    for off in range(3):
        m = bd > off
        np.maximum.at(sk, bk[m] + off, bl[m])
    np.save(WK / "bond_sk.npy", sk)
    grp_tok = S["GallX"]
    gaps = []
    d4 = cen[(cen.output == "L4b") & ~cen.std_ok]
    for r in d4.itertuples():
        i = r.ridx
        ev = Ev(D, S, i, "corr")
        s, e = ev.s, ev.e
        cov_fail = r.std_bf < 0.8
        for p in range(s, e):
            if sk[p] > 0:
                continue
            rec_a, test_a, n_a, E_a, L_a = ev.pair_rec(p, p + 1)
            best = (-1, 0.0, False, False, -1, -1)
            for x in range(max(s, p - 2), p + 1):
                for y in range(p + 1, min(e, x + 3) + 1):
                    rc_, ts_, nn, EE, LL = ev.pair_rec(x, y)
                    if nn > best[0]:
                        best = (nn, LL, rc_, ts_, x - s, y - s)
            tx, ty = int(D.tok[p]), int(D.tok[p + 1])
            mk = lambda k: D.marker_names[D.topm[k]] if D.topm[k] >= 0 else "-"
            gaps.append(dict(run_id=r.run_id, stratum=runs.stratum.iat[i], cov_fail=bool(cov_fail),
                             fail_type=r.fail_type, c_label=r.c_label, rel=p - s, run_len=e - s + 1,
                             left=mk(p), right=mk(p + 1), strand_flip=bool(D.strand[p] != D.strand[p + 1]),
                             adj_n_other=n_a, adj_lambda=round(L_a, 1), adj_recurrent=rec_a, adj_testable=test_a,
                             best_n_other=best[0], best_lambda=round(best[1], 1), best_recurrent=best[2],
                             best_pair=f"{best[4]}-{best[5]}", tok_groups_left=int(grp_tok[tx]),
                             tok_groups_right=int(grp_tok[ty]),
                             cut_of_best=bool(r.std_a >= 0 and (p == r.std_b or p + 1 == r.std_a))))
    g = pd.DataFrame(gaps)
    g.to_csv(WK / "gaps_L4b.tsv", sep="\t", index=False)
    log("gaps", len(g), "in runs", g.run_id.nunique())
    log("addendum done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    WK.mkdir(parents=True, exist_ok=True)
    {"repro": cmd_repro, "sample": cmd_sample, "stats": cmd_stats, "classify": cmd_classify,
     "cases": cmd_cases, "full": cmd_full, "annot": cmd_annot, "addendum": cmd_addendum}[cmd](None)
