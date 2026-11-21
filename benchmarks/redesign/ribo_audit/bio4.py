#!/usr/bin/env python
"""ribo_audit part 4: the frozen cluster-level ribosomal run definition (predeclared4.json) on lut, DPANN and SR-VP.

usage (biotite login node, one thread):
  bio4.py repro   DS            standing recovered / resolved / any-fit of the three outputs (+ ribosomal_check equality)
  bio4.py ext     DS READING    cluster-level targets (READING union | domain): recovered, resolved, any-fit, chance;
                                standing chance with the same procedure
  bio4.py verdict DS DEF        part-2 strict verdicts for every run unresolved under DEF (cluster | standing), 3 outputs
  bio4.py insert  DS            L15 -> alpha-block insert of every bacterial run (output-independent)
DS = lut | dpann | srvp. Writes $RD/ribo_audit/work4/<DS>/.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bio as Bm  # noqa: E402  (part-2 rules, read-only reuse)
import audit as A  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RD = A.RD
W4 = A.AD / "work4"
log = A.log
EXTRA_NAMES = {"K02601": "nusG", "K03043": "rpoB", "K03046": "rpoC", "K02358": "tufA", "K02355": "fusA", "K00939": "adk",
               "K01265": "map", "K02518": "infA", "K03073": "secE", "K01883": "cysS", "K03685": "rnc", "K03218": "rlmB",
               "K16786": "ecfA1", "K16787": "ecfA2", "K16785": "ecfT", "K06173": "truA"}
OUTS = ("Z", "WA", "DH")
LUT = RD / "slice_lut/testset/package_k8"
SRV = RD / "ws_srvp_bins/package"
DPK = RD / "ws_dpann/package"
CFG = {
    "lut": dict(genes=LUT / "universe/genes.parquet", contigs=LUT / "universe/contigs.parquet", genomes=LUT / "universe/genomes.parquet",
                marked=LUT / "gold/ribosomal_marked.parquet", runs=LUT / "gold/ribosomal_runs.parquet",
                groups=RD / "long_loci/frozen/groups/emb_hom_average_t0.75.parquet", flags=LUT / "universe/gene_flags.parquet",
                annot=W4 / "lut/annot", base=W4 / "lut/annot/genes_annot.parquet",
                out={"Z": RD / "long_loci/frozen/loci/lut_Z/L4bp_t0.02.parquet", "WA": RD / "long_loci/frozen/loci/lut_WA/L4b_t0.2.parquet",
                     "DH": RD / "long_loci/frozen/loci/lut_engine/DH.parquet"},
                target={"Z": (857, 811), "WA": (858, 788), "DH": (None, 668)}),
    "dpann": dict(genes=DPK / "universe/genes.parquet", contigs=DPK / "universe/contigs.parquet", genomes=DPK / "universe/genomes.parquet",
                  marked=RD / "ribosomal/gold/marked.parquet", runs=RD / "ribosomal/gold/runs.parquet",
                  groups=RD / "lineagefree/groups/emb_hom_average_t0.75.parquet", flags=W4 / "dpann/annot/flags.parquet",
                  annot=W4 / "dpann/annot", base=W4 / "dpann/annot/genes_annot.parquet",
                  out={"Z": RD / "long_loci/loci/dpann_Z/L4bp_t0.02.parquet", "WA": RD / "long_loci/loci/dpann_WA/L4b_t0.2.parquet",
                       "DH": RD / "boundary/loci/rescue3/DH_coreH_res0.3_f0.5.parquet"},
                  target={"Z": (1681, 1542), "WA": (1685, 1594), "DH": (1686, 1512)}),
    "srvp": dict(genes=SRV / "universe/genes.parquet", contigs=SRV / "universe/contigs.parquet", genomes=SRV / "universe/genomes.parquet",
                 marked=SRV / "ribosomal/marked.parquet", runs=SRV / "ribosomal/runs.parquet",
                 groups=RD / "long_loci/frozen2/groups/emb_hom_average_t0.75.parquet", flags=SRV / "universe/gene_flags.parquet",
                 annot=A.AD / "work3/annot", base=RD / "long_loci/frozen2/annot/genes_annot.parquet",
                 out={"Z": RD / "long_loci/frozen2/loci/srvp_Z/L4bp_t0.02.parquet", "WA": RD / "long_loci/frozen2/loci/srvp_WA/L4b_t0.2.parquet",
                      "DH": RD / "long_loci/frozen2/loci/srvp_engine/DH.parquet"},
                 target={"Z": (802, 528), "WA": (803, 498), "DH": (None, 313)}),
}


# ------------------------------------------------------------------------------------------- data
class Data4:
    def __init__(self, ds):
        c = CFG[ds]
        self.ds = ds
        g = pd.read_parquet(c["genes"], columns=["gid", "genome_code", "contig_code", "position_index", "strand", "length",
                                                 "start", "end"]).sort_values("gid").reset_index(drop=True)
        assert np.array_equal(g.gid.to_numpy(), np.arange(len(g)))
        self.n = len(g)
        self.cc = g.contig_code.to_numpy(np.int64)
        self.gc = g.genome_code.to_numpy(np.int64)
        self.pos = g.position_index.to_numpy(np.int64)
        self.strand = g.strand.to_numpy(np.int8)
        self.length = g.length.to_numpy(np.int64)
        self.start = g.start.to_numpy(np.int64)
        self.end = g.end.to_numpy(np.int64)
        ct = pd.read_parquet(c["contigs"]).set_index("contig_code")
        self.cfirst = ct.first_gid.reindex(self.cc).to_numpy(np.int64)
        self.clast = self.cfirst + ct.n_genes.reindex(self.cc).to_numpy(np.int64) - 1
        gm = pd.read_parquet(c["genomes"]).sort_values("genome_code").reset_index(drop=True)
        assert np.array_equal(gm.genome_code.to_numpy(), np.arange(len(gm)))
        dom = gm.domain.to_numpy() if "domain" in gm.columns else gm.tax.str.split(";").str[0].str.replace("d__", "").to_numpy()
        self.genomes = gm
        self.domain_of_genome = dom
        m = pd.read_parquet(c["groups"], columns=["genome_code", "group"]).sort_values("genome_code")
        assert np.array_equal(m.genome_code.to_numpy(), np.arange(len(gm)))
        self.grp = pd.factorize(m.group.to_numpy())[0].astype(np.int64)
        self.G = int(self.grp.max()) + 1
        mk = pd.read_parquet(c["marked"])
        self.marked = np.zeros(self.n, bool)
        self.marked[mk.gid.to_numpy()] = True
        self.marker = pd.Series(mk.marker.to_numpy(), index=mk.gid.to_numpy())
        r = pd.read_parquet(c["runs"])
        if "m16" in r.columns and ds == "srvp":
            r = r[(r.run_set == "g2") & r.m16]
        else:
            r = r[(r.run_set == "g2") & (r.n_marked >= 16)]
        r = r.reset_index(drop=True)
        gcode = r.genome_code.to_numpy()
        r["domain"] = dom[gcode]
        closed = gm.n_contigs.to_numpy()[gcode] == 1
        r["gtype"] = np.where(r.domain == "Archaea", "Archaea", np.where(closed, "Bact_closed", "Bact_multi"))
        r["stratum"] = r.gtype
        r["group"] = self.grp[gcode]
        self.runs = r
        fl = pd.read_parquet(c["flags"]).sort_values("gid")
        assert len(fl) == self.n
        pa = fl.partial.astype(str).to_numpy()
        self.partial = (pa != "00") & fl.partial.notna().to_numpy()
        self.truncated = fl.truncated.fillna(False).astype(bool).to_numpy() if "truncated" in fl.columns else np.zeros(self.n, bool)


class Bio4(Bm.Bio):
    """Part-3 annotation merge (Bio3) on any dataset; READING union = supplement for every genome (as in part 3),
    domain = supplement only for Bacteria genomes."""
    def __init__(self, ds, reading="union"):
        t0 = time.time()
        self.D = D = Data4(ds)
        c = CFG[ds]
        self.start, self.end = D.start, D.end
        base = pd.read_parquet(c["base"]).sort_values("gid").reset_index(drop=True)
        assert len(base) == D.n
        ex = pd.read_parquet(c["annot"] / "extra_ko.parquet")
        wp = pd.read_parquet(c["annot"] / "window_pfam.parquet")
        win = np.zeros(D.n, bool)
        win[np.load(c["annot"] / "window_gids.npy")] = True
        self.window = win
        ko = base.ko.to_numpy(dtype=object).copy()
        ko_name = base.ko_name.to_numpy(dtype=object).copy()
        cls = base.cls.to_numpy(dtype=object).copy()
        lit = base.lit.fillna("").to_numpy(dtype=object).copy()
        bact = D.domain_of_genome[D.gc] == "Bacteria"
        n_sup = 0
        for gid, k in zip(ex.gid.to_numpy(), ex.ko.to_numpy()):
            if reading == "domain" and not bact[gid]:
                continue
            if cls[gid] == "X":
                cls[gid] = "N"
                lit[gid] = EXTRA_NAMES[k]
                n_sup += 1
            if not isinstance(ko[gid], str):
                ko[gid] = k
                ko_name[gid] = EXTRA_NAMES[k]
        arch = np.full(D.n, None, dtype=object)
        top = np.full(D.n, None, dtype=object)
        arch[wp.gid.to_numpy()] = wp.arch.to_numpy()
        top[wp.gid.to_numpy()] = wp.top.to_numpy()
        self.an = pd.DataFrame(dict(gid=np.arange(D.n), ko=ko, ko_name=ko_name, arch=arch, top=top, cls=cls, lit=lit))
        self.cls = cls.astype(str)
        self.member = np.isin(self.cls, ["R", "N"])
        self.isR = self.cls == "R"
        self.lit = lit
        self.ko = ko
        self.top = top
        kos = pd.Series(ko)
        archs = pd.Series(arch)
        has_ko, has_arch = kos.notna().to_numpy(), archs.notna().to_numpy()
        self.unannot = win & ~has_ko & ~has_arch
        ident = pd.Series(np.where(has_ko, "K:" + kos.fillna("").astype(str), np.where(has_arch, "P:" + archs.fillna("").astype(str), None)),
                          dtype=object)
        self.ident = pd.factorize(ident)[0].astype(np.int64)      # -1 where neither
        n = D.n
        same = D.cc[:-1] == D.cc[1:]
        self.dist = np.full(n - 1, 10 ** 9, np.int64)
        self.dist[same] = (self.start[1:] - self.end[:-1] - 1)[same]
        sstr = D.strand[:-1] == D.strand[1:]
        mm = self.member[:-1] & self.member[1:]
        self.link = same & sstr & ((self.dist <= Bm.D_OP) | (mm & (self.dist <= Bm.D_MEM)))
        cs, ce, nm, nr = Bm.build_components(self.link, self.member, self.isR, Bm.X_CUT)
        self.cs, self.ce, self.nm, self.nr = cs, ce, nm, nr
        self.comp_of = np.full(n, -1, np.int64)
        for i, (s, e) in enumerate(zip(cs, ce)):
            self.comp_of[s:e + 1] = i
        kc = pd.factorize(kos)[0]
        tc = pd.factorize(pd.Series(top))[0]
        adj = same & sstr & (((kc[:-1] == kc[1:]) & (kc[:-1] >= 0)) | ((tc[:-1] == tc[1:]) & (tc[:-1] >= 0)))
        self.anom = np.zeros(n, bool)
        self.anom[:-1] |= adj
        self.anom[1:] |= adj
        key = np.where(has_ko, kos.fillna("").astype(str).to_numpy(), np.where(self.isR, lit.astype(str), ""))
        df = pd.DataFrame(dict(key=key, length=D.length))
        med = df[df.key != ""].groupby("key").length.median()
        mkm = pd.Series(key).map(med).to_numpy(dtype=float)
        outl = self.member & np.isfinite(mkm) & ((D.length > 2 * mkm) | (D.length < 0.5 * mkm))
        flag = self.member & (D.partial | D.truncated)
        self.anom |= outl | flag
        self.anom_kind = np.full(n, "", dtype=object)
        self.anom_kind[np.flatnonzero(self.anom)] = "split"
        self.anom_kind[np.flatnonzero(outl)] = "len"
        self.anom_kind[np.flatnonzero(flag)] = "partial"
        log(ds, reading, "components", len(cs), "R", int(self.isR.sum()), "N", int((self.cls == "N").sum()),
            "supplement X->N", n_sup, "window unannotated", int(self.unannot.sum()), "of", int(win.sum()),
            "anomalies", int(self.anom.sum()), f"{time.time()-t0:.0f}s")

    def span_units(self, s, e):
        """Bm.Bio.run_units on an arbitrary span."""
        D = self.D
        mk = [k for k in range(s, e + 1) if D.marked[k]]
        comps = sorted(set(int(self.comp_of[k]) for k in mk if self.comp_of[k] >= 0))
        single = [k for k in mk if self.comp_of[k] < 0]
        n_units = len(comps) + len(single)
        if n_units == 1 and comps:
            targets = [(int(self.cs[comps[0]]), int(self.ce[comps[0]]))]
        else:
            targets = [(int(self.cs[c]), int(self.ce[c])) for c in comps if self.nm[c] >= Bm.MIN_TARGET_MEMBERS]
        return mk, comps, single, n_units, targets


class LociF(A.Loci):
    """lut frozen scorer reading (fscore.load_loci): loci with e > s, duplicates dropped."""
    def __init__(self, path):
        import pyarrow.parquet as pq
        sc, ec = ("gid_s", "gid_e") if "gid_s" in pq.read_schema(path).names else ("s", "e")
        df = pd.read_parquet(path, columns=[sc, ec])
        s, e = df[sc].to_numpy(np.int64), df[ec].to_numpy(np.int64)
        k = e > s
        se = pd.DataFrame(dict(s=s[k], e=e[k])).drop_duplicates().sort_values(["s", "e"]).reset_index(drop=True)
        self.a, self.b = se.s.to_numpy(np.int64), se.e.to_numpy(np.int64)
        self.maxlen = int((self.b - self.a + 1).max())
        self.n = len(self.a)


def loci(ds):
    rd = json.load(open(W4 / ds / "repro4.json")) if (W4 / ds / "repro4.json").exists() else {}
    reading = rd.get("loci_reading", "stored")
    return {n: (LociF(CFG[ds]["out"][n]) if reading == "filtered" else A.Loci(CFG[ds]["out"][n])) for n in OUTS}, reading


def ext_targets(B):
    """Part-3 cmd_extdef chain rule."""
    D = B.D
    mk = np.flatnonzero(B.member)
    new = np.r_[True, (np.diff(mk) > 3) | (D.cc[mk[1:]] != D.cc[mk[:-1]])]
    cs = mk[new]
    ce = mk[np.r_[new[1:], True]]
    chain_of = np.full(D.n, -1, np.int64)
    for i, (s, e) in enumerate(zip(cs, ce)):
        chain_of[s:e + 1] = i
    c = chain_of[D.runs.gid_start.to_numpy()]
    return cs[c].astype(np.int64), ce[c].astype(np.int64)


# ------------------------------------------------------------------------------------------- repro
def cmd_repro(ds):
    D = Data4(ds)
    runs = D.runs
    (W4 / ds).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(RD / "ribosomal/code"))
    import ribosomal_check as RC
    res = dict(runs=int(len(runs)), by_stratum=runs.gtype.value_counts().to_dict())
    for reading, cls_ in (("stored", A.Loci), ("filtered", LociF)):
        rr = {}
        for n in OUTS:
            L = cls_(CFG[ds]["out"][n])
            df = pd.DataFrame([A.standard(L, s, e) for s, e in zip(runs.gid_start, runs.gid_end)])
            sc = RC.score(dict(a=L.a, b=L.b, mod=np.full(L.n, -1, np.int64), maxlen=L.maxlen, n=L.n),
                          runs.gid_start.to_numpy(), runs.gid_end.to_numpy())
            df["any_fit"] = [Bm.any_fit(L, s, e) for s, e in zip(runs.gid_start, runs.gid_end)]
            df.insert(0, "run_id", runs.run_id.to_numpy())
            df["gtype"] = runs.gtype.to_numpy()
            df["domain"] = runs.domain.to_numpy()
            df.to_parquet(W4 / ds / f"std4_{reading}_{n}.parquet", index=False)
            rr[n] = dict(loci=int(L.n), recovered=int(df.recovered.sum()), resolved=int(df.resolved.sum()),
                         any_fit=int(df.any_fit.sum()), equal_rc=bool(np.array_equal(sc.resolved.to_numpy(), df.resolved.to_numpy())),
                         by_stratum=df.groupby("gtype")[["recovered", "resolved", "any_fit"]].sum().astype(int).to_dict("index"),
                         target=CFG[ds]["target"][n])
            t = CFG[ds]["target"][n]
            rr[n]["reproduced"] = bool((t[0] is None or t[0] == rr[n]["recovered"]) and t[1] == rr[n]["resolved"])
        res[reading] = rr
        log(ds, reading, {n: (v["recovered"], v["resolved"], v["any_fit"], v["reproduced"], v["equal_rc"]) for n, v in rr.items()})
    ok = {r: all(res[r][n]["reproduced"] for n in OUTS) for r in ("stored", "filtered")}
    res["loci_reading"] = "stored" if ok["stored"] else ("filtered" if ok["filtered"] else "stored")
    res["reproduced"] = ok
    json.dump(res, open(W4 / ds / "repro4.json", "w"), indent=1)
    for n in OUTS:
        pd.read_parquet(W4 / ds / f"std4_{res['loci_reading']}_{n}.parquet").to_parquet(W4 / ds / f"std4_{n}.parquet", index=False)
    log(ds, "loci reading", res["loci_reading"], ok)


# ------------------------------------------------------------------------------------------- cluster-level scoring
def chance_pass(D, Ls, spans):
    rng = np.random.default_rng(17)
    ch = {n: {} for n in OUTS}
    for gt, (ts, te) in zip(D.runs.gtype.to_numpy(), spans):
        ts, te = int(ts), int(te)
        cf, cl = int(D.cfirst[ts]), int(D.clast[ts])
        ln = te - ts + 1
        cand = [x for x in range(cf, cl - ln + 2) if x + ln - 1 < ts or x > te]
        if cand:
            for st in rng.choice(cand, size=5, replace=True):
                for n in OUTS:
                    d = ch[n].setdefault(gt, [0, 0])
                    d[0] += int(Bm.score_target(Ls[n], int(st), int(st) + ln - 1)["ok"])
                    d[1] += 1
    return ch


def cmd_ext(ds, reading):
    t0 = time.time()
    B = Bio4(ds, reading)
    D = B.D
    runs = D.runs
    Ls, lr = loci(ds)
    ts, te = ext_targets(B)
    df = pd.DataFrame(dict(run_id=runs.run_id.to_numpy(), gtype=runs.gtype.to_numpy(), domain=runs.domain.to_numpy(),
                           run_s=runs.gid_start.to_numpy(), run_e=runs.gid_end.to_numpy(), ext_s=ts, ext_e=te))
    df["run_len"] = df.run_e - df.run_s + 1
    df["ext_len"] = df.ext_e - df.ext_s + 1
    df["added_left"] = df.run_s - df.ext_s
    df["added_right"] = df.ext_e - df.run_e
    std = {n: pd.read_parquet(W4 / ds / f"std4_{n}.parquet") for n in OUTS}
    for n in OUTS:
        st = [A.standard(Ls[n], int(a), int(b)) for a, b in zip(ts, te)]
        df[f"{n}_recovered"] = [x["recovered"] for x in st]
        df[f"{n}_resolved"] = [x["resolved"] for x in st]
        df[f"{n}_anyfit"] = [Bm.any_fit(Ls[n], int(a), int(b)) for a, b in zip(ts, te)]
        df[f"{n}_std_resolved"] = std[n].resolved.to_numpy()
        df[f"{n}_std_anyfit"] = std[n].any_fit.to_numpy()
    ch = chance_pass(D, Ls, zip(ts, te))
    ch0 = chance_pass(D, Ls, zip(runs.gid_start.to_numpy(), runs.gid_end.to_numpy()))
    df.to_parquet(W4 / ds / f"ext4_{reading}.parquet", index=False)
    out = dict(dataset=ds, reading=reading, loci_reading=lr, strata={})
    groups = [(g, df[df.gtype == g]) for g in sorted(df.gtype.unique())] + [(d, df[df.domain == d]) for d in sorted(df.domain.unique())]
    for g, x in groups:
        gts = [g] if g in df.gtype.unique() else sorted(x.gtype.unique())
        row = dict(runs=int(len(x)), run_len_median=float(x.run_len.median()), ext_len_median=float(x.ext_len.median()),
                   added_left_median=float(x.added_left.median()), added_right_median=float(x.added_right.median()),
                   ext_equals_run=int((x.ext_len == x.run_len).sum()))
        for n in OUTS:
            w1 = sum(ch[n].get(t, [0, 0])[0] for t in gts)
            w2 = sum(ch[n].get(t, [0, 0])[1] for t in gts)
            v1 = sum(ch0[n].get(t, [0, 0])[0] for t in gts)
            v2 = sum(ch0[n].get(t, [0, 0])[1] for t in gts)
            row[n] = dict(std_resolved=int(x[f"{n}_std_resolved"].sum()), std_anyfit=int(x[f"{n}_std_anyfit"].sum()),
                          cl_recovered=int(x[f"{n}_recovered"].sum()), cl_resolved=int(x[f"{n}_resolved"].sum()),
                          cl_anyfit=int(x[f"{n}_anyfit"].sum()), cl_chance=w1 / max(w2, 1), cl_chance_windows=w2,
                          std_chance=v1 / max(v2, 1), std_chance_windows=v2,
                          gained=int((x[f"{n}_resolved"] & ~x[f"{n}_std_resolved"]).sum()),
                          lost=int((~x[f"{n}_resolved"] & x[f"{n}_std_resolved"]).sum()))
        out["strata"][g] = row
    json.dump(out, open(W4 / ds / f"ext4_{reading}.json", "w"), indent=1)
    log(ds, reading, json.dumps({g: {n: (v[n]["std_resolved"], v[n]["cl_resolved"], v[n]["cl_anyfit"], round(v[n]["cl_chance"], 4))
                                     for n in OUTS} for g, v in out["strata"].items()}), f"{time.time()-t0:.0f}s")


# ------------------------------------------------------------------------------------------- verdicts
def cmd_verdict(ds, definition, reading="union"):
    t0 = time.time()
    B = Bio4(ds, reading)
    D = B.D
    runs = D.runs
    Ls, _ = loci(ds)
    E = pd.read_parquet(W4 / ds / f"ext4_{reading}.parquet")
    if definition == "cluster":
        spans = list(zip(E.ext_s.to_numpy(), E.ext_e.to_numpy()))
        okcol = {n: E[f"{n}_resolved"].to_numpy() for n in OUTS}
    else:
        spans = list(zip(runs.gid_start.to_numpy(), runs.gid_end.to_numpy()))
        okcol = {n: E[f"{n}_std_resolved"].to_numpy() for n in OUTS}
    info = [B.span_units(int(s), int(e)) for s, e in spans]
    cases = []
    for i, (s, e) in enumerate(spans):
        s, e = int(s), int(e)
        mk, comps, single, n_units, targets = info[i]
        for n in OUTS:
            if okcol[n][i]:
                continue
            L = Ls[n]
            st = A.standard(L, s, e)
            c = dict(run_id=int(runs.run_id.iat[i]), ridx=i, output=n, gtype=runs.gtype.iat[i], genome_code=int(runs.genome_code.iat[i]),
                     span_s=s, span_e=e, span_len=e - s + 1, run_len=int(runs.gid_end.iat[i] - runs.gid_start.iat[i] + 1),
                     std_bf=float(st["best_frac"]), std_prec=float(st["precision"]) if np.isfinite(st["precision"]) else None,
                     n_units=n_units, n_targets=len(targets), targets=";".join(f"{a - s}..{b - s}" for a, b in targets))
            decisive, kind = [], None
            if not targets:
                verdict = 7
                c["why"] = "no qualifying unit"
            else:
                if n_units > 1:
                    uid = [int(B.comp_of[k]) if B.comp_of[k] >= 0 else -10 - k for k in mk]
                    for x, y, ux, uy in zip(mk[:-1], mk[1:], uid[:-1], uid[1:]):
                        if ux != uy:
                            p = next((q for q in range(x, y) if not B.link[q]), x)
                            decisive.append(("glue", p))
                    kind = "glue"
                else:
                    ts, te = targets[0]
                    sc = Bm.score_target(L, ts, te)
                    c.update(C_s=ts - s, C_e=te - s, C_len=te - ts + 1, bio_bf=sc["bf"], bio_prec=sc["prec"],
                             best_s=(sc["a"] - s) if sc["a"] >= 0 else None, best_e=(sc["b"] - s) if sc["b"] >= 0 else None)
                    if sc["ok"]:
                        kind = "ok"
                    elif Bm.any_fit(L, ts, te):
                        kind = "fit"
                    elif sc["bf"] < 0.8:
                        kind = "split"
                        if sc["a"] > ts:
                            decisive.append(("cut", sc["a"] - 1))
                        if 0 <= sc["b"] < te:
                            decisive.append(("cut", sc["b"]))
                        if sc["a"] < 0:
                            decisive.append(("cut", ts))
                    else:
                        kind = "ext"
                        if sc["a"] < ts:
                            decisive.append(("edge", ts - 1))
                        if sc["b"] > te:
                            decisive.append(("edge", te))
                dat = []
                for (kd, p) in decisive:
                    for x in (p - 1, p, p + 1, p + 2):
                        if D.cfirst[s] <= x <= D.clast[s] and B.anom[x]:
                            dat.append(f"{x - s}:{B.anom_kind[x]}")
                if dat and kind in ("split", "glue"):
                    verdict = 6
                    c["why"] = "gene-call anomaly at the decisive point: " + ",".join(sorted(set(dat)))
                else:
                    verdict = {"glue": 2, "ok": 3, "fit": 5, "split": 1, "ext": 4}[kind]
            c["verdict"] = verdict
            c["decisive"] = decisive
            cases.append(c)
    IH = Bm.Inherit(B, [(p, p + 1) for c in cases for (_, p) in c["decisive"]])
    for c in cases:
        info_j = []
        for (kd, p) in c["decisive"]:
            h = IH.junction(p, p + 1, c["genome_code"])
            info_j.append(dict(kind=kd, rel=p - c["span_s"], left=Bm.gene_label(B, p), right=Bm.gene_label(B, p + 1),
                               left_cls=B.cls[p], right_cls=B.cls[p + 1], same_strand=bool(D.strand[p] == D.strand[p + 1]),
                               dist=int(B.dist[p]) if B.dist[p] < 10 ** 8 else None, **h))
        c["junction_txt"] = " ; ".join(
            f"{j['kind']}@{j['rel']} {j['left']}|{j['right']} {'same' if j['same_strand'] else 'OPP'} d={j['dist']} "
            f"grp {j['own_adj']}/{j['own_both']} othergrp {j['other_groups_adj']}" for j in info_j)
        if c["verdict"] == 2:
            c["subtype"] = "+".join(sorted({"strand" if not j["same_strand"] else
                                            ("mem>500" if j["left_cls"] in "RN" and j["right_cls"] in "RN" else "spacer")
                                            for j in info_j}))
        if c.get("C_s") is not None and c.get("best_s") is not None and c["verdict"] in (4, 5):
            s = c["span_s"]
            cs_, ce_ = s + c["C_s"], s + c["C_e"]
            ext = [k for k in range(s + c["best_s"], s + c["best_e"] + 1) if not (cs_ <= k <= ce_)]
            cl = pd.Series([B.cls[k] for k in ext]).value_counts().to_dict()
            c["beyond_RNX"] = f"{cl.get('R', 0)}/{cl.get('N', 0)}/{cl.get('X', 0)}"
            if c["verdict"] == 4:
                c["subtype"] = "4b_other_unit_members" if (cl.get("R", 0) + cl.get("N", 0)) >= 0.5 * max(len(ext), 1) else "4a_unrelated"
            c["best_beyond_C"] = ",".join(Bm.gene_label(B, k) for k in ext[:15]) + (" ..." if len(ext) > 15 else "")
        if c.get("C_s") is not None and c["verdict"] == 3:
            s = c["span_s"]
            c["C_beyond_span"] = ",".join(f"{Bm.gene_label(B, k)}[{B.cls[k]}]" for k in range(s + c["C_s"], s + c["C_e"] + 1)
                                          if not (s <= k <= c["span_e"]))
        c["case_id"] = f"{ds}-{definition}-{c['output']}-{c['run_id']}"
        del c["decisive"]
    # diagrams (cluster definition only; window +-4 around span, targets and best locus)
    diag = []
    if definition == "cluster":
        for c in cases:
            L = Ls[c["output"]]
            s, e = c["span_s"], c["span_e"]
            tg = info[c["ridx"]][4]
            lo = min([s] + [t[0] for t in tg]) - 4
            hi = max([e] + [t[1] for t in tg]) + 4
            j, _ = L.best(s, e)
            if j >= 0:
                lo, hi = min(lo, int(L.a[j]) - 1), max(hi, int(L.b[j]) + 1)
            lo, hi = max(lo, s - 40, int(D.cfirst[s])), min(hi, e + 40, int(D.clast[s]))
            diag.append(f"{c['case_id']}  {c['gtype']}  extended run {e - s + 1} genes (run {c['run_len']})  std bf {c['std_bf']:.2f} "
                        f"prec {c['std_prec'] if c['std_prec'] is None else round(c['std_prec'], 2)} -> VERDICT {c['verdict']}  targets "
                        f"{c['targets']} (window starts at {lo - s})")
            for key in ("junction_txt", "why", "best_beyond_C", "C_beyond_span"):
                if c.get(key):
                    diag.append(f"    {key}: {c[key]}")
            diag.append("    genes: " + " ".join(Bm.gene_label(B, k) + ("*" if s <= k <= e else "") for k in range(lo, hi + 1)))
            diag += ["    " + x for x in Bm.diagram(B, lo, hi, s, e, tg, L, j)]
    cdf = pd.DataFrame(cases)
    cdf.to_csv(W4 / ds / f"cases4_{definition}_{reading}.tsv", sep="\t", index=False)
    if diag:
        (W4 / ds / f"diagrams4_{definition}_{reading}.txt").write_text(
            "Legend as in part 2/3; upper case = extended-run span; '*' in the gene line = inside the span.\n\n" + "\n".join(diag) + "\n")
    summ = cdf.groupby(["output", "gtype", "verdict"]).size().unstack(fill_value=0) if len(cdf) else pd.DataFrame()
    json.dump({f"{a}|{b}": {int(k): int(v) for k, v in r.items()} for (a, b), r in summ.iterrows()},
              open(W4 / ds / f"verdicts4_{definition}_{reading}.json", "w"), indent=1)
    log(ds, definition, reading, summ.to_dict("index"), f"{time.time()-t0:.0f}s")


# ------------------------------------------------------------------------------------------- insert
ALPHA = {"S13", "S11", "S4", "L17", "rpoD"}


def cmd_insert(ds):
    t0 = time.time()
    B = Bio4(ds, "union")
    D = B.D
    runs = D.runs
    lit = np.array([str(x) for x in B.lit], dtype=object)
    isL15 = np.isin(lit, ["L15", "L27A"]) & B.isR
    isA = (np.isin(lit, list(ALPHA)) & B.member) | (pd.Series(B.ko).astype(str).to_numpy() == "K03040")
    spc = np.isin(lit, ["L30", "S5"]) & B.isR
    rows = []
    for r in runs.itertuples():
        if r.domain != "Bacteria":
            continue
        s, e = int(r.gid_start), int(r.gid_end)
        cf, cl = int(D.cfirst[s]), int(D.clast[s])
        l15 = [k for k in range(s, e + 1) if isL15[k]]
        row = dict(run_id=r.run_id, gtype=r.gtype, genome_code=int(r.genome_code), n_L15=len(l15),
                   phylum=D.genomes.phylum.iat[int(r.genome_code)], tclass=D.genomes["class"].iat[int(r.genome_code)])
        if not l15:
            row["cls"] = "no L15 in run"
            rows.append(row)
            continue
        best = None
        for k in l15:
            left_spc = any(spc[x] for x in range(max(cf, k - 3), k))
            right_spc = any(spc[x] for x in range(k + 1, min(cl, k + 3) + 1))
            sides = [+1] if (left_spc and not right_spc) else ([-1] if (right_spc and not left_spc) else [+1, -1])
            for sd in sides:
                for d in range(1, 16):
                    x = k + sd * d
                    if x < cf or x > cl:
                        break
                    if isA[x]:
                        if best is None or d < best[2]:
                            best = (k, sd, d, x)
                        break
        if best is None:
            k = l15[0]
            near_end = min(k - cf, cl - k) < 15
            row["cls"] = "alpha not within 15 (contig end within 15)" if near_end else "alpha not within 15 (contig continues)"
            rows.append(row)
            continue
        k, sd, d, x = best
        between = list(range(k + sd, x, sd))
        unm = [q for q in between if not D.marked[q]]
        row.update(cls="alpha found", insert_all=len(between), insert_unmarked=len(unm), alpha_gene=Bm.gene_label(B, x),
                   insert_genes="-".join(Bm.gene_label(B, q) for q in between), alpha_in_run=bool(s <= x <= e))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["ins_class"] = np.where(df.cls != "alpha found", df.cls,
                               df.insert_unmarked.fillna(-1).astype(int).clip(upper=5).astype(str).replace({"5": ">=5"}))
    df.to_parquet(W4 / ds / "insert4.parquet", index=False)
    log(ds, "insert classes", df.ins_class.value_counts().to_dict(), f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    cmd, ds = sys.argv[1], sys.argv[2]
    (W4 / ds).mkdir(parents=True, exist_ok=True)
    if cmd == "repro":
        cmd_repro(ds)
    elif cmd == "ext":
        cmd_ext(ds, sys.argv[3] if len(sys.argv) > 3 else "union")
    elif cmd == "verdict":
        cmd_verdict(ds, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "union")
    elif cmd == "insert":
        cmd_insert(ds)
