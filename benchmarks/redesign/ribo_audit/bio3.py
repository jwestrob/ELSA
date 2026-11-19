#!/usr/bin/env python
"""ribo_audit part 3: the part-2 accurate-locus rules on the SR-VP genome-bin set (predeclared3.json).

usage (biotite login node, one thread; RIBO_VARIANT=strict|loose):
  bio3.py repro      standing recovered / resolved of PRIMARY (Z), SECONDARY (WA), ENGINE (DH) by domain and genome type
  bio3.py verdict    accurate targets for all 803 runs, accurate scoring, verdicts for every unresolved run, diagrams
  bio3.py rearr      gene-order variability inside runs across lineage groups, core-marker orders, rare junctions,
                     conserved stretch around runs, over-extension contents
  bio3.py chance     random windows of target length (5 per run, seed 17)
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bio as Bm  # noqa: E402  (part-2 rules, read-only reuse)
import audit as A  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RD = A.RD
PKG = RD / "ws_srvp_bins/package"
FZ = RD / "long_loci/frozen2"
W3 = A.AD / "work3"
VARIANT = Bm.VARIANT
WV = W3 if VARIANT == "strict" else W3 / VARIANT
OUT3 = {"PRIMARY": FZ / "loci/srvp_Z/L4bp_t0.02.parquet", "SECONDARY": FZ / "loci/srvp_WA/L4b_t0.2.parquet",
        "ENGINE": FZ / "loci/srvp_engine/DH.parquet"}
ORD3 = ("PRIMARY", "SECONDARY", "ENGINE")
EXTRA_NAMES = {"K02601": "nusG", "K03043": "rpoB", "K03046": "rpoC", "K02358": "tufA", "K02355": "fusA", "K00939": "adk",
               "K01265": "map", "K02518": "infA", "K03073": "secE", "K01883": "cysS", "K03685": "rnc", "K03218": "rlmB",
               "K16786": "ecfA1", "K16787": "ecfA2", "K16785": "ecfT", "K06173": "truA"}
CORE = ("L3", "L4", "L23", "L2", "S19", "L22", "S3", "L29", "S17", "L14", "L24", "L5", "S14", "S8", "L6", "L18", "S5", "L15")
log = A.log


# ------------------------------------------------------------------------------------------- data
class D3:
    """SR-VP universe with the attributes the part-2 code reads."""
    def __init__(self):
        g = pd.read_parquet(PKG / "universe/genes.parquet",
                            columns=["gid", "genome_code", "contig_code", "position_index", "strand", "length", "start", "end"]
                            ).sort_values("gid").reset_index(drop=True)
        assert np.array_equal(g.gid.to_numpy(), np.arange(len(g)))
        self.n = len(g)
        self.cc = g.contig_code.to_numpy(np.int64)
        self.gc = g.genome_code.to_numpy(np.int64)
        self.pos = g.position_index.to_numpy(np.int64)
        self.strand = g.strand.to_numpy(np.int8)
        self.length = g.length.to_numpy(np.int64)
        self.start = g.start.to_numpy(np.int64)
        self.end = g.end.to_numpy(np.int64)
        c = pd.read_parquet(PKG / "universe/contigs.parquet").set_index("contig_code")
        self.cfirst = c.first_gid.reindex(self.cc).to_numpy(np.int64)
        self.clast = self.cfirst + c.n_genes.reindex(self.cc).to_numpy(np.int64) - 1
        gm = pd.read_parquet(PKG / "universe/genomes.parquet").sort_values("genome_code").reset_index(drop=True)
        assert np.array_equal(gm.genome_code.to_numpy(), np.arange(len(gm)))
        self.genomes = gm
        m = pd.read_parquet(FZ / "groups/emb_hom_average_t0.75.parquet").sort_values("genome_code")
        assert np.array_equal(m.genome_code.to_numpy(), np.arange(len(gm)))
        self.grp = pd.factorize(m.group.to_numpy())[0].astype(np.int64)
        self.G = int(self.grp.max()) + 1
        mk = pd.read_parquet(PKG / "ribosomal/marked.parquet")
        self.marked = np.zeros(self.n, bool)
        self.marked[mk.gid.to_numpy()] = True
        self.marker = pd.Series(mk.marker.to_numpy(), index=mk.gid.to_numpy())
        r = pd.read_parquet(PKG / "ribosomal/runs.parquet")
        r = r[(r.run_set == "g2") & r.m16].reset_index(drop=True)
        r["group"] = self.grp[r.genome_code.to_numpy()]
        r["closed"] = (gm.n_contigs.to_numpy()[r.genome_code.to_numpy()] == 1)
        r["gtype"] = np.where(r.domain == "Archaea", "Archaea", np.where(r.closed, "Bact_closed", "Bact_multi"))
        r["stratum"] = r.gtype
        self.runs = r
        fl = pd.read_parquet(PKG / "universe/gene_flags.parquet", columns=["gid", "partial", "truncated"]).sort_values("gid")
        self.partial = (fl.partial.astype(str).to_numpy() != "00") & fl.partial.notna().to_numpy()
        self.truncated = fl.truncated.fillna(False).astype(bool).to_numpy()
        self.domain_of_genome = gm.domain.to_numpy()


class Bio3(Bm.Bio):
    def __init__(self):
        self.D = D3()
        D = self.D
        self.start, self.end = D.start, D.end
        base = pd.read_parquet(FZ / "annot/genes_annot.parquet").sort_values("gid").reset_index(drop=True)
        ex = pd.read_parquet(W3 / "annot/extra_ko.parquet")
        wp = pd.read_parquet(W3 / "annot/window_pfam.parquet")
        win = np.zeros(D.n, bool)
        win[np.load(W3 / "annot/window_gids.npy")] = True
        self.window = win
        ko = base.ko.to_numpy(dtype=object).copy()
        ko_name = base.ko_name.to_numpy(dtype=object).copy()
        cls = base.cls.to_numpy(dtype=object).copy()
        lit = base.lit.fillna("").to_numpy(dtype=object).copy()
        exs = pd.Series(ex.ko.to_numpy(), index=ex.gid.to_numpy())
        for gid, k in exs.items():
            if cls[gid] == "X":
                cls[gid] = "N"
                lit[gid] = EXTRA_NAMES[k]
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
        self.unannot = win & np.array([not isinstance(x, str) for x in ko]) & np.array([not isinstance(x, str) for x in arch])
        ident = np.where([isinstance(x, str) for x in ko], ["K:" + str(x) for x in ko],
                         np.where([isinstance(x, str) for x in arch], ["P:" + str(x) for x in arch], ""))
        codes, _ = pd.factorize(pd.Series(ident).replace("", np.nan))
        self.ident = codes.astype(np.int64)
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
        kov = np.array([x if isinstance(x, str) else "" for x in ko])
        topv = np.array([x if isinstance(x, str) else "" for x in top])
        adj = same & sstr & (((kov[:-1] == kov[1:]) & (kov[:-1] != "")) | ((topv[:-1] == topv[1:]) & (topv[:-1] != "")))
        self.anom = np.zeros(n, bool)
        self.anom[:-1] |= adj
        self.anom[1:] |= adj
        key = np.where(kov != "", kov, np.where(self.isR, lit.astype(str), ""))
        df = pd.DataFrame(dict(key=key, length=D.length))
        med = df[df.key != ""].groupby("key").length.median()
        mk = pd.Series(key).map(med).to_numpy(dtype=float)
        outl = self.member & np.isfinite(mk) & ((D.length > 2 * mk) | (D.length < 0.5 * mk))
        flag = self.member & (D.partial | D.truncated)
        self.anom |= outl | flag
        self.anom_kind = np.full(n, "", dtype=object)
        self.anom_kind[np.flatnonzero(self.anom)] = "split"
        self.anom_kind[np.flatnonzero(outl)] = "len"
        self.anom_kind[np.flatnonzero(flag)] = "partial"
        log("components", len(cs), "N genes", int((self.cls == "N").sum()), "R", int(self.isR.sum()),
            "window unannotated", int(self.unannot.sum()), "of", int(win.sum()), "anomalies", int(self.anom.sum()))


# ------------------------------------------------------------------------------------------- repro
def cmd_repro(a):
    D = D3()
    runs = D.runs
    sys.path.insert(0, str(RD / "ribosomal/code"))
    import ribosomal_check as RC
    res = {}
    for n in ORD3:
        L = A.Loci(OUT3[n])
        df = pd.DataFrame([A.standard(L, s, e) for s, e in zip(runs.gid_start, runs.gid_end)])
        sc = RC.score(dict(a=L.a, b=L.b, mod=np.full(L.n, -1, np.int64), maxlen=L.maxlen, n=L.n),
                      runs.gid_start.to_numpy(), runs.gid_end.to_numpy())
        anyok = [Bm.any_fit(L, s, e) for s, e in zip(runs.gid_start, runs.gid_end)]
        df["any_fit"] = anyok
        df.insert(0, "run_id", runs.run_id.to_numpy())
        df["gtype"] = runs.gtype.to_numpy()
        df["domain"] = runs.domain.to_numpy()
        df.to_parquet(W3 / f"std3_{n}.parquet", index=False)
        out = dict(equal_rc=bool(np.array_equal(sc.resolved.to_numpy(), df.resolved.to_numpy())))
        for key, d in [("all", df)] + list(df.groupby("domain")) + list(df.groupby("gtype")):
            out[key] = dict(runs=int(len(d)), recovered=int(d.recovered.sum()), resolved=int(d.resolved.sum()),
                            any_fit=int(d.any_fit.sum()))
        res[n] = out
        log(n, out)
    json.dump(res, open(W3 / "repro3.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------- verdict (part-2 procedure)
def cmd_verdict(a):
    t0 = time.time()
    B = Bio3()
    D = B.D
    runs = D.runs
    Ls = {n: A.Loci(OUT3[n]) for n in ORD3}
    std = {n: pd.read_parquet(W3 / f"std3_{n}.parquet") for n in ORD3}
    runinfo, rows = [], []
    for i in range(len(runs)):
        s, e, mk, comps, single, n_units, targets = B.run_units(i)
        runinfo.append(dict(ridx=i, run_id=int(runs.run_id.iat[i]), s=s, e=e, n_units=n_units, targets=targets,
                            n_targets=len(targets), gtype=runs.gtype.iat[i], genome_code=int(runs.genome_code.iat[i])))
        for n in ORD3:
            sc = [Bm.score_target(Ls[n], ts, te) for (ts, te) in targets]
            rows.append(dict(run_id=int(runs.run_id.iat[i]), output=n, gtype=runs.gtype.iat[i],
                             std_ok=bool(std[n].resolved.iat[i]), bio_ok=bool(sc and all(x["ok"] for x in sc)),
                             n_units=n_units, n_targets=len(targets)))
    bio = pd.DataFrame(rows)
    bio.to_parquet(WV / "bio3_scores.parquet", index=False)
    log("accurate scoring", bio.groupby(["output", "gtype"])[["std_ok", "bio_ok"]].sum().to_dict("index"))
    cases = []
    for ri in runinfo:
        s, e = ri["s"], ri["e"]
        for n in ("PRIMARY", "SECONDARY"):
            st = std[n].iloc[ri["ridx"]]
            if bool(st.resolved):
                continue
            L = Ls[n]
            c = dict(run_id=ri["run_id"], ridx=ri["ridx"], output=n, gtype=ri["gtype"], genome_code=ri["genome_code"],
                     run_s=s, run_e=e, run_len=e - s + 1, std_bf=float(st.best_frac), std_prec=float(st.precision),
                     n_units=ri["n_units"], n_targets=ri["n_targets"],
                     targets=";".join(f"{ts - s}..{te - s}" for ts, te in ri["targets"]))
            decisive, verdict, kind = [], None, None
            if ri["n_targets"] == 0:
                verdict = 7
                c["why"] = "no qualifying unit"
            else:
                if ri["n_units"] > 1:
                    mk = [k for k in range(s, e + 1) if D.marked[k]]
                    uid = [int(B.comp_of[k]) if B.comp_of[k] >= 0 else -10 - k for k in mk]
                    for x, y, ux, uy in zip(mk[:-1], mk[1:], uid[:-1], uid[1:]):
                        if ux != uy:
                            p = next((q for q in range(x, y) if not B.link[q]), x)
                            decisive.append(("glue", p))
                    kind = "glue"
                else:
                    ts, te = ri["targets"][0]
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
        info = []
        for (kd, p) in c["decisive"]:
            h = IH.junction(p, p + 1, c["genome_code"])
            info.append(dict(kind=kd, rel=p - c["run_s"], left=Bm.gene_label(B, p), right=Bm.gene_label(B, p + 1),
                             left_cls=B.cls[p], right_cls=B.cls[p + 1], same_strand=bool(D.strand[p] == D.strand[p + 1]),
                             dist=int(B.dist[p]) if B.dist[p] < 10 ** 8 else None, **h))
        c["junctions"] = info
        c["junction_txt"] = " ; ".join(
            f"{j['kind']}@{j['rel']} {j['left']}|{j['right']} {'same' if j['same_strand'] else 'OPP'} d={j['dist']} "
            f"grp {j['own_adj']}/{j['own_both']} othergrp {j['other_groups_adj']}" for j in info)
        if c["verdict"] == 2:
            c["subtype"] = "+".join(sorted({"strand" if not j["same_strand"] else
                                            ("mem>500" if j["left_cls"] in "RN" and j["right_cls"] in "RN" else "spacer")
                                            for j in info}))
        del c["decisive"]
    # C beyond run / best beyond C
    extrows = []
    for c in cases:
        if c.get("C_s") is None:
            continue
        s = c["run_s"]
        ts, te = s + c["C_s"], s + c["C_e"]
        c["C_beyond_run"] = ",".join(f"{Bm.gene_label(B, k)}[{B.cls[k]}]" for k in range(ts, te + 1) if not (s <= k <= c["run_e"]))
        for k in range(ts, te + 1):
            if not (s <= k <= c["run_e"]):
                extrows.append(dict(case_key=f"{c['output']}-{c['run_id']}", output=c["output"], run_id=c["run_id"],
                                    gtype=c["gtype"], verdict=c["verdict"], where="C_beyond_run", gid=k, rel=k - s,
                                    side="left" if k < s else "right", cls=B.cls[k], lit=B.lit[k], ko=B.ko[k],
                                    ko_name=B.an.ko_name.iat[k], top=B.top[k], unannot=bool(B.unannot[k])))
        if c.get("best_s") is not None and c["verdict"] in (4, 5):
            bs, be = s + c["best_s"], s + c["best_e"]
            ext = [k for k in range(bs, be + 1) if not (ts <= k <= te)]
            for k in ext:
                extrows.append(dict(case_key=f"{c['output']}-{c['run_id']}", output=c["output"], run_id=c["run_id"],
                                    gtype=c["gtype"], verdict=c["verdict"], where="best_beyond_C", gid=k, rel=k - s,
                                    side="left" if k < ts else "right", cls=B.cls[k], lit=B.lit[k], ko=B.ko[k],
                                    ko_name=B.an.ko_name.iat[k], top=B.top[k], unannot=bool(B.unannot[k]),
                                    in_window=bool(B.window[k])))
            cl = pd.Series([B.cls[k] for k in ext]).value_counts().to_dict()
            c["n_beyond_C"] = len(ext)
            c["beyond_RNX"] = f"{cl.get('R', 0)}/{cl.get('N', 0)}/{cl.get('X', 0)}"
            if c["verdict"] == 4:
                c["subtype"] = "4b_other_unit_members" if (cl.get("R", 0) + cl.get("N", 0)) >= 0.5 * max(len(ext), 1) else "4a_unrelated"
            c["best_beyond_C"] = ",".join(Bm.gene_label(B, k) for k in ext[:15]) + (" ..." if len(ext) > 15 else "")
    # diagrams
    diag, lay = [], []
    for c in cases:
        L = Ls[c["output"]]
        s, e = c["run_s"], c["run_e"]
        tg = runinfo[c["ridx"]]["targets"]
        lo = min([s] + [t[0] for t in tg]) - 4
        hi = max([e] + [t[1] for t in tg]) + 4
        j, ov = L.best(s, e)
        if j >= 0:
            lo, hi = min(lo, int(L.a[j]) - 1), max(hi, int(L.b[j]) + 1)
        lo, hi = max(lo, s - 40, int(D.cfirst[s])), min(hi, e + 40, int(D.clast[s]))
        cid = f"{c['output']}-{c['run_id']}"
        c["case_id"] = cid
        diag.append(f"{cid}  {c['gtype']}  run {e - s + 1} genes  std bf {c['std_bf']:.2f} prec {c['std_prec']:.2f}  "
                    f"-> VERDICT {c['verdict']}  targets {c['targets']} (window starts at {lo - s})")
        for key in ("junction_txt", "why", "best_beyond_C"):
            if c.get(key):
                diag.append(f"    {key}: {c[key]}")
        if c.get("C_beyond_run") and c["verdict"] == 3:
            diag.append("    C beyond run: " + c["C_beyond_run"])
        diag += ["    " + x for x in Bm.diagram(B, lo, hi, s, e, tg, L, j)]
        for k in range(lo, hi + 1):
            lay.append(dict(case_id=cid, rel=k - s, gid=k, strand=int(D.strand[k]), start=int(B.start[k]), end=int(B.end[k]),
                            dist_next=(int(B.dist[k]) if k < D.n - 1 and B.dist[k] < 10 ** 8 else None),
                            linked_next=bool(B.link[k]) if k < D.n - 1 else False, cls=B.cls[k], lit=B.lit[k], ko=B.ko[k],
                            pfam=B.an.arch.iat[k], marked=bool(D.marked[k]), in_run=bool(s <= k <= e),
                            in_target=bool(any(t0 <= k <= t1 for t0, t1 in tg)), anomaly=B.anom_kind[k],
                            loci=";".join(f"{int(L.a[x]) - s}..{int(L.b[x]) - s}" for x in L.overlapping(k, k))))
    pd.DataFrame(extrows).to_parquet(WV / "ext3.parquet", index=False)
    pd.DataFrame([{k: v for k, v in c.items() if k != "junctions"} for c in cases]).to_csv(WV / "cases3.tsv", sep="\t", index=False)
    json.dump([dict(case_id=c["case_id"], junctions=c["junctions"]) for c in cases], open(WV / "junctions3.json", "w"), default=str)
    pd.DataFrame(lay).to_csv(WV / "layouts3.tsv", sep="\t", index=False)
    with open(WV / "diagrams3.txt", "w") as fh:
        fh.write("Legend as in part 2 (R ribosomal protein, N literature member incl. bacterial cluster genes, X other, U no\n"
                 "KO and no Pfam inside the annotated windows; UPPER CASE = run span; ' ' linked, '/' not linked, '|' strand\n"
                 "change; line 3 T = accurate target; line 4 B = best locus of the output, o = other loci).\n\n")
        fh.write("\n".join(diag) + "\n")
    pd.DataFrame([{k: (v if k != "targets" else ";".join(f"{x}..{y}" for x, y in v)) for k, v in r.items()} for r in runinfo]
                 ).to_parquet(WV / "runinfo3.parquet", index=False)
    cdf = pd.DataFrame(cases)
    log("verdicts", cdf.groupby(["output", "gtype", "verdict"]).size().unstack(fill_value=0).to_dict("index"), f"{time.time()-t0:.0f}s")


# ------------------------------------------------------------------------------------------- rearrangement and stretch
def cmd_rearr(a):
    t0 = time.time()
    B = Bio3()
    D = B.D
    runs = D.runs
    dom_g = D.domain_of_genome
    name = np.array([str(x) if B.isR[i] else "" for i, x in enumerate(B.lit)], dtype=object)
    # group sets per marker name and per adjacent marker pair (<= 3 positions, one contig), genome-wide
    Rg = np.flatnonzero(B.isR)
    hold = {}
    for gid in Rg:
        hold.setdefault(name[gid], set()).add(int(D.grp[D.gc[gid]]))
    adj = {}
    for d in (1, 2, 3):
        i = Rg[Rg + d < D.n]
        j = i + d
        ok = B.isR[j] & (D.cc[i] == D.cc[j])
        for x, y in zip(i[ok], j[ok]):
            k = tuple(sorted((name[x], name[y])))
            adj.setdefault(k, set()).add(int(D.grp[D.gc[x]]))
    grp_dom = {}
    for gcode in range(len(dom_g)):
        grp_dom[int(D.grp[gcode])] = dom_g[gcode]

    def share(a_, b_, q, dom):
        k = tuple(sorted((a_, b_)))
        both = {h for h in (hold.get(a_, set()) & hold.get(b_, set())) if h != q and grp_dom[h] == dom}
        if not both:
            return np.nan, 0
        ad = {h for h in adj.get(k, set()) if h in both}
        return len(ad) / len(both), len(both)

    prow, rrow = [], []
    for i, r in enumerate(runs.itertuples()):
        s, e = int(r.gid_start), int(r.gid_end)
        q = int(r.group)
        dom = r.domain
        mk = [k for k in range(s, e + 1) if D.marked[k] and B.isR[k]]
        shs = []
        for x, y in zip(mk[:-1], mk[1:]):
            sh, nb = share(name[x], name[y], q, dom)
            prow.append(dict(run_id=r.run_id, gtype=r.gtype, x=x, y=y, a=name[x], b=name[y], share=sh, n_groups=nb,
                             rel_x=x - s, rel_y=y - s))
            if np.isfinite(sh):
                shs.append(sh)
        core = [name[k] for k in mk if name[k] in CORE]
        dedup = [c for j_, c in enumerate(core) if j_ == 0 or c != core[j_ - 1]]
        canon = min(tuple(dedup), tuple(reversed(dedup))) if dedup else ()
        rrow.append(dict(run_id=r.run_id, gtype=r.gtype, group=q, n_pairs=len(mk) - 1,
                         median_share=float(np.median(shs)) if shs else np.nan, min_share=float(np.min(shs)) if shs else np.nan,
                         n_rare=int(sum(1 for x in shs if x < 0.10)), core_order="-".join(canon), n_core=len(set(core))))
    P = pd.DataFrame(prow)
    R = pd.DataFrame(rrow)
    P.to_parquet(W3 / "pair_shares.parquet", index=False)
    R.to_parquet(W3 / "run_order.parquet", index=False)
    log("rearrangement tables", len(P), len(R), f"{time.time()-t0:.0f}s")
    # conserved stretch (identity pairs <= 2 positions, >= 25% of other same-domain groups holding both)
    idt = B.ident
    win = B.window
    ids_hold = {}
    wg = np.flatnonzero(win & (idt >= 0))
    for gid in wg:
        ids_hold.setdefault(int(idt[gid]), set()).add(int(D.grp[D.gc[gid]]))
    padj = {}
    for d in (1, 2):
        i = wg[wg + d < D.n]
        j = i + d
        ok = win[j] & (idt[j] >= 0) & (D.cc[i] == D.cc[j])
        for x, y in zip(i[ok], j[ok]):
            k = (min(idt[x], idt[y]), max(idt[x], idt[y]))
            padj.setdefault(k, set()).add(int(D.grp[D.gc[x]]))

    def conserved(x, y, q, dom):
        if idt[x] < 0 or idt[y] < 0:
            return False
        both = {h for h in (ids_hold.get(int(idt[x]), set()) & ids_hold.get(int(idt[y]), set())) if h != q and grp_dom[h] == dom}
        if len(both) < 3:
            return False
        k = (min(idt[x], idt[y]), max(idt[x], idt[y]))
        return len({h for h in padj.get(k, set()) if h in both}) >= 0.25 * len(both)

    srow = []
    for r in runs.itertuples():
        s, e = int(r.gid_start), int(r.gid_end)
        q, dom = int(r.group), r.domain
        cf, cl = int(D.cfirst[s]), int(D.clast[s])
        left = 0
        k = s
        while k - 1 >= cf and win[k - 1] and conserved(k - 1, k, q, dom):
            left += 1
            k -= 1
        right = 0
        k = e
        while k + 1 <= cl and win[k + 1] and conserved(k, k + 1, q, dom):
            right += 1
            k += 1
        srow.append(dict(run_id=r.run_id, gtype=r.gtype, run_len=e - s + 1, stretch_left=left, stretch_right=right,
                         at_left_end=bool(s - left == cf), at_right_end=bool(e + right == cl)))
    S = pd.DataFrame(srow)
    S.to_parquet(W3 / "stretch.parquet", index=False)
    log("stretch", S.groupby("gtype")[["stretch_left", "stretch_right", "run_len"]].median().to_dict("index"), f"{time.time()-t0:.0f}s")


# ------------------------------------------------------------------------------------------- chance
def cmd_chance(a):
    D = D3()
    ri = pd.read_parquet(WV / "runinfo3.parquet")
    rng = np.random.default_rng(17)
    Ls = {n: A.Loci(OUT3[n]) for n in ORD3}
    res = {n: {} for n in ORD3}
    for r in ri.itertuples():
        if not r.targets:
            continue
        cf, cl = int(D.cfirst[r.s]), int(D.clast[r.s])
        for t in r.targets.split(";"):
            a_, b_ = map(int, t.split(".."))
            ln = b_ - a_ + 1
            cand = [x for x in range(cf, cl - ln + 2) if x + ln - 1 < r.s or x > r.e]
            if not cand:
                continue
            for st in rng.choice(cand, size=5, replace=True):
                for n in ORD3:
                    ok = Bm.score_target(Ls[n], int(st), int(st) + ln - 1)["ok"]
                    d = res[n].setdefault(r.gtype, [0, 0])
                    d[0] += int(ok)
                    d[1] += 1
    out = {n: {g: dict(resolved=v[0], windows=v[1], chance=v[0] / max(v[1], 1)) for g, v in dd.items()} for n, dd in res.items()}
    json.dump(out, open(WV / "chance3.json", "w"), indent=1)
    log(out)


# ------------------------------------------------------------------------------------------- cluster-level definition
def cmd_extdef(a):
    """POST HOC candidate definition: runs rebuilt from marker = R or N (literature cluster members) with the same
    <= 2-gene gap rule; the extended run containing each original run is the target; standard best-locus 80 / 80."""
    B = Bio3()
    D = B.D
    runs = D.runs
    mk = np.flatnonzero(B.member)
    new = np.r_[True, (np.diff(mk) > 3) | (D.cc[mk[1:]] != D.cc[mk[:-1]])]
    cid = np.cumsum(new) - 1
    cs = mk[new]
    ce = mk[np.r_[new[1:], True]]
    chain_of = np.full(D.n, -1, np.int64)
    for i, (s, e) in enumerate(zip(cs, ce)):
        chain_of[s:e + 1] = i
    Ls = {n: A.Loci(OUT3[n]) for n in ORD3}
    rng = np.random.default_rng(17)
    rows, chance = [], {n: {} for n in ORD3}
    for r in runs.itertuples():
        c = chain_of[r.gid_start]
        ts, te = int(cs[c]), int(ce[c])
        row = dict(run_id=r.run_id, gtype=r.gtype, run_len=r.gid_end - r.gid_start + 1, ext_len=te - ts + 1,
                   added_left=int(r.gid_start - ts), added_right=int(te - r.gid_end))
        for n in ORD3:
            row[n] = bool(Bm.score_target(Ls[n], ts, te)["ok"])
        rows.append(row)
        cf, cl = int(D.cfirst[r.gid_start]), int(D.clast[r.gid_start])
        ln = te - ts + 1
        cand = [x for x in range(cf, cl - ln + 2) if x + ln - 1 < ts or x > te]
        if cand:
            for st in rng.choice(cand, size=5, replace=True):
                for n in ORD3:
                    d = chance[n].setdefault(r.gtype, [0, 0])
                    d[0] += int(Bm.score_target(Ls[n], int(st), int(st) + ln - 1)["ok"])
                    d[1] += 1
    df = pd.DataFrame(rows)
    df.to_parquet(W3 / "extdef.parquet", index=False)
    out = {n: {g: dict(resolved=int(df[df.gtype == g][n].sum()), runs=int((df.gtype == g).sum()),
                       chance=chance[n].get(g, [0, 1])[0] / max(chance[n].get(g, [0, 1])[1], 1)) for g in df.gtype.unique()}
           for n in ORD3}
    out["extension"] = df.groupby("gtype")[["run_len", "ext_len", "added_left", "added_right"]].median().to_dict("index")
    json.dump(out, open(W3 / "extdef.json", "w"), indent=1)
    log(out)


if __name__ == "__main__":
    W3.mkdir(parents=True, exist_ok=True)
    WV.mkdir(parents=True, exist_ok=True)
    {"repro": cmd_repro, "verdict": cmd_verdict, "rearr": cmd_rearr, "chance": cmd_chance, "extdef": cmd_extdef}[sys.argv[1]](None)
