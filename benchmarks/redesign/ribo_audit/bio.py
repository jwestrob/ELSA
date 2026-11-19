#!/usr/bin/env python
"""ribo_audit part 2: the biologically accurate locus of each ribosomal run (rules: predeclared2.json).

usage (biotite login node, one thread):
  bio.py annot     per-gene classes R / N / X from Pfam + KOfam by the literature rules -> work2/genes_annot.parquet
  bio.py repro     standing recovered / resolved of TOK, L4b, WA_L4b, Z_L4bp (own scorer = ribosomal_check.score)
  bio.py verdict   accurate loci for all 1,686 runs, bio scoring, verdicts for unresolved runs, inheritance, diagrams
  bio.py bridge    bridging rules (i) and (ii) on WA_L4b and L4b
  bio.py chance    random windows with target-matched lengths (chance level of the corrected scoring)
"""
import os
import re
import sys
import json
import time
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit as A  # noqa: E402  (Data, Loci, state helpers; sets numba env)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

RD = A.RD
VARIANT = os.environ.get("RIBO_VARIANT", "strict")      # strict = predeclared; loose = POST HOC (500 nt for every pair)
W2 = A.AD / "work2"
WV = W2 if VARIANT == "strict" else W2 / VARIANT
OUT2 = {"TOK": RD / "boundary/loci/rescue3/DH_coreH_res0.3_f0.5.parquet",
        "L4b": RD / "long_loci/loci/dpann_group/L4b_t0.1.parquet",
        "WA_L4b": RD / "long_loci/loci/dpann_WA/L4b_t0.2.parquet",
        "Z_L4bp": RD / "long_loci/loci/dpann_Z/L4bp_t0.02.parquet"}
BONDS = {"L4b": RD / "long_loci/nodes/dpann_group/bonds_obs.npz",
         "WA_L4b": RD / "long_loci/nodes/dpann_WA/bonds_obs.npz",
         "Z_L4bp": RD / "long_loci/nodes/dpann_Z/bonds_obs.npz"}
TARGET2 = {"TOK": (1686, 1512), "L4b": (1680, 1598), "WA_L4b": (1685, 1594), "Z_L4bp": (1681, 1542)}
ORD2 = ("TOK", "L4b", "WA_L4b", "Z_L4bp")
D_OP, D_MEM = (150, 500) if VARIANT == "strict" else (500, 500)   # linkage distances (nt): any pair / member pair
X_CUT = 3                       # >= 3 consecutive X genes cut a component
MIN_TARGET_MEMBERS = 3          # components scored as separate targets in multi-unit runs
log = A.log

# literature members (predeclared2.json annotation_mapping_rules)
N_RULES = [
    ("secY", {"K03076"}, {"SecY"}), ("rnp1", {"K03538"}, {"RNase_P-MRP_p29"}), ("sui1", {"K03113"}, {"SUI1"}),
    ("rpoD", {"K03047"}, {"RNA_pol_L", "RNA_pol_A_bac"}), ("rpoN", {"K03058"}, {"RNA_pol_N"}), ("rpoK", {"K03055"}, {"RNA_pol_Rpb6"}),
    ("eno", {"K01689"}, {"Enolase_N", "Enolase_C"}), ("cmk", set(), {"Cytidylate_kin", "Cytidylate_kin2"}),
    ("cbf5", {"K03177", "K11130"}, {"TruB_N", "TruB_C_2", "DKCLD"}), ("ndk", {"K00940"}, {"NDK"}),
    ("infB", set(), {"IF-2"}), ("eIF2", {"K03242", "K03237", "K03238"}, set()), ("nusA", {"K02600"}, set()),
    ("EF1a", {"K03231"}, set()), ("rpoHBAEFP", {"K03053", "K13798", "K03040", "K03041", "K03042", "K03049", "K03051",
                                                "K03059"}, set()),
    ("spt5", {"K15172"}, set()), ("secE", set(), {"SecE"}), ("eIF6", {"K03264"}, set())]


def short_rp(ko_name, top):
    if isinstance(ko_name, str):
        m = re.search(r"ribosomal protein (\S+)", ko_name)
        if m:
            return m.group(1)
    if isinstance(top, str) and re.match(r"(?i)^ribosom(al)?_", top):
        return re.sub(r"(?i)^ribosom(al)?_", "", top)
    return "RP?"


# ------------------------------------------------------------------------------------------- annot
def cmd_annot(a):
    import duckdb
    D = A.Data(protein_ids=True)
    con = duckdb.connect(str(A.DB), read_only=True)
    con.execute("SET threads=2")
    kegg = dict(con.execute("SELECT accession, any_value(name) FROM annotations WHERE source='kegg' GROUP BY 1").fetchall())
    ko = con.execute("SELECT protein_id, accession, score FROM annotations WHERE source='kofam'").df()
    pf = con.execute("SELECT protein_id, name, start_aa, score FROM annotations WHERE source='pfam'").df()
    con.close()
    log("kofam rows", len(ko), "pfam rows", len(pf))
    pid2gid = pd.Series(np.arange(D.n), index=D.pid)
    ko["gid"] = pid2gid.reindex(ko.protein_id.to_numpy()).to_numpy()
    ko = ko.dropna(subset=["gid"])
    ko["gid"] = ko.gid.astype(np.int64)
    ko = ko.sort_values("score", ascending=False).drop_duplicates("gid")
    pf["gid"] = pid2gid.reindex(pf.protein_id.to_numpy()).to_numpy()
    pf = pf.dropna(subset=["gid"])
    pf["gid"] = pf.gid.astype(np.int64)
    arch = pf.sort_values(["gid", "start_aa"]).groupby("gid").name.agg(",".join)
    top = pf.sort_values("score", ascending=False).drop_duplicates("gid").set_index("gid").name
    g = pd.DataFrame({"gid": np.arange(D.n)})
    g["ko"] = pd.Series(ko.accession.to_numpy(), index=ko.gid.to_numpy()).reindex(g.gid).to_numpy()
    g["ko_name"] = g.ko.map(kegg)
    g["arch"] = arch.reindex(g.gid).to_numpy()
    g["top"] = top.reindex(g.gid).to_numpy()
    R = D.marked | g.ko_name.fillna("").str.contains("ribosomal protein").to_numpy()
    lit = np.array([""] * D.n, dtype=object)
    for i in np.flatnonzero(R):
        lit[i] = short_rp(g.ko_name.iat[i], g.top.iat[i])
    # literature members: rules in priority order, KO first then Pfam
    lab = pd.Series([""] * D.n, dtype=object)
    for name, kos, pfs in reversed(N_RULES):          # reversed so that earlier rules overwrite later ones
        hit = np.zeros(D.n, bool)
        if kos:
            hit |= g.ko.isin(kos).to_numpy()
        if pfs:
            gg = pf[pf.name.isin(pfs)].gid.unique()
            hit[gg] = True
        lab[hit] = name
    efg = pf[pf.name.isin(["EFG_C", "EFG_IV"])].gid.unique()
    gtp = set(pf[pf.name == "GTP_EFTU"].gid.unique())
    ef2 = np.array([x for x in efg if x in gtp], dtype=np.int64)
    lab0 = lab.to_numpy()
    for x in ef2:
        if lab0[x] == "":
            lab0[x] = "EF2"
    N = (lab0 != "") & ~R
    lit[N] = lab0[N]
    g["cls"] = np.where(R, "R", np.where(N, "N", "X"))
    g["lit"] = lit
    g["unannot"] = g.ko.isna().to_numpy() & g.arch.isna().to_numpy()
    ident = np.where(g.ko.notna(), "K:" + g.ko.fillna(""), np.where(g.arch.notna(), "P:" + g.arch.fillna(""), ""))
    codes, uniq = pd.factorize(pd.Series(ident).replace("", np.nan))
    g["ident"] = codes.astype(np.int64)
    g.to_parquet(W2 / "genes_annot.parquet", index=False)
    log("annot", g.cls.value_counts().to_dict(), "unannotated", int(g.unannot.sum()), "identities", len(uniq))


# ------------------------------------------------------------------------------------------- helpers
@njit(cache=True)
def build_components(link, member, isR, xcut):
    """Global components: chains of linked genes, cut at >= xcut consecutive non-members, trimmed to members."""
    n = len(member)
    cs = np.empty(n, np.int64)
    ce = np.empty(n, np.int64)
    nm = np.empty(n, np.int64)
    nr = np.empty(n, np.int64)
    m = 0
    a = 0
    while a < n:
        b = a
        while b < n - 1 and link[b]:
            b += 1
        # chain a..b
        i = a
        while i <= b:
            # piece starts at i, ends before a run of >= xcut X
            j = i
            xrun = 0
            last_end = -1
            while j <= b:
                if member[j]:
                    xrun = 0
                    last_end = j
                else:
                    xrun += 1
                    if xrun >= xcut:
                        break
                j += 1
            # piece i..min(j, b) trimmed
            p0 = i
            p1 = last_end
            while p0 <= p1 and not member[p0]:
                p0 += 1
            if p1 >= p0:
                cnt = 0
                cr = 0
                for k in range(p0, p1 + 1):
                    if member[k]:
                        cnt += 1
                        if isR[k]:
                            cr += 1
                if cnt >= 2 and cr >= 1:
                    cs[m] = p0
                    ce[m] = p1
                    nm[m] = cnt
                    nr[m] = cr
                    m += 1
            if j > b:
                break
            # skip the X run
            while j <= b and not member[j]:
                j += 1
            i = j
        a = b + 1
    return cs[:m], ce[:m], nm[:m], nr[:m]


class Bio:
    def __init__(self):
        self.D = A.Data()
        D = self.D
        g = pd.read_parquet(A.PKG / "universe/genes.parquet", columns=["gid", "start", "end"]).sort_values("gid")
        self.start = g.start.to_numpy(np.int64)
        self.end = g.end.to_numpy(np.int64)
        an = pd.read_parquet(W2 / "genes_annot.parquet")
        self.an = an
        self.cls = an.cls.to_numpy()
        self.member = np.isin(self.cls, ["R", "N"])
        self.isR = self.cls == "R"
        self.lit = an.lit.to_numpy()
        self.ident = an.ident.to_numpy(np.int64)
        self.unannot = an.unannot.to_numpy()
        self.ko = an.ko.to_numpy()
        self.top = an.top.to_numpy()
        n = D.n
        same = D.cc[:-1] == D.cc[1:]
        self.dist = np.full(n - 1, 10 ** 9, np.int64)
        self.dist[same] = (self.start[1:] - self.end[:-1] - 1)[same]
        sstr = D.strand[:-1] == D.strand[1:]
        mm = self.member[:-1] & self.member[1:]
        self.link = same & sstr & ((self.dist <= D_OP) | (mm & (self.dist <= D_MEM)))
        cs, ce, nm, nr = build_components(self.link, self.member, self.isR, X_CUT)
        self.cs, self.ce, self.nm, self.nr = cs, ce, nm, nr
        self.comp_of = np.full(n, -1, np.int64)
        for i, (s, e) in enumerate(zip(cs, ce)):
            self.comp_of[s:e + 1] = i
        # anomalies (d): split ORFs and length outliers
        topv = an.top.fillna("").to_numpy()
        kov = an.ko.fillna("").to_numpy()
        adj = same & sstr & (((kov[:-1] == kov[1:]) & (kov[:-1] != "")) | ((topv[:-1] == topv[1:]) & (topv[:-1] != "")))
        self.anom = np.zeros(n, bool)
        self.anom[:-1] |= adj
        self.anom[1:] |= adj
        key = np.where(kov != "", kov, np.where(self.isR, self.lit, ""))
        df = pd.DataFrame(dict(key=key, length=D.length))
        med = df[df.key != ""].groupby("key").length.median()
        mk = pd.Series(key).map(med).to_numpy(dtype=float)
        outl = self.member & np.isfinite(mk) & ((D.length > 2 * mk) | (D.length < 0.5 * mk))
        self.anom |= outl
        self.anom_kind = np.where(outl, "len", "") .astype(object)
        self.anom_kind[np.flatnonzero(self.anom & ~outl)] = "split"
        log("components", len(cs), "genes in components", int((ce - cs + 1).sum()), "anomalies", int(self.anom.sum()))

    def run_units(self, i):
        D = self.D
        s, e = int(D.runs.gid_start.iat[i]), int(D.runs.gid_end.iat[i])
        mk = [k for k in range(s, e + 1) if D.marked[k]]
        comps = sorted(set(int(self.comp_of[k]) for k in mk if self.comp_of[k] >= 0))
        single = [k for k in mk if self.comp_of[k] < 0]
        n_units = len(comps) + len(single)
        if n_units == 1 and comps:
            targets = [(int(self.cs[comps[0]]), int(self.ce[comps[0]]))]
        else:
            targets = [(int(self.cs[c]), int(self.ce[c])) for c in comps if self.nm[c] >= MIN_TARGET_MEMBERS]
        return s, e, mk, comps, single, n_units, targets


def score_target(L, ts, te):
    j, ov = L.best(ts, te)
    if j < 0:
        return dict(best=-1, bf=0.0, prec=np.nan, ok=False, a=-1, b=-1)
    ln = L.length(j)
    bf = ov / (te - ts + 1)
    prec = ov / ln
    return dict(best=j, bf=bf, prec=prec, ok=bool(bf >= 0.8 and prec >= 0.8), a=int(L.a[j]), b=int(L.b[j]))


def any_fit(L, ts, te):
    for k in L.overlapping(ts, te):
        ov = A.ovl(L.a[k], L.b[k], ts, te)
        if ov / (te - ts + 1) >= 0.8 and ov / L.length(k) >= 0.8:
            return True
    return False


def bond_sk(path, n):
    z = np.load(path)
    k = z["k"]
    d = z["d"] if "d" in z.files else np.ones(len(k), np.int64)
    lam = z["lam"] if "lam" in z.files else np.ones(len(k))
    sk = np.zeros(n)
    for off in range(int(d.max())):
        m = d > off
        np.maximum.at(sk, k[m] + off, lam[m])
    return sk


# ------------------------------------------------------------------------------------------- repro
def cmd_repro(a):
    D = A.Data()
    runs = D.runs
    sys.path.insert(0, str(RD / "ribosomal/code"))
    import ribosomal_check as RC
    res = {}
    for n in ORD2:
        L = A.Loci(OUT2[n])
        df = pd.DataFrame([A.standard(L, s, e) for s, e in zip(runs.gid_start, runs.gid_end)])
        sc = RC.score(dict(a=L.a, b=L.b, mod=np.full(L.n, -1, np.int64), maxlen=L.maxlen, n=L.n),
                      runs.gid_start.to_numpy(), runs.gid_end.to_numpy())
        res[n] = dict(loci=L.n, recovered=int(df.recovered.sum()), resolved=int(df.resolved.sum()),
                      rc_resolved=int(sc.resolved.sum()), equal=bool(np.array_equal(sc.resolved.to_numpy(), df.resolved.to_numpy())),
                      target=TARGET2[n], reproduced=bool((int(df.recovered.sum()), int(df.resolved.sum())) == TARGET2[n]))
        df.insert(0, "run_id", runs.run_id.to_numpy())
        df.to_parquet(W2 / f"std_{n}.parquet", index=False)
        log(n, res[n])
    json.dump(res, open(W2 / "repro2.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------- inheritance
class Inherit:
    """Identity-pair adjacency (<= 2 positions) across genomes and lineage groups, for queried pairs."""
    def __init__(self, B, pairs):
        D = B.D
        idt = B.ident
        M = int(idt.max()) + 2
        self.M = M
        q = set()
        for (x, y) in pairs:
            if idt[x] >= 0 and idt[y] >= 0:
                q.add(min(idt[x], idt[y]) * M + max(idt[x], idt[y]))
        self.q = np.array(sorted(q), np.int64)
        keys, gens = [], []
        for d in (1, 2):
            i = np.arange(D.n - d)
            ok = (D.cc[i] == D.cc[i + d]) & (idt[i] >= 0) & (idt[i + d] >= 0)
            i = i[ok]
            k = np.minimum(idt[i], idt[i + d]) * M + np.maximum(idt[i], idt[i + d])
            keep = np.isin(k, self.q)
            keys.append(k[keep])
            gens.append(D.gc[i[keep]])
        kk = np.concatenate(keys)
        gg = np.concatenate(gens)
        u = np.unique(kk * 4096 + gg)
        self.adj = {}
        for x in u:
            self.adj.setdefault(int(x // 4096), set()).add(int(x % 4096))
        ids = set()
        for x in self.q:
            ids.add(int(x // M))
            ids.add(int(x % M))
        sel = np.isin(idt, np.array(sorted(ids), np.int64))
        ug = np.unique(idt[sel] * 4096 + D.gc[sel])
        self.has = {}
        for x in ug:
            self.has.setdefault(int(x // 4096), set()).add(int(x % 4096))
        self.grp = D.grp
        self.B = B

    def junction(self, x, y, genome):
        idt = self.B.ident
        if idt[x] < 0 or idt[y] < 0:
            return dict(own_adj=None, own_both=None, other_groups_adj=None)
        key = min(idt[x], idt[y]) * self.M + max(idt[x], idt[y])
        q = self.grp[genome]
        adj = self.adj.get(int(key), set()) - {genome}
        both = (self.has.get(int(idt[x]), set()) & self.has.get(int(idt[y]), set())) - {genome}
        own_adj = sum(1 for h in adj if self.grp[h] == q)
        own_both = sum(1 for h in both if self.grp[h] == q)
        og = len(set(int(self.grp[h]) for h in adj if self.grp[h] != q))
        return dict(own_adj=own_adj, own_both=own_both, other_groups_adj=og)


# ------------------------------------------------------------------------------------------- verdict
def gene_label(B, k):
    c = B.cls[k]
    if c == "R":
        return B.lit[k]
    if c == "N":
        return B.lit[k]
    t = B.top[k]
    return (t if isinstance(t, str) else ("unannot" if B.unannot[k] else str(B.ko[k])))


def diagram(B, lo, hi, s, e, targets, L, best):
    """Three aligned lines: genes (upper = run span; R/N/X/U) with junction marks (' ' linked, '/' distance break,
    '|' strand change); T = accurate target; output: B best locus, o other locus."""
    D = B.D
    l1, l2, l3, l4 = [], [], [], []
    tset = set()
    for (ts, te) in targets:
        tset.update(range(ts, te + 1))
    cov = {}
    for j in L.overlapping(lo, hi):
        for k in range(max(lo, L.a[j]), min(hi, L.b[j]) + 1):
            cov[k] = "o"
    if best >= 0:
        for k in range(max(lo, L.a[best]), min(hi, L.b[best]) + 1):
            cov[k] = "B"
    for k in range(lo, hi + 1):
        ch = {"R": "r", "N": "n", "X": "x"}[B.cls[k]]
        if B.cls[k] == "X" and B.unannot[k]:
            ch = "u"
        if s <= k <= e:
            ch = ch.upper()
        if k < hi:
            if D.strand[k] != D.strand[k + 1]:
                jm = "|"
            elif not B.link[k]:
                jm = "/"
            else:
                jm = " "
        else:
            jm = " "
        l1.append(ch + jm)
        l2.append((">" if D.strand[k] > 0 else "<") + " ")
        l3.append(("T" if k in tset else ".") + " ")
        l4.append(cov.get(k, ".") + " ")
    pre = "|" if lo == B.D.cfirst[lo] else " "
    post = "|" if hi == B.D.clast[hi] else ""
    return [pre + "".join(l1) + post, " " + "".join(l2), " " + "".join(l3), " " + "".join(l4)]


def cmd_verdict(a):
    t0 = time.time()
    B = Bio()
    D = B.D
    runs = D.runs
    Ls = {n: A.Loci(OUT2[n]) for n in ORD2}
    std = {n: pd.read_parquet(W2 / f"std_{n}.parquet") for n in ORD2}
    sks = {n: bond_sk(BONDS[n], D.n) for n in BONDS}
    # per-run units / targets
    runinfo = []
    for i in range(len(runs)):
        s, e, mk, comps, single, n_units, targets = B.run_units(i)
        runinfo.append(dict(ridx=i, run_id=int(runs.run_id.iat[i]), s=s, e=e, n_marked=len(mk), n_comp_units=len(comps),
                            n_single_units=len(single), n_units=n_units, targets=targets, n_targets=len(targets),
                            stratum=runs.stratum.iat[i], genome_code=int(runs.genome_code.iat[i])))
    # decisive junction pairs to query inheritance
    rows, pairs_needed = [], []
    for ri in runinfo:
        for n in ORD2:
            L = Ls[n]
            st = std[n].iloc[ri["ridx"]]
            sc = [score_target(L, ts, te) for (ts, te) in ri["targets"]]
            right = bool(len(sc) > 0 and all(x["ok"] for x in sc))
            rows.append(dict(run_id=ri["run_id"], ridx=ri["ridx"], output=n, std_ok=bool(st.resolved), bio_ok=right,
                             n_targets=ri["n_targets"], n_units=ri["n_units"]))
    bio = pd.DataFrame(rows)
    bio.to_parquet(WV / "bio_scores.parquet", index=False)
    log("bio scoring done", f"{time.time()-t0:.0f}s", bio.groupby("output")[["std_ok", "bio_ok"]].sum().to_dict())
    # verdicts for unresolved (standing)
    cases = []
    for ri in runinfo:
        s, e = ri["s"], ri["e"]
        for n in ORD2:
            st = std[n].iloc[ri["ridx"]]
            if bool(st.resolved):
                continue
            L = Ls[n]
            c = dict(run_id=ri["run_id"], ridx=ri["ridx"], output=n, stratum=ri["stratum"], genome_code=ri["genome_code"],
                     run_s=s, run_e=e, run_len=e - s + 1, std_bf=float(st.best_frac), std_prec=float(st.precision),
                     n_units=ri["n_units"], n_targets=ri["n_targets"], targets=";".join(f"{ts - s}..{te - s}" for ts, te in ri["targets"]))
            decisive = []
            verdict = None
            if ri["n_targets"] == 0:
                verdict = 7
                c["why"] = "no qualifying unit (marked genes in no component with >= 2 members incl. a ribosomal protein, or units all < 3 members)"
            if verdict is None:
                if ri["n_units"] > 1:
                    # glue junctions: consecutive marked genes in different units
                    mk = [k for k in range(s, e + 1) if D.marked[k]]
                    uid = [int(B.comp_of[k]) if B.comp_of[k] >= 0 else -10 - k for k in mk]
                    for x, y, ux, uy in zip(mk[:-1], mk[1:], uid[:-1], uid[1:]):
                        if ux != uy:
                            # the break inside x..y: first non-linked boundary
                            p = next((q for q in range(x, y) if not B.link[q]), x)
                            decisive.append(("glue", p))
                    kind = "glue"
                else:
                    ts, te = ri["targets"][0]
                    sc = score_target(L, ts, te)
                    c.update(C_s=ts - s, C_e=te - s, C_len=te - ts + 1, bio_bf=sc["bf"], bio_prec=sc["prec"],
                             best_s=(sc["a"] - s) if sc["a"] >= 0 else None, best_e=(sc["b"] - s) if sc["b"] >= 0 else None)
                    if sc["ok"]:
                        kind = "ok"
                    elif any_fit(L, ts, te):
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
                # data check
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
                    if dat:
                        c["data_flag"] = ",".join(sorted(set(dat)))
            c["verdict"] = verdict
            c["decisive"] = decisive
            cases.append(c)
    # inheritance for decisive junctions
    pairs = [(p, p + 1) for c in cases for (_, p) in c["decisive"]]
    IH = Inherit(B, pairs)
    for c in cases:
        info = []
        for (kd, p) in c["decisive"]:
            h = IH.junction(p, p + 1, c["genome_code"])
            sk = {n: round(float(sks[n][p]), 1) for n in sks}
            info.append(dict(kind=kd, rel=p - c["run_s"], left=gene_label(B, p), right=gene_label(B, p + 1),
                             left_cls=B.cls[p], right_cls=B.cls[p + 1], same_strand=bool(D.strand[p] == D.strand[p + 1]),
                             dist=int(B.dist[p]) if B.dist[p] < 10 ** 8 else None, linked=bool(B.link[p]),
                             unannot=bool(B.unannot[p] or B.unannot[p + 1]), **h,
                             bond_this=sk.get(c["output"]) if c["output"] in sks else None))
        c["junctions"] = info
        if c["verdict"] == 2:
            ty = set()
            for j in info:
                ty.add("strand" if not j["same_strand"] else ("mem>500" if (j["left_cls"] in "RN" and j["right_cls"] in "RN") else "spacer"))
            c["subtype"] = "+".join(sorted(ty))
        c["junction_txt"] = " ; ".join(
            f"{j['kind']}@{j['rel']} {j['left']}|{j['right']} {'same' if j['same_strand'] else 'OPP'} d={j['dist']} "
            f"grp {j['own_adj']}/{j['own_both']} othergrp {j['other_groups_adj']} bond {j['bond_this']}" for j in info)
        del c["decisive"]
    # verdict-3 / 4 detail: genes of C beyond the run, genes of best beyond C
    for c in cases:
        if c.get("C_s") is None:
            continue
        s = c["run_s"]
        ts, te = s + c["C_s"], s + c["C_e"]
        beyond = [k for k in range(ts, te + 1) if not (s <= k <= c["run_e"])]
        c["C_beyond_run"] = ",".join(f"{gene_label(B, k)}[{B.cls[k]}]" for k in beyond)
        if c.get("best_s") is not None and c["verdict"] == 4:
            bs, be = s + c["best_s"], s + c["best_e"]
            ext = [k for k in range(bs, be + 1) if not (ts <= k <= te)]
            cl = pd.Series([B.cls[k] for k in ext]).value_counts().to_dict()
            c["subtype"] = "4b_other_unit_members" if (cl.get("R", 0) + cl.get("N", 0)) >= 0.5 * max(len(ext), 1) else "4a_unrelated"
            c["best_beyond_C"] = f"{len(ext)} genes R/N/X {cl.get('R', 0)}/{cl.get('N', 0)}/{cl.get('X', 0)}: " + ",".join(
                gene_label(B, k) for k in ext[:12]) + (" ..." if len(ext) > 12 else "")
    # diagrams and layouts
    diag, lay = [], []
    for c in cases:
        L = Ls[c["output"]]
        s, e = c["run_s"], c["run_e"]
        ri = runinfo[c["ridx"]]
        tg = ri["targets"]
        lo = min([s] + [t[0] for t in tg]) - 4
        hi = max([e] + [t[1] for t in tg]) + 4
        j, ov = L.best(s, e)
        if j >= 0:
            lo = min(lo, int(L.a[j]) - 1)
            hi = max(hi, int(L.b[j]) + 1)
        lo = max(lo, s - 40, int(D.cfirst[s]))
        hi = min(hi, e + 40, int(D.clast[s]))
        cid = f"{c['output']}-{c['run_id']}"
        c["case_id"] = cid
        lines = diagram(B, lo, hi, s, e, tg, L, j)
        c["diag_lo"] = lo - s
        diag.append(f"{cid}  {c['stratum']}  run {e - s + 1} genes  std bf {c['std_bf']:.2f} prec {c['std_prec']:.2f}  "
                    f"-> VERDICT {c['verdict']}  targets {c['targets']} (window starts at {lo - s})")
        if c.get("junction_txt"):
            diag.append("    " + c["junction_txt"])
        if c.get("why"):
            diag.append("    " + c["why"])
        if c.get("best_beyond_C"):
            diag.append("    best beyond C: " + c["best_beyond_C"])
        if c.get("C_beyond_run") and c["verdict"] == 3:
            diag.append("    C beyond run: " + c["C_beyond_run"])
        diag += ["    " + x for x in lines]
        for k in range(lo, hi + 1):
            lay.append(dict(case_id=cid, rel=k - s, gid=k, strand=int(D.strand[k]), start=int(B.start[k]), end=int(B.end[k]),
                            dist_next=(int(B.dist[k]) if k < D.n - 1 and B.dist[k] < 10 ** 8 else None),
                            linked_next=bool(B.link[k]) if k < D.n - 1 else False, cls=B.cls[k], lit=B.lit[k],
                            ko=B.ko[k], ko_name=B.an.ko_name.iat[k], pfam=B.an.arch.iat[k], marked=bool(D.marked[k]),
                            in_run=bool(s <= k <= e), in_target=bool(any(t0 <= k <= t1 for t0, t1 in tg)),
                            anomaly=B.anom_kind[k], comp=int(B.comp_of[k]),
                            loci=";".join(f"{int(L.a[x]) - s}..{int(L.b[x]) - s}" for x in L.overlapping(k, k)),
                            **{f"bond_{n}": round(float(sks[n][k]), 1) for n in sks}))
    cdf = pd.DataFrame([{k: v for k, v in c.items() if k != "junctions"} for c in cases])
    cdf.to_csv(WV / "cases2.tsv", sep="\t", index=False)
    json.dump([dict(case_id=c["case_id"], junctions=c["junctions"]) for c in cases], open(WV / "junctions2.json", "w"),
              indent=0, default=str)
    pd.DataFrame(lay).to_csv(WV / "layouts2.tsv", sep="\t", index=False)
    with open(WV / "diagrams2.txt", "w") as fh:
        fh.write("Legend line 1: gene class (R ribosomal protein: Ribosomal_* Pfam or KO ribosomal protein; N literature member;\n"
                 "X other; U no Pfam and no KO), UPPER CASE inside the run span; junction mark after each gene: ' ' linked\n"
                 "(same strand, operon spacing), '/' not linked (distance or X-cut rule), '|' strand change; '|' at the ends =\n"
                 "contig end. Line 2 strand. Line 3 T = accurate target locus (C or components). Line 4 output loci: B = the\n"
                 "output's best locus for the run, o = other loci. Junction notes: grp a/b = other genomes of the own group\n"
                 "with the same adjacency / holding both genes; othergrp = other groups with the adjacency; bond = this\n"
                 "output's s_k (reported only).\n\n")
        fh.write("\n".join(diag) + "\n")
    rinfo = pd.DataFrame([{k: (v if k != "targets" else ";".join(f"{a_}..{b_}" for a_, b_ in v)) for k, v in r.items()} for r in runinfo])
    rinfo.to_parquet(WV / "runinfo.parquet", index=False)
    log("verdicts", cdf.groupby(["output", "verdict"]).size().unstack(fill_value=0).to_dict("index"), f"{time.time()-t0:.0f}s")


# ------------------------------------------------------------------------------------------- bridge (answer B)
def top_level(L, n):
    """gene -> id of the maximal locus containing it (-1 none); maximal loci of a nested set are disjoint."""
    order = np.lexsort((-(L.b - L.a), L.a))
    top = np.full(n, -1, np.int64)
    tops = []
    reach = -1
    for j in order:
        if L.a[j] > reach:
            tops.append(j)
            top[L.a[j]:L.b[j] + 1] = len(tops) - 1
            reach = L.b[j]
        elif L.b[j] > reach:          # partial overlap (not nested): extend the current top
            top[reach + 1:L.b[j] + 1] = len(tops) - 1
            reach = L.b[j]
    return top


def cmd_bridge(a):
    t0 = time.time()
    B = Bio()
    D = B.D
    tok = D.tok
    G = D.G
    cases = pd.read_csv(WV / "cases2.tsv", sep="\t")
    inst = pd.read_parquet(A.PKG / "gold/instances.parquet", columns=["instance_id", "caller"])
    mem = pd.read_parquet(A.PKG / "gold/members.parquet")
    nm = mem.groupby("instance_id").size()
    mem = mem[mem.instance_id.isin(nm[nm >= 2].index)]
    gene_inst = {}
    for r in mem.itertuples():
        gene_inst.setdefault(int(r.gid), set()).add(r.instance_id)
    out = {}
    for n in ("WA_L4b", "L4b"):
        L = A.Loci(OUT2[n])
        sk = bond_sk(BONDS[n], D.n)
        top = top_level(L, D.n)
        # candidate junctions: a locus ends at k, another starts at k + 1, same contig, no locus spans k|k+1, unbonded
        span = np.zeros(D.n + 1, np.int64)
        np.add.at(span, L.a, 1)
        np.add.at(span, L.b, -1)
        span = np.cumsum(span)[:D.n]                     # loci with a <= k < b
        ends = np.zeros(D.n, bool)
        ends[L.b] = True
        starts = np.zeros(D.n + 1, bool)
        starts[L.a] = True
        k = np.flatnonzero(ends[:-1] & starts[1:D.n] & (D.cc[:-1] == D.cc[1:]) & (span[:-1] == 0) & (sk[:-1] == 0))
        log(n, "candidate junctions", len(k))
        tx, ty = tok[k], tok[k + 1]
        T = int(tok.max()) + 1
        pk = np.minimum(tx, ty) * T + np.maximum(tx, ty)
        need = np.unique(pk)
        # rule (ii): own-group adjacency (<= 3 positions) in other genomes
        adjg = {}
        for d in (1, 2, 3):
            i = np.arange(D.n - d)
            ok = D.cc[i] == D.cc[i + d]
            i = i[ok]
            kk = np.minimum(tok[i], tok[i + d]) * T + np.maximum(tok[i], tok[i + d])
            keep = np.isin(kk, need)
            for key, gg in zip(kk[keep], D.gc[i[keep]]):
                adjg.setdefault(int(key), set()).add(int(gg))
        # rule (i): co-membership in one top-level locus in other groups
        intop = top >= 0
        ut = np.unique(tok[intop] * 10 ** 7 + top[intop])
        needtok = set(np.unique(np.r_[tx, ty]).tolist())
        tl = {}
        for x in ut:
            t_ = int(x // 10 ** 7)
            if t_ in needtok:
                tl.setdefault(t_, set()).add(int(x % 10 ** 7))
        top_genome = {}
        for j in range(D.n):
            pass
        # genome of each top locus
        tg = np.full(int(top.max()) + 1, -1, np.int64)
        tg[top[intop]] = D.gc[intop]
        rows = []
        for kk_, a_, b_, key in zip(k, tx, ty, pk):
            g_own = int(D.gc[kk_])
            q = int(D.grp[g_own])
            co = tl.get(int(a_), set()) & tl.get(int(b_), set())
            co_groups = set(int(D.grp[tg[t]]) for t in co) - {q}
            adj_own = sum(1 for h in adjg.get(int(key), set()) if h != g_own and D.grp[h] == q)
            same_str = bool(D.strand[kk_] == D.strand[kk_ + 1])
            dist = int(B.dist[kk_])
            # curated: instances overlapping the two top-level loci
            lt, rt = int(top[kk_]), int(top[kk_ + 1])
            li, rgt = set(), set()
            for x in np.flatnonzero(top[max(0, kk_ - 300):kk_ + 1] == lt) + max(0, kk_ - 300):
                li |= gene_inst.get(int(x), set())
            for x in np.flatnonzero(top[kk_ + 1:kk_ + 302] == rt) + kk_ + 1:
                rgt |= gene_inst.get(int(x), set())
            rows.append(dict(k=int(kk_), genome=g_own, group=q, co_groups=len(co_groups), adj_own=adj_own,
                             same_strand=same_str, dist=dist, member_pair=bool(B.member[kk_] and B.member[kk_ + 1]),
                             cur_left=len(li), cur_right=len(rgt), cur_diff=bool(li and rgt and not (li & rgt)),
                             cur_same=bool(li & rgt)))
        J = pd.DataFrame(rows)
        J["rule_i"] = J.co_groups >= 3
        J["rule_ii"] = J.adj_own >= 1
        J["rule_iii"] = J.same_strand & (J.dist <= 150)
        J["rule_i_iii"] = J.rule_i & J.rule_iii
        J["rule_ii_iii"] = J.rule_ii & J.rule_iii
        J.to_parquet(WV / f"bridge_junctions_{n}.parquet", index=False)
        res = dict(candidates=int(len(J)))
        fire = {r: set(J.k[J[r]].tolist()) for r in ("rule_i", "rule_ii", "rule_iii", "rule_i_iii", "rule_ii_iii")}
        # harm on the ribosomal set: firing junctions at the outer edges of accurate targets (would extend past them)
        ri = pd.read_parquet(WV / "runinfo.parquet")
        edges = set()
        inner = set()
        for t in ri.targets:
            if not t:
                continue
            for x in t.split(";"):
                a0, b0 = map(int, x.split(".."))
                edges.add(a0 - 1)
                edges.add(b0)
                inner.update(range(a0, b0))
        for r in fire:
            d = J[J[r]]
            res[r] = dict(fires=int(len(d)), merges_two_curated=int(d.cur_diff.sum()), joins_one_curated=int(d.cur_same.sum()),
                          fires_touching_curated=int(((d.cur_left > 0) | (d.cur_right > 0)).sum()),
                          fires_at_target_edges=int(sum(1 for x in fire[r] if x in edges)),
                          fires_inside_targets=int(sum(1 for x in fire[r] if x in inner)))
        res["candidates_at_target_edges"] = int(sum(1 for x in J.k if x in edges))
        res["candidates_inside_targets"] = int(sum(1 for x in J.k if x in inner))
        # repairs of verdict-1 cases
        v1 = cases[(cases.output == n) & (cases.verdict == 1)]
        rep = {r: 0 for r in fire}
        detail = []
        for c in v1.itertuples():
            s = int(c.run_s)
            ts, te = s + int(c.C_s), s + int(c.C_e)
            idx = L.overlapping(ts, te)
            iv = sorted(set((int(L.a[j]), int(L.b[j])) for j in idx))
            row = dict(case_id=c.case_id)
            for r, fs in fire.items():
                # merge adjacent intervals across firing junctions, iteratively
                ivs = list(iv)
                changed = True
                while changed:
                    changed = False
                    new = set(ivs)
                    for (x0, x1) in ivs:
                        for (y0, y1) in ivs:
                            if y0 == x1 + 1 and x1 in fs and (x0, y1) not in new:
                                new.add((x0, y1))
                                changed = True
                    ivs = sorted(new)
                best, bov, blen = None, 0, 10 ** 9
                for (x0, x1) in ivs:
                    ov = A.ovl(x0, x1, ts, te)
                    ln = x1 - x0 + 1
                    if ov > bov or (ov == bov and ln < blen):
                        best, bov, blen = (x0, x1), ov, ln
                ok = bool(best and bov / (te - ts + 1) >= 0.8 and bov / blen >= 0.8)
                row[r] = ok
                rep[r] += int(ok)
            detail.append(row)
        res["verdict1_cases"] = int(len(v1))
        res["repaired"] = rep
        out[n] = res
        pd.DataFrame(detail).to_csv(WV / f"bridge_repairs_{n}.tsv", sep="\t", index=False)
        log(n, res, f"{time.time()-t0:.0f}s")
    json.dump(out, open(WV / "bridge.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------- chance (answer C)
def cmd_chance(a):
    B = Bio()
    D = B.D
    ri = pd.read_parquet(WV / "runinfo.parquet")
    ctrl = pd.read_parquet(A.GOLD / "controls_random.parquet")
    ctrl = ctrl[ctrl.draw < 5]
    st_of = ctrl.groupby("run_id").gid_start.apply(list).to_dict()
    Ls = {n: A.Loci(OUT2[n]) for n in ORD2}
    res = {n: [0, 0] for n in ORD2}
    std = {n: [0, 0] for n in ORD2}
    runs = D.runs.set_index("run_id")
    for rid, starts in st_of.items():
        if rid not in runs.index:
            continue
        ln = int(runs.loc[rid, "gid_end"] - runs.loc[rid, "gid_start"] + 1)
        for st in starts:
            en = st + ln - 1
            if en > D.clast[st]:
                continue
            for n in ORD2:
                sc = score_target(Ls[n], st, en)
                std[n][0] += int(sc["ok"])
                std[n][1] += 1
    for r in ri.itertuples():
        if not r.targets:
            continue
        for t in r.targets.split(";"):
            a_, b_ = map(int, t.split(".."))
            ln = b_ - a_ + 1
            for st in st_of.get(r.run_id, []):
                en = st + ln - 1
                if en > D.clast[st]:
                    continue
                for n in ORD2:
                    sc = score_target(Ls[n], st, en)
                    res[n][0] += int(sc["ok"])
                    res[n][1] += 1
    out = {n: dict(resolved=v[0], windows=v[1], chance=v[0] / max(v[1], 1), standing_windows=std[n][1],
                   standing_chance=std[n][0] / max(std[n][1], 1)) for n, v in res.items()}
    json.dump(out, open(WV / "chance2.json", "w"), indent=1)
    log(out)


if __name__ == "__main__":
    W2.mkdir(parents=True, exist_ok=True)
    WV.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    {"annot": cmd_annot, "repro": cmd_repro, "verdict": cmd_verdict, "bridge": cmd_bridge, "chance": cmd_chance}[cmd](None)
