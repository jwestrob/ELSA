#!/usr/bin/env python
"""lut_loci s02 (Mac): carboxylate-center site test on apo models and crystal checks (tests_rule.json T_seq, T_struct,
T_crystal_check). Rule parameters are copied from tests_rule.json and must not be changed after results are read.

usage: s02_sites.py models   -> data/sites_apo.tsv (one row per model)
       s02_sites.py crystals -> data/sites_crystal.tsv (one row per protein-bound metal of each crystal)
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.spatial import cKDTree

B = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/lut_loci"
GRID, CLASH, OCOUNT_R, OCOUNT_MIN = 0.7, 2.0, 3.2, 2
REACH_C, REACH_P = 4.2, 5.2
PLDDT_MIN, GAP_MAX, DE_MIN, PASS_CONS = 70.0, 0.20, 0.80, 4
CARBO = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
CARBC = {"ASP": "CG", "GLU": "CD"}
PREC = {"ASP": "CB", "GLU": "CG"}
METALS = {"LA", "CA", "ND", "DY", "CE", "PR", "SM", "EU", "GD", "TB", "HO", "ER", "TM", "YB", "LU", "Y"}
AA3 = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K",
           MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")
EF = re.compile(r"(?=(D[^W][DNS][^ILVFYW][DENSTG][DNQGHRK][^GP][LIVMC][DENQSTAGC].{2}[DE]))")
EF_RELAXED = re.compile(r"(?=(D.[DNS].[DNSEG]G.{5}[DE]))")


def parse(path, chains=None):
    A = []
    for line in open(path):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[16] not in (" ", "A"):
            continue
        ch = line[21]
        if chains and ch not in chains and line.startswith("ATOM"):
            continue
        A.append(dict(het=line.startswith("HETATM"), name=line[12:16].strip(), resn=line[17:20].strip(), chain=ch,
                      resi=int(line[22:26]), x=float(line[30:38]), y=float(line[38:46]), z=float(line[46:54]),
                      b=float(line[60:66]) if line[60:66].strip() else np.nan,
                      el=(line[76:78].strip().upper() if len(line) >= 78 else "")))
    return pd.DataFrame(A)


def centers(P, conserved=None, plddt=None, use_plddt=True):
    """P: protein atoms (no HETATM). Returns center table: xyz, reaching residues, capacity, site pLDDT, conserved count."""
    xyz = P[["x", "y", "z"]].to_numpy()
    heavy = P.el.ne("H").to_numpy()
    carbO = P[P.apply(lambda r: r.resn in CARBO and r["name"] in CARBO[r.resn], axis=1)]
    if len(carbO) == 0:
        return pd.DataFrame()
    oxyz = carbO[["x", "y", "z"]].to_numpy()
    lo, hi = oxyz.min(0) - 3.5, oxyz.max(0) + 3.5
    g = np.mgrid[lo[0]:hi[0]:GRID, lo[1]:hi[1]:GRID, lo[2]:hi[2]:GRID].reshape(3, -1).T
    th = cKDTree(xyz[heavy])
    g = g[np.array([len(x) == 0 for x in th.query_ball_point(g, CLASH)])]
    to = cKDTree(oxyz)
    cnt = np.array([len(x) for x in to.query_ball_point(g, OCOUNT_R)])
    g = g[cnt >= OCOUNT_MIN]
    if len(g) == 0:
        return pd.DataFrame()
    res = P[P.resn.isin(CARBO)]
    keys = res[["chain", "resi", "resn"]].drop_duplicates()
    cc, pc, rid = [], [], []
    for k in keys.itertuples():
        r = res[(res.chain == k.chain) & (res.resi == k.resi)]
        c1 = r[r["name"] == CARBC[k.resn]]
        c2 = r[r["name"] == PREC[k.resn]]
        if len(c1) == 0 and len(c2) == 0:
            continue
        cc.append(c1[["x", "y", "z"]].to_numpy()[0] if len(c1) else np.array([1e9] * 3))
        pc.append(c2[["x", "y", "z"]].to_numpy()[0] if len(c2) else np.array([1e9] * 3))
        rid.append((k.chain, k.resi, k.resn))
    cc, pc = np.array(cc), np.array(pc)
    d1 = np.linalg.norm(g[:, None, :] - cc[None, :, :], axis=2)
    d2 = np.linalg.norm(g[:, None, :] - pc[None, :, :], axis=2)
    reach = (d1 <= REACH_C) | (d2 <= REACH_P)
    rows = []
    for i in range(len(g)):
        idx = np.flatnonzero(reach[i])
        if len(idx) == 0:
            continue
        rr = [rid[j] for j in idx]
        pl = min(plddt.get(r[1], np.nan) for r in rr) if plddt else np.nan
        nc = sum(1 for r in rr if conserved is None or conserved.get(r[1], False))
        rows.append(dict(x=g[i, 0], y=g[i, 1], z=g[i, 2], cap=len(rr), cons=nc, site_plddt=pl,
                         res=";".join(f"{dict(ASP="D", GLU="E")[r[2]]}{r[1]}" for r in rr)))
    return pd.DataFrame(rows)


def conservation(fam, gid, seq):
    """conserved acidic positions of the representative (1-based) from msa/<fam>_aln.faa; also column stats."""
    al = {r.id: str(r.seq) for r in SeqIO.parse(f"{B}/msa/{fam}_aln.faa", "fasta")}
    a = al[str(gid)]
    M = np.array([list(s) for s in al.values()])
    cons, pos, n_cons = {}, 0, 0
    for j, c in enumerate(a):
        if c == "-":
            continue
        pos += 1
        col = M[:, j]
        ng = col[col != "-"]
        gap = 1 - len(ng) / len(col)
        de = np.isin(ng, ["D", "E"]).mean() if len(ng) else 0
        ok = gap <= GAP_MAX and de >= DE_MIN
        cons[pos] = bool(ok)
        n_cons += ok
    assert pos == len(seq), (fam, gid, pos, len(seq))
    allres = "".join(s.replace("-", "") for s in al.values())
    return cons, n_cons, len(al), (allres.count("D") + allres.count("E")) / max(1, len(allres))


def models():
    S = pd.read_csv(B + "/sel/selection.tsv", sep="\t")
    rows = []
    for fa in SeqIO.parse(B + "/folds_in/apo.faa", "fasta"):
        nm = fa.id
        role, fam, gid = nm.split("__")
        seq = str(fa.seq)
        P = parse(f"{B}/folds/apo/{nm}.pdb")
        P = P[~P.het]
        pl = P[P["name"] == "CA"].set_index("resi").b.to_dict()
        if role == "posref":
            cons, ncons, nfam, defrac = None, np.nan, 0, np.nan
        else:
            cons, ncons, nfam, defrac = conservation(fam, gid, seq)
        C = centers(P, cons, pl)
        r = dict(name=nm, role=role, family=fam, gid=gid, length=len(seq), mean_plddt=round(float(np.mean(list(pl.values()))), 1),
                 fam_n=nfam, fam_DE_frac=round(defrac, 3) if nfam else np.nan,
                 cons_acidic_cols=ncons, cons_acidic_per100=round(100 * ncons / len(seq), 2) if nfam else np.nan,
                 ef_strict=len(EF.findall(seq)), ef_relaxed=len(EF_RELAXED.findall(seq)),
                 ilvfmawc=round(sum(seq.count(c) for c in "ILVFMAWC") / len(seq), 3))
        if len(C):
            ok = C[C.site_plddt >= PLDDT_MIN]
            r["score_struct"] = int(ok.cap.max()) if len(ok) else 0
            r["score_struct_any_plddt"] = int(C.cap.max())
            r["score_cons"] = int(ok.cons.max()) if len(ok) else 0
            if len(ok):
                b = ok.sort_values(["cons", "cap", "site_plddt"], ascending=False).iloc[0]
                r.update(best_res=b.res, best_cap=int(b.cap), best_cons=int(b.cons), best_site_plddt=round(b.site_plddt, 1),
                         best_xyz=f"{b.x:.2f},{b.y:.2f},{b.z:.2f}")
                xyz = P[["x", "y", "z"]].to_numpy()
                d = np.linalg.norm(xyz - np.array([b.x, b.y, b.z]), axis=1)
                near = P[d <= 3.0]
                r["best_O_within3"] = int((near.el == "O").sum())
                r["best_amideO_bbO_within3"] = int(((near["name"] == "O") | ((near.resn == "ASN") & (near["name"] == "OD1")) |
                                                    ((near.resn == "GLN") & (near["name"] == "OE1"))).sum())
                r["centers_ge4_per100"] = round(100 * ok.groupby("res").cap.max().ge(4).sum() / len(seq), 2)
        else:
            r.update(score_struct=0, score_cons=0, score_struct_any_plddt=0)
        if role == "posref":
            r["score_cons"] = r.get("score_struct", 0)
            r["note"] = "structure-only: no collection family; score_cons set to score_struct (tests_rule positives_reference)"
        r["PASS_site"] = bool(r.get("score_cons", 0) >= PASS_CONS)
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(B + "/data/sites_apo.tsv", sep="\t", index=False)
    return D


def crystals():
    rows = []
    for pdb, path in [("8FNS", "crystals/8FNS.pdb"), ("8DQ2", "crystals/8DQ2.pdb"), ("1CLL", "crystals/1CLL.pdb"),
                      ("9VY4", "crystals/9VY4.pdb"), ("6OC5", "crystals/6OC5.pdb")]:
        A = parse(f"{B}/{path}")
        P = A[(~A.het)]
        M = A[A.het & A.resn.isin(METALS)]
        # protein-bound metals: >= 2 protein O within 3.0 A
        pxyz = P[["x", "y", "z"]].to_numpy()
        tp = cKDTree(pxyz)
        for ch in sorted(P.chain.unique()):
            Pc = P[P.chain == ch]
            C = centers(Pc, None, None)
            for m in M.itertuples():
                nb = tp.query_ball_point([m.x, m.y, m.z], 3.0)
                nbP = P.iloc[nb]
                if (nbP.chain == ch).sum() < 2 or (nbP.el == "O").sum() < 2:
                    continue
                if len(C):
                    d = np.linalg.norm(C[["x", "y", "z"]].to_numpy() - np.array([m.x, m.y, m.z]), axis=1)
                    w = C[d <= 2.0]
                    cap = int(w.cap.max()) if len(w) else 0
                else:
                    cap = 0
                rows.append(dict(pdb=pdb, chain=ch, metal=m.resn, metal_resi=m.resi, protein_O_within3=int((nbP.el == "O").sum()),
                                 max_capacity_within2A=cap, recovered=cap >= PASS_CONS))
    D = pd.DataFrame(rows)
    D.to_csv(B + "/data/sites_crystal.tsv", sep="\t", index=False)
    return D


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    if sys.argv[1] == "models":
        models()
    else:
        print(crystals().groupby(["pdb", "metal"]).agg(n=("recovered", "size"), recovered=("recovered", "sum"),
                                                      cap_median=("max_capacity_within2A", "median")))
