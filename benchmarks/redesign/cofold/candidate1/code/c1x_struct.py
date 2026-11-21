"""Task 3 (local, seconds): the B-C contact at C's conserved His and Cys, across the ABC models.

Models (deep dive, folds_c1abc/pdb): c1_abc_<slot> (single sequence), c1_abcmsa_<slot> (block-paired MSA),
c1_monoC_<slot> (C alone). Slot 4 (104-aa C, trimer ipTM 0.38) is reported separately; the 11 passing slots are
1-3 and 5-12.
His / Cys of C: C instance alignment (deepdive/work/c1_aln_out/C_inst.aln.fa) columns 76 (H, 98.4 %) and 135
(C, 96.1 %), mapped to residue numbers per instance.
Per model:
  SASA (Shrake-Rupley, Biopython, probe 1.4 A, 100 points): side-chain SASA (atoms beyond N, CA, C, O) of His and
     Cys in (i) the C-alone model, (ii) chain C cut out of the complex, (iii) B+C (A removed), (iv) A+B+C;
     relative SASA = residue SASA / Tien 2013 max (His 224, Cys 167).
  B residues with any heavy atom <= 4.5 A from a His or Cys side-chain heavy atom; their B alignment columns
     (deepdive/work/aln/c1B_inst.aln.fa) and conservation (frequency of the most common residue, 268 rows).
  Backbone H-bonds (Kabsch-Sander energy < -0.5 kcal/mol, amide H placed from the preceding C=O) within and
     between chains; DSSP-lite secondary structure (H: two consecutive i->i+4 turns; E: bridge partner) per
     chain; inter-chain B-C bridges (parallel / antiparallel) and the C residues they pair with.
  Cys SG to the nearest B backbone carbonyl carbon; His ring N to Cys SG; His CB - Cys CB; min A-atom distance to
     His / Cys side chains.
Writes work/struct/struct_models.tsv, struct_bcols.tsv, struct_summary.json.
"""
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.SeqUtils import seq1

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent.parent
DD = HERE.parent / "deepdive" / "work"
O = HERE / "work" / "struct"
O.mkdir(parents=True, exist_ok=True)
HCOL, CCOL = 76, 135
MAXASA = {"HIS": 224.0, "CYS": 167.0}
BB = {"N", "CA", "C", "O"}


def rf(p):
    d, h = {}, None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            h = line[1:].split()[0]
            d[h] = []
        elif h:
            d[h].append(line)
    return {k: "".join(v) for k, v in d.items()}


def col_map(aln_row):
    """residue index (1-based) -> alignment column"""
    m, k = {}, 0
    for c, a in enumerate(aln_row):
        if a != "-":
            k += 1
            m[k] = c
    return m


def sub(struct, chains):
    s = Structure("s")
    m = Model(0)
    s.add(m)
    for ch in struct[0]:
        if ch.id in chains:
            m.add(ch.copy())
    return s


def sc_sasa(struct, chain, resnum):
    ShrakeRupley().compute(struct[0], level="A")
    r = struct[0][chain][resnum]
    sc = sum(a.sasa for a in r if a.get_id() not in BB)
    tot = sum(a.sasa for a in r)
    return sc, tot / MAXASA[r.get_resname()]


def hbond_matrix(chains):
    """chains: list of (chain_id, residues). Returns flat residue list and E (acceptor CO i, donor NH j)."""
    res = []
    for cid, rs in chains:
        prev = None
        for r in rs:
            if not all(x in r for x in ("N", "CA", "C", "O")):
                prev = None
                continue
            N, C, O_ = r["N"].coord, r["C"].coord, r["O"].coord
            H = None
            if prev is not None and r.get_resname() != "PRO":
                v = prev["C"].coord - prev["O"].coord
                H = N + v / np.linalg.norm(v)
            res.append(dict(chain=cid, num=r.id[1], name=r.get_resname(), N=N, C=C, O=O_, H=H))
            prev = r
    n = len(res)
    Cc = np.array([x["C"] for x in res])
    Oo = np.array([x["O"] for x in res])
    Nn = np.array([x["N"] for x in res])
    Hh = np.array([x["H"] if x["H"] is not None else [np.nan] * 3 for x in res])
    rON = np.linalg.norm(Oo[:, None] - Nn[None], axis=-1)
    rCH = np.linalg.norm(Cc[:, None] - Hh[None], axis=-1)
    rOH = np.linalg.norm(Oo[:, None] - Hh[None], axis=-1)
    rCN = np.linalg.norm(Cc[:, None] - Nn[None], axis=-1)
    E = 0.084 * 332 * (1 / rON + 1 / rCH - 1 / rOH - 1 / rCN)
    E = np.nan_to_num(E, nan=0.0)
    np.fill_diagonal(E, 0.0)
    return res, E


def dssp_lite(res, E):
    n = len(res)
    hb = E < -0.5
    same = np.array([[res[i]["chain"] == res[j]["chain"] for j in range(n)] for i in range(n)])
    idx = {(x["chain"], x["num"]): i for i, x in enumerate(res)}

    def nxt(i, k):
        j = idx.get((res[i]["chain"], res[i]["num"] + k))
        return j

    ss = ["-"] * n
    turn4 = [False] * n
    for i in range(n):
        j = nxt(i, 4)
        turn4[i] = j is not None and hb[i, j]
    for i in range(1, n):
        p = nxt(i, -1)
        if p is not None and turn4[p] and turn4[i]:
            for k in range(4):
                j = nxt(i, k)
                if j is not None:
                    ss[j] = "H"
    bridges = []
    for i in range(n):
        im, ip = nxt(i, -1), nxt(i, 1)
        for j in range(n):
            if same[i, j] and abs(res[i]["num"] - res[j]["num"]) < 3:
                continue
            jm, jp = nxt(j, -1), nxt(j, 1)
            par = (im is not None and ip is not None and hb[im, j] and hb[j, ip]) or \
                  (jm is not None and jp is not None and hb[jm, i] and hb[i, jp])
            anti = (hb[i, j] and hb[j, i]) or (im is not None and ip is not None and jm is not None and jp is not None
                                                and hb[im, jp] and hb[jm, ip])
            if par or anti:
                bridges.append((i, j, "P" if par else "A"))
    for i, j, _ in bridges:
        if ss[i] != "H":
            ss[i] = "E"
    return ss, bridges, hb


def cons_table(aln):
    rows = list(aln.values())
    L = len(rows[0])
    out = {}
    for c in range(L):
        cnt = Counter(r[c] for r in rows)
        occ = 1 - cnt.get("-", 0) / len(rows)
        top = [x for x in cnt.most_common() if x[0] != "-"]
        out[c] = (top[0][0], top[0][1] / len(rows), occ) if top else ("-", 0.0, 0.0)
    return out


def main():
    sel = pd.read_csv(DD / "c1_selected.tsv", sep="\t")
    caln = rf(DD / "c1_aln_out" / "C_inst.aln.fa")
    baln = rf(DD / "aln" / "c1B_inst.aln.fa")
    bcons = cons_table(baln)
    ccons = cons_table(caln)
    P = PDBParser(QUIET=True)
    rows, brows = [], []
    for _, s in sel.iterrows():
        slot, inst = int(s.slot), s.instance_id
        cmap, bmap = col_map(caln[inst]), col_map(baln[inst])
        inv = {v: k for k, v in cmap.items()}
        hnum, cnum = inv.get(HCOL), inv.get(CCOL)
        for kind in ("abc", "abcmsa"):
            f = DD / "folds_c1abc" / "pdb" / f"c1_{kind}_{slot:02d}.pdb"
            st = P.get_structure(kind, f)
            ch = {c.id: c for c in st[0]}
            cseq = "".join(seq1(r.get_resname()) for r in ch["C"])
            assert cseq == caln[inst].replace("-", ""), (slot, "C seq mismatch")
            bseq = "".join(seq1(r.get_resname()) for r in ch["B"])
            assert bseq == baln[inst].replace("-", ""), (slot, "B seq mismatch")
            H = ch["C"][hnum] if hnum else None
            Cy = ch["C"][cnum] if cnum else None
            r = dict(slot=slot, instance=inst, kind=kind, lenC=len(cseq), lenB=len(bseq), his=hnum, cys=cnum,
                     his_aa=H.get_resname() if H else None, cys_aa=Cy.get_resname() if Cy else None)
            if H is None or Cy is None:
                rows.append(r)
                continue
            r["sep"] = cnum - hnum
            r["hisCB_cysCB"] = float(np.linalg.norm(H["CB"].coord - Cy["CB"].coord))
            r["hisN_cysSG"] = float(min(np.linalg.norm(H[a].coord - Cy["SG"].coord) for a in ("ND1", "NE2")))
            # SASA
            mono = P.get_structure("m", DD / "folds_c1abc" / "pdb" / f"c1_monoC_{slot:02d}.pdb")
            mono_ch = list(mono[0])[0].id
            for lab, struct, chid in (("mono", mono, mono_ch), ("Ccut", sub(st, {"C"}), "C"),
                                      ("BC", sub(st, {"B", "C"}), "C"), ("ABC", sub(st, {"A", "B", "C"}), "C")):
                for nm, num in (("his", hnum), ("cys", cnum)):
                    sc, rel = sc_sasa(struct, chid, num)
                    r[f"{nm}_sc_{lab}"] = round(sc, 1)
                    r[f"{nm}_rel_{lab}"] = round(rel, 3)
            for nm in ("his", "cys"):
                a, b = r[f"{nm}_sc_Ccut"], r[f"{nm}_sc_BC"]
                r[f"{nm}_buried_by_B"] = round(1 - b / a, 3) if a > 0 else None
            # contacts of B with His / Cys side chains
            bat = [(rr, a) for rr in ch["B"] for a in rr if a.element != "H"]
            aat = [a for rr in ch["A"] for a in rr if a.element != "H"]
            for nm, R in (("his", H), ("cys", Cy)):
                scat = [a for a in R if a.get_id() not in BB]
                close = {}
                for rr, a in bat:
                    d = min(np.linalg.norm(a.coord - x.coord) for x in scat)
                    if d <= 4.5:
                        k = rr.id[1]
                        close[k] = min(close.get(k, 9e9), d)
                r[f"{nm}_Bcontacts"] = ";".join(f"{seq1(ch['B'][k].get_resname())}{k}(col{bmap[k]})" for k in sorted(close))
                r[f"{nm}_minA"] = round(float(min(min(np.linalg.norm(a.coord - x.coord) for x in scat) for a in aat)), 1)
                for k, d in close.items():
                    c = bmap[k]
                    brows.append(dict(slot=slot, kind=kind, target=nm, Bres=k, Baa=seq1(ch["B"][k].get_resname()),
                                      Bcol=c, col_top=bcons[c][0], col_freq=round(bcons[c][1], 3),
                                      col_occ=round(bcons[c][2], 3), dmin=round(float(d), 2)))
            # carbonyl next to Cys SG
            best = min(((np.linalg.norm(rr["C"].coord - Cy["SG"].coord), rr.id[1]) for rr in ch["B"] if "C" in rr))
            r["cysSG_Bcarbonyl"] = round(float(best[0]), 2)
            r["cysSG_Bcarbonyl_res"] = f"{seq1(ch['B'][best[1]].get_resname())}{best[1]}(col{bmap[best[1]]})"
            # H-bonds and secondary structure
            res, E = hbond_matrix([(c, list(ch[c])) for c in ("A", "B", "C")])
            ss, bridges, hb = dssp_lite(res, E)
            ssd = {(x["chain"], x["num"]): ss[i] for i, x in enumerate(res)}
            r["ssB"] = "".join(ssd.get(("B", rr.id[1]), "-") for rr in ch["B"])
            r["ssC"] = "".join(ssd.get(("C", rr.id[1]), "-") for rr in ch["C"])
            bc = [(res[i]["num"], res[j]["num"], t) for i, j, t in bridges if res[i]["chain"] == "B" and res[j]["chain"] == "C"]
            r["BC_bridges"] = ";".join(f"B{a}-C{b}{t}" for a, b, t in sorted(bc))
            nhb = [(res[i], res[j]) for i in range(len(res)) for j in range(len(res))
                   if hb[i, j] and {res[i]["chain"], res[j]["chain"]} == {"B", "C"}]
            r["BC_bb_hbonds"] = len(nhb)
            # C strand neighbourhood of Cys: C residues within +-3 of Cys and His paired by B bridges
            r["Cys_strand_paired_by_B"] = any(abs(b - cnum) <= 3 for _, b, _ in bc)
            r["His_strand_paired_by_B"] = any(abs(b - hnum) <= 3 for _, b, _ in bc)
            r["C_ss_at_his"] = ssd.get(("C", hnum))
            r["C_ss_at_cys"] = ssd.get(("C", cnum))
            # SS of B residues touching His/Cys
            ks = sorted({int(x.split("(")[0][1:]) for x in (r["his_Bcontacts"] + ";" + r["cys_Bcontacts"]).split(";") if x})
            r["ss_of_contacting_B"] = "".join(ssd.get(("B", k), "-") for k in ks)
            rows.append(r)
    M = pd.DataFrame(rows)
    Bc = pd.DataFrame(brows)
    M.to_csv(O / "struct_models.tsv", sep="\t", index=False)
    Bc.to_csv(O / "struct_bcontacts.tsv", sep="\t", index=False)
    ok = M[(M.slot != 4) & M.his.notna() & M.cys.notna()]
    S = {}
    for kind, x in ok.groupby("kind"):
        d = {}
        for c in ["sep", "hisCB_cysCB", "hisN_cysSG", "his_rel_mono", "cys_rel_mono", "his_rel_Ccut", "cys_rel_Ccut",
                  "his_rel_BC", "cys_rel_BC", "his_rel_ABC", "cys_rel_ABC", "his_sc_Ccut", "cys_sc_Ccut", "his_sc_BC",
                  "cys_sc_BC", "his_buried_by_B", "cys_buried_by_B", "his_minA", "cys_minA", "cysSG_Bcarbonyl",
                  "BC_bb_hbonds"]:
            d[c] = [round(float(x[c].min()), 3), round(float(x[c].median()), 3), round(float(x[c].max()), 3)]
        d["Cys_strand_paired_by_B"] = int(x.Cys_strand_paired_by_B.sum())
        d["His_strand_paired_by_B"] = int(x.His_strand_paired_by_B.sum())
        d["n"] = len(x)
        d["C_ss_at_his"] = x.C_ss_at_his.value_counts().to_dict()
        d["C_ss_at_cys"] = x.C_ss_at_cys.value_counts().to_dict()
        S[kind] = d
        bk = Bc[(Bc.kind == kind) & (Bc.slot != 4)]
        S[kind + "_Bcols"] = (bk.groupby(["target", "Bcol", "col_top", "col_freq"]).slot.nunique()
                              .reset_index().sort_values(["target", "slot"], ascending=[True, False])
                              .to_dict("records"))
    S["C_cons_H"] = ccons[HCOL]
    S["C_cons_C"] = ccons[CCOL]
    json.dump(S, open(O / "struct_summary.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 120)
    print(json.dumps({k: v for k, v in S.items()}, indent=0, default=str)[:6000])
    print(M[M.kind == "abc"][["slot", "his", "cys", "sep", "hisCB_cysCB", "hisN_cysSG", "his_rel_mono", "cys_rel_mono",
                              "his_rel_BC", "cys_rel_BC", "his_buried_by_B", "cys_buried_by_B", "his_Bcontacts",
                              "cys_Bcontacts", "cysSG_Bcarbonyl_res", "cysSG_Bcarbonyl"]].to_string())
    print(M[M.kind == "abc"][["slot", "ssB", "ss_of_contacting_B", "BC_bridges", "BC_bb_hbonds", "C_ss_at_his",
                              "C_ss_at_cys"]].to_string())


if __name__ == "__main__":
    main()
