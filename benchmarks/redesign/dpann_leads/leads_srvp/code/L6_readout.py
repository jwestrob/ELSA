#!/usr/bin/env python
"""Step 5: residue readout and PASS call per US-align pair (rule: assay_rule.json).
In place = model residue identical to the curated UniProt residue and the aligned pair marked ':' (C-alpha < 5 A).
PASS = TM2 (normalized by the reference family chain) >= 0.50, all catalytic in place, binding share >= 0.50, mean pLDDT of
the model residues aligned to the catalytic residues >= 70 (any catalytic residue unaligned fails).
usage: L6_readout.py COLLECTION MODE (copy of dpann_leads/invisible/code/s05_readout.py, paths only)   writes data/readout_<mode>.tsv and data/residues_<mode>.tsv; prints the control table."""
import sys
from pathlib import Path
import pandas as pd

import json
C, MODE = sys.argv[1], sys.argv[2]
H = Path(json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]["local_out"].replace(
    "R/", "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/"))
_C = json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]
UA = (Path(_C["fold_dir"]) if _C["fold_dir"] else H / "folds") / "usalign"     # srvp: alignments live under the fold folder
U = pd.read_csv(UA / f"usalign_{MODE}.tsv", sep="\t", keep_default_na=False)
S = pd.read_csv(H / f"data/selection_{MODE}.tsv", sep="\t")
F = pd.read_csv(H / "data/ref_features.tsv", sep="\t", keep_default_na=False)
REFS = pd.read_csv(H / "data/references_testable.tsv", sep="\t").set_index("func")
path_of = dict(zip(S.protein_id.astype(str), S.fold_path))
U["protein_id"] = U.protein_id.astype(str)


def ca_list(p):
    out = []
    for l in open(p):
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            out.append((int(l[22:26]), float(l[60:66])))
    return out


refca = {}
for f in REFS.index:
    ca = ca_list(H / REFS.loc[f, "ref_file"])
    refca[f] = {resnum - 1: i for i, (resnum, b) in enumerate(ca)}     # trimmed chain index (0-based) -> CA-list position
TM_MIN, BIND_MIN, PL_MIN = 0.50, 0.50, 70.0
out, res_rows = [], []
for r in U.itertuples():
    rec = dict(pair_id=r.pair_id, protein_id=r.protein_id, func=r.func, role=r.role, pair=r.pair, ref_func=r.ref_func)
    if r.TM2 == "" or not r.aln_ref:
        rec["status"] = "no_alignment"; out.append(rec); continue
    plddt = [b for _, b in ca_list(path_of[r.protein_id])]
    am, mk, ar = r.aln_model, r.aln_marks, r.aln_ref
    col_of_ref, model_at_col = {}, {}
    i1 = i2 = -1
    for cidx, (x, y) in enumerate(zip(am, ar)):
        if x != "-":
            i1 += 1
            model_at_col[cidx] = i1
        if y != "-":
            i2 += 1
            col_of_ref[i2] = cidx
    feats = F[(F.func == r.ref_func) & (F.chain_index != "unresolved")]
    cat_ok, cat_n, bind_ok, bind_n, cat_pl, cat_unaligned = 0, 0, 0, 0, [], 0
    for x in feats.itertuples():
        k = int(x.chain_index)
        pos = refca[r.ref_func].get(k)
        col = col_of_ref.get(pos) if pos is not None else None
        m = am[col] if col is not None else "-"
        mark = mk[col] if col is not None else ""
        mi = model_at_col.get(col) if col is not None else None
        inplace = (m == x.residue) and mark == ":"
        if x.feature == "Active site":
            cat_n += 1; cat_ok += inplace
            if mi is None:
                cat_unaligned += 1
            else:
                cat_pl.append(plddt[mi])
        else:
            bind_n += 1; bind_ok += inplace
        res_rows.append(dict(pair_id=r.pair_id, protein_id=r.protein_id, role=r.role, pair=r.pair, ref_func=r.ref_func, feature=x.feature,
                             uniprot_pos=x.uniprot_pos, ref_residue=x.residue, ligand=x.ligand, model_residue=m, mark=mark,
                             model_pos=(mi + 1) if mi is not None else "", plddt=plddt[mi] if mi is not None else "", in_place=inplace))
    site_pl = sum(cat_pl) / len(cat_pl) if cat_pl and not cat_unaligned else 0.0
    tm2 = float(r.TM2)
    rec.update(status="ok", TM_model=float(r.TM1), TM_ref=tm2, RMSD=float(r.RMSD), Lali=int(r.Lali), mean_plddt=sum(plddt) / len(plddt),
               catalytic=f"{cat_ok}/{cat_n}", binding=f"{bind_ok}/{bind_n}", cat_all=cat_ok == cat_n and cat_n > 0,
               bind_share=bind_ok / bind_n if bind_n else 0.0, site_plddt=round(site_pl, 1),
               fold_match=tm2 >= TM_MIN)
    rec["PASS"] = bool(rec["fold_match"] and rec["cat_all"] and rec["bind_share"] >= BIND_MIN and site_pl >= PL_MIN)
    out.append(rec)
O = pd.DataFrame(out)
O.to_csv(H / f"data/readout_{MODE}.tsv", sep="\t", index=False)
pd.DataFrame(res_rows).to_csv(H / f"data/residues_{MODE}.tsv", sep="\t", index=False)
O["set"] = O.role.where(O.pair == "own", "negA")
O.loc[(O.role == "candidate") & (O.pair == "own"), "set"] = "candidate"
pd.set_option("display.width", 250)
print(O.groupby(["func", "set"]).agg(n=("PASS", "size"), PASS=("PASS", "sum"), fold=("fold_match", "sum"), cat=("cat_all", "sum"),
                                     TM_med=("TM_ref", "median")).unstack("set").to_string())
print(O.groupby("set").agg(n=("PASS", "size"), PASS=("PASS", "sum"), fold=("fold_match", "sum"), cat=("cat_all", "sum")).to_string())
