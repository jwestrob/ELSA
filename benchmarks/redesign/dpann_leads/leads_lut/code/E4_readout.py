#!/usr/bin/env python
"""Extended tier step 4 (local): site read-out and PASS per pair (assay_rule_extended.json).
In place = model residue identical to the reference-member residue at the structurally aligned position and marked ':'
(C-alpha < 5 A). PASS = TM normalized by the trimmed reference >= 0.50 AND in-place share >= 0.90 (all positions when 5 or
fewer) AND mean pLDDT of the aligned model residues at the site >= 70. TM to the experimental chain is attached (not used).
usage: E4_readout.py COLLECTION   writes data/ext/readout_ext.tsv and data/ext/residues_ext.tsv"""
import json, sys
from pathlib import Path
import pandas as pd

C = sys.argv[1]
H = Path(json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]["local_out"].replace(
    "R/", "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/"))
E = H / "data/ext"
U = pd.read_csv(E / "usalign_ext.tsv", sep="\t", keep_default_na=False, dtype={"protein_id": str})
S = pd.read_csv(E / "selection_ext.tsv", sep="\t", dtype={"protein_id": str})
F = pd.read_csv(E / "ref_features.tsv", sep="\t", keep_default_na=False)
RX = pd.read_csv(E / "references_ext.tsv", sep="\t", keep_default_na=False).set_index("func")
path_of = dict(zip(S.protein_id, S.fold_path))


def ca_list(p):
    return [(int(l[22:26]), float(l[60:66])) for l in open(p) if l.startswith("ATOM") and l[12:16].strip() == "CA"]


refca = {f: {num - 1: i for i, (num, b) in enumerate(ca_list(RX.loc[f, "ref_file"]))} for f in RX.index}
exp_tm = {(r.protein_id, r.func): r.TM2 for r in U[U.pair == "exp"].itertuples()}
out, res_rows = [], []
for r in U[U.pair.isin(["own", "negA"])].itertuples():
    rec = dict(pair_id=r.pair_id, protein_id=r.protein_id, func=r.func, role=r.role, pair=r.pair, ref_func=r.ref_func)
    if r.TM2 == "" or not r.aln_ref:
        rec["status"] = "no_alignment"; out.append(rec); continue
    plddt = [b for _, b in ca_list(path_of[r.protein_id])]
    am, mk, ar = r.aln_model, r.aln_marks, r.aln_ref
    col_of_ref, model_at_col = {}, {}
    i1 = i2 = -1
    for c, (x, y) in enumerate(zip(am, ar)):
        if x != "-":
            i1 += 1; model_at_col[c] = i1
        if y != "-":
            i2 += 1; col_of_ref[i2] = c
    feats = F[F.func == r.ref_func]
    ok = n = 0; pl = []
    for x in feats.itertuples():
        pos = refca[r.ref_func].get(int(x.chain_index))
        col = col_of_ref.get(pos) if pos is not None else None
        m = am[col] if col is not None else "-"
        mark = mk[col] if col is not None else ""
        mi = model_at_col.get(col) if col is not None else None
        inplace = (m == x.residue) and mark == ":"
        n += 1; ok += inplace
        if mi is not None:
            pl.append(plddt[mi])
        res_rows.append(dict(pair_id=r.pair_id, protein_id=r.protein_id, role=r.role, pair=r.pair, ref_func=r.ref_func, kind=x.kind,
                             member_pos=x.uniprot_pos, ref_residue=x.residue, note=x.ligand, model_residue=m, mark=mark,
                             model_pos=(mi + 1) if mi is not None else "", plddt=plddt[mi] if mi is not None else "", in_place=inplace))
    share = ok / n if n else 0.0
    need = 1.0 if n <= 5 else 0.9
    site_pl = sum(pl) / len(pl) if pl else 0.0
    tm2 = float(r.TM2)
    rec.update(status="ok", TM_model=float(r.TM1), TM_ref=tm2, RMSD=float(r.RMSD), Lali=int(r.Lali), mean_plddt=sum(plddt) / len(plddt),
               site=f"{ok}/{n}", site_share=round(share, 3), site_plddt=round(site_pl, 1), fold_match=tm2 >= 0.5,
               site_ok=share >= need - 1e-9, TM_exp=exp_tm.get((r.protein_id, r.func), "") if r.pair == "own" else "")
    rec["PASS"] = bool(rec["fold_match"] and rec["site_ok"] and site_pl >= 70)
    out.append(rec)
O = pd.DataFrame(out)
O.to_csv(E / "readout_ext.tsv", sep="\t", index=False)
pd.DataFrame(res_rows).to_csv(E / "residues_ext.tsv", sep="\t", index=False)
O["set"] = O.role.where(O.pair == "own", "negA")
O.loc[(O.role == "candidate") & (O.pair == "own"), "set"] = "candidate"
pd.set_option("display.width", 250)
print(O.groupby(["func", "set"]).agg(n=("PASS", "size"), PASS=("PASS", "sum"), fold=("fold_match", "sum"), site=("site_ok", "sum")).unstack("set").to_string())
print(O.groupby("set").agg(n=("PASS", "size"), PASS=("PASS", "sum"), fold=("fold_match", "sum"), site=("site_ok", "sum")).to_string())
