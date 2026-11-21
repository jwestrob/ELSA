#!/usr/bin/env python3
"""gapscreen workup (Mac): for the five leads taken through rungs 2-5, pick three labeled DPANN genes (lead clade, tier T1,
gap genomes first, best weak-hit E-value, distinct sequences), a positive control (DPANN protein with a stored row for the
family at its gathering / KO threshold and the KO) and a negative control (DPANN protein of similar length from an
unrelated family with a different fold); write the fold input, reference chains, and the rung-3 residue table from a
mafft alignment with the curated UniProt record (features fetched by accession into ref/).
Writes data/workup_genes.tsv, data/fold_input.faa, data/rung3_residues.tsv, ref/<pdb>_A.pdb, data/aln_<lead>.faa."""
import json
import subprocess
from pathlib import Path

import duckdb
import pandas as pd

H = Path(__file__).resolve().parents[1]
D, REF = H / "data", H / "ref"
LEADS = {  # lead: (KO, clade, family label, positive-control Pfam (None = KO row only), UniProt reference, PDB, negative family)
    "L1_dihydroorotase": ("K01465", "Micrarchaeales", "Amidohydro_1", "Amidohydro_1", "P05020", "1J79", "Glycos_transf_1"),
    "L2_beta_carbonic_anhydrase": ("K01673", "Micrarchaeales", "Pro_CA", "Pro_CA", "P61517", "1I6P", "Methyltransf_25"),
    "L3_ADP_serine_kinase": ("K23370", "Anstonellales", "L-serine kinase (ADP)", None, "Q5JD03", "5X0B", "Methyltransf_11"),
    "L4_small_alarmone_synthetase": ("K07816", "Woesearchaeales", "RelA_SpoT", "RelA_SpoT", "O31611", "5DEC", "Glycos_transf_1"),
    "L5_gamma_glutamylcysteine_ligase_GCS2": ("K01919", "Woesearchaeales", "GCS2", "GCS2", "P77213", "1R8G", "Aminotran_1_2"),
}
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
S = pd.read_parquet(D / "genes_function.parquet")
rows = []
for lead, (ko, clade, lab, pf, up, pdb, neg) in LEADS.items():
    g = S[(S.ko_top == ko) & (S.order == clade) & (S.tier == "T1")].copy()
    seq = dict(con.execute("SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN (SELECT unnest(?::VARCHAR[]))",
                           [g.protein_id.tolist()]).fetchall())
    g["seq"] = g.protein_id.map(seq)
    g = g.sort_values(["gap", "proposed_evalue"], ascending=[False, True]).drop_duplicates("seq")
    for r in g.head(3).itertuples():
        rows.append(dict(lead=lead, role="labeled", protein_id=r.protein_id, bin_id=r.bin_id, length=r.length, gap=r.gap,
                         tier=r.tier, evalue=r.proposed_evalue, seq=r.seq.rstrip("*")))
    L0 = int(g.length.median())
    q = ("""SELECT p.protein_id, p.bin_id, p.sequence_length AS length, p.sequence FROM d.proteins p
            WHERE p.protein_id IN (SELECT protein_id FROM d.annotations WHERE source IN ('kofam','kegg') AND accession = ?)""" +
         (""" AND p.protein_id IN (SELECT protein_id FROM d.annotations WHERE source = 'pfam' AND "name" = ?)""" if pf else ""))
    pc = con.execute(q, [ko, pf] if pf else [ko]).df()
    pc["clade"] = pc.bin_id.map(pd.read_parquet(D / "bins.parquet").set_index("bin_id")["order"])
    pc["d"] = (pc.length - L0).abs()
    pc = pc.sort_values(["clade", "d", "protein_id"], key=lambda s: s.ne(clade) if s.name == "clade" else s).head(1)
    for r in pc.itertuples():
        rows.append(dict(lead=lead, role="positive_control", protein_id=r.protein_id, bin_id=r.bin_id, length=r.length,
                         seq=r.sequence.rstrip("*"), clade=r.clade))
    nc = con.execute("""SELECT p.protein_id, p.bin_id, p.sequence_length AS length, p.sequence FROM d.proteins p
                        WHERE p.protein_id IN (SELECT protein_id FROM d.annotations WHERE source = 'pfam' AND "name" = ?)
                        AND p.sequence_length BETWEEN ? AND ? ORDER BY hash(p.protein_id || 'gapscreen') LIMIT 1""",
                     [neg, int(L0 * 0.85), int(L0 * 1.15)]).df()
    for r in nc.itertuples():
        rows.append(dict(lead=lead, role="negative_control", protein_id=r.protein_id, bin_id=r.bin_id, length=r.length,
                         seq=r.sequence.rstrip("*"), neg_family=neg))
W = pd.DataFrame(rows)
W.to_csv(D / "workup_genes.tsv", sep="\t", index=False)
with open(D / "fold_input.faa", "w") as f:
    for r in W.drop_duplicates("protein_id").itertuples():
        f.write(f">{r.protein_id}\n{r.seq}\n")
# reference chains (chain A, ATOM records) and rung 3 residue table
res_rows = []
for lead, (ko, clade, lab, pf, up, pdb, neg) in LEADS.items():
    lines = [l for l in open(REF / f"{pdb}.pdb") if l.startswith("ATOM") and l[21] == "A"]
    open(REF / f"{pdb}_A.pdb", "w").write("".join(lines) + "END\n")
    j = json.load(open(REF / f"{up}.json"))
    useq = j["sequence"]["value"]
    feats = [(f["type"], f["location"]["start"]["value"], f.get("ligand", {}).get("name", ""), f.get("description", ""))
             for f in j.get("features", []) if f["type"] in ("Active site", "Binding site")]
    sub = W[(W.lead == lead) & (W.role != "negative_control")]
    fa = D / f"aln_{lead}.in.faa"
    with open(fa, "w") as f:
        f.write(f">{up}\n{useq}\n")
        for r in sub.itertuples():
            f.write(f">{r.protein_id}\n{r.seq}\n")
    aln = subprocess.run(["mafft", "--quiet", "--localpair", "--maxiterate", "1000", "--thread", "2", str(fa)],
                         capture_output=True, text=True, check=True).stdout
    open(D / f"aln_{lead}.faa", "w").write(aln)
    A, k = {}, None
    for l in aln.splitlines():
        if l.startswith(">"):
            k = l[1:].split()[0]
            A[k] = ""
        else:
            A[k] += l.strip()
    ref = A[up]
    col_of, pos = {}, 0
    for c, ch in enumerate(ref):
        if ch != "-":
            pos += 1
            col_of[pos] = c
    for t, p, ligand, desc in feats:
        c = col_of[p]
        rec = dict(lead=lead, reference=up, feature=t, position=p, ligand=ligand or desc, ref_residue=ref[c].upper())
        for r in sub.itertuples():
            s = A[r.protein_id]
            ch = s[c].upper()
            prefix = (r.role[:3] + ":") if r.role != "labeled" else ""
            rec[prefix + r.protein_id] = ch
        res_rows.append(rec)
    if not feats:
        res_rows.append(dict(lead=lead, reference=up, feature="none in curated record", position=None, ligand="", ref_residue=""))
R = pd.DataFrame(res_rows)
R.to_csv(D / "rung3_residues.tsv", sep="\t", index=False)
print(W[["lead", "role", "protein_id", "bin_id", "length"]].to_string(index=False))
print(len(W.drop_duplicates("protein_id")), "sequences to fold")
