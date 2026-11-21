#!/usr/bin/env python
"""Extended tier step 6 (local driver + assembly): genes that pass the extended tier in a kept family.
usage: E6_leads.py COLLECTION [--housekeeping K1,K2]
Runs r5_context.py on biotite for the passing genes and writes leads_extended.tsv (one row per passing gene; brief columns
first). Wording: 'conserved-core residues in place'. Verdict WEAK, or CURATED-SITE when a reviewed record (accession given)
marks site positions as active or binding residues and all of them are in place in the gene; '-housekeeping' and
'-redundant' (genome carries the function by Pfam GA or picked-profile KOfam) are appended; PASS-NAMED for already-named genes."""
import argparse, json, subprocess
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("collection"); ap.add_argument("--housekeeping", default="")
a = ap.parse_args()
C = a.collection
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CODE = Path(__file__).resolve().parent
CFG = json.load(open(CODE / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
E = H / "data/ext"
RS = CFG["remote"].replace("RD/", RD + "/")
B, PY = "jwestrob@biotite.berkeley.edu", "/home/jwestrob/miniconda3/envs/elsa/bin/python"
HK = set(x for x in a.housekeeping.split(",") if x)
O = pd.read_csv(E / "readout_ext.tsv", sep="\t", dtype={"protein_id": str})
RES = pd.read_csv(E / "residues_ext.tsv", sep="\t", keep_default_na=False, dtype={"protein_id": str})
st = json.load(open(E / "stats_ext.json"))
RX = pd.read_csv(E / "references_ext.tsv", sep="\t", keep_default_na=False).set_index("func")
pop = pd.read_parquet(H / "data/population.parquet").assign(protein_id=lambda d: d.gid.astype(str))
GF = pd.read_parquet(H / "data/genes_function.parquet", columns=["gid", "func", "genome", "order"])
S = pd.read_csv(H / "data/sets_per_function.tsv", sep="\t", keep_default_na=False)
K = pd.read_csv(R / "benchmarks/redesign/dpann_leads/gapscreen/data/kegg_ko.tsv", sep="\t", keep_default_na=False).set_index("ko")
P = O[(O.role == "candidate") & (O.pair == "own") & (O.PASS == True) & O.func.isin(st["kept_families"])].copy()  # noqa: E712
P = P.merge(pop.drop(columns=["func"]), on="protein_id", how="left")
if P.empty:
    pd.DataFrame(columns=["lead"]).to_csv(H / "leads_extended.tsv", sep="\t", index=False)
    raise SystemExit("no passing genes")
li = []
for r in P.itertuples():
    s = S[(S.func == r.func) & (S.label == r.proposed_label_positional_homology)].iloc[0]
    li.append(dict(gid=int(r.protein_id), func=r.func, koset=s.koset, pfset=s.pfset))
pd.DataFrame(li).to_csv(E / "leads_in_ext.tsv", sep="\t", index=False)
subprocess.run(["rsync", "-a", str(E / "leads_in_ext.tsv"), f"{B}:{RS}/data/"], check=True)
subprocess.run(["rsync", "-a", f"{CODE}/", f"{B}:{RS}/code/"], check=True)
subprocess.run(["ssh", B, f"cd {RS} && {PY} code/r5_context.py --collection {C} --leads {RS}/data/leads_in_ext.tsv --threads 4"], check=True)
subprocess.run(["rsync", "-a", f"{B}:{RS}/data/leads_context.tsv", str(E / "leads_context_ext.tsv")], check=True)
subprocess.run(["rsync", "-a", f"{B}:{RS}/data/leads_neighbors.tsv", str(E / "leads_neighbors_ext.tsv")], check=True)
X = pd.read_csv(E / "leads_context_ext.tsv", sep="\t", keep_default_na=False).assign(protein_id=lambda d: d.gid.astype(str))
P = P.merge(X.drop(columns=["gid", "genome"]), on="protein_id", how="left", suffixes=("", "_ctx"))
rows = []
for r in P.itertuples():
    f = r.func
    rr = RES[RES.pair_id == r.pair_id]
    cur = rr[rr.kind == "curated"]
    if r.named:
        verdict = "PASS-NAMED"
    elif len(cur) and cur.in_place.astype(str).eq("True").all():
        verdict = "CURATED-SITE"
    else:
        verdict = "WEAK"
    if not r.named:
        if f in HK:
            verdict += "-housekeeping"
        elif str(r.function_absent_otherwise) in ("False", "0"):
            verdict += "-redundant"
    rel = GF[GF.func == f]
    rel_o = rel[rel.order == r.order]
    rows.append(dict(lead=r.protein_id, family_label=r.proposed_label_name, clade=r.order, genes=len(rel_o), genomes=rel_o.genome.nunique(),
                     clade_genomes=r.order_genomes, median_length=r.seq_len, tier_T1=0, support_median=round(r.support, 3),
                     catalytic_residues_ok=f"conserved-core {r.site} in place" + (f"; curated {int(cur.in_place.astype(str).eq('True').sum())}/{len(cur)} ({RX.loc[f, 'curated_record']})" if len(cur) else "; no curated residue in the site"),
                     structure_TM_to_reference=round(r.TM_ref, 3), function_absent_otherwise=r.function_absent_otherwise,
                     embedding_only_share="n/a (no alignment lists)", verdict=verdict,
                     one_line_reason="conserved-core residues in place (extended tier); activity UNVERIFIED",
                     function=f"{f} {K.name.get(f, '')[:70]}", phylum=r.phylum, genus=r.genus, species=r.species, genome=r.genome,
                     tier=r.tier, votes=f"{r.votes_for}/{r.supporting_genomes}", site_plddt=r.site_plddt,
                     TM_experimental=r.TM_exp, reference_member=RX.loc[f, "ref_member"], exp_chain=RX.loc[f, "exp_chain"],
                     other_kofam_cutga=r.other_kofam_cutga, other_pfam_ga=r.other_pfam_ga,
                     order_genomes_with_family_pfam=r.order_genomes_with_family_pfam,
                     related_labeled=f"{len(rel)} genes / {rel.genome.nunique()} genomes", neighbors=r.neighbors, ilvfmawc=r.ilvfmawc))
L = pd.DataFrame(rows).sort_values(["verdict", "function"])
L.to_csv(H / "leads_extended.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(L[["lead", "function", "clade", "genus", "structure_TM_to_reference", "catalytic_residues_ok", "function_absent_otherwise", "verdict"]].to_string(index=False))
