#!/usr/bin/env python
"""Leads, both assays (local driver + assembly; replaces L8_leads.py's assembly for srvp, using the same remote context code).
usage: E5_leads.py COLLECTION [--housekeeping K1,K2]
Rows: every gene that passes the fixed assay in an informative family, and every gene that passes the extended tier in a kept
family (both novelty strata). Remote: r5_context.py (taxonomy, completeness, +-5 neighbors with Pfam names, picked-profile
KOfam search of the lead genomes for KOSET = the KO + KOs sharing its EC, Pfam GA rows of PFSET) and r7_gold.py (DefenseFinder /
TXSScan membership of the gene and its neighbors).
Verdicts. Fixed assay: SUPPORTED (unannotated, function otherwise absent, not housekeeping), SUPPORTED-housekeeping,
SUPPORTED-redundant (genome carries the function by Pfam GA or picked KOfam), PASS-NAMED (already named). Extended tier:
WEAK (conserved-core residues in place; no curated record identifies the site residues as catalytic or binding),
WEAK-redundant, WEAK-housekeeping, PASS-NAMED-ext. Columns 'curated_site_in_place' list curated positions that are in place.
Writes leads.tsv (brief columns first), data/leads_context_all.tsv, data/leads_neighbors_all.tsv, data/leads_gold.tsv,
data/neighbor_recurrence.tsv."""
import argparse, json, subprocess
from collections import Counter
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
RS = CFG["remote"].replace("RD/", RD + "/")
B, PY = "jwestrob@biotite.berkeley.edu", "/home/jwestrob/miniconda3/envs/elsa/bin/python"
HK = set(x for x in a.housekeeping.split(",") if x)
K = pd.read_csv(R / "benchmarks/redesign/dpann_leads/gapscreen/data/kegg_ko.tsv", sep="\t", keep_default_na=False).set_index("ko")
pop = pd.read_parquet(H / "data/population.parquet")
GF = pd.read_parquet(H / "data/genes_function.parquet", columns=["gid", "func", "tier", "genome", "order"])
GT = pd.read_csv(H / "data/genomes_tax.tsv", sep="\t").set_index("genome")
S = pd.read_csv(H / "data/sets_per_function.tsv", sep="\t", keep_default_na=False)
# fixed assay passes
O = pd.read_csv(H / "data/readout_full.tsv", sep="\t", dtype={"protein_id": str})
inf = set(json.load(open(H / "data/stats.json"))["informative_families"])
RES = pd.read_csv(H / "data/residues_full.tsv", sep="\t", keep_default_na=False, dtype={"protein_id": str})
REFS = pd.read_csv(H / "data/references_testable.tsv", sep="\t").set_index("func")
fx = O[(O.role == "candidate") & (O.pair == "own") & (O.PASS == True) & O.func.isin(inf)].copy()  # noqa: E712
fx["assay"] = "fixed"
fx["site_text"] = [" ".join(f"{x.ref_residue}{x.uniprot_pos}->{x.model_residue}{x.model_pos}" for x in RES[(RES.pair_id == p) & (RES.feature == "Active site")].itertuples())
                   for p in fx.pair_id]
fx["site_detail"] = fx.binding.map(lambda b: f"binding {b}")
fx["reference"] = [f"{REFS.loc[f, 'ref_acc']} {REFS.loc[f, 'pdb_chain']}" for f in fx.func]
fx["curated_site_in_place"] = ""
fx["TM_exp"] = None
# extended tier passes
EO = pd.read_csv(H / "data/ext/readout.tsv", sep="\t", dtype={"protein_id": str})
EF = pd.read_csv(H / "data/ext/families_ext.tsv", sep="\t")
kept = set(EF[EF.kept].func)
ERES = pd.read_csv(H / "data/ext/residues.tsv", sep="\t", keep_default_na=False, dtype={"protein_id": str})
ex = EO[(EO.set == "candidate") & (EO.PASS == True) & EO.func.isin(kept)].copy()  # noqa: E712
ex["assay"] = "extended"
ex["site_text"] = ex.site.map(lambda s: f"conserved-core {s} in place")
ex["site_detail"] = ""
efi = EF.set_index("func")
ex["reference"] = [f"member {efi.loc[f, 'ref_member']} (ESMFold2, collection) | exp {efi.loc[f, 'exp_chain'] if isinstance(efi.loc[f, 'exp_chain'], str) else '-'}" for f in ex.func]
ex["curated_site_in_place"] = [" ".join(f"{x.ref_residue}{x.member_pos}({x.note.split()[0]} {x.note.split()[1] if len(x.note.split()) > 1 else ''})"
                                        for x in ERES[(ERES.pair_id == p) & (ERES.kind == "curated") & (ERES.in_place == True)].itertuples())  # noqa: E712
                               for p in ex.pair_id]
P = pd.concat([fx, ex], ignore_index=True)
P = P.merge(pop.drop(columns=["func", "protein_id"]).assign(protein_id=pop.gid.astype(str)).drop(columns=["gid"]), on="protein_id", how="left")
if P.empty:
    raise SystemExit("no passing genes")
li = []
for r in P.drop_duplicates("protein_id").itertuples():
    s = S[(S.func == r.func) & (S.label == r.proposed_label_positional_homology)].iloc[0]
    li.append(dict(gid=int(r.protein_id), func=r.func, koset=s.koset, pfset=s.pfset))
pd.DataFrame(li).to_csv(H / "data/leads_in_all.tsv", sep="\t", index=False)
(H / "data/leads_gids.txt").write_text("\n".join(str(x["gid"]) for x in li) + "\n")
subprocess.run(["rsync", "-a", str(H / "data/leads_in_all.tsv"), str(H / "data/leads_gids.txt"), f"{B}:{RS}/data/"], check=True)
subprocess.run(["rsync", "-a", str(CODE) + "/", f"{B}:{RS}/code/"], check=True)
subprocess.run(["ssh", B, f"cd {RS} && {PY} code/r5_context.py --collection {C} --leads {RS}/data/leads_in_all.tsv --threads 4 && "
                          f"{PY} code/r7_gold.py --collection {C} --gids {RS}/data/leads_gids.txt --out {RS}/data/leads_gold.tsv > /dev/null"], check=True)
for f, g in (("leads_context.tsv", "leads_context_all.tsv"), ("leads_neighbors.tsv", "leads_neighbors_all.tsv"), ("leads_gold.tsv", "leads_gold.tsv")):
    subprocess.run(["rsync", "-a", f"{B}:{RS}/data/{f}", str(H / "data" / g)], check=True)
X = pd.read_csv(H / "data/leads_context_all.tsv", sep="\t", keep_default_na=False).rename(columns={"gid": "protein_id"})
X["protein_id"] = X.protein_id.astype(str)
GD = pd.read_csv(H / "data/leads_gold.tsv", sep="\t", keep_default_na=False).rename(columns={"gid": "protein_id"})
GD["protein_id"] = GD.protein_id.astype(str)
P = P.merge(X.drop(columns=["genome"]), on="protein_id", how="left", suffixes=("", "_ctx")).merge(GD, on="protein_id", how="left")
NB = pd.read_csv(H / "data/leads_neighbors_all.tsv", sep="\t", keep_default_na=False)
NB["lead"] = NB.lead.astype(str)
rec_rows = []
for (asy, f), d in P.groupby(["assay", "func"]):
    ids = set(d.protein_id); cnt = Counter()
    for lead, g in NB[NB.lead.isin(ids)].groupby("lead"):
        cnt.update({x for v in g.pfam for x in str(v).split(",") if x and x != "-"})
    for k, v in cnt.most_common(10):
        rec_rows.append(dict(assay=asy, func=f, pfam=k, leads_with_neighbor=f"{v}/{len(ids)}"))
pd.DataFrame(rec_rows).to_csv(H / "data/neighbor_recurrence.tsv", sep="\t", index=False)
rows = []
for r in P.itertuples():
    f = r.func
    rel = GF[GF.func == f]
    rel_o = rel[rel.order == r.order]
    named = bool(r.named)
    redundant = str(r.function_absent_otherwise) in ("False", "0")
    if r.assay == "fixed":
        v = "PASS-NAMED" if named else "SUPPORTED-housekeeping" if f in HK else "SUPPORTED-redundant" if redundant else "SUPPORTED"
    else:
        v = "PASS-NAMED-ext" if named else "WEAK-housekeeping" if f in HK else "WEAK-redundant" if redundant else "WEAK"
    o = r.order
    rows.append(dict(lead=r.protein_id, family_label=r.proposed_label_name, clade=o, genes=len(rel_o), genomes=rel_o.genome.nunique(),
                     clade_genomes=int((GT.ord == o).sum()), median_length=r.seq_len, tier_T1=0, support_median=round(r.support, 3),
                     catalytic_residues_ok=r.site_text, structure_TM_to_reference=round(r.TM_ref, 3),
                     function_absent_otherwise=r.function_absent_otherwise, embedding_only_share="n/a (no alignment lists)",
                     verdict=v, one_line_reason="", assay=r.assay, stratum="named" if named else "unannotated",
                     function=f"{f} {K.name.get(f, '')[:70]}", phylum=r.phylum, tclass=GT["class"].get(r.genome, ""),
                     family=GT.family.get(r.genome, ""), genus=GT.genus.get(r.genome, ""), genome=r.genome, tier=r.tier,
                     votes=f"{r.votes_for}/{r.supporting_genomes}", partner_ko=f"{r.ko_top} {r.ko_top_n}/{r.n_carriers_ko2} (share {r.ko_share:.2f})",
                     site_detail=r.site_detail, curated_site_in_place=r.curated_site_in_place, TM_by_model=round(r.TM_model, 3),
                     TM_experimental=r.TM_exp if r.assay == "extended" else "", site_plddt=r.site_plddt, mean_plddt=round(r.mean_plddt, 1),
                     reference=r.reference, named_detail=r.nov_detail if named else "", own_kofam_cutga=r.own_kofam_cutga,
                     other_kofam_cutga=r.other_kofam_cutga, other_pfam_ga=r.other_pfam_ga,
                     order_genomes_with_family_pfam=getattr(r, "order_genomes_with_family_pfam", ""),
                     related_labeled=f"{len(rel)} genes / {rel.genome.nunique()} genomes",
                     completeness=GT.completeness.get(r.genome, ""), clade_completeness_median=round(GT[GT.ord == o].completeness.median(), 1),
                     curated_system_own=r.own_system, curated_system_neighbors=r.neighbor_systems, neighbors=r.neighbors, ilvfmawc=r.ilvfmawc))
L = pd.DataFrame(rows)
rank = {"SUPPORTED": 0, "SUPPORTED-redundant": 1, "SUPPORTED-housekeeping": 2, "WEAK": 3, "WEAK-redundant": 4, "WEAK-housekeeping": 5,
        "PASS-NAMED": 6, "PASS-NAMED-ext": 7}
L = L.assign(_r=L.verdict.map(rank)).sort_values(["_r", "function", "structure_TM_to_reference"], ascending=[True, True, False]).drop(columns="_r")
L.to_csv(H / "leads.tsv", sep="\t", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 60)
print(L[["lead", "assay", "function", "phylum", "clade", "structure_TM_to_reference", "catalytic_residues_ok", "site_plddt",
         "function_absent_otherwise", "curated_system_own", "verdict"]].to_string(index=False))
