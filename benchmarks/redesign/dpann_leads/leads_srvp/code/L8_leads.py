#!/usr/bin/env python
"""Leads pipeline step 5 (local driver + assembly): every gene that passes the assay in an informative family.
usage: L8_leads.py COLLECTION [--housekeeping K00001,K00002,...]
Runs r5_context.py on biotite for the passing genes (unannotated and already-named strata), then writes leads.tsv
(brief columns first; one row per passing gene) and data/neighbor_recurrence.tsv.
Verdicts: SUPPORTED (unannotated, no other source gives the function in the genome, not housekeeping),
SUPPORTED-housekeeping (KOs passed with --housekeeping: routine functions any annotator names from context),
SUPPORTED-redundant (the genome carries the function by Pfam GA or by a picked-profile KOfam hit),
PASS-NAMED (already named by a relaxed own hit or CDD). Activity, substrate and pathway role stay UNVERIFIED."""
import argparse, json, subprocess
from collections import Counter
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("collection"); ap.add_argument("--housekeeping", default="")
a = ap.parse_args()
C = a.collection
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CFG = json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
RS = CFG["remote"].replace("RD/", RD + "/")
B = "jwestrob@biotite.berkeley.edu"
PY = "/home/jwestrob/miniconda3/envs/elsa/bin/python"
HK = set(x for x in a.housekeeping.split(",") if x)
O = pd.read_csv(H / "data/readout_full.tsv", sep="\t")
RES = pd.read_csv(H / "data/residues_full.tsv", sep="\t", keep_default_na=False)
st = json.load(open(H / "data/stats.json"))
inf = set(st["informative_families"])
pop = pd.read_parquet(H / "data/population.parquet")
GF = pd.read_parquet(H / "data/genes_function.parquet", columns=["gid", "func", "tier", "genome", "order"])
RC = pd.read_csv(H / "data/candidate_ranking.tsv", sep="\t")
REFS = pd.read_csv(H / "data/references_testable.tsv", sep="\t").set_index("func")
S = pd.read_csv(H / "data/sets_per_function.tsv", sep="\t", keep_default_na=False)
K = pd.read_csv(R / "benchmarks/redesign/dpann_leads/gapscreen/data/kegg_ko.tsv", sep="\t", keep_default_na=False).set_index("ko")
P = O[(O.role == "candidate") & (O.pair == "own") & (O.PASS == True) & O.func.isin(inf)].copy()  # noqa: E712
P = P.merge(pop.drop(columns=["func", "protein_id"]).rename(columns={"gid": "protein_id"}), on="protein_id", how="left")
if P.empty:
    pd.DataFrame(columns=["lead"]).to_csv(H / "leads.tsv", sep="\t", index=False)
    raise SystemExit("no passing genes")
li = []
for r in P.itertuples():
    s = S[(S.func == r.func) & (S.label == r.proposed_label_positional_homology)].iloc[0]
    li.append(dict(gid=r.protein_id, func=r.func, koset=s.koset, pfset=s.pfset))
pd.DataFrame(li).to_csv(H / "data/leads_in.tsv", sep="\t", index=False)
subprocess.run(["rsync", "-a", str(H / "data/leads_in.tsv"), f"{B}:{RS}/data/"], check=True)
subprocess.run(["rsync", "-a", str(Path(__file__).resolve().parent) + "/", f"{B}:{RS}/code/"], check=True)
subprocess.run(["ssh", B, f"cd {RS} && {PY} code/r5_context.py --collection {C} --leads {RS}/data/leads_in.tsv --threads 4"], check=True)
for f in ("leads_context.tsv", "leads_neighbors.tsv"):
    subprocess.run(["rsync", "-a", f"{B}:{RS}/data/{f}", str(H / "data") + "/"], check=True)
X = pd.read_csv(H / "data/leads_context.tsv", sep="\t", keep_default_na=False).rename(columns={"gid": "protein_id"})
P = P.merge(X.drop(columns=["genome"]), on="protein_id", how="left", suffixes=("", "_ctx"))
NB = pd.read_csv(H / "data/leads_neighbors.tsv", sep="\t", keep_default_na=False)
rec_rows = []
for f, d in P.groupby("func"):
    ids = set(d.protein_id)
    cnt = Counter()
    for lead, g in NB[NB.lead.isin(ids)].groupby("lead"):
        cnt.update({x for v in g.pfam for x in str(v).split(",") if x and x != "-"})
    for k, v in cnt.most_common(10):
        rec_rows.append(dict(func=f, pfam=k, leads_with_neighbor=f"{v}/{len(ids)}"))
pd.DataFrame(rec_rows).to_csv(H / "data/neighbor_recurrence.tsv", sep="\t", index=False)
rows = []
for r in P.itertuples():
    f = r.func
    rr = RES[RES.pair_id == r.pair_id]
    cat = rr[rr.feature == "Active site"]
    cats = " ".join(f"{x.ref_residue}{x.uniprot_pos}->{x.model_residue}{x.model_pos}" for x in cat.itertuples())
    rel = GF[GF.func == f] if "func" in GF.columns else GF.iloc[0:0]
    rel_o = rel[rel.order == r.order]
    named = bool(r.named)
    redundant = str(r.function_absent_otherwise) in ("False", "0")
    verdict = "PASS-NAMED" if named else "SUPPORTED-housekeeping" if f in HK else "SUPPORTED-redundant" if redundant else "SUPPORTED"
    rows.append(dict(lead=r.protein_id, family_label=r.proposed_label_name, clade=r.order, genes=len(rel_o), genomes=rel_o.genome.nunique(),
                     clade_genomes=r.order_genomes, median_length=r.seq_len, tier_T1=0, support_median=round(r.support, 3),
                     catalytic_residues_ok=cats, structure_TM_to_reference=round(r.TM_ref, 3),
                     function_absent_otherwise=r.function_absent_otherwise, embedding_only_share="n/a (no alignment lists)",
                     verdict=verdict, one_line_reason="",
                     function=f"{f} {K.name.get(f, '')[:70]}", phylum=r.phylum, tclass=r.tclass, family=r.family, genus=r.genus,
                     species=r.species, genome=r.genome, tier=r.tier, votes=f"{r.votes_for}/{r.supporting_genomes}",
                     partner_ko=f"{r.ko_top} {r.ko_top_n}/{r.n_carriers_ko2} (share {r.ko_share:.2f})", binding=r.binding,
                     TM_by_model=round(r.TM_model, 3), site_plddt=r.site_plddt, mean_plddt=round(r.mean_plddt, 1),
                     reference=f"{REFS.loc[f, 'ref_acc']} {REFS.loc[f, 'pdb_chain']}", named_detail=r.nov_detail if named else "",
                     own_kofam_cutga=r.own_kofam_cutga, other_kofam_cutga=r.other_kofam_cutga, other_pfam_ga=r.other_pfam_ga,
                     order_genomes_with_family_pfam=r.order_genomes_with_family_pfam,
                     related_labeled=f"{len(rel)} genes / {rel.genome.nunique()} genomes",
                     cluster90_size=int(RC[(RC.protein_id == r.protein_id) & (RC.func == f)].cluster_size.iloc[0]),
                     completeness=r.completeness_ctx if hasattr(r, "completeness_ctx") else r.completeness, neighbors=r.neighbors,
                     ilvfmawc=r.ilvfmawc))
L = pd.DataFrame(rows)
rank = {"SUPPORTED": 0, "SUPPORTED-redundant": 1, "SUPPORTED-housekeeping": 2, "PASS-NAMED": 3}
L = L.assign(_r=L.verdict.map(rank)).sort_values(["_r", "function", "structure_TM_to_reference"], ascending=[True, True, False]).drop(columns="_r")
L.to_csv(H / "leads.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(L[["lead", "function", "clade", "genus", "structure_TM_to_reference", "catalytic_residues_ok", "binding", "site_plddt",
         "function_absent_otherwise", "order_genomes_with_family_pfam", "verdict"]].to_string(index=False))
print(pd.DataFrame(rec_rows).to_string(index=False))
