#!/usr/bin/env python
"""galt step 1: census of the GalT-like family in the Sharur DPANN database.

Sets
  lab76   every gene with slot label PF01087 (GalP_UDP_transf) in slot_rescue/pass (all tiers).
  set64   the 64 labeled genes of the invisible workstream (its first-pass novel population, label GalP_UDP_transf,
          tier != contradicted); 47 carry a CDD COG1085 best hit ('named64'), 17 carry no family row ('unann17').
  lead15  the 15 folded, passing, unannotated genes (invisible/leads.tsv, named_by_amendment False).
  named   DPANN proteins placed in the family by a database source: Pfam PF01087 or PF02744 row (Sharur Pfam run),
          KO K00965 (kofam or kegg), or a CDD best hit to a GalT-specific model (cd00608 GalT, COG1085 GalT,
          PRK11720 galT). COG4468 (class II GalT) is recorded separately.
  ap4a    DPANN proteins with Pfam PF19327 (Ap4A_phos_N) or PF09830 (ATP_transf): Ap4A phosphorylase branch.
Writes data/census_proteins.tsv, data/census.faa (named + lab76 + ap4a), data/genomes.tsv (bin, order, completeness).
"""
from pathlib import Path
import duckdb
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
INV = H.parent / "invisible"
c = duckdb.connect()
c.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
c.execute("SET threads=4")

P = pd.read_parquet(H.parents[1] / "slot_rescue/pass/dpann_slot_rescue_pass.parquet")
lab76 = P[P.proposed_label_positional_homology == "PF01087"].copy()
nov = pd.read_parquet(INV / "data/novel.parquet")
set64 = nov[(nov.proposed_label_name == "GalP_UDP_transf") & (nov.tier != "contradicted")].copy()
L = pd.read_csv(INV / "leads.tsv", sep="\t")
lead15 = L[(L.family_label == "GalP_UDP_transf") & (~L.named_by_amendment)].lead.tolist()
assert len(set64) == 64 and len(lead15) == 15 and set(lead15) <= set(set64.protein_id)

ann = c.execute("""SELECT protein_id, source, accession FROM d.annotations
  WHERE accession IN ('PF01087','PF02744','K00965','PF19327','PF09830')""").fetchdf()
cdd = c.execute("""SELECT protein_id, accession, evalue FROM d.protein_resemblance
  WHERE accession IN ('cd00608','COG1085','PRK11720','COG4468','COG4360')""").fetchdf()
flags = {}
for r in ann.itertuples():
    flags.setdefault(r.protein_id, set()).add(f"{r.source}:{r.accession}")
for r in cdd.itertuples():
    flags.setdefault(r.protein_id, set()).add(f"cdd:{r.accession}")
GALT_KEYS = {"pfam:PF01087", "pfam:PF02744", "kofam:K00965", "kegg:K00965", "cdd:cd00608", "cdd:COG1085", "cdd:PRK11720"}
AP4A_KEYS = {"pfam:PF19327", "pfam:PF09830", "cdd:COG4360"}
ids = set(flags) | set(lab76.protein_id)
c.register("q", pd.DataFrame({"protein_id": sorted(ids)}))
prot = c.execute("""SELECT p.protein_id, p.bin_id, p.contig_id, p.gene_index, p.strand, p.sequence_length length, p.sequence
  FROM d.proteins p JOIN q USING(protein_id)""").fetchdf()
bins = c.execute("SELECT bin_id, completeness, contamination, taxonomy FROM d.bins").fetchdf()
for k, pre in [("phylum", "p__"), ("class_", "c__"), ("order", "o__"), ("family", "f__"), ("genus", "g__")]:
    bins[k] = bins.taxonomy.str.extract(pre + r"([^;]*)")[0].fillna("")
bins.drop(columns="taxonomy").to_csv(H / "data/genomes.tsv", sep="\t", index=False)
prot = prot.merge(bins[["bin_id", "completeness", "phylum", "order", "family", "genus"]], on="bin_id", how="left")
prot["fl"] = prot.protein_id.map(lambda p: " ".join(sorted(flags.get(p, []))))
prot["named"] = prot.protein_id.map(lambda p: bool(flags.get(p, set()) & GALT_KEYS))
prot["ap4a"] = prot.protein_id.map(lambda p: bool(flags.get(p, set()) & AP4A_KEYS))
prot["cog4468"] = prot.protein_id.map(lambda p: "cdd:COG4468" in flags.get(p, set()))
prot["lab76"] = prot.protein_id.isin(lab76.protein_id)
prot["set64"] = prot.protein_id.isin(set64.protein_id)
prot["lead15"] = prot.protein_id.isin(lead15)
prot["unann17"] = prot.set64 & ~prot.named
prot["pfam_row"] = prot.fl.str.contains("pfam:PF0108|pfam:PF02744")
prot["ko_row"] = prot.fl.str.contains("K00965")
prot["group"] = "other"
prot.loc[prot.named & ~prot.lab76, "group"] = "named_unlabeled"
prot.loc[prot.lab76 & prot.named, "group"] = "labeled_named"
prot.loc[prot.lab76 & ~prot.named, "group"] = "labeled_unannotated"
prot.loc[prot.lead15, "group"] = "lead15"
prot.loc[prot.ap4a & ~prot.named & ~prot.lab76, "group"] = "ap4a_branch"
lab_cols = ["protein_id", "tier_slot", "votes_for", "supporting_genomes", "alignment_label", "alignment_lists_same_label"]
lab76 = lab76.rename(columns={"basis": "basis_"})
tiers = pd.read_parquet(H.parents[1] / "slot_rescue/verify/verification_per_protein.parquet", columns=["protein_id", "tier"])
lab76 = lab76.merge(tiers.rename(columns={"tier": "tier_slot"}), on="protein_id", how="left")
prot = prot.merge(lab76[lab_cols], on="protein_id", how="left")
prot = prot[prot.group != "other"]
prot.drop(columns="sequence").to_csv(H / "data/census_proteins.tsv", sep="\t", index=False)
with open(H / "data/census.faa", "w") as fh:
    for r in prot.itertuples():
        fh.write(f">{r.protein_id}\n{r.sequence.rstrip('*')}\n")
print(prot.group.value_counts().to_string())
print("set64 named / unannotated:", int((prot.set64 & prot.named).sum()), int(prot.unann17.sum()))
print("lab76 named:", int((prot.lab76 & prot.named).sum()), "of", int(prot.lab76.sum()))
print("named flags (non-labeled):")
nn = prot[prot.group == "named_unlabeled"]
for k in sorted(GALT_KEYS):
    print(" ", k, int(nn.fl.str.contains(k, regex=False).sum()))
print("named length quartiles", nn.length.quantile([.1, .25, .5, .75, .9]).round().tolist())
print("named orders", nn.order.value_counts().head(15).to_dict())
print("lab76 orders", prot[prot.lab76].order.value_counts().to_dict())
print("ap4a", prot[prot.ap4a].order.value_counts().to_dict(), prot[prot.ap4a].length.tolist())
