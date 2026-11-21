"""Genome overlap between the earlier Spicy LanMs genome set (2,547 assemblies in its Sharur database; GTDB r220
taxonomy file) and the 1,032-genome lut collection (GTDB r232 species representatives).
Levels: same assembly (accession number, GCA/GCF and version ignored), same GTDB r232 species cluster (via
sp_clusters.tsv membership), shared genus and family (GTDB r232 names where the assembly is in a r232 cluster, else
the r220 names of the taxonomy file). Read-only. Writes overlap_*.tsv and overlap.json in the working directory."""
import json, re
import duckdb
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PKG = RD + "/slice_lut/testset/package_k8"
TAX = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/spicy_lams/all_genomes_taxonomy.tsv"
DB = "/groups/banfield/users/jwestrob/bin/Sharur/data/spicy_lams/sharur.duckdb"
SPC = "/groups/banfield/projects/environmental/sr/srvp2020/gtdb/rp16_run_r232/sp_clusters.tsv"

num = lambda a: re.sub(r"^(RS_|GB_)?GC[AF]_", "", str(a)).split(".")[0]
con = duckdb.connect(DB, read_only=True)
bins = con.execute("select bin_id from bins").fetchdf()
bins["acc"] = bins.bin_id.str.extract(r"^(GC[AF]_\d+\.\d+)")[0]
bins["num"] = bins.acc.map(num)
pr = con.execute("select bin_id, count(*) n from proteins group by 1").fetchdf()
bins = bins.merge(pr, on="bin_id", how="left")
tax = pd.read_csv(TAX, sep="\t")
tax["num"] = tax.assembly_acc.map(num)
sp = pd.read_csv(SPC, sep="\t", usecols=["Representative genome", "GTDB taxonomy", "Clustered genomes"])
rows = []
for rep, tx, mem in sp.itertuples(index=False):
    for m in str(mem).split(","):
        rows.append((num(m), num(rep), tx))
memb = pd.DataFrame(rows, columns=["num", "rep_num", "r232_tax"]).drop_duplicates("num")
gm = pd.read_parquet(PKG + "/universe/genomes.parquet")
gm["num"] = gm.genome.map(num)
gm["genus"] = gm.tax.str.extract(r"g__([^;]*)")[0]
gm["family"] = gm.tax.str.extract(r"f__([^;]*)")[0]

S = bins.merge(tax[["num", "source", "family", "genus", "species"]].drop_duplicates("num"), on="num", how="left")
S = S.merge(memb, on="num", how="left")
S["family_r232"] = S.r232_tax.str.extract(r"f__([^;]*)")[0].fillna(S.family)
S["genus_r232"] = S.r232_tax.str.extract(r"g__([^;]*)")[0].fillna(S.genus)
out = dict(spicy_bins=int(len(S)), spicy_unique_accession_numbers=int(S.num.nunique()),
           spicy_bins_with_proteins=int((S.n.fillna(0) > 0).sum()),
           spicy_source_counts=S.drop_duplicates("num").source.value_counts(dropna=False).to_dict(),
           spicy_in_r232_cluster=int(S.drop_duplicates("num").rep_num.notna().sum()))
U = S.drop_duplicates("num")
lut_nums = set(gm.num)
U = U.assign(same_assembly=U.num.isin(lut_nums), same_species_cluster=U.rep_num.isin(lut_nums))
out["spicy_assemblies_that_are_lut_genomes"] = int(U.same_assembly.sum())
out["spicy_assemblies_in_a_lut_species_cluster"] = int(U.same_species_cluster.sum())
lut_hit_species = set(U.loc[U.same_species_cluster, "rep_num"])
gm["spicy_same_assembly"] = gm.num.isin(set(U.num))
gm["spicy_species_cluster"] = gm.num.isin(lut_hit_species)
out["lut_genomes_that_are_spicy_assemblies"] = int(gm.spicy_same_assembly.sum())
out["lut_genomes_whose_species_cluster_holds_a_spicy_assembly"] = int(gm.spicy_species_cluster.sum())
out["lut_by_layer"] = gm.groupby("layer").agg(n=("genome", "size"), same_assembly=("spicy_same_assembly", "sum"),
                                              species_cluster=("spicy_species_cluster", "sum")).reset_index().to_dict("records")
sg, sf = set(U.genus_r232.dropna()), set(U.family_r232.dropna())
lg, lf = set(gm.genus.dropna()), set(gm.family.dropna())
out["genera"] = dict(spicy=len(sg), lut=len(lg), shared=len(sg & lg),
                     lut_genomes_in_shared_genera=int(gm.genus.isin(sg).sum()))
out["families"] = dict(spicy=len(sf), lut=len(lf), shared=len(sf & lf), shared_list=sorted(sf & lf),
                       lut_genomes_in_shared_families=int(gm.family.isin(sf).sum()))
fam = gm.groupby("family").agg(lut_genomes=("genome", "size"), lut_same_assembly=("spicy_same_assembly", "sum"),
                               lut_species_cluster=("spicy_species_cluster", "sum")).reset_index()
fam = fam.merge(U.groupby("family_r232").size().rename("spicy_assemblies").reset_index().rename(columns={"family_r232": "family"}),
                on="family", how="outer").fillna(0).sort_values(["lut_genomes", "spicy_assemblies"], ascending=False)
fam.to_csv("overlap_families.tsv", sep="\t", index=False)
gen = gm.groupby("genus").agg(lut_genomes=("genome", "size"), lut_same_assembly=("spicy_same_assembly", "sum"),
                              lut_species_cluster=("spicy_species_cluster", "sum")).reset_index()
gen = gen.merge(U.groupby("genus_r232").size().rename("spicy_assemblies").reset_index().rename(columns={"genus_r232": "genus"}),
                on="genus", how="outer").fillna(0)
gen = gen[(gen.lut_genomes > 0) & (gen.spicy_assemblies > 0)].sort_values("lut_genomes", ascending=False)
gen.to_csv("overlap_genera_shared.tsv", sep="\t", index=False)
gm.loc[gm.spicy_same_assembly | gm.spicy_species_cluster,
       ["genome", "genome_code", "layer", "family", "genus", "spicy_same_assembly", "spicy_species_cluster"]].to_csv(
    "overlap_lut_genomes.tsv", sep="\t", index=False)
json.dump(out, open("overlap.json", "w"), indent=1, default=int)
print(json.dumps({k: v for k, v in out.items() if k != "families"}, indent=1, default=int))
print({k: v for k, v in out["families"].items() if k != "shared_list"})
print(fam.head(25).to_string())
