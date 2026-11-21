"""PI summary table (local): larger modules plus three two-gene references from the first pass, sorted by module size
(median members, largest first). Columns: genes (median [min-max]), instances, genome groups, shift-null mean, then
recovered / resolved (chance) for SEC, PRI, DH, SYN. Writes pi_summary_posthoc.tsv. All values POST HOC."""
import pandas as pd
L = pd.read_csv("large_scores.tsv", sep="\t"); C = pd.read_csv("large_counts.tsv", sep="\t").set_index("module")
F = pd.read_csv("score_modules.tsv", sep="\t"); FC = pd.read_csv("module_counts.tsv", sep="\t").set_index("module")
FG = pd.read_csv("groups_by_module.tsv", sep="\t").set_index("module")
F["output"] = F.output.map({"SECONDARY_embedding_list_tree_Z_L4bp_t0.02": "SEC", "PRIMARY_family_bond_tree_WA_L4b_t0.2": "PRI",
                            "engine_DH": "DH", "synton_faithful": "SYN"})
rows = []
def add(m, g, genes, med, inst, groups, null):
    r = dict(module=m, genes=genes, _med=med, instances=inst, genome_groups=groups, shift_null_mean=null)
    for _, x in g.iterrows():
        r[x.output] = f"{x.recovered} ({x.chance_recovered:.0f}) / {x.resolved} ({x.chance_resolved:.0f})"
    rows.append(r)
for m, g in L.groupby("module", sort=False):
    mn, md, mx = C.loc[m, "members_min_median_max"].split("/")
    add(m, g, f"{md} [{mn}-{mx}]", int(md), int(C.loc[m, "instances"]), int(C.loc[m, "genome_groups"]), C.loc[m, "null_mean"])
for m in ["M03_pqqdh_rieske", "M04_pqqdh_lactamaseB", "M06_pqqdh_ankyrin"]:
    add(m, F[F.module == m], "2", 2, int(FC.loc[m, "instances"]), int(FG.loc[m, "genome_groups"]), FC.loc[m, "null_mean"])
T = pd.DataFrame(rows).sort_values(["_med", "instances"], ascending=[False, False]).drop(columns="_med")
T.insert(0, "status", "POST HOC")
T.to_csv("pi_summary_posthoc.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(T.drop(columns="status").to_string(index=False))
P = pd.read_parquet("large_score_per_instance.parquet")
print(P[(P.output == "SEC") & (P.family_groups == 1) & (P.n_members >= 3)].module.value_counts().to_dict())
