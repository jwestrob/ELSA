"""One-table summary of the POST HOC scoring (local; reads score_modules.tsv, module_counts.tsv, groups_by_module.tsv).
Writes first_scoring_posthoc.tsv: per module, instances, genera, genome groups, shift-null instances, then for each
output recovered / resolved (chance in brackets) and excess [95%]."""
import pandas as pd
R = pd.read_csv("score_modules.tsv", sep="\t")
C = pd.read_csv("module_counts.tsv", sep="\t").set_index("module")
G = pd.read_csv("groups_by_module.tsv", sep="\t").set_index("module")
short = {"SECONDARY_embedding_list_tree_Z_L4bp_t0.02": "SEC", "PRIMARY_family_bond_tree_WA_L4b_t0.2": "PRI",
         "engine_DH": "DH", "synton_faithful": "SYN"}
rows = []
for m, g in R.groupby("module", sort=False):
    r = dict(module=m, instances=int(C.loc[m, "instances"]), genera=int(C.loc[m, "genera"]),
             genome_groups=int(G.loc[m, "genome_groups"]), shift_null_mean=C.loc[m, "null_mean"])
    for _, x in g.iterrows():
        k = short[x.output]
        r[f"{k}_recovered"] = f"{x.recovered} ({x.chance_recovered:.0f})"
        r[f"{k}_resolved"] = f"{x.resolved} ({x.chance_resolved:.0f})"
        r[f"{k}_excess"] = f"{x.excess:+.3f} [{x.excess_lo:+.3f},{x.excess_hi:+.3f}]"
    rows.append(r)
T = pd.DataFrame(rows)
T.insert(0, "status", "POST HOC (lut collection already used as the frozen untouched test)")
T.to_csv("first_scoring_posthoc.tsv", sep="\t", index=False)
print(T.drop(columns="status").to_string())
