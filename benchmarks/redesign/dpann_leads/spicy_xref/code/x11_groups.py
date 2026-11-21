"""Genome-group spread of each module (frozen run genome groups, emb_hom 0.75 rule) and recovery split by whether
an instance's module also occurs in another genome group (lists exclude a gene's own group, so a module confined to
one group has no list partners). Reads module_instances.parquet and score_per_instance.parquet. Writes
groups_by_module.tsv and recovery_by_group_spread.tsv."""
import pandas as pd
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
G = pd.read_parquet(RD + "/long_loci/frozen/groups/emb_hom_average_t0.75.parquet")
print(G.columns.tolist(), G.head(2).to_dict("records"))
gcol = [c for c in G.columns if c not in ("genome", "genome_code")][0]
key = "genome" if "genome" in G.columns else "genome_code"
I = pd.read_parquet("module_instances.parquet")
I["group"] = I[key].map(dict(zip(G[key], G[gcol])))
by = I.groupby("module").agg(instances=("instance_id", "size"), genome_groups=("group", "nunique"),
                             largest_group_share=("group", lambda s: s.value_counts(normalize=True).iloc[0]))
by.to_csv("groups_by_module.tsv", sep="\t")
print(by.to_string())
ng = I.groupby("module").group.nunique()
cnt = I.groupby(["module", "group"]).size()
I["other_groups"] = [ng[m] - 1 for m in I.module]
P = pd.read_parquet("score_per_instance.parquet").merge(I[["instance_id", "other_groups", "group"]], on="instance_id")
P["spread"] = (P.other_groups > 0).map({True: "module in >= 2 groups", False: "module in 1 group"})
# instance-level: does another genome group carry the same module?
mg = I.groupby("module").group.apply(set)
P["module_elsewhere"] = [len(mg[m] - {g}) > 0 for m, g in zip(P.module, P.group)]
S = P.groupby(["output", "module", "module_elsewhere"]).agg(n=("instance_id", "size"), recovered=("recovered", "sum"),
                                                            resolved=("resolved", "sum")).reset_index()
S.to_csv("recovery_by_group_spread.tsv", sep="\t", index=False)
print(S[S.output.str.startswith("SECONDARY")].to_string())
