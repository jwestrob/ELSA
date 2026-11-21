"""Strata by family genome groups (1, 2, >= 3), recomputed without the first pass's wide-anchor duplicates (M11w,
M12w, M13w) so no instance is counted twice. Two-gene = first-pass instances with 2 members; larger = this pass's
instances with >= 3 members. Writes strata_by_family_groups.tsv (POST HOC)."""
from pathlib import Path
import numpy as np
import pandas as pd
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
FZ = RD / "long_loci/frozen"
TOK = pd.read_parquet(FZ / "tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet").token.to_numpy()
grp = pd.read_parquet(FZ / "groups/emb_hom_average_t0.75.parquet"); GRPG = dict(zip(grp.genome, grp.group))


def family_groups(df):
    out = pd.Series(0, index=df.index)
    for m, sub in df.groupby("module"):
        toks = [set(t) for t in sub.tokens]; grs = sub.group.to_numpy()
        for a, ia in enumerate(sub.index):
            out[ia] = len({grs[b] for b in range(len(sub)) if len(toks[a] & toks[b]) / len(toks[a] | toks[b]) >= 0.5})
    return out


F = pd.read_parquet("module_instances.parquet")
F = F[~F.module.str.contains("w_")].copy()
F["tokens"] = F.members.map(lambda v: [int(TOK[x]) for x in v]); F["group"] = F.genome.map(GRPG)
F["family_groups"] = family_groups(F)
FP = pd.read_parquet("score_per_instance.parquet").merge(F[["instance_id", "family_groups", "n_members"]], on="instance_id")
FP["output"] = FP.output.map({"SECONDARY_embedding_list_tree_Z_L4bp_t0.02": "SEC", "PRIMARY_family_bond_tree_WA_L4b_t0.2": "PRI",
                              "engine_DH": "DH", "synton_faithful": "SYN"})
PA = pd.read_parquet("large_score_per_instance.parquet")
cols = ["output", "family_groups", "recovered", "resolved", "chance_recovered", "chance_resolved"]
X = pd.concat([FP[FP.n_members == 2][cols].assign(size="2 genes"), PA[PA.n_members >= 3][cols].assign(size=">= 3 genes")])
X["stratum"] = X.family_groups.clip(upper=3).map({1: "1 group", 2: "2 groups", 3: ">= 3 groups"})
SS = X.groupby(["size", "stratum", "output"]).agg(instances=("recovered", "size"), recovered=("recovered", "sum"), resolved=("resolved", "sum"),
                                               chance_recovered=("chance_recovered", "sum"), chance_resolved=("chance_resolved", "sum")).reset_index()
SS["chance_recovered"] = SS.chance_recovered.round(1); SS["chance_resolved"] = SS.chance_resolved.round(1)
SS.insert(0, "status", "POST HOC")
SS.to_csv("strata_by_family_groups.tsv", sep="\t", index=False)
print(SS.to_string())
