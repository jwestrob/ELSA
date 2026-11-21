"""Gold type/subtype inventory (login node, one thread, seconds)."""
import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
I = pd.read_parquet(f"{RD}/ws_dpann/package/gold/instances.parquet")
I["nm"] = I.member_gids.map(len)
for c, g in I.groupby("caller"):
    g2 = g[g.nm >= 2]
    print("==", c, len(g), ">=2 members:", len(g2), "n_contigs>1:", int((g2.n_contigs > 1).sum()))
    print(" system_type top:", g2.system_type.value_counts().head(25).to_dict())
    print(" n types", g2.system_type.nunique(), "n subtypes", g2.system_subtype.nunique())
    print(" subtype top:", g2.system_subtype.value_counts().head(40).to_dict())
    print(" members q:", np.quantile(g2.nm, [.5, .9, .99, 1]).tolist(), "span q:", np.quantile(g2.span_genes, [.5, .9, .99, 1]).tolist())
df = I[(I.caller == "DefenseFinder") & (I.nm >= 2)]
x = df.groupby("system_type").system_subtype.nunique()
print("DF types with >1 subtype:", int((x > 1).sum()), x[x > 1].to_dict())
print(df[df.system_type != df.system_subtype][["system_type", "system_subtype"]].drop_duplicates().head(60).to_string())
