"""Local helper: writes the SR-VP site definitions (positions only) and the already-named stratum (gids only) that
p4_srvp_remote.py reads on biotite. No sequences involved."""
import json, pandas as pd
D = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads"
s = pd.read_csv(D + "/leads_srvp/data/ext/site_ref.tsv", sep="\t")
out = {}
for f in ["K06152", "K20930"]:
    x = s[s.func == f]
    out[f] = dict(ref_member=int(x.ref_member.iloc[0]), positions=[[int(p), r, k] for p, r, k in zip(x.member_pos, x.residue, x.kind)])
json.dump(out, open(D + "/pseudo/data/srvp_sites.json", "w"), indent=1)
p = pd.read_parquet(D + "/leads_srvp/data/population.parquet", columns=["gid", "func", "named"])
p[p.func.isin(["K06152", "K20930"])].to_csv(D + "/pseudo/data/srvp_named_stratum.tsv", sep="\t", index=False)
print(out)
