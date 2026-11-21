#!/usr/bin/env python3
"""Check B sample (predeclared_verify.json): first 100 proteins per stratum order with a qualifying partner; null pairs
(partner of another sampled protein, different label, partner length within 20%, else 30%; default_rng(20261121));
fold input FASTA (sequences read-only from the Sharur DPANN DB). Writes data/B_sample.tsv, data/B_pairs.tsv, B/fold_input.faa."""
from pathlib import Path
import duckdb, numpy as np, pandas as pd
H = Path(__file__).resolve().parents[1]
P = pd.read_csv(H / "data/B_partners.tsv", sep="\t")
O = pd.read_csv(H / "data/B_order.tsv", sep="\t")
P = P.merge(O[["stratum", "order", "label_name", "length", "votes_for", "vote_share", "confidence_bin"]], on=["stratum", "order"])
S = pd.concat([g[g.partner_gid.notna()].sort_values("order").head(100) for _, g in P.groupby("stratum")]).reset_index(drop=True)
S["partner_gid"] = S.partner_gid.astype(int)
rng = np.random.default_rng(20261121)
null, widen = [], []
for i, r in S.iterrows():
    ok = None
    for tol in (0.2, 0.3):
        c = S[(S.index != i) & (S.label != r.label) & ((S.partner_length - r.partner_length).abs() <= tol * r.partner_length)]
        if len(c):
            ok = (c.index[rng.integers(len(c))], tol)
            break
    null.append(S.at[ok[0], "partner_protein_id"] if ok else "")
    widen.append(ok[1] if ok else np.nan)
S["null_partner_protein_id"] = null
S["null_tolerance"] = widen
S.to_csv(H / "data/B_sample.tsv", sep="\t", index=False)
pairs = pd.concat([pd.DataFrame(dict(kind="true", stratum=S.stratum, protein_id=S.protein_id, partner=S.partner_protein_id, label=S.label)),
                   pd.DataFrame(dict(kind="null", stratum=S.stratum, protein_id=S.protein_id, partner=S.null_partner_protein_id, label=S.label))])
pairs = pairs[pairs.partner != ""]
pairs.to_csv(H / "data/B_pairs.tsv", sep="\t", index=False)
ids = list(dict.fromkeys(list(S.protein_id) + list(S.partner_protein_id)))
con = duckdb.connect("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb", read_only=True)
seq = dict(con.execute("SELECT protein_id, sequence FROM proteins WHERE protein_id IN (SELECT unnest(?::VARCHAR[]))", [ids]).fetchall())
assert len(seq) == len(ids)
(H / "B").mkdir(exist_ok=True)
with open(H / "B/fold_input.faa", "w") as f:
    for p in ids:
        f.write(f">{p}\n{seq[p].rstrip('*')}\n")
L = {p: len(seq[p].rstrip('*')) for p in ids}
print("sample", len(S), S.groupby("stratum").size().to_dict(), "folds", len(ids), "max length", max(L.values()),
      "internal *", sum('*' in seq[p].rstrip('*') for p in ids), "null pairs", int((S.null_partner_protein_id != "").sum()),
      "widened", int((S.null_tolerance == 0.3).sum()), "unique labels", S.label.nunique())
