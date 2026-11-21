#!/usr/bin/env python3
"""Cut test, diagnostic (exploratory, after the predeclared rates): is a wrong-family or false join a sibling token
of the original's own homologs, or a different family?

usage: wrong_join_check.py prep|classify <repair dir>
prep:     writes cut/orig_pref.tsv, the explicit pairs (original uid, joined target uid) for every wrong-family
          positive and false-join negative under variants base, a, c.
classify: reads cut/orig_hits.parquet (the intact original aligned to the joined target, -e 1e-5 -a) and calls a join
          'same_homolog' when the intact original aligns to that target at E <= 1e-5 covering >= 80% of both
          proteins, else 'different'. Writes cut/wrong_join.tsv and cut/wrong_join.json.
"""
import json, sys
import numpy as np, pandas as pd

mode, R = sys.argv[1], sys.argv[2]
O = pd.read_parquet(f"{R}/cut/outcomes.parquet")
rows = []
for v in ("base", "a", "c"):
    k = O[f"outcome_{v}"].isin(["wrong_family", "false_join"])
    rows.append(pd.DataFrame({"variant": v, "fid": O.index[k], "kind": O.kind[k].values, "uid0": O.uid[k].values,
                              "t": O[f"t_{v}"][k].astype(np.int64).values}))
W = pd.concat(rows, ignore_index=True)
if mode == "prep":
    W[["uid0", "t"]].drop_duplicates().sort_values(["uid0", "t"]).assign(a=0, b=0).to_csv(
        f"{R}/cut/orig_pref.tsv", sep="\t", header=False, index=False)
    print("pairs", len(W[["uid0", "t"]].drop_duplicates()))
else:
    A = pd.read_parquet(f"{R}/cut/orig_hits.parquet")
    A = A[A.evalue <= 1e-5]
    A["full"] = ((A.qend.astype(int) - A.qstart.astype(int) + 1) >= 0.8 * A.qlen.astype(int)) & \
                ((A.tend.astype(int) - A.tstart.astype(int) + 1) >= 0.8 * A.tlen.astype(int))
    full = set(zip(A.q[A.full].astype(np.int64), A.t[A.full].astype(np.int64)))
    anyhit = set(zip(A.q.astype(np.int64), A.t.astype(np.int64)))
    W["class"] = [("same_homolog" if (u, t) in full else ("partial_homolog" if (u, t) in anyhit else "no_hit"))
                  for u, t in zip(W.uid0, W.t)]
    W.to_csv(f"{R}/cut/wrong_join.tsv", sep="\t", index=False)
    s = W.groupby(["variant", "kind", "class"]).size().unstack(fill_value=0)
    print(s)
    json.dump({f"{v}|{k}": r.to_dict() for (v, k), r in s.iterrows()}, open(f"{R}/cut/wrong_join.json", "w"), indent=1, default=int)
