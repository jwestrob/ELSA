#!/usr/bin/env python
"""Pilot gate (assay_rule.json): positives pass >= 50%, negative A <= 10%, negative B <= 10%, one-sided Fisher p < 0.01 for
positives vs each negative set. Families whose positive-control pass share < 0.5 are uninformative. Writes data/pilot_gate.json."""
import json
from pathlib import Path
import pandas as pd
from scipy.stats import fisher_exact
H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/invisible")
O = pd.read_csv(H / "data/readout_pilot.tsv", sep="\t")
O["set"] = O.role.where(O.pair == "own", "negA")
O.loc[(O.role == "candidate") & (O.pair == "own"), "set"] = "candidate"
cnt = {s: (int(d.PASS.sum()), len(d)) for s, d in O.groupby("set")}
p = lambda a, b: fisher_exact([[a[0], a[1] - a[0]], [b[0], b[1] - b[0]]], alternative="greater")[1]
fam = O[O.set == "positive"].groupby("func").PASS.agg(["sum", "size"])
g = dict(counts=cnt, fisher_pos_vs_negA=p(cnt["positive"], cnt["negA"]), fisher_pos_vs_negB=p(cnt["positive"], cnt["negB"]),
         positive_share=cnt["positive"][0] / cnt["positive"][1],
         positives_per_family={f: f"{int(r['sum'])}/{int(r['size'])}" for f, r in fam.iterrows()},
         uninformative=[f for f, r in fam.iterrows() if r["sum"] / r["size"] < 0.5])
g["gate_pass"] = bool(g["positive_share"] >= 0.5 and cnt["negA"][0] / cnt["negA"][1] <= 0.1 and cnt["negB"][0] / cnt["negB"][1] <= 0.1
                      and g["fisher_pos_vs_negA"] < 0.01 and g["fisher_pos_vs_negB"] < 0.01)
json.dump(g, open(H / "data/pilot_gate.json", "w"), indent=1)
print(json.dumps(g, indent=1))
