#!/usr/bin/env python
"""Exact comparison of two recurrence result directories (significant pairs, rules, boundary support, modules, loci)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

a, b = Path(sys.argv[1]), Path(sys.argv[2])
out = {}
for f in ("significant_pairs.parquet", "token_modules.parquet", "loci_all_genomes.parquet"):
    x, y = pd.read_parquet(a / f), pd.read_parquet(b / f)
    cols = [c for c in y.columns if c in x.columns]
    out[f] = dict(rows=(len(x), len(y)), columns_compared=cols,
                  equal=bool(len(x) == len(y) and all(np.array_equal(x[c].to_numpy(), y[c].to_numpy()) for c in cols)))
x, y = pd.read_csv(a / "rules.tsv", sep="\t"), pd.read_csv(b / "rules.tsv", sep="\t")
out["rules.tsv"] = dict(equal=bool(x[["rule", "observed", "null_mean"]].equals(y[["rule", "observed", "null_mean"]])))
out["boundary_support.npy"] = dict(equal=bool(np.array_equal(np.load(a / "boundary_support.npy"), np.load(b / "boundary_support.npy"))))
sa, sb = json.load(open(a / "summary.json")), json.load(open(b / "summary.json"))
keys = ("rule", "fdr", "significant_pairs", "significant_pairs_cross_phylum", "genes_covered", "token_pairs_L2", "L_quantiles", "modules", "loci_all_genomes")
out["summary"] = {k: (sa.get(k) == sb.get(k)) for k in keys}
out["all_equal"] = all(v["equal"] for k, v in out.items() if isinstance(v, dict) and "equal" in v) and all(out["summary"].values())
print(json.dumps(out, indent=1))
