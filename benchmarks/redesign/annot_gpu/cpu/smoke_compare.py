#!/usr/bin/env python3
"""Compare the smoke runs of the sharded chain with the previous agent's CPU validation table (Pfam).

Usage: smoke_compare.py VAL_TSV META_TSV SMOKE1_DIR SMOKE3_DIR OUT.json

The validation table (job 119815, 4,000 proteins searched as one set) is converted with the same function as
merge.py. smoke1 (one shard) must match it on every column, and its raw Aksha table must be byte-identical.
smoke3 (three shards) must match on every column except the three E-value columns, which depend on the
number of targets in a search; for those the ratio smoke3 / validation is reported per shard.
Single process, small files (login node).
"""
import hashlib
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge  # noqa: E402

EVAL = ["evalue", "c_evalue", "i_evalue"]
KEY = ["protein_id", "accession", "env_from", "env_to", "ali_from", "ali_to", "hmm_from", "hmm_to"]


def multiset_equal(a, b, cols):
    x = a[cols].sort_values(cols, kind="stable").reset_index(drop=True)
    y = b[cols].sort_values(cols, kind="stable").reset_index(drop=True)
    return len(x) == len(y) and bool((x == y).all().all())


def main():
    val_tsv, meta_tsv, s1, s3, out = sys.argv[1:6]
    meta = merge.load_meta(meta_tsv)
    val = merge.to_annotation(merge.read_hits(val_tsv), meta, label="validation")
    res = {"validation_tsv": val_tsv, "validation_rows": len(val),
           "validation_proteins_with_hit": int(val["gid"].nunique())}
    non_e = [c for c in merge.OUT_COLS if c not in EVAL]

    p1 = pq.read_table(f"{s1}/PFAM.parquet").to_pandas()
    raw1 = f"{s1}/search/shard_000/PFAM_hits_df.tsv"
    res["smoke1"] = {
        "rows": len(p1),
        "all_columns_identical": multiset_equal(p1, val, merge.OUT_COLS),
        "raw_tsv_bytes_identical": open(raw1, "rb").read() == open(val_tsv, "rb").read(),
        "raw_tsv_sha256": hashlib.sha256(open(raw1, "rb").read()).hexdigest(),
        "status": open(f"{s1}/STATUS").read().strip(),
    }

    p3 = pq.read_table(f"{s3}/PFAM.parquet").to_pandas()
    r3 = {"rows": len(p3),
          "non_evalue_columns_identical": multiset_equal(p3, val, non_e),
          "all_columns_identical": multiset_equal(p3, val, merge.OUT_COLS),
          "status": open(f"{s3}/STATUS").read().strip()}
    shards = pd.read_csv(f"{s3}/shards/shards.tsv", sep="\t")
    n_all = int(json.load(open(f"{s3}/shards/SHARDS.json"))["proteins"])
    j = p3.merge(val, on=KEY, suffixes=("_s", "_v"), how="inner", validate="one_to_one")
    r3["joined_rows"] = len(j)
    per = []
    for _, row in shards.iterrows():
        m = j[(j["gid_s"] >= row["first_gid"]) & (j["gid_s"] <= row["last_gid"])]
        d = {"shard": int(row["shard"]), "proteins": int(row["proteins"]), "rows": len(m),
             "z_ratio_shard_over_all": round(int(row["proteins"]) / n_all, 4)}
        for c in EVAL:
            ok = (m[f"{c}_v"] > 0) & (m[f"{c}_s"] > 0)
            ratio = m.loc[ok, f"{c}_s"] / m.loc[ok, f"{c}_v"]
            d[f"{c}_ratio_median"] = round(float(ratio.median()), 4) if len(ratio) else None
            d[f"{c}_ratio_min"] = round(float(ratio.min()), 4) if len(ratio) else None
            d[f"{c}_ratio_max"] = round(float(ratio.max()), 4) if len(ratio) else None
            d[f"{c}_rows_equal"] = int((m[f"{c}_s"] == m[f"{c}_v"]).sum())
        per.append(d)
    r3["per_shard_evalue_ratio"] = per
    res["smoke3"] = r3
    res["rows_identical"] = bool(res["smoke1"]["all_columns_identical"] and r3["non_evalue_columns_identical"]
                                 and len(p1) == len(val) == len(p3))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["rows_identical"] else 1)


if __name__ == "__main__":
    main()
