#!/usr/bin/env python3
"""Convert one Aksha hit table (<DB>_hits_df.tsv) to the annotation Parquet.

Usage: to_parquet.py DB HITS_TSV META_TSV N_PROTEINS OUT.parquet OUT_SUMMARY.json
Columns: protein_id (FASTA header, string), gid (int64, same value; headers are 0-based row indices),
accession, name, [description (PFAM) | ko_threshold, ko_score_type (KOFAM)], bitscore (full sequence),
evalue (full sequence), dom_bitscore, c_evalue, i_evalue, hmm_from, hmm_to, ali_from, ali_to, env_from, env_to.
One row per reported domain, as Aksha writes them; rows sorted by gid, env_from, accession.
"""
import json
import sys

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq


def main():
    db, tsv, meta_tsv, n_prot, out, summary = sys.argv[1:7]
    n_prot = int(n_prot)
    types = {
        "sequence_id": pa.string(), "hmm_name": pa.string(),
        "bitscore": pa.float64(), "evalue": pa.float64(), "c_evalue": pa.float64(),
        "i_evalue": pa.float64(), "dom_bitscore": pa.float64(),
        "env_from": pa.int32(), "env_to": pa.int32(), "ali_from": pa.int32(),
        "ali_to": pa.int32(), "hmm_from": pa.int32(), "hmm_to": pa.int32(),
    }
    t = pcsv.read_csv(
        tsv,
        parse_options=pcsv.ParseOptions(delimiter="\t"),
        convert_options=pcsv.ConvertOptions(column_types=types),
    )
    df = t.to_pandas()
    meta = pcsv.read_csv(
        meta_tsv,
        parse_options=pcsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pcsv.ConvertOptions(strings_can_be_null=False,
                                            column_types={"threshold": pa.string()}),
    ).to_pandas()
    gid = df["sequence_id"].astype("int64")
    assert (gid.astype(str) == df["sequence_id"]).all(), "protein_id is not a plain integer gid"
    assert gid.min() >= 0 and gid.max() < n_prot, "gid outside the FASTA row range"
    out_df = df.rename(columns={"sequence_id": "protein_id", "hmm_name": "name"})
    out_df.insert(1, "gid", gid)
    if db == "PFAM":
        m = meta[["name", "accession", "description"]]
    else:
        m = meta[["name", "accession", "threshold", "score_type", "definition"]].rename(
            columns={"threshold": "ko_threshold", "score_type": "ko_score_type"})
    out_df = out_df.merge(m, on="name", how="left", validate="many_to_one")
    unmapped = int(out_df["accession"].isna().sum())
    assert unmapped == 0, f"{unmapped} rows with a model missing from the metadata"
    if db == "KOFAM":
        # name = ko_list definition; accession = KO number
        out_df["name"] = out_df.pop("definition")
    cols = ["protein_id", "gid", "accession", "name"]
    cols += ["description"] if db == "PFAM" else ["ko_threshold", "ko_score_type"]
    cols += ["bitscore", "evalue", "dom_bitscore", "c_evalue", "i_evalue",
             "hmm_from", "hmm_to", "ali_from", "ali_to", "env_from", "env_to"]
    out_df = out_df[cols].sort_values(["gid", "env_from", "accession"], kind="stable").reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(out_df, preserve_index=False), out, compression="zstd")
    s = {
        "db": db, "tsv": tsv, "parquet": out,
        "rows": int(len(out_df)),
        "proteins_with_hit": int(out_df["gid"].nunique()),
        "proteins_in_fasta": n_prot,
        "models_hit": int(out_df["accession"].nunique()),
        "protein_model_pairs": int(out_df[["gid", "accession"]].drop_duplicates().shape[0]),
    }
    with open(summary, "w") as fh:
        json.dump(s, fh, indent=1)
    print(json.dumps(s))


if __name__ == "__main__":
    main()
