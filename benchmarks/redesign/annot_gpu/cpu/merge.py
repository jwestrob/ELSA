#!/usr/bin/env python3
"""Check a sharded Aksha Pfam run and merge the shard tables into OUTDIR/PFAM.parquet.

Usage: merge.py --outdir O --meta pfam_meta.tsv --threads N

Layout under O: shards/ (shard.py output), search/shard_NNN/PFAM_hits_df.tsv (Aksha), search/shard_NNN.done.
Checks, all required:
  * shards.tsv rows 0..K-1; protein counts sum to SHARDS.json proteins; gids strictly increasing within and
    across shards (with consecutive_gids: first gid 0, last gid N-1, no gap);
  * each shard file has the recorded SHA-256 and exactly the recorded headers, in order;
  * the source FASTA still has the recorded SHA-256 and the concatenated shard files reproduce it;
  * each shard has a .done marker from Aksha 0.2.3 whose loaded-sequence count equals the shard's proteins and
    whose hit-row count equals the rows read from its table;
  * every hit's protein_id is a header of its own shard; every model name maps to a Pfam accession.
Output columns: protein_id (FASTA header, string), gid (int64, same value), accession, name, description,
bitscore and evalue (full sequence), dom_bitscore, c_evalue, i_evalue, hmm_from, hmm_to, ali_from, ali_to,
env_from, env_to; one row per domain as Aksha reports it; sorted by gid, env_from, accession.
Parallel: one process per shard plus two for the source hashes; pyarrow threads split the remaining CPUs.
On a failed check: writes O/merge_fail_reason.txt and exits 1.
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

BLOCK = 1 << 24
HIT_COLS = ["sequence_id", "hmm_name", "bitscore", "evalue", "c_evalue", "i_evalue", "env_from", "env_to",
            "dom_bitscore", "ali_from", "ali_to", "hmm_from", "hmm_to"]
HIT_TYPES = {
    "sequence_id": pa.string(), "hmm_name": pa.string(),
    "bitscore": pa.float64(), "evalue": pa.float64(), "c_evalue": pa.float64(),
    "i_evalue": pa.float64(), "dom_bitscore": pa.float64(),
    "env_from": pa.int32(), "env_to": pa.int32(), "ali_from": pa.int32(),
    "ali_to": pa.int32(), "hmm_from": pa.int32(), "hmm_to": pa.int32(),
}
OUT_COLS = ["protein_id", "gid", "accession", "name", "description", "bitscore", "evalue", "dom_bitscore",
            "c_evalue", "i_evalue", "hmm_from", "hmm_to", "ali_from", "ali_to", "env_from", "env_to"]
OUT_SCHEMA = pa.schema(
    [("protein_id", pa.string()), ("gid", pa.int64()), ("accession", pa.string()), ("name", pa.string()),
     ("description", pa.string())]
    + [(c, pa.float64()) for c in ("bitscore", "evalue", "dom_bitscore", "c_evalue", "i_evalue")]
    + [(c, pa.int32()) for c in ("hmm_from", "hmm_to", "ali_from", "ali_to", "env_from", "env_to")]
)
AKSHA_VERSION = "aksha 0.2.3"


class CheckFailed(Exception):
    pass


def check(cond, msg):
    if not cond:
        raise CheckFailed(msg)


def set_threads(n):
    pa.set_cpu_count(n)
    pa.set_io_thread_count(n)


def load_meta(path):
    m = pcsv.read_csv(
        path,
        parse_options=pcsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pcsv.ConvertOptions(strings_can_be_null=False),
    ).to_pandas()
    check(list(m.columns[:3]) == ["name", "accession", "description"], f"unexpected metadata columns {list(m.columns)}")
    check(m["name"].is_unique, "metadata names are not unique")
    return m[["name", "accession", "description"]]


def read_hits(tsv):
    with open(tsv) as fh:
        header = fh.readline().rstrip("\n").split("\t")
    check(header == HIT_COLS, f"{tsv}: header {header} differs from the Aksha table layout")
    t = pcsv.read_csv(
        tsv,
        parse_options=pcsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pcsv.ConvertOptions(column_types=HIT_TYPES, strings_can_be_null=False),
    )
    return t.to_pandas()


def to_annotation(df, meta, allowed_gids=None, label=""):
    """Aksha hit rows -> annotation rows (the column set of the previous agent's to_parquet.py, PFAM branch)."""
    gid = df["sequence_id"].astype("int64")
    check(bool((gid.astype(str) == df["sequence_id"]).all()), f"{label}: a protein_id is not a plain integer gid")
    if allowed_gids is not None and len(df):
        check(bool(np.isin(gid.to_numpy(), allowed_gids).all()), f"{label}: a hit names a protein outside its shard")
    out = df.rename(columns={"sequence_id": "protein_id", "hmm_name": "name"})
    out.insert(1, "gid", gid)
    out = out.merge(meta, on="name", how="left", validate="many_to_one")
    check(int(out["accession"].isna().sum()) == 0, f"{label}: a model name is missing from the Pfam metadata")
    out = out[OUT_COLS].sort_values(["gid", "env_from", "accession"], kind="stable").reset_index(drop=True)
    return out


def fasta_headers(path):
    h = hashlib.sha256()
    gids = []
    with open(path, "rb") as fh:
        data = fh.read()
    h.update(data)
    for line in data.split(b"\n"):
        if line.startswith(b">"):
            gids.append(int(line[1:]))
    return h.hexdigest(), np.asarray(gids, dtype=np.int64)


def shard_job(o, row, meta_path, consecutive):
    s = int(row["shard"])
    tag = f"shard {s:03d}"
    faa = f"{o}/shards/shard_{s:03d}.faa"
    sha, gids = fasta_headers(faa)
    check(sha == row["sha256"], f"{tag}: FASTA SHA-256 differs from shards.tsv")
    n = int(row["proteins"])
    check(len(gids) == n, f"{tag}: {len(gids)} records, shards.tsv says {n}")
    check(n == 0 or (int(gids[0]) == int(row["first_gid"]) and int(gids[-1]) == int(row["last_gid"])),
          f"{tag}: first/last gid differ from shards.tsv")
    check(bool(np.all(np.diff(gids) > 0)), f"{tag}: gids are not strictly increasing")
    if consecutive:
        check(n == 0 or int(gids[-1]) - int(gids[0]) + 1 == n, f"{tag}: gids are not consecutive")
    done_path = f"{o}/search/shard_{s:03d}.done"
    check(os.path.exists(done_path), f"{tag}: no .done marker (search unfinished or failed)")
    done = json.load(open(done_path))
    check(done["aksha"] == AKSHA_VERSION, f"{tag}: searched with {done['aksha']}, expected {AKSHA_VERSION}")
    check(int(done["proteins"]) == n and int(done["loaded"]) == n,
          f"{tag}: Aksha loaded {done['loaded']} sequences of {n}")
    tsv = f"{o}/search/shard_{s:03d}/PFAM_hits_df.tsv"
    check(os.path.exists(tsv), f"{tag}: {tsv} is missing")
    df = read_hits(tsv)
    check(len(df) == int(done["hit_rows"]), f"{tag}: {len(df)} rows read, the marker recorded {done['hit_rows']}")
    out = to_annotation(df, load_meta(meta_path), allowed_gids=gids, label=tag)
    piece = f"{o}/merge_tmp/piece_{s:03d}.parquet"
    pq.write_table(pa.Table.from_pandas(out, schema=OUT_SCHEMA, preserve_index=False), piece, compression="zstd")
    return {"shard": s, "proteins": n, "rows": len(out), "proteins_with_hit": int(out["gid"].nunique()),
            "piece": piece, "first_gid": int(row["first_gid"]), "last_gid": int(row["last_gid"]),
            "wall_s": done.get("wall_s"), "user_s": done.get("user_s"), "sys_s": done.get("sys_s"),
            "max_rss_kb": done.get("max_rss_kb"), "host": done.get("host"), "job": done.get("job"),
            "start_utc": done.get("start_utc"), "end_utc": done.get("end_utc")}


def sha256_files(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as fh:
            while True:
                b = fh.read(BLOCK)
                if not b:
                    break
                h.update(b)
    return h.hexdigest()


def read_shards_tsv(path):
    rows = []
    with open(path) as fh:
        cols = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            rows.append(dict(zip(cols, line.rstrip("\n").split("\t"))))
    return rows


def run(o, meta_path, threads):
    info = json.load(open(f"{o}/shards/SHARDS.json"))
    rows = read_shards_tsv(f"{o}/shards/shards.tsv")
    k = int(info["shards"])
    check([int(r["shard"]) for r in rows] == list(range(k)), "shards.tsv does not list shards 0..K-1")
    check(sum(int(r["proteins"]) for r in rows) == int(info["proteins"]), "shard protein counts do not sum to N")
    for a, b in zip(rows, rows[1:]):
        check(int(b["first_gid"]) > int(a["last_gid"]), f"shards {a['shard']} and {b['shard']} overlap")
        if info["consecutive_gids"]:
            check(int(b["first_gid"]) == int(a["last_gid"]) + 1, f"gap between shards {a['shard']} and {b['shard']}")
    if info["consecutive_gids"]:
        check(int(rows[0]["first_gid"]) == 0 and int(rows[-1]["last_gid"]) == int(info["proteins"]) - 1,
              "shards do not span gid 0..N-1")
    meta = load_meta(meta_path)  # validated once here; workers reload it
    os.makedirs(f"{o}/merge_tmp", exist_ok=True)
    workers = max(1, min(threads, k + 2))
    shard_paths = [f"{o}/shards/shard_{s:03d}.faa" for s in range(k)]
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=set_threads,
                                initargs=(max(1, threads // workers),)) as ex:
        f_src = ex.submit(sha256_files, [info["source"]])
        f_cat = ex.submit(sha256_files, shard_paths)
        f_sh = [ex.submit(shard_job, o, r, meta_path, info["consecutive_gids"]) for r in rows]
        results = [f.result() for f in f_sh]
        check(f_src.result() == info["source_sha256"], "the source FASTA changed since sharding")
        check(f_cat.result() == info["source_sha256"], "the concatenated shards differ from the source FASTA")
    set_threads(threads)
    tables = [pq.read_table(r["piece"]) for r in results]
    table = pa.concat_tables(tables)
    n_rows = sum(r["rows"] for r in results)
    check(table.num_rows == n_rows, "row count changed while concatenating")
    tmp = f"{o}/PFAM.parquet.tmp"
    pq.write_table(table, tmp, compression="zstd")
    check(pq.ParquetFile(tmp).metadata.num_rows == n_rows, "PFAM.parquet row count differs after writing")
    os.replace(tmp, f"{o}/PFAM.parquet")
    for r in results:
        os.remove(r["piece"])
    os.rmdir(f"{o}/merge_tmp")
    gid = table.column("gid").to_numpy()
    acc = table.column("accession").to_numpy(zero_copy_only=False)
    pairs = len(set(zip(gid.tolist(), acc.tolist())))
    phit = int(np.unique(gid).size)
    summary = {
        "database": "PFAM (Pfam 38.1, 27,481 models) only",
        "parquet": f"{o}/PFAM.parquet",
        "source": info["source"], "source_sha256": info["source_sha256"],
        "proteins": int(info["proteins"]), "shards": k,
        "rows": n_rows, "proteins_with_hit": phit,
        "models_hit": int(len(set(acc.tolist()))), "protein_model_pairs": pairs,
        "checks": "all passed: shard files and headers, source and concatenation SHA-256, Aksha 0.2.3 markers, "
                  "loaded = shard proteins, hits inside their shard, metadata join",
        "search_cpu_s": round(sum(float(r["user_s"] or 0) + float(r["sys_s"] or 0) for r in results), 1),
        "per_shard": [{k2: v for k2, v in r.items() if k2 != "piece"} for r in results],
        "status_fields": f"rows {n_rows} proteins_with_hit {phit} shards {k} database PFAM only",
        "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(f"{o}/PFAM_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k2: v for k2, v in summary.items() if k2 != "per_shard"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    try:
        run(a.outdir, a.meta, a.threads)
    except CheckFailed as e:
        with open(f"{a.outdir}/merge_fail_reason.txt", "w") as fh:
            fh.write(f"check failed: {e}\n")
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 (record any error as the STATUS reason)
        with open(f"{a.outdir}/merge_fail_reason.txt", "w") as fh:
            fh.write(f"merge error: {type(e).__name__}: {e}\n")
        raise


if __name__ == "__main__":
    main()
