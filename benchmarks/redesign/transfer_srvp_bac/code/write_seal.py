#!/usr/bin/env python
"""Write package/SEAL.txt: SHA-256 of every gold table and of the universe, the source database's identity, the date.

usage: write_seal.py --pkg DIR --verify-receipt JSON --source-file PATH --source-size N --source-mtime STR
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import pyarrow.parquet as pq


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(p: Path):
    if p.suffix == ".parquet":
        return pq.ParquetFile(p).metadata.num_rows
    if p.suffix == ".faa":
        with open(p) as f:
            return sum(1 for line in f if line.startswith(">"))
    return ""


ap = argparse.ArgumentParser()
ap.add_argument("--pkg", type=Path, required=True)
ap.add_argument("--source-file", required=True)
ap.add_argument("--source-size", type=int, required=True)
ap.add_argument("--source-mtime", required=True)
a = ap.parse_args()
P = a.pkg
v = json.loads((P / "receipts" / "verify.json").read_text())
universe = ["universe/genes.parquet", "universe/contigs.parquet", "universe/genomes.parquet", "universe/gene_flags.parquet",
            "universe/proteins.faa", "split/genome_halves.parquet", "annotation_flags.parquet"]
gold = ["gold/instances.parquet", "gold/members.parquet", "ribosomal/runs.parquet", "ribosomal/marked.parquet",
        "ribosomal/controls_random.parquet", "ribosomal/controls_transplant.parquet"]
lines = [
    "SR-VP bacteria (PacBio) held-out test package: SEAL",
    f"written_utc\t{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
    f"date\t{time.strftime('%Y-%m-%d', time.gmtime())}",
    "",
    "[source database]",
    f"source_file\t{a.source_file}",
    f"source_size_bytes\t{a.source_size}",
    f"source_mtime\t{a.source_mtime}",
    f"source_sha256_before_transfer\t{v['expected_sha256']}",
    f"copy\t{v['db']}",
    f"copy_sha256\t{v['file_sha256']}",
    f"copy_size_bytes\t{v['size']}",
    f"copy_matches_source\t{v['sha_match']}",
    f"sharur_seal_dataset_id\t{v['seal_id']}",
    f"sharur_seal_generated_at\t{v['seal_generated']}",
    f"sharur_seal_strength\t{v.get('seal_strength')}",
    f"table_row_counts_match_sharur_seal\t{not v['row_count_mismatches']}",
    "",
    "[status]",
    "No neighbourhood, recurrence, word or locus method has been run on this universe, and nothing has been",
    "scored against its gold tables. The gold tables were read only to count instances and spans.",
    "",
    "[files]\tsha256\tbytes\trows\tpath",
]
for grp, files in (("universe", universe), ("gold", gold)):
    for f in files:
        p = P / f
        lines.append(f"{grp}\t{sha(p)}\t{p.stat().st_size}\t{rows(p)}\t{f}")
tmp = P / ".SEAL.txt.tmp"
tmp.write_text("\n".join(lines) + "\n")
tmp.rename(P / "SEAL.txt")
print("\n".join(lines))
