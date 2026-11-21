#!/usr/bin/env python
"""Write MANIFEST.tsv and MANIFEST.json for the DPANN package: every file, its rows, bytes and SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(path: Path):
    if path.suffix == ".parquet":
        return pq.ParquetFile(path).metadata.num_rows, "parquet rows"
    if path.suffix == ".npy":
        return int(np.load(path, mmap_mode="r").shape[0]), "array rows"
    if path.suffix in (".tsv", ".csv"):
        with open(path) as f:
            return sum(1 for _ in f) - 1, "data lines (header excluded)"
    if path.suffix == ".faa":
        with open(path) as f:
            return sum(1 for line in f if line.startswith(">")), "FASTA records"
    if path.suffix in (".py", ".sh", ".sbatch", ".md", ".json"):
        with open(path) as f:
            return sum(1 for _ in f), "text lines"
    return None, "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    files = sorted(p for p in a.pkg.rglob("*") if p.is_file() and not p.name.startswith(".")
                   and p.name not in ("MANIFEST.tsv", "MANIFEST.json"))

    def one(p):
        n, unit = rows(p)
        return dict(path=str(p.relative_to(a.pkg)), rows=n, row_unit=unit, bytes=p.stat().st_size, sha256=sha(p))

    with ThreadPoolExecutor(a.threads) as ex:
        recs = list(ex.map(one, files))
    with open(a.pkg / "MANIFEST.tsv", "w") as f:
        f.write("path\trows\trow_unit\tbytes\tsha256\n")
        for r in recs:
            f.write(f"{r['path']}\t{'' if r['rows'] is None else r['rows']}\t{r['row_unit']}\t{r['bytes']}\t{r['sha256']}\n")
    (a.pkg / "MANIFEST.json").write_text(json.dumps(dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                         files=recs, n_files=len(recs),
                                                         total_bytes=sum(r["bytes"] for r in recs)), indent=2) + "\n")
    for r in recs:
        print(r["path"], r["rows"], r["bytes"], r["sha256"][:12])


if __name__ == "__main__":
    main()
