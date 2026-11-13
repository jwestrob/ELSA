#!/usr/bin/env python3
"""MANIFEST.tsv (path, rows, bytes, sha256) for a slice package (workstream slice_lut).

usage: 11_manifest.py PKG_DIR
"""
import hashlib, os, sys
import pyarrow.parquet as pq

P = sys.argv[1]
rows = []
for root, _, files in os.walk(P):
    for f in sorted(files):
        p = os.path.join(root, f)
        rel = os.path.relpath(p, P)
        if rel.startswith(".") or rel == "MANIFEST.tsv":
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(16 << 20), b""):
                h.update(b)
        if f.endswith(".parquet"):
            n = pq.ParquetFile(p).metadata.num_rows
        elif f.endswith(".faa"):
            n = sum(1 for line in open(p) if line.startswith(">"))
        else:
            n = ""
        rows.append((rel, n, os.path.getsize(p), h.hexdigest()))
with open(os.path.join(P, "MANIFEST.tsv"), "w") as out:
    out.write("path\trows\tbytes\tsha256\n")
    for r in sorted(rows):
        out.write("\t".join(map(str, r)) + "\n")
print("\n".join("\t".join(map(str, r[:3])) for r in sorted(rows)))
