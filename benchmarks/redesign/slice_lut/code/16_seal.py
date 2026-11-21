#!/usr/bin/env python3
"""Seal a slice test-set package: SHA-256 of the universe, FASTA, truth tables, split and selection table.

usage: 16_seal.py PKG_DIR
Writes PKG_DIR/seal.json and PKG_DIR/MANIFEST.tsv (path, rows, bytes, sha256 of every file).
"""
import hashlib, json, os, sys, time
import pyarrow.parquet as pq

P = sys.argv[1]
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()
sealed = ["selection.tsv", "split.parquet", "proteins_gid.faa"] + \
    [f"universe/{x}" for x in sorted(os.listdir(f"{P}/universe"))] + [f"gold/{x}" for x in sorted(os.listdir(f"{P}/gold"))]
files = {f: sha(f"{P}/{f}") for f in sealed}
combined = hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
seal = dict(package=P, sealed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), files=files, seal_sha256=combined,
            rule="sha256 over the sorted lines '<path>\\t<sha256>\\n' of the files listed")
json.dump(seal, open(f"{P}/seal.json", "w"), indent=1)
rows = []
for root, _, fs in os.walk(P):
    for f in sorted(fs):
        p = os.path.join(root, f); rel = os.path.relpath(p, P)
        if rel.startswith(".") or rel == "MANIFEST.tsv":
            continue
        n = pq.ParquetFile(p).metadata.num_rows if f.endswith(".parquet") else (sum(1 for l in open(p) if l.startswith(">")) if f.endswith(".faa") else "")
        rows.append((rel, n, os.path.getsize(p), files.get(rel) or sha(p)))
with open(f"{P}/MANIFEST.tsv", "w") as out:
    out.write("path\trows\tbytes\tsha256\n")
    for r in sorted(rows):
        out.write("\t".join(map(str, r)) + "\n")
print(json.dumps(dict(seal_sha256=combined, n_files=len(files))), flush=True)
