#!/usr/bin/env python3
"""Split a protein FASTA into K contiguous shards, byte for byte.

Usage: shard.py SRC.faa N_EXPECTED K OUTDIR [--consecutive]

Every header must be '>' followed by a non-negative integer and nothing else, and headers must be strictly
increasing. With --consecutive the header of record r must equal r (gid = 0..N-1, the package scheme).
Shard s receives the next N//K (+1 for the first N%K shards) records; lines are copied unchanged, so the
concatenation of the shards equals the source file. After writing, the shards are read back from disk and the
SHA-256 of their concatenation is checked against the SHA-256 of the source.

Writes OUTDIR/shard_NNN.faa, OUTDIR/shards.tsv (shard, first_gid, last_gid, proteins, residues, bytes, sha256)
and OUTDIR/SHARDS.json. Single process, streaming (login node).
"""
import datetime
import hashlib
import json
import os
import sys

BLOCK = 1 << 24


def fail(msg):
    sys.exit(f"SHARD FAILED: {msg}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    consecutive = "--consecutive" in sys.argv[1:]
    src, n_exp, k, outdir = args[0], int(args[1]), int(args[2]), args[3]
    if os.path.exists(outdir) and os.listdir(outdir):
        fail(f"{outdir} exists and is not empty")
    os.makedirs(outdir, exist_ok=True)
    sizes = [n_exp // k + (1 if s < n_exp % k else 0) for s in range(k)]
    ends = [sum(sizes[: s + 1]) for s in range(k)]  # record index one past each shard

    src_h = hashlib.sha256()
    stats = []
    s, rec, prev = 0, 0, -1
    cur = None

    def open_shard(idx):
        path = os.path.join(outdir, f"shard_{idx:03d}.faa")
        return {"shard": idx, "path": path, "fh": open(path, "wb", buffering=BLOCK), "h": hashlib.sha256(),
                "first": None, "last": None, "n": 0, "residues": 0, "bytes": 0}

    def close_shard(c):
        c["fh"].close()
        stats.append({"shard": c["shard"], "first_gid": c["first"], "last_gid": c["last"], "proteins": c["n"],
                      "residues": c["residues"], "bytes": c["bytes"], "sha256": c["h"].hexdigest()})

    cur = open_shard(0)
    first_line = True
    with open(src, "rb", buffering=BLOCK) as fh:
        for line in fh:
            src_h.update(line)
            if line.startswith(b">"):
                if rec == ends[s]:
                    close_shard(cur)
                    s += 1
                    cur = open_shard(s)
                tok = line[1:].rstrip(b"\n")
                if not tok.isdigit() or (len(tok) > 1 and tok.startswith(b"0")):
                    fail(f"record {rec}: header {line[:60]!r} is not a plain non-negative integer")
                g = int(tok)
                if consecutive and g != rec:
                    fail(f"record {rec}: header gid {g} differs from its row index")
                if g <= prev:
                    fail(f"record {rec}: header gid {g} is not greater than the previous gid {prev}")
                prev = g
                rec += 1
                if cur["first"] is None:
                    cur["first"] = g
                cur["last"] = g
                cur["n"] += 1
            else:
                if first_line:
                    fail("the file does not start with a header line")
                cur["residues"] += len(line.rstrip(b"\n"))
            first_line = False
            cur["fh"].write(line)
            cur["h"].update(line)
            cur["bytes"] += len(line)
    close_shard(cur)
    if rec != n_exp:
        fail(f"{rec} records in {src}, expected {n_exp}")
    if len(stats) != k or [x["proteins"] for x in stats] != sizes:
        fail(f"shard sizes {[x['proteins'] for x in stats]} differ from the plan {sizes}")
    src_sha = src_h.hexdigest()

    # Read the shards back from disk: their concatenation must reproduce the source bytes.
    cat_h = hashlib.sha256()
    for x in stats:
        h = hashlib.sha256()
        with open(os.path.join(outdir, f"shard_{x['shard']:03d}.faa"), "rb") as fh:
            while True:
                b = fh.read(BLOCK)
                if not b:
                    break
                h.update(b)
                cat_h.update(b)
        if h.hexdigest() != x["sha256"]:
            fail(f"shard {x['shard']} read back with a different SHA-256")
    if cat_h.hexdigest() != src_sha:
        fail("the concatenated shards differ from the source file")

    cols = ["shard", "first_gid", "last_gid", "proteins", "residues", "bytes", "sha256"]
    with open(os.path.join(outdir, "shards.tsv"), "w") as fo:
        fo.write("\t".join(cols) + "\n")
        for x in stats:
            fo.write("\t".join(str(x[c]) for c in cols) + "\n")
    info = {
        "source": os.path.abspath(src),
        "source_sha256": src_sha,
        "source_bytes": os.path.getsize(src),
        "proteins": rec,
        "residues": sum(x["residues"] for x in stats),
        "consecutive_gids": consecutive,
        "shards": k,
        "proteins_per_shard_min": min(sizes),
        "proteins_per_shard_max": max(sizes),
        "concatenation_sha256_equals_source": True,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": os.path.abspath(__file__),
    }
    with open(os.path.join(outdir, "SHARDS.json"), "w") as fo:
        json.dump(info, fo, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
