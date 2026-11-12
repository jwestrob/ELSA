#!/usr/bin/env python
"""Write one FASTA of all 4,257,531 universe proteins, header = gid, in gid order.

Sequences come straight from the MMseqs databases the pair tables were built on
(query: querydb keyed by the crosswalk key; reference: keggclusterdb keyed by
source_sequence_db_key), so the universe's mmseqs_key is the lookup key on each side.
Every sequence length is checked against the universe's ELSA length.
"""
import argparse
import json
import time

import duckdb
import numpy as np

from common import N, NGENES, NREF, UNIVERSE, stamp

DBS = {"query": N / "query/querydb", "ref": N / "reference/keggclusterdb"}


def read_db(path):
    con = duckdb.connect()
    idx = con.execute(f"""SELECT key, off, len FROM read_csv('{path}.index', delim='\t', header=false,
                          columns={{'key':'BIGINT','off':'BIGINT','len':'BIGINT'}})""").fetchnumpy()
    with open(path, "rb") as fh:
        data = fh.read()
    return idx, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    con = duckdb.connect()
    u = con.execute(f"SELECT gid, side, mmseqs_key, length FROM read_parquet('{UNIVERSE}') ORDER BY gid").fetchnumpy()
    assert len(u["gid"]) == NGENES and np.array_equal(u["gid"], np.arange(NGENES))
    seqs = [None] * NGENES
    stats = {}
    for side, path in DBS.items():
        idx, data = read_db(path)
        pos = {int(k): (int(o), int(l)) for k, o, l in zip(idx["key"], idx["off"], idx["len"])}
        rows = np.flatnonzero(u["side"] == side)
        bad_len = 0
        missing = 0
        for r in rows:
            k = int(u["mmseqs_key"][r])
            if k not in pos:
                missing += 1
                continue
            o, l = pos[k]
            s = data[o:o + l].rstrip(b"\x00").rstrip(b"\n")
            if len(s) != int(u["length"][r]):
                bad_len += 1
            seqs[r] = s
        stats[side] = dict(rows=int(len(rows)), db_entries=int(len(idx["key"])), missing=missing,
                           length_mismatch=bad_len)
        stamp(side, stats[side])
        del data
    assert all(v["missing"] == 0 and v["length_mismatch"] == 0 for v in stats.values()), stats
    assert stats["ref"]["rows"] == NREF
    with open(a.out, "wb", buffering=1 << 24) as fh:
        for g in range(NGENES):
            fh.write(b">%d\n" % g)
            fh.write(seqs[g])
            fh.write(b"\n")
    stats["seconds"] = round(time.time() - t0, 1)
    stats["residues"] = int(sum(len(s) for s in seqs))
    with open(a.out + ".stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    stamp("FASTA written", a.out, stats)


if __name__ == "__main__":
    main()
