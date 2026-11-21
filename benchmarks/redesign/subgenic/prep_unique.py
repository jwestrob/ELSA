#!/usr/bin/env python3
"""Deduplicate the DPANN proteins to unique sequences and write the query lists.

Outputs (in --out):
  uniq.faa          one record per unique sequence, header = uid (0-based, order of first gid)
  gid2uid.parquet   gid, uid                                   (3,585,303 rows)
  uniq.parquet      uid, length, rep_gid, n_genes, n_genomes, genome_code_first
  lists/            uid lists: giants (>= 2046 aa), bench20k (random non-giant), bench_giant200
  prep.json         counts and checks
"""
import argparse, json, time, hashlib
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--faa", required=True)
ap.add_argument("--genes", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--giant", type=int, default=2046)
ap.add_argument("--seed", type=int, default=17)
a = ap.parse_args()
t0 = time.time()

seq2uid = {}
uid_seq = []
gid_uid = []
gids = []
n_star = 0
cur_gid, chunks = None, []

def flush():
    global n_star
    s = "".join(chunks)
    if s.endswith("*"):
        s = s[:-1]; n_star += 1
    u = seq2uid.get(s)
    if u is None:
        u = len(uid_seq); seq2uid[s] = u; uid_seq.append(s)
    gids.append(cur_gid); gid_uid.append(u)

with open(a.faa) as fh:
    for line in fh:
        if line.startswith(">"):
            if cur_gid is not None:
                flush()
            cur_gid = int(line[1:].split()[0]); chunks = []
        else:
            chunks.append(line.strip())
    if cur_gid is not None:
        flush()

gids = np.asarray(gids, dtype=np.int64)
gid_uid = np.asarray(gid_uid, dtype=np.int64)
assert np.all(np.diff(gids) == 1) and gids[0] == 0, "FASTA is not in contiguous gid order"
U = len(uid_seq)
lens = np.fromiter((len(s) for s in uid_seq), dtype=np.int64, count=U)

genes = pq.read_table(a.genes, columns=["gid", "genome_code", "length"]).to_pandas().sort_values("gid")
assert len(genes) == len(gids) and np.array_equal(genes.gid.values, gids)
# universe `length` counts a stored stop for some genomes: it is FASTA length + 1 there
ldiff = genes.length.values - lens[gid_uid]
assert np.isin(ldiff, [0, 1]).all(), f"unexpected length differences {np.unique(ldiff)}"
n_ldiff1 = int((ldiff == 1).sum())
gc = genes.genome_code.values
# genomes whose universe length is +1 for every gene (stored stop) vs mixed
gdiff = np.bincount(gc, weights=(ldiff == 1)) / np.bincount(gc)
n_g_all = int((gdiff == 1).sum()); n_g_mixed = int(((gdiff > 0) & (gdiff < 1)).sum())

rep_gid = np.full(U, -1, dtype=np.int64)
order = np.argsort(gid_uid, kind="stable")
first = np.r_[0, np.flatnonzero(np.diff(gid_uid[order])) + 1]
rep_gid[gid_uid[order][first]] = gids[order][first]
n_genes = np.bincount(gid_uid, minlength=U)
pairs = np.unique(np.stack([gid_uid, gc], 1), axis=0)
n_genomes = np.bincount(pairs[:, 0], minlength=U)
gfirst = gc[rep_gid]

with open(f"{a.out}/uniq.faa", "w") as out:
    for u, s in enumerate(uid_seq):
        out.write(f">{u}\n{s}\n")
pq.write_table(pa.table({"gid": gids, "uid": gid_uid}), f"{a.out}/gid2uid.parquet")
pq.write_table(pa.table({"uid": np.arange(U), "length": lens, "rep_gid": rep_gid, "n_genes": n_genes,
                         "n_genomes": n_genomes, "genome_code_first": gfirst}), f"{a.out}/uniq.parquet")

import os
os.makedirs(f"{a.out}/lists", exist_ok=True)
rng = np.random.default_rng(a.seed)
giants = np.flatnonzero(lens >= a.giant)
nong = np.flatnonzero(lens < a.giant)
bench = np.sort(rng.choice(nong, 20000, replace=False))
bench_g = np.sort(rng.choice(giants, 200, replace=False))
# length control for the repeat measurement: random non-giants of 1,000-2,045 aa, same count as giants
mid = np.flatnonzero((lens >= 1000) & (lens < a.giant))
lenctl = np.sort(rng.choice(mid, min(len(giants), len(mid)), replace=False))
for name, arr in [("giants", giants), ("bench20k", bench), ("bench_giant200", bench_g), ("lenctl", lenctl),
                  ("nongiants", nong)]:
    np.savetxt(f"{a.out}/lists/{name}.uid", arr, fmt="%d")

res = {"genes": int(len(gids)), "unique": int(U), "trailing_star_stripped": int(n_star),
       "universe_length_is_fasta_plus1_genes": n_ldiff1, "genomes_all_plus1": n_g_all, "genomes_mixed_plus1": n_g_mixed,
       "giants_unique_by_universe_length": int(len(np.unique(gid_uid[genes.length.values >= a.giant]))),
       "residues_all_genes": int(lens[gid_uid].sum()), "residues_unique": int(lens.sum()),
       "giants_unique": int(len(giants)), "giants_genes": int(np.isin(gid_uid, giants).sum()),
       "longest": int(lens.max()), "unique_lt160": int((lens < 160).sum()),
       "residues_giants": int(lens[giants].sum()), "residues_nongiants": int(lens[nong].sum()),
       "bench20k_residues": int(lens[bench].sum()), "bench_giant200_residues": int(lens[bench_g].sum()),
       "lenctl_n": int(len(lenctl)), "unique_in_ge2_genomes": int((n_genomes >= 2).sum()),
       "seconds": round(time.time() - t0, 1)}
json.dump(res, open(f"{a.out}/prep.json", "w"), indent=1)
print(json.dumps(res, indent=1))
