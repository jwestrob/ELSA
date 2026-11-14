#!/usr/bin/env python3
"""Rescue error, part 1: gene truncation flags and the simulated-truncation (cut) test set.

usage: cut_test_make.py <repair dir> <subgenic dir> <package dir>
Writes <repair dir>/partial.parquet  gid, partial, left_trunc, right_trunc, trunc, intact_n, intact_c, flag_source
       <repair dir>/cut/frag.faa     header = fragment id
       <repair dir>/cut/frag_meta.parquet, <repair dir>/cut/make.json
Design: predeclared.json amendment_1_rescue_error.cut_test.
"""
import json, os, sys, time
import numpy as np, pandas as pd, duckdb

R, SG, P = sys.argv[1:4]
t0 = time.time()
os.makedirs(f"{R}/cut", exist_ok=True)
con = duckdb.connect(); con.execute(f"SET threads={os.environ.get('SLURM_CPUS_ON_NODE', '2')}")
con.execute(f"ATTACH '{P}/../input/sharur.duckdb' AS s (READ_ONLY)")
F = con.execute(f"""SELECT g.gid, p.partial, g.strand, g.start, g."end" stop, ct.length_nt
   FROM '{P}/universe/genes.parquet' g LEFT JOIN s.proteins p USING (protein_id)
   JOIN '{P}/universe/contigs.parquet' ct USING (contig_code) ORDER BY g.gid""").df()
assert len(F) == 3585303 and (F.gid.values == np.arange(len(F))).all()
has = F.partial.notna().values
pl = F.partial.fillna("xx").values
left = np.where(has, np.char.startswith(pl.astype(str), "1"), F.start.values <= 3)
right = np.where(has, np.char.endswith(pl.astype(str), "1"), F.stop.values >= F.length_nt.values - 2)
plus = F.strand.values > 0
n_missing = np.where(plus, left, right); c_missing = np.where(plus, right, left)
out = pd.DataFrame({"gid": F.gid.values, "partial": F.partial.values, "left_trunc": left, "right_trunc": right,
                    "trunc": left | right, "intact_n": ~n_missing, "intact_c": ~c_missing,
                    "flag_source": np.where(has, "partial", "contig_end")})
out.to_parquet(f"{R}/partial.parquet", index=False)
flags = {"genes": int(len(out)), "partial_present": int(has.sum()), "trunc": int(out.trunc.sum()),
         "trunc_from_partial": int((out.trunc & has).sum()), "trunc_from_contig_end": int((out.trunc & ~has).sum()),
         "no_intact_side": int((~out.intact_n & ~out.intact_c).sum())}

G = pd.read_parquet(f"{R}/genes.parquet")
G["partial"] = F.partial.values
rng = np.random.default_rng(17)
LB = pd.cut(G.len, [0, 150, 300, 600, 10 ** 9], right=False, labels=["<150", "150-299", "300-599", ">=600"])
FS = pd.cut(G.tsize, [3, 5, 10, 100, 1000, 10 ** 9], right=False, labels=["3-4", "5-9", "10-99", "100-999", ">=1000"])
complete = (G.partial == "00").values & (G.len.values >= 34)
rows, avail = [], {}
for fs in FS.cat.categories:
    for lb in LB.cat.categories:
        cand = np.flatnonzero(complete & (FS == fs).values & (LB == lb).values)
        avail[f"pos|{fs}|{lb}"] = int(len(cand))
        pick = np.sort(rng.choice(cand, min(300, len(cand)), replace=False))
        rows += [("pos", fs, lb, g) for g in pick]
for lb in LB.cat.categories:
    cand = np.flatnonzero(complete & (G.tsize.values == 1) & (LB == lb).values)
    avail[f"neg|1|{lb}"] = int(len(cand))
    pick = np.sort(rng.choice(cand, min(500, len(cand)), replace=False))
    rows += [("neg", "1", lb, g) for g in pick]
M = pd.DataFrame(rows, columns=["kind", "fam_stratum", "len_stratum", "gid"])
M["fid"] = np.arange(len(M))
M["uid"] = G.uid.values[M.gid.values]; M["token"] = G.token.values[M.gid.values]
M["tsize"] = G.tsize.values[M.gid.values]; M["plen"] = G.len.values[M.gid.values]
M["side_kept"] = np.where(rng.random(len(M)) < 0.5, "N", "C")
frac = rng.uniform(0.15, 0.95, len(M))
M["klen"] = np.minimum(np.maximum(30, np.rint(frac * M.plen.values)).astype(int), M.plen.values)
M["kept_frac"] = M.klen / M.plen
need = set(M.uid.tolist()); seqs = {}
with open(f"{SG}/uniq.faa") as fh:
    u = None
    for line in fh:
        if line[0] == ">":
            u = int(line[1:])
        elif u in need:
            seqs[u] = line.strip()
with open(f"{R}/cut/frag.faa", "w") as out_f:
    for fid, u, side, k, L in zip(M.fid, M.uid, M.side_kept, M.klen, M.plen):
        s = seqs[u]; assert len(s) == L
        out_f.write(f">{fid}\n{s[:k] if side == 'N' else s[L - k:]}\n")
M.to_parquet(f"{R}/cut/frag_meta.parquet", index=False)
res = {"flags": flags, "available_per_stratum": avail, "positives": int((M.kind == "pos").sum()),
       "negatives": int((M.kind == "neg").sum()), "frag_len_q10_q50_q90": M.klen.quantile([0.1, 0.5, 0.9]).tolist(),
       "seconds": round(time.time() - t0, 1)}
json.dump(res, open(f"{R}/cut/make.json", "w"), indent=1)
print(json.dumps(res, indent=1))
