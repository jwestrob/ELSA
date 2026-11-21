#!/usr/bin/env python
"""galt step 6 (biotite login node, single-threaded serial): reconstruct, for the 76 GalP_UDP_transf-labeled genes,
the embedding list (Z arm: ws_calib table, z >= 10.6, own genome group excluded, ranked by z) and the alignment list
(A arm: MMseqs2 kNN aln__k1000, own group excluded, depth 1,000), and the slot supporters of each arm exactly as
slot_rescue/code/pass_run.py slot() selects them (best-ranked list partner per genome that lies 1-3 genes from a list
partner of a bonded neighbor). Code path: score.py z_chunk / a_chunk / support, the stored bonds (bonds_obs.npz).
usage: g06_voters_remote.py QUERY_IDS.txt OUT_DIR
Writes OUT_DIR/lists.tsv (query, arm, rank, partner, z) for the first 300 list entries and OUT_DIR/supporters.tsv
(query, arm, partner, rank)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue/code")
import score as S

G = S.G
q_ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]
OUT = sys.argv[2]
gn = pd.read_parquet(S.GENES, columns=["gid", "protein_id", "genome_code", "contig_code"])
n = len(gn)
G["n"] = n
G["gc"] = gn.genome_code.to_numpy(np.int64)
G["cid"] = gn.contig_code.to_numpy(np.int64)
G["g_start"] = np.r_[0, np.flatnonzero(np.diff(G["gc"])) + 1].astype(np.int64)
lm = pd.read_parquet(S.GROUPS, columns=["genome_code", "group"]).sort_values("genome_code")
G["lin_of_genome"] = pd.factorize(lm.group.to_numpy())[0].astype(np.int64)
G["grp"] = G["lin_of_genome"]
G["lin"] = G["lin_of_genome"][G["gc"]]
uq = np.load(S.EMB + "/uid_of_row.npy").astype(np.int64)
G["uq"] = uq
pid = gn.protein_id.to_numpy()
pid2gid = pd.Series(np.arange(n), index=pid)
T_ = S.RD + "/calib/table/"
G["zoff"] = np.r_[0, np.cumsum(np.load(T_ + "counts.npy").astype(np.int64))]
G["zG"] = np.memmap(T_ + "genome.bin", np.int16, "r")
G["zL"] = np.memmap(T_ + "local.bin", np.int16, "r")
G["zz"] = np.memmap(T_ + "z.bin", np.float16, "r")
G["I"] = np.load(S.RD + "/vocab_lists/mm/aln__k1000__I.npy", mmap_mode="r")
okq = uq >= 0
G["members"] = np.flatnonzero(okq)[np.argsort(uq[okq], kind="stable")].astype(np.int64)
G["mptr"] = np.r_[0, np.cumsum(np.bincount(uq[okq], minlength=G["I"].shape[0]))].astype(np.int64)
BP, BV, BK = {}, {}, {}
for arm in ("Z", "A"):
    z = np.load(f"{S.RD}/long_loci/nodes/dpann_{arm}/bonds_obs.npz")
    k, d = z["k"].astype(np.int64), z["d"].astype(np.int64)
    e = np.r_[k, k + d]
    bi = np.r_[np.arange(len(k)), np.arange(len(k))]
    o = np.argsort(e, kind="stable")
    BP[arm] = np.r_[0, np.cumsum(np.bincount(e, minlength=n))].astype(np.int64)
    BV[arm] = bi[o].astype(np.int64)
    BK[arm] = (k, d)


def zlist_with_z(g):
    u = uq[g]
    if u < 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    s, e = G["zoff"][u], G["zoff"][u + 1]
    z = np.asarray(G["zz"][s:e]).astype(np.float32)
    gg = np.asarray(G["zG"][s:e]).astype(np.int64)
    keep = np.flatnonzero((z >= S.ZCUT) & (G["lin_of_genome"][gg] != G["lin"][g]))
    o = keep[np.argsort(-z[keep], kind="mergesort")]
    p = G["g_start"][gg[o]] + np.asarray(G["zL"][s:e]).astype(np.int64)[o]
    return p, z[o]


cache = {}


def getl(arm, g):
    if (arm, g) not in cache:
        fn = S.z_chunk if arm == "Z" else S.a_chunk
        cache[(arm, g)] = fn(np.array([g], np.int64))[1].astype(np.int64)
    return cache[(arm, g)]


lrows, srows = [], []
for q in q_ids:
    if q not in pid2gid.index:
        continue
    g = int(pid2gid[q])
    zp, zz = zlist_with_z(g)
    for r, (p_, z_) in enumerate(zip(zp[:300], zz[:300])):
        lrows.append((q, "Z", r, pid[p_], round(float(z_), 2)))
    ap = getl("A", g)
    for r, p_ in enumerate(ap[:300]):
        lrows.append((q, "A", r, pid[p_], ""))
    for arm in ("Z", "A"):
        P = getl(arm, g)
        k, d = BK[arm]
        best = {}
        for b in BV[arm][BP[arm][g]:BP[arm][g + 1]]:
            j, dd = int(k[b]), int(d[b])
            other = j + dd if j == g else j
            ns, ngs, ug, pos = S.support(P, getl(arm, other))
            for G_, p_ in zip(ug.tolist(), pos.tolist()):
                if G_ not in best or p_ < best[G_]:
                    best[G_] = p_
        for p_ in sorted(best.values()):
            srows.append((q, arm, pid[P[p_]], p_))
pd.DataFrame(lrows, columns=["query", "arm", "rank", "partner", "z"]).to_csv(f"{OUT}/lists.tsv", sep="\t", index=False)
pd.DataFrame(srows, columns=["query", "arm", "partner", "rank"]).to_csv(f"{OUT}/supporters.tsv", sep="\t", index=False)
print(len(q_ids), "queries;", len(lrows), "list rows;", len(srows), "supporter rows")
