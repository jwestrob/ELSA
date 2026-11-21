#!/usr/bin/env python
"""Biotite login node, serial (same setup as slot_rescue/verify/code/partnersB.py). For each of the 287 LacAB_rpiB-labeled
genes, recompute the Z-arm slot call and write every supporter (the best-positioned partner per supporting genome, the
genes that vote) with its rank. Also writes the same for annotated PF02502 carriers in Micrarchaeota given in
carriers.txt (reverse direction: what sits at their slot in partner genomes).
usage: 02_partners_remote.py genes.txt carriers.txt out_prefix"""
import sys, time
import numpy as np, pandas as pd
SRD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue"
sys.path.insert(0, SRD + "/code")
import score as S  # noqa
import pass_run as PR  # noqa
G = S.G; T0 = time.time()

def setup():
    gn = pd.read_parquet(S.GENES, columns=["gid", "protein_id", "length"])
    n = len(gn); G["n"] = n
    g2 = pd.read_parquet(S.GENES, columns=["genome_code", "contig_code"])
    G["gc"] = g2.genome_code.to_numpy(np.int64); G["cid"] = g2.contig_code.to_numpy(np.int64)
    G["cstart"] = np.r_[0, np.flatnonzero(np.diff(G["cid"])) + 1].astype(np.int64)
    G["g_start"] = np.r_[0, np.flatnonzero(np.diff(G["gc"])) + 1].astype(np.int64)
    lm = pd.read_parquet(S.GROUPS, columns=["genome_code", "group"]).sort_values("genome_code")
    G["lin_of_genome"] = pd.factorize(lm.group.to_numpy())[0].astype(np.int64)
    G["grp"] = G["lin_of_genome"]; G["lin"] = G["lin_of_genome"][G["gc"]]
    G["uq"] = np.load(S.EMB + "/uid_of_row.npy").astype(np.int64)
    pid2gid = pd.Series(np.arange(n), index=gn.protein_id.to_numpy())
    T = pd.read_csv(SRD + "/data/truth.tsv", sep="\t", keep_default_na=False)
    C = pd.read_csv(SRD + "/data/controls.tsv", sep="\t", keep_default_na=False)
    A = pd.read_parquet(SRD + "/data/annot.parquet")
    kp = pd.read_csv(SRD + "/data/ko_proxy.tsv", sep="\t")
    proxy = kp.groupby("ko").pf.apply(sorted).to_dict()
    vocab = sorted({x for v in A.pfam for x in v} | {x for v in A.ko for x in v} | set(kp.pf))
    for s_ in list(T.R_pfam) + list(T.R_ko) + list(C.R_pfam) + list(C.R_ko):
        vocab += [x for x in str(s_).split(";") if x]
    vocab = sorted(set(vocab)); lid = {x: i for i, x in enumerate(vocab)}; G["nlab"] = len(vocab)
    A["gid"] = A.protein_id.map(pid2gid).astype(np.int64)
    ep = A[["gid", "pfam"]].explode("pfam").dropna().rename(columns={"pfam": "lab"})
    ek = A[["gid", "ko"]].explode("ko").dropna().rename(columns={"ko": "lab"})
    lab = pd.concat([ep, ek], ignore_index=True); lab["i"] = lab.lab.map(lid).astype(np.int64)
    lab = lab.drop_duplicates(["gid", "i"]).sort_values(["gid", "i"], kind="stable")
    ek["key"] = ek.lab.map(lambda k: proxy.get(k, [k]))
    kk = ek[["gid", "key"]].explode("key").rename(columns={"key": "lab"})
    key = pd.concat([ep, kk], ignore_index=True); key["i"] = key.lab.map(lid).astype(np.int64)
    key = key.drop_duplicates(["gid", "i"]).sort_values(["gid", "i"], kind="stable")
    G["LABP"] = np.r_[0, np.cumsum(np.bincount(lab.gid.to_numpy(), minlength=n))].astype(np.int64)
    G["KEYP"] = np.r_[0, np.cumsum(np.bincount(key.gid.to_numpy(), minlength=n))].astype(np.int64)
    G["LABV"] = lab.i.to_numpy(np.int64); G["KEYV"] = key.i.to_numpy(np.int64)
    T_ = S.RD + "/calib/table/"
    G["zoff"] = np.r_[0, np.cumsum(np.load(T_ + "counts.npy").astype(np.int64))]
    G["zG"] = np.memmap(T_ + "genome.bin", np.int16, "r"); G["zL"] = np.memmap(T_ + "local.bin", np.int16, "r")
    G["zz"] = np.memmap(T_ + "z.bin", np.float16, "r")
    G["BONDS"], G["BPTR"], G["BVAL"], G["INLOC"] = {}, {}, {}, {}
    z = np.load(f"{S.RD}/long_loci/nodes/dpann_Z/bonds_obs.npz")
    k, d = z["k"].astype(np.int64), z["d"].astype(np.int64)
    G["BONDS"]["Z"] = (k, d, z["n"], z["ng"], z["lam"])
    e = np.r_[k, k + d]; bi = np.r_[np.arange(len(k)), np.arange(len(k))]; o = np.argsort(e, kind="stable")
    G["BPTR"]["Z"] = np.r_[0, np.cumsum(np.bincount(e, minlength=n))].astype(np.int64)
    G["BVAL"]["Z"] = bi[o].astype(np.int64)
    L = pd.read_parquet(f"{S.RD}/long_loci/loci/dpann_Z/{PR.ZOUT}.parquet")
    cov = np.zeros(n + 1, np.int64); np.add.at(cov, L.s.to_numpy(np.int64), 1); np.add.at(cov, L.e.to_numpy(np.int64) + 1, -1)
    G["INLOC"]["Z"] = np.cumsum(cov)[:n] > 0
    print(f"[{time.time() - T0:.0f}s] setup", flush=True)
    return gn, pid2gid, vocab

def run(pids, gn, pid2gid, vocab, out):
    rows, cache, summ = [], {}, []
    for q in pids:
        if q not in pid2gid.index:
            summ.append(dict(query=q, status="no_gid")); continue
        g = int(pid2gid[q]); r = PR.slot("Z", g, None, cache)
        summ.append(dict(query=q, gid=g, cls=r["cls"], lab=vocab[r["lab"]] if r["lab"] >= 0 else "", n_support=r["n_support"], nb=r["nb"], bad=r["bad"], listlen=r["listlen"]))
        for i, p in enumerate(r["Pp"].tolist()):
            rows.append((q, i + 1, int(p), gn.protein_id.iat[p]))
    pd.DataFrame(rows, columns=["query", "rank", "partner_gid", "partner_protein_id"]).to_csv(out + "_supporters.tsv", sep="\t", index=False)
    pd.DataFrame(summ).to_csv(out + "_summary.tsv", sep="\t", index=False)
    print(f"[{time.time() - T0:.0f}s] {out}: {len(pids)} queries, {len(rows)} supporter rows", flush=True)

if __name__ == "__main__":
    gn, pid2gid, vocab = setup()
    run([l.strip() for l in open(sys.argv[1]) if l.strip()], gn, pid2gid, vocab, sys.argv[3] + "_labeled")
    run([l.strip() for l in open(sys.argv[2]) if l.strip()], gn, pid2gid, vocab, sys.argv[3] + "_carriers")
