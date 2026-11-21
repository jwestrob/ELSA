#!/usr/bin/env python
"""slot_rescue POST HOC diagnostic (not predeclared): list depth versus the positional filter.

Same query rows, lists, labels and vote rule as score.py (imported; its G is filled here). Per row and arm:
  full      vote over every list partner (one vote per partner)
  fullg     vote over the best-ranked partner of every genome in the list (the slot call's pooling, no adjacency filter)
  combfull  vote over the best-ranked slot-consistent partner of every genome (adjacency to a partner of any gene within
            3 positions; no bond significance, no locus)
Writes results/posthoc.parquet.

usage: posthoc.py --threads T
"""
import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import score as S

G = S.G


def best_per_genome(P):
    if len(P) == 0:
        return np.zeros(0, np.int64)
    _, first = np.unique(G["gc"][P], return_index=True)
    return np.sort(first)


def work(rows):
    out = []
    cid, n = G["cid"], G["n"]
    for r in rows:
        g = int(r["gid"])
        for arm in ("Z", "A"):
            P = S.get_list(arm, g)
            right = S.right_mask(P, r["Rids"])
            rec = dict(row_id=r["row_id"], arm=arm)
            c, lab, cR, co, na = S.vote(P, right)
            rec.update(full_class=c, full_label=lab, full_cR=cR, full_cother=co)
            pos = best_per_genome(P)
            c, lab, cR, co, na = S.vote(P[pos], right[pos])
            rec.update(fullg_class=c, fullg_label=lab, fullg_cR=cR, fullg_cother=co, fullg_n=len(pos))
            nb = [g + d for d in S.OFFS if 0 <= g + d < n and cid[g + d] == cid[g]]
            ol = [S.get_list(arm, k) for k in nb]
            ol = np.concatenate(ol) if ol else np.zeros(0, np.int64)
            msk = np.isin(P, S.marked_around(ol)) if len(P) else np.zeros(0, bool)
            idx = np.flatnonzero(msk)
            sub = idx[best_per_genome(P[idx])] if len(idx) else idx
            c, lab, cR, co, na = S.vote(P[sub], right[sub])
            rec.update(combfull_class=c, combfull_label=lab, combfull_cR=cR, combfull_cother=co, combfull_n=len(sub))
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    th = a.threads
    t0 = time.time()
    g = pd.read_parquet(S.GENES, columns=["gid", "protein_id", "genome_code", "contig_code"])
    n = len(g)
    G["n"] = n
    G["gc"] = g.genome_code.to_numpy(np.int64)
    G["cid"] = g.contig_code.to_numpy(np.int64)
    G["g_start"] = np.r_[0, np.flatnonzero(np.diff(G["gc"])) + 1].astype(np.int64)
    lm = pd.read_parquet(S.GROUPS, columns=["genome_code", "group"]).sort_values("genome_code")
    G["lin_of_genome"] = pd.factorize(lm.group.to_numpy())[0].astype(np.int64)
    G["grp"] = G["lin_of_genome"]
    G["lin"] = G["lin_of_genome"][G["gc"]]
    uq = np.load(S.EMB + "/uid_of_row.npy").astype(np.int64)
    G["uq"] = uq
    pid2gid = pd.Series(np.arange(n), index=g.protein_id.to_numpy())
    T = pd.read_csv(S.SRD + "/data/truth.tsv", sep="\t", keep_default_na=False)
    C = pd.read_csv(S.SRD + "/data/controls.tsv", sep="\t", keep_default_na=False)
    Qr = pd.concat([T[["protein_id", "R_pfam", "R_ko"]], C[["protein_id", "R_pfam", "R_ko"]]], ignore_index=True)
    Qr["row_id"] = np.arange(len(Qr))
    Qr["gid"] = Qr.protein_id.map(pid2gid).astype(np.int64)
    A = pd.read_parquet(S.SRD + "/data/annot.parquet")
    kp = pd.read_csv(S.SRD + "/data/ko_proxy.tsv", sep="\t")
    proxy = kp.groupby("ko").pf.apply(sorted).to_dict()
    vocab = sorted({x for v in A.pfam for x in v} | {x for v in A.ko for x in v} | set(kp.pf))
    for s_ in list(Qr.R_pfam) + list(Qr.R_ko):
        vocab += [x for x in str(s_).split(";") if x]
    vocab = sorted(set(vocab))
    lid = {x: i for i, x in enumerate(vocab)}
    G["nlab"] = len(vocab)
    A["gid"] = A.protein_id.map(pid2gid).astype(np.int64)
    ep = A[["gid", "pfam"]].explode("pfam").dropna().rename(columns={"pfam": "lab"})
    ek = A[["gid", "ko"]].explode("ko").dropna().rename(columns={"ko": "lab"})
    lab = pd.concat([ep, ek], ignore_index=True)
    lab["i"] = lab.lab.map(lid).astype(np.int64)
    lab = lab.drop_duplicates(["gid", "i"]).sort_values(["gid", "i"], kind="stable")
    ek["key"] = ek.lab.map(lambda k: proxy.get(k, [k]))
    kk = ek[["gid", "key"]].explode("key").rename(columns={"key": "lab"})
    key = pd.concat([ep, kk], ignore_index=True)
    key["i"] = key.lab.map(lid).astype(np.int64)
    key = key.drop_duplicates(["gid", "i"]).sort_values(["gid", "i"], kind="stable")
    G["LABP"] = np.r_[0, np.cumsum(np.bincount(lab.gid.to_numpy(), minlength=n))].astype(np.int64)
    G["KEYP"] = np.r_[0, np.cumsum(np.bincount(key.gid.to_numpy(), minlength=n))].astype(np.int64)
    G["LABV"] = lab.i.to_numpy(np.int64)
    G["KEYV"] = key.i.to_numpy(np.int64)
    Qr["Rids"] = [np.array(sorted({lid[x] for x in (str(p).split(";") + str(k).split(";")) if x}), np.int64)
                  for p, k in zip(Qr.R_pfam, Qr.R_ko)]
    qg = np.unique(Qr.gid.to_numpy())
    x = qg[:, None] + np.r_[0, S.OFFS][None, :]
    ok = (x >= 0) & (x < n)
    xc = np.where(ok, x, 0)
    ok &= G["cid"][xc] == G["cid"][qg][:, None]
    need = np.unique(x[ok])
    G["need"] = need
    T_ = S.RD + "/calib/table/"
    G["zoff"] = np.r_[0, np.cumsum(np.load(T_ + "counts.npy").astype(np.int64))]
    G["zG"] = np.memmap(T_ + "genome.bin", np.int16, "r")
    G["zL"] = np.memmap(T_ + "local.bin", np.int16, "r")
    G["zz"] = np.memmap(T_ + "z.bin", np.float16, "r")
    G["LP"], G["LV"] = {}, {}
    G["LP"]["Z"], G["LV"]["Z"] = S.build_lists("Z", need, th)
    G["I"] = np.load(S.RD + "/vocab_lists/mm/aln__k1000__I.npy", mmap_mode="r")
    okq = uq >= 0
    G["members"] = np.flatnonzero(okq)[np.argsort(uq[okq], kind="stable")].astype(np.int64)
    G["mptr"] = np.r_[0, np.cumsum(np.bincount(uq[okq], minlength=G["I"].shape[0]))].astype(np.int64)
    G["LP"]["A"], G["LV"]["A"] = S.build_lists("A", need, th)
    print(f"[{time.time() - t0:.1f}s] lists Z {len(G['LV']['Z'])} A {len(G['LV']['A'])}", flush=True)
    recs = Qr[["row_id", "gid", "Rids"]].to_dict("records")
    chunks = [recs[i::th * 4] for i in range(th * 4)]
    with Pool(th) as pool:
        res = pool.map(work, chunks)
    out = pd.DataFrame([r for part in res for r in part]).sort_values(["row_id", "arm"])
    for c in [c for c in out.columns if c.endswith("_label")]:
        out[c] = [vocab[v] if v >= 0 else "" for v in out[c]]
    out.to_parquet(S.SRD + "/results/posthoc.parquet", index=False)
    print(f"[{time.time() - t0:.1f}s] done rows {len(out)}", flush=True)


if __name__ == "__main__":
    main()
