#!/usr/bin/env python
"""Check B partner extraction (predeclared_verify.json): for the first 200 proteins of each stratum order, recompute the
slot call (pass code, Z lists) and write the best-ranked supporter whose own stored rows carry the proposed label itself.
Single-threaded (login node, serial). Writes verify/data/B_partners.tsv.

usage: partnersB.py
"""
import sys
import time

import numpy as np
import pandas as pd

SRD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue"
sys.path.insert(0, SRD + "/code")
import score as S  # noqa: E402
import pass_run as PR  # noqa: E402

G = S.G
T0 = time.time()


def main():
    gn = pd.read_parquet(S.GENES, columns=["gid", "protein_id", "length"])
    n = len(gn)
    G["n"] = n
    g2 = pd.read_parquet(S.GENES, columns=["genome_code", "contig_code"])
    G["gc"] = g2.genome_code.to_numpy(np.int64)
    G["cid"] = g2.contig_code.to_numpy(np.int64)
    G["cstart"] = np.r_[0, np.flatnonzero(np.diff(G["cid"])) + 1].astype(np.int64)
    G["g_start"] = np.r_[0, np.flatnonzero(np.diff(G["gc"])) + 1].astype(np.int64)
    lm = pd.read_parquet(S.GROUPS, columns=["genome_code", "group"]).sort_values("genome_code")
    G["lin_of_genome"] = pd.factorize(lm.group.to_numpy())[0].astype(np.int64)
    G["grp"] = G["lin_of_genome"]
    G["lin"] = G["lin_of_genome"][G["gc"]]
    uq = np.load(S.EMB + "/uid_of_row.npy").astype(np.int64)
    G["uq"] = uq
    pid2gid = pd.Series(np.arange(n), index=gn.protein_id.to_numpy())
    T = pd.read_csv(SRD + "/data/truth.tsv", sep="\t", keep_default_na=False)
    C = pd.read_csv(SRD + "/data/controls.tsv", sep="\t", keep_default_na=False)
    A = pd.read_parquet(SRD + "/data/annot.parquet")
    kp = pd.read_csv(SRD + "/data/ko_proxy.tsv", sep="\t")
    proxy = kp.groupby("ko").pf.apply(sorted).to_dict()
    vocab = sorted({x for v in A.pfam for x in v} | {x for v in A.ko for x in v} | set(kp.pf))
    for s_ in list(T.R_pfam) + list(T.R_ko) + list(C.R_pfam) + list(C.R_ko):
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
    T_ = S.RD + "/calib/table/"
    G["zoff"] = np.r_[0, np.cumsum(np.load(T_ + "counts.npy").astype(np.int64))]
    G["zG"] = np.memmap(T_ + "genome.bin", np.int16, "r")
    G["zL"] = np.memmap(T_ + "local.bin", np.int16, "r")
    G["zz"] = np.memmap(T_ + "z.bin", np.float16, "r")
    G["BONDS"], G["BPTR"], G["BVAL"], G["INLOC"] = {}, {}, {}, {}
    z = np.load(f"{S.RD}/long_loci/nodes/dpann_Z/bonds_obs.npz")
    k, d = z["k"].astype(np.int64), z["d"].astype(np.int64)
    G["BONDS"]["Z"] = (k, d, z["n"], z["ng"], z["lam"])
    e = np.r_[k, k + d]
    bi = np.r_[np.arange(len(k)), np.arange(len(k))]
    o = np.argsort(e, kind="stable")
    G["BPTR"]["Z"] = np.r_[0, np.cumsum(np.bincount(e, minlength=n))].astype(np.int64)
    G["BVAL"]["Z"] = bi[o].astype(np.int64)
    L = pd.read_parquet(f"{S.RD}/long_loci/loci/dpann_Z/{PR.ZOUT}.parquet")
    cov = np.zeros(n + 1, np.int64)
    np.add.at(cov, L.s.to_numpy(np.int64), 1)
    np.add.at(cov, L.e.to_numpy(np.int64) + 1, -1)
    G["INLOC"]["Z"] = np.cumsum(cov)[:n] > 0
    print(f"[{time.time() - T0:.0f}s] setup", flush=True)

    B = pd.read_csv(SRD + "/verify/data/B_order.tsv", sep="\t")
    rows, cache, bad = [], {}, 0
    for r in B.itertuples(index=False):
        res = PR.slot("Z", int(r.gid), None, cache)
        bad += res["bad"]
        same = res["cls"] == "other" and vocab[res["lab"]] == r.label
        Pp = res["Pp"]
        li = lid.get(r.label, -1)
        carry = [i for i, p in enumerate(Pp.tolist()) if li in set(G["LABV"][G["LABP"][p]:G["LABP"][p + 1]].tolist())]
        rec = dict(stratum=r.stratum, order=r.order, gid=r.gid, protein_id=r.protein_id, label=r.label, call_reproduced=same,
                   n_supporters=len(Pp), n_supporters_with_label=len(carry))
        if carry:
            p = int(Pp[carry[0]])
            rec.update(partner_gid=p, partner_protein_id=gn.protein_id.iat[p], partner_length=int(gn.length.iat[p]),
                       partner_supporter_index=carry[0] + 1, partner_genome_code=int(G["gc"][p]))
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(SRD + "/verify/data/B_partners.tsv", sep="\t", index=False)
    print(f"[{time.time() - T0:.0f}s] rows {len(out)}; calls reproduced {int(out.call_reproduced.sum())}; with partner "
          f"{int(out.partner_gid.notna().sum())}; bond gate mismatches {bad}", flush=True)
    print(out.groupby("stratum").agg(n=("gid", "size"), with_partner=("partner_gid", lambda x: int(x.notna().sum()))).to_string())


if __name__ == "__main__":
    main()
