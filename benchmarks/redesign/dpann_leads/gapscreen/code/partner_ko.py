#!/usr/bin/env python
"""gapscreen step 1 (biotite, Slurm): for every labeled gene of the pass, recompute the slot call with the pass code
(slot_rescue/code/pass_run.py, Z lists) and tally the own KOs of the supporting partners that carried the winning label.

Per gene: n_supporters, n_carriers (supporters whose keys hold the winning label), n_carriers_ko (carriers with >= 1 KO),
ko_top (plurality KO among carriers; each carrier counts each of its KOs once; ties -> lowest KO), ko_top_n, ko_second,
ko_second_n, ko_counts (top 6 'KO:n'), rep_partner_gid (best-ranked carrier with ko_top), call_reproduced.
Writes OUT/partner_ko.parquet and OUT/partner_ko_run.json.

usage: partner_ko.py --threads T --out DIR [--smoke N]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

SRD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slot_rescue"
sys.path.insert(0, SRD + "/code")
import score as S  # noqa: E402
import pass_run as PR  # noqa: E402

G = S.G
T0 = time.time()


def setup():
    gn = pd.read_parquet(S.GENES, columns=["gid", "protein_id", "genome_code", "contig_code"])
    n = len(gn)
    G["n"] = n
    G["gc"] = gn.genome_code.to_numpy(np.int64)
    G["cid"] = gn.contig_code.to_numpy(np.int64)
    G["cstart"] = np.r_[0, np.flatnonzero(np.diff(G["cid"])) + 1].astype(np.int64)
    G["g_start"] = np.r_[0, np.flatnonzero(np.diff(G["gc"])) + 1].astype(np.int64)
    lm = pd.read_parquet(S.GROUPS, columns=["genome_code", "group"]).sort_values("genome_code")
    G["lin_of_genome"] = pd.factorize(lm.group.to_numpy())[0].astype(np.int64)
    G["grp"] = G["lin_of_genome"]
    G["lin"] = G["lin_of_genome"][G["gc"]]
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
    vocab = sorted(set(vocab))           # identical construction to pass_run.py (label ids, tie-breaks)
    lid = {x: i for i, x in enumerate(vocab)}
    G["nlab"] = len(vocab)
    G["ISKO"] = np.array([v.startswith("K") for v in vocab])
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
    return vocab, lid


def task(items):
    cache, out = {}, []
    vocab = G["VOCAB"]
    for g, kw in items:
        r = PR.slot("Z", g, None, cache)
        Pp = r["Pp"]
        carriers = [p for p in Pp.tolist() if kw in set(G["KEYV"][G["KEYP"][p]:G["KEYP"][p + 1]].tolist())]
        cnt, first = Counter(), {}
        nko = 0
        for p in carriers:
            kos = [x for x in G["LABV"][G["LABP"][p]:G["LABP"][p + 1]].tolist() if G["ISKO"][x]]
            if kos:
                nko += 1
            for x in set(kos):
                cnt[x] += 1
                first.setdefault(x, p)
        top = sorted(cnt.items(), key=lambda t: (-t[1], t[0]))
        out.append((g, int(r["cls"] == "other" and r["lab"] == kw), len(Pp), len(carriers), nko,
                    vocab[top[0][0]] if top else "", top[0][1] if top else 0,
                    vocab[top[1][0]] if len(top) > 1 else "", top[1][1] if len(top) > 1 else 0,
                    ";".join(f"{vocab[k]}:{c}" for k, c in top[:6]), int(first[top[0][0]]) if top else -1, int(r["bad"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    vocab, lid = setup()
    G["VOCAB"] = vocab
    P = pd.read_parquet(SRD + "/pass/dpann_slot_rescue_pass.parquet", columns=["protein_id", "gid", "proposed_label_positional_homology"])
    if a.smoke:
        P = P.head(a.smoke)
    items = list(zip(P.gid.astype(int).tolist(), [lid[x] for x in P.proposed_label_positional_homology]))
    print(f"[{time.time() - T0:.0f}s] setup; genes {len(items)}", flush=True)
    chunks = [items[i::a.threads * 16] for i in range(a.threads * 16)]
    chunks = [sorted(c) for c in chunks if c]
    with Pool(a.threads) as pool:
        res = [r for part in pool.map(task, chunks) for r in part]
    R = pd.DataFrame(res, columns=["gid", "call_reproduced", "n_supporters", "n_carriers", "n_carriers_ko", "ko_top", "ko_top_n",
                                   "ko_second", "ko_second_n", "ko_counts", "rep_partner_gid", "bond_gate_bad"])
    R = P.merge(R, on="gid", how="left")
    R.to_parquet(f"{a.out}/partner_ko.parquet", index=False)
    run = dict(genes=len(R), reproduced=int(R.call_reproduced.sum()), bond_gate_bad=int(R.bond_gate_bad.sum()),
               with_carrier_ko=int((R.n_carriers_ko > 0).sum()), seconds=round(time.time() - T0, 1))
    json.dump(run, open(f"{a.out}/partner_ko_run.json", "w"), indent=1)
    print(json.dumps(run), flush=True)
    if run["reproduced"] != len(R) or run["bond_gate_bad"]:
        raise SystemExit("reproduction or bond gate failed")


if __name__ == "__main__":
    main()
