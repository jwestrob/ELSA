#!/usr/bin/env python
"""Withheld-family test design (predeclared test_W): positives = within-instance consecutive pairs of each curated
system type with >= 10 of them (all genomes); negatives = 20,000 random no-gold adjacencies (seed 17), replicated
per type so that each type's negatives are scored under that type's withholding.

  L2 mask   MMseqs2 families of every member gene of every instance of the type
  L2 ctl    the same number of families drawn at random within the same family-size decile (seed 17), outside
            the type's own families
Writes $RD/kinds/withheld/{design.npz, types.tsv}.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
OUT = RD / "kinds/withheld"
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"{time.time() - T0:7.0f}s", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    tokM = np.load(RD / "kinds/prep/tokM.npy")
    TM = int(tokM.max()) + 1
    fsize = np.bincount(tokM, minlength=TM)
    adj = pd.read_parquet(RD / "grammar/prep/adj1.parquet", columns=["i", "genome_code", "gold_none"])
    row_of_left = pd.Series(np.arange(len(adj)), index=adj.i.to_numpy())
    inst = pd.read_parquet(PKG / "gold/instances.parquet", columns=["instance_id", "caller", "system_type"])
    mem = pd.read_parquet(PKG / "gold/members.parquet").merge(inst, on=["instance_id", "caller"])
    mem["type"] = mem.caller + ":" + mem.system_type.astype(str)
    # within-instance consecutive pairs
    m2 = mem[["instance_id", "caller", "gid", "type"]].copy()
    m2["nxt"] = m2.gid + 1
    pairs = m2.merge(m2[["instance_id", "caller", "gid"]].rename(columns=dict(gid="nxt")), on=["instance_id", "caller", "nxt"])
    pairs["row"] = row_of_left.reindex(pairs.gid.to_numpy()).to_numpy()
    pairs = pairs[pairs.row.notna()].drop_duplicates(["row", "type"])
    pairs["row"] = pairs.row.astype(np.int64)
    cnt = pairs.groupby("type").row.nunique()
    types = sorted(cnt[cnt >= 10].index)
    log("types with >= 10 within-instance adjacencies", len(types), "of", len(cnt))
    neg_rows = rng.choice(np.flatnonzero(adj.gold_none.to_numpy()), 20000, replace=False)
    deciles = np.quantile(fsize[fsize > 0], np.linspace(0, 1, 11))
    dec_of = np.clip(np.searchsorted(deciles, fsize, side="right") - 1, 0, 9)
    by_dec = [np.flatnonzero((dec_of == d) & (fsize > 0)) for d in range(10)]
    mask_l2 = np.zeros((len(types), TM), np.bool_)
    mask_ct = np.zeros((len(types), TM), np.bool_)
    qrow, qt, qpos = [], [], []
    trows = []
    for t, ty in enumerate(types):
        fams = np.unique(tokM[mem.gid[mem.type == ty].to_numpy()])
        mask_l2[t, fams] = True
        for d in range(10):
            k = int((dec_of[fams] == d).sum())
            if k:
                pool = by_dec[d][~mask_l2[t, by_dec[d]]]
                mask_ct[t, rng.choice(pool, k, replace=False)] = True
        pr = np.unique(pairs.row[pairs.type == ty].to_numpy())
        qrow += [pr, neg_rows]
        qt += [np.full(len(pr) + len(neg_rows), t)]
        qpos += [np.ones(len(pr), bool), np.zeros(len(neg_rows), bool)]
        trows.append(dict(type=ty, t=t, n_pos=len(pr), n_families_withheld=len(fams),
                          genes_withheld=int(fsize[fams].sum()), genes_ctl=int(fsize[mask_ct[t]].sum())))
    qrow = np.concatenate(qrow)
    qt = np.concatenate(qt).astype(np.int32)
    qpos = np.concatenate(qpos)
    qa = adj.i.to_numpy()[qrow].astype(np.int64)
    np.savez(OUT / "design.npz", qa=qa, qb=qa + 1, qrow=qrow, qtype=qt, is_pos=qpos, mask_l2=mask_l2, mid_l2=qt,
             mask_ct=mask_ct, mid_ct=qt, neg_rows=neg_rows)
    tt = pd.DataFrame(trows)
    tt.to_csv(OUT / "types.tsv", sep="\t", index=False)
    log(tt.to_string(index=False))
    log("queries", len(qa))


if __name__ == "__main__":
    main()
