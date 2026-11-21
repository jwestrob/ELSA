#!/usr/bin/env python
"""Task 2: merged gene universe in which each adjacent broken-call-like split pair is one gene position.

Input: package universe ($RD/ws_dpann/package/universe), gold instances, rescue3 tokens, split pairs
($RD/repair/results/split_pairs.parquet: ga, gb = adjacent gids, same strand, distance 1).
Output under $RD/boundary/merged/:
  pkg/universe/{genes,contigs,genomes}.parquet  merged universe (gid = rank over genome, contig, position)
  pkg/gold/instances.parquet                    gold remapped (members deduplicated on merged positions)
  tokens_rescue3_merged.parquet                 gid, token: a merged position takes the rescue3 token of its
                                                longest piece (ties: lowest old gid)
  old2new.npy                                   old gid -> new gid (both pieces map to the merged gid)
  ribo_runs.parquet                             ribosomal g2/m16 runs remapped (gid_start, gid_end)
  instance_sets.parquet                         per instance: n_members old/new, contains_pair (both pieces of a
                                                pair among its members), touched (any member is a piece),
                                                span_has_piece (a piece inside the first-to-last member span)
  build.json                                    counts
Chains (a gene in two pairs) collapse into one position: every gb merges into the position of gb - 1.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
OUT = RD / "boundary/merged"
TOK = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    (OUT / "pkg/universe").mkdir(parents=True, exist_ok=True)
    (OUT / "pkg/gold").mkdir(parents=True, exist_ok=True)
    g = pd.read_parquet(PKG / "universe/genes.parquet").sort_values("gid").reset_index(drop=True)
    n = len(g)
    assert np.array_equal(g.gid.to_numpy(), np.arange(n))
    sp = pd.read_parquet(RD / "repair/results/split_pairs.parquet", columns=["ga", "gb"])
    ga, gb = sp.ga.to_numpy(np.int64), sp.gb.to_numpy(np.int64)
    assert (gb == ga + 1).all(), "split pairs must be adjacent (gb = ga + 1)"
    cc = g.contig_code.to_numpy()
    st = g.strand.to_numpy()
    assert (cc[ga] == cc[gb]).all() and (st[ga] == st[gb]).all()
    drop = np.zeros(n, dtype=bool)
    drop[gb] = True
    old2new = np.cumsum(~drop) - 1
    n_new = int(old2new[-1] + 1)
    in_pair = np.zeros(n, dtype=bool)
    in_pair[ga] = True
    in_pair[gb] = True
    res = dict(pairs=int(len(sp)), genes_in_pairs=int(in_pair.sum()), genes_in_two_pairs=int(2 * len(sp) - in_pair.sum()),
               positions_removed=int(drop.sum()), genes_old=n, genes_new=n_new)
    log(res)

    # merged gene table
    grp = old2new
    length = g.length.to_numpy()
    agg = pd.DataFrame(dict(new=grp, start=g.start.to_numpy(), end=g.end.to_numpy(), length=length))
    first = np.r_[0, np.flatnonzero(np.diff(grp)) + 1]
    keep = g.iloc[first].reset_index(drop=True).copy()
    keep["start"] = agg.groupby("new").start.min().to_numpy()
    keep["end"] = agg.groupby("new").end.max().to_numpy()
    keep["length"] = agg.groupby("new").length.sum().to_numpy()
    keep["n_pieces"] = np.bincount(grp)
    keep["gid"] = np.arange(n_new)
    pos_new = np.empty(n_new, dtype=np.int64)
    ccn = keep.contig_code.to_numpy()
    chg = np.r_[0, np.flatnonzero(np.diff(ccn)) + 1]
    run_id = np.cumsum(np.r_[True, np.diff(ccn) != 0]) - 1
    pos_new = np.arange(n_new) - chg[run_id]
    keep["position_index"] = pos_new
    keep.to_parquet(OUT / "pkg/universe/genes.parquet", index=False)
    np.save(OUT / "old2new.npy", old2new)

    contigs = pd.read_parquet(PKG / "universe/contigs.parquet").sort_values("first_gid").reset_index(drop=True)
    cnt = keep.groupby("contig_code").size()
    fg = keep.groupby("contig_code").gid.min()
    contigs["n_genes"] = cnt.reindex(contigs.contig_code).to_numpy()
    contigs["first_gid"] = fg.reindex(contigs.contig_code).to_numpy()
    assert contigs.n_genes.notna().all()
    contigs.to_parquet(OUT / "pkg/universe/contigs.parquet", index=False)
    genomes = pd.read_parquet(PKG / "universe/genomes.parquet")
    gcnt = keep.groupby("genome_code").size()
    res["genome_n_genes_matches_old"] = bool((genomes.set_index("genome_code").n_genes.sort_index().to_numpy()
                                              == g.groupby("genome_code").size().sort_index().to_numpy()).all())
    genomes["n_genes"] = gcnt.reindex(genomes.genome_code).to_numpy()
    genomes.to_parquet(OUT / "pkg/universe/genomes.parquet", index=False)

    # tokens: longest piece's rescue3 token
    t = pd.read_parquet(TOK).sort_values("gid")
    assert np.array_equal(t.gid.to_numpy(), np.arange(n))
    tok = t.token.to_numpy()
    order = np.lexsort((np.arange(n), -length, grp))  # by group, longest first, lowest gid
    firsts = order[np.r_[True, np.diff(grp[order]) != 0]]
    new_tok = tok[firsts]
    assert len(new_tok) == n_new
    pd.DataFrame(dict(gid=np.arange(n_new), token=new_tok)).to_parquet(OUT / "tokens_rescue3_merged.parquet", index=False)
    pa_ = tok[ga] != tok[gb]
    res["pairs_pieces_with_different_rescue3_tokens"] = int(pa_.sum())

    # gold
    inst = pd.read_parquet(PKG / "gold/instances.parquet")
    pos_of_new = keep.position_index.to_numpy()
    rows = []
    new_members, new_pos = [], []
    for r in inst.itertuples():
        m = np.asarray(r.member_gids, dtype=np.int64)
        nm = np.unique(old2new[m])
        new_members.append(nm.tolist())
        new_pos.append(pos_of_new[nm].tolist())
        mm = np.unique(m)
        nxt = mm[:-1] + 1
        # (x, x + 1) is a split pair exactly when x + 1 is a dropped gb
        contains = bool(np.any((mm[1:] == nxt) & drop[np.minimum(nxt, n - 1)]))
        lo, hi = int(m.min()), int(m.max())
        rows.append((r.instance_id, len(np.unique(m)), len(nm), contains, bool(in_pair[m].any()),
                     bool(in_pair[lo:hi + 1].any())))
    sets = pd.DataFrame(rows, columns=["instance_id", "n_old", "n_new", "contains_pair", "touched", "span_has_piece"])
    inst2 = inst.copy()
    inst2["member_gids"] = new_members
    inst2["member_positions"] = new_pos
    inst2["n_members_resolved"] = [len(x) for x in new_members]
    inst2["start_idx"] = [min(x) for x in new_pos]
    inst2["end_idx"] = [max(x) for x in new_pos]
    inst2["span_genes"] = inst2.end_idx - inst2.start_idx + 1
    inst2.to_parquet(OUT / "pkg/gold/instances.parquet", index=False)
    sets.to_parquet(OUT / "instance_sets.parquet", index=False)
    ok = inst.resolved.fillna(False) & inst.order_trusted.fillna(False)
    s2 = sets[ok.to_numpy()]
    res["instances_min2_old"] = int((s2.n_old >= 2).sum())
    res["instances_min2_new"] = int((s2.n_new >= 2).sum())
    res["instances_leaving_min2"] = int(((s2.n_old >= 2) & (s2.n_new < 2)).sum())
    res["leaving_by_caller"] = inst[ok][((s2.n_old >= 2) & (s2.n_new < 2)).to_numpy()].caller.value_counts().to_dict()
    res["contains_pair_min2_new"] = int((s2.contains_pair & (s2.n_new >= 2)).sum())
    res["touched_min2_old"] = int((s2.touched & (s2.n_old >= 2)).sum())
    res["unaffected_min2"] = int((~s2.span_has_piece & (s2.n_old >= 2)).sum())

    # ribosomal runs
    sys.path.insert(0, str(RD / "ribosomal/code"))
    runs = pd.read_parquet(RD / "ribosomal/gold/runs.parquet")
    runs = runs[(runs.run_set == "g2") & (runs.n_marked >= 16)].reset_index(drop=True)
    runs["gid_start"] = old2new[runs.gid_start.to_numpy()]
    runs["gid_end"] = old2new[runs.gid_end.to_numpy()]
    runs.to_parquet(OUT / "ribo_runs.parquet", index=False)
    res["seconds"] = round(time.time() - t0, 1)
    with open(OUT / "build.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    log(res)


if __name__ == "__main__":
    main()
