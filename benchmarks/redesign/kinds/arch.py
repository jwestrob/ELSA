#!/usr/bin/env python
"""Architectures as objects: the high-kappa region of pair space among pairs the token engine cannot call.

H = consecutive pairs, all genomes, token-engine uncallable, not homologous, kappa above the random same-genome
pair q99.9 and R >= 3. Edge p -- p' when p' is in p's support set (selected depth). Leiden modularity (seed 17).
Per architecture (>= 5 pairs): pairs, distinct MMseqs2 family pairs, genomes, orders, phyla (evaluation),
length profile, share with a singleton member, share both unannotated, observed Pfam domains per side (short /
long protein) over all member genes, curated calls on member genes. Also where the grammar critic's top-30
low-cosine types fall.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kinds as KD  # noqa: E402

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
K = RD / "kinds"
OUT = K / "arch"
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"{time.time() - T0:7.0f}s", *a, flush=True)


def topdom(gids, dom, k=4):
    c = Counter()
    for g in set(gids):
        for d in dom.get(g, ()):
            c[d] += 1
    n = len(set(gids))
    return "; ".join(f"{d} {v}/{n}" for d, v in c.most_common(k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--max-h", type=int, default=150_000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summ = json.load(open(K / "main/summary_main.json"))
    sel = summ["selected"]
    D = int(sel.split("_")[1][1:])
    q999 = summ["random_kappa_q50_q99_q999"][2]
    df = pd.read_parquet(K / "main/scores.parquet")
    H = df[df.uncallable & ~df.homologous & (df.kappa > q999) & (df.R >= 3)].copy()
    log("H", len(H), "selected", sel, "q999", q999)
    if len(H) > a.max_h:
        H = H.sort_values("kappa", ascending=False).iloc[:a.max_h]
    H["row"] = H.index.to_numpy()
    d = KD.Data()
    ptr, left = KD.support_lists(H.i.to_numpy(np.int64), H.j.to_numpy(np.int64), d.uq, d.I, D, d.mptr, d.members,
                                 d.gc, d.cid, d.U, d.n, d.bufcap)
    log("support lists", len(left))
    pos_of_left = pd.Series(np.arange(len(H)), index=H.i.to_numpy())
    src = np.repeat(np.arange(len(H)), np.diff(ptr))
    dst = pos_of_left.reindex(left).to_numpy()
    ok = ~np.isnan(dst)
    e = np.c_[src[ok], dst[ok].astype(np.int64)]
    e = e[e[:, 0] != e[:, 1]]
    e = np.unique(np.sort(e, axis=1), axis=0)
    log("edges within H", len(e))
    import igraph as ig
    import leidenalg as la
    gph = ig.Graph(n=len(H), edges=e.tolist())
    part = la.find_partition(gph, la.ModularityVertexPartition, seed=17)
    H["arch"] = np.array(part.membership)
    H["degree"] = np.array(gph.degree())
    gm = pd.read_parquet(PKG / "universe/genomes.parquet").sort_values("genome_code").reset_index(drop=True)
    H["phylum"] = gm.phylum.to_numpy()[H.genome_code]
    H["ord"] = gm.ord.to_numpy()[H.genome_code]
    L = np.load(K / "prep/length.npy")
    la_, lb_ = L[H.i], L[H.j]
    H["short_gid"] = np.where(la_ <= lb_, H.i, H.j)
    H["long_gid"] = np.where(la_ <= lb_, H.j, H.i)
    H["len_short"] = np.minimum(la_, lb_)
    H["len_long"] = np.maximum(la_, lb_)
    # observed domains
    genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "protein_id"])
    import duckdb
    con = duckdb.connect(str(RD / "ws_dpann/input/sharur.duckdb"), read_only=True)
    con.execute(f"SET threads={a.threads}")
    pf = con.execute("SELECT protein_id, name FROM annotations WHERE source='pfam'").fetchdf()
    pf = pf.merge(genes, on="protein_id")
    want = set(H.short_gid) | set(H.long_gid)
    pf = pf[pf.gid.isin(want)]
    dom = pf.groupby("gid").name.apply(lambda s: sorted(set(s))).to_dict()
    mem = pd.read_parquet(PKG / "gold/members.parquet").merge(
        pd.read_parquet(PKG / "gold/instances.parquet", columns=["instance_id", "caller", "system_type"]),
        on=["instance_id", "caller"])
    gold = (mem.caller + ":" + mem.system_type.astype(str)).groupby(mem.gid).apply(lambda s: sorted(set(s))).to_dict()
    rows = []
    for c, g in H.groupby("arch"):
        if len(g) < 5:
            continue
        fp = set(zip(np.minimum(g.ta, g.tb), np.maximum(g.ta, g.tb)))
        gl = Counter(t for x in list(g.short_gid) + list(g.long_gid) for t in gold.get(x, ()))
        rows.append(dict(arch=int(c), pairs=len(g), family_pairs=len(fp), genomes=g.genome_code.nunique(),
                         orders=g.ord.nunique(), phyla=g.phylum.nunique(),
                         phyla_list=";".join(f"{k}:{v}" for k, v in g.phylum.value_counts().items()),
                         len_short_med=float(g.len_short.median()), len_long_med=float(g.len_long.median()),
                         len_short_q10_q90=f"{g.len_short.quantile(.1):.0f}-{g.len_short.quantile(.9):.0f}",
                         len_long_q10_q90=f"{g.len_long.quantile(.1):.0f}-{g.len_long.quantile(.9):.0f}",
                         frac_singleton=float(g.singleton_either.mean()), frac_unannotated_both=float(g.unannotated_both.mean()),
                         frac_gold_member=float((~g.gold_none).mean()), kappa_med=float(g.kappa.median()),
                         R_med=float(g.R.median()), same_strand=float((g.config == 0).mean()),
                         dom_short=topdom(g.short_gid, dom), dom_long=topdom(g.long_gid, dom),
                         frac_short_pfam=float(np.mean([x in dom for x in g.short_gid])),
                         frac_long_pfam=float(np.mean([x in dom for x in g.long_gid])),
                         curated_calls="; ".join(f"{k} {v}" for k, v in gl.most_common(3)),
                         example=f"{g.i.iloc[0]}-{g.j.iloc[0]}"))
    A = pd.DataFrame(rows).sort_values("family_pairs", ascending=False)
    A.to_csv(OUT / "architectures.tsv", sep="\t", index=False)
    H[["row", "i", "j", "genome_code", "arch", "degree", "kappa", "R", "R0", "ta", "tb", "len_short", "len_long",
       "singleton_either", "eng_L", "unannotated_both", "gold_none", "phylum", "ord"]].to_parquet(OUT / "H_pairs.parquet",
                                                                                                 index=False)
    # grammar critic top-30 low-cosine types
    gt = pd.read_csv(RD / "grammar/eval/learned_top30_lowcos.tsv", sep="\t")
    ex = gt.example.str.split("->", expand=True).astype(np.int64)
    lo = np.minimum(ex[0], ex[1]).to_numpy()
    lut = pd.Series(H.arch.to_numpy(), index=H.i.to_numpy())
    gt["arch"] = lut.reindex(lo).to_numpy()
    rowlut = pd.Series(np.arange(len(df)), index=df.i.to_numpy())
    rr = rowlut.reindex(lo).to_numpy()
    gt["kappa"] = np.where(np.isnan(rr), np.nan, df.kappa.to_numpy()[np.nan_to_num(rr).astype(int)])
    gt["R"] = np.where(np.isnan(rr), np.nan, df.R.to_numpy()[np.nan_to_num(rr).astype(int)])
    gt["uncallable"] = np.where(np.isnan(rr), np.nan, df.uncallable.to_numpy()[np.nan_to_num(rr).astype(int)])
    gt["homologous"] = np.where(np.isnan(rr), np.nan, df.homologous.to_numpy()[np.nan_to_num(rr).astype(int)])
    gt[["type", "example", "len_a_med", "len_b_med", "uv_cluster", "kappa", "R", "uncallable", "homologous", "arch",
        "nbr_domains_a", "nbr_domains_b"]].to_csv(OUT / "grammar_top30_lowcos_arch.tsv", sep="\t", index=False)
    s = dict(H=int(len(H)), edges=int(len(e)), communities=int(H.arch.nunique()),
             architectures_ge5=int(len(A)), pairs_in_ge5=int(A.pairs.sum()) if len(A) else 0,
             grammar_top30_in_H=int(gt.arch.notna().sum()),
             grammar_pattern28_arch_counts=gt[gt.uv_cluster == gt.uv_cluster.mode().iloc[0]].arch.value_counts(dropna=False).to_dict(),
             q999=q999, depth=D)
    json.dump(s, open(OUT / "arch_summary.json", "w"), indent=1, default=str)
    log(s)
    log(A.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
