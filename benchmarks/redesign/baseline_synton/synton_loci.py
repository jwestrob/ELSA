#!/usr/bin/env python
"""Turn the pipeline's syntons into locus intervals and collect run statistics (no gold is read here).

Inputs (run directory of compute_syntons.ipynb, parameters X=12, J=0.15, G=2, T=80):
  results/syntons_all_12_0.15_2_80.parquet   (synton, gene)
  results/ab50_with_pvalues_12_0.15.parquet  (A50, B50, k_obs, lambda, p_value, fdr_q_value, ..., sig_score)
  results/ab_12_0.15_2_80.parquet            (a, b, k_obs, sig_score) gene links
  results/context_clusters_12_0.15.parquet   (representative_50, rep_nc_soft, gene)
  data/genes.parquet, data/clusters.parquet  (adapters)
Outputs (run_dir/post/):
  loci.parquet        one row per synton: genome, contig, contig_code, start, end (position_index), n_genes, span, side
  loci_paperq.parquet same with links restricted to pairs with q < 1e-5 (only if that removes any link)
  stats.json
"""
import argparse
import json
import time
from pathlib import Path

import duckdb
import networkx as nx
import numpy as np
import pandas as pd

TAG = "12_0.15"
STAG = "12_0.15_2_80"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def q(x, qs=(.1, .25, .5, .75, .9, .99, 1.0)):
    x = np.asarray(x)
    return [float(v) for v in np.quantile(x, qs)] if len(x) else []


def intervals(syn, uni):
    """syn: DataFrame(synton, gid) -> one interval per synton."""
    g = uni.loc[syn.gid.to_numpy()]
    d = pd.DataFrame(dict(synton=syn.synton.to_numpy(), contig_code=g.contig_code.to_numpy(),
                          pos=g.position_index.to_numpy(), genome=g.genome.to_numpy(), contig=g.contig.to_numpy(),
                          side=g.side.to_numpy()))
    agg = d.groupby("synton").agg(contig_code=("contig_code", "first"), n_contigs=("contig_code", "nunique"),
                                  genome=("genome", "first"), contig=("contig", "first"), side=("side", "first"),
                                  start=("pos", "min"), end=("pos", "max"), n_genes=("pos", "size")).reset_index()
    assert (agg.n_contigs == 1).all(), "a synton spans two replicons"
    agg["span"] = agg.end - agg.start + 1
    return agg.drop(columns="n_contigs")


def gene_cover(loci, uni_n, uni):
    """genes covered by the union of intervals (holes included)."""
    key = uni.contig_code.to_numpy().astype(np.int64) * 10**7 + uni.position_index.to_numpy()
    order = np.argsort(key, kind="stable")
    skey = key[order]
    lo = np.searchsorted(skey, loci.contig_code.to_numpy().astype(np.int64) * 10**7 + loci.start.to_numpy())
    hi = np.searchsorted(skey, loci.contig_code.to_numpy().astype(np.int64) * 10**7 + loci.end.to_numpy(), side="right")
    d = np.zeros(uni_n + 1, dtype=np.int64)
    np.add.at(d, lo, 1)
    np.add.at(d, hi, -1)
    cov = np.zeros(uni_n, dtype=bool)
    cov[order] = np.cumsum(d)[:uni_n] > 0
    return cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--universe", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    post = run / "post"
    post.mkdir(exist_ok=True)
    t0 = time.time()
    uni = pd.read_parquet(Path(a.universe) / "genes.parquet")
    if "side" not in uni:
        uni["side"] = "ref"
    uni = uni.sort_values("gid").set_index("gid", drop=False)
    n = len(uni)
    con = duckdb.connect()
    con.execute(f"set threads={max(1, int(__import__('os').environ.get('SLURM_CPUS_ON_NODE', '4')))}")
    st = dict(run=str(run), genes=n)

    # ---- usable context (their definition: replicons with >= 48 genes; window 13..24 on each side)
    clen = uni.groupby("contig_code").size()
    L = clen.loc[uni.contig_code.to_numpy()].to_numpy()
    p = uni.position_index.to_numpy()
    left = np.clip(p - 12, 0, 12)            # offsets 13..24 to the left exist when p - d >= 0
    right = np.clip((L - 1 - p) - 12, 0, 12)
    on48 = L >= 48
    st["usable_context"] = dict(
        genes_on_replicons_ge48=int(on48.sum()), frac_on_replicons_ge48=float(on48.mean()),
        genes_nonempty_window=int((on48 & (left + right > 0)).sum()),
        frac_nonempty_window=float((on48 & (left + right > 0)).mean()),
        genes_full_window=int((on48 & (left + right == 24)).sum()),
        frac_full_window=float((on48 & (left + right == 24)).mean()),
        replicons=int(len(clen)), replicons_ge48=int((clen >= 48).sum()))
    log("usable context", st["usable_context"])

    # ---- families and context clusters
    cl = pd.read_parquet(run / "data/clusters.parquet")
    fam = cl.representative_50.value_counts()
    st["families"] = dict(fam50=int(len(fam)), fam50_singletons=int((fam == 1).sum()),
                          fam50_size_quantiles=q(fam.to_numpy()))
    cc = con.execute(f"""select count(*) genes, count(distinct rep_nc_soft) ctx, count(distinct representative_50) fams
                         from '{run}/results/context_clusters_{TAG}.parquet'""").fetchone()
    per_fam = con.execute(f"""select count(distinct rep_nc_soft) from '{run}/results/context_clusters_{TAG}.parquet'
                              group by representative_50""").fetchnumpy()
    per_fam = list(per_fam.values())[0]
    st["context_clusters"] = dict(genes=int(cc[0]), context_clusters=int(cc[1]), fam50_with_contexts=int(cc[2]),
                                  per_fam50_quantiles=q(per_fam), fam50_with_1_context=int((per_fam == 1).sum()))
    log("contexts", st["context_clusters"])

    # ---- associations
    ab = pd.read_parquet(run / f"results/ab50_with_pvalues_{TAG}.parquet")
    sig = ab.sig_score > 80
    st["associations"] = dict(
        tested_pairs=int(len(ab)), pairs_q_lt_005=int((ab.fdr_q_value < 0.05).sum()),
        pairs_q_lt_1e5=int((ab.fdr_q_value < 1e-5).sum()), significant_score_gt80=int(sig.sum()),
        significant_score_gt80_and_q_ge_1e5=int((sig & (ab.fdr_q_value >= 1e-5)).sum()),
        k_obs_of_significant_quantiles=q(ab.k_obs[sig].to_numpy()) if sig.any() else [],
        min_k_obs_significant=int(ab.k_obs[sig].min()) if sig.any() else None,
        k_obs_quantiles=q(ab.k_obs.to_numpy()))
    log("associations", st["associations"])

    links = pd.read_parquet(run / f"results/ab_{STAG}.parquet")
    st["links"] = dict(rows=int(len(links)), distinct_gene_pairs=int(len(links[["a", "b"]].drop_duplicates())))

    # ---- syntons -> intervals
    syn = pd.read_parquet(run / f"results/syntons_all_{STAG}.parquet")
    syn["gid"] = syn.gene.str.slice(1).astype(np.int64)
    loci = intervals(syn, uni)
    fam_path = run / f"results/syntons_{STAG}_clustered_50.parquet"      # their synton families (seed = family id)
    if fam_path.exists():
        sf = pd.read_parquet(fam_path).rename(columns={"seed": "family"})
        loci = loci.merge(sf[["synton", "family"]], on="synton", how="left")
        loci["family"] = loci.family.fillna(-1).astype(np.int64)
    loci.to_parquet(post / "loci.parquet", index=False)
    cov = gene_cover(loci, n, uni)
    isq = (uni.side == "query").to_numpy()
    in_syn = np.zeros(n, dtype=bool)
    in_syn[syn.gid.to_numpy()] = True
    qloc = loci[loci.side == "query"]
    st["syntons"] = dict(
        syntons=int(len(loci)), genes_in_syntons=int(in_syn.sum()), frac_genes_in_syntons=float(in_syn.mean()),
        genes_in_intervals=int(cov.sum()), frac_genes_in_intervals=float(cov.mean()),
        query_genes=int(isq.sum()), query_genes_in_syntons=int((in_syn & isq).sum()),
        frac_query_genes_in_syntons=float((in_syn & isq).sum() / max(1, isq.sum())),
        frac_query_genes_in_intervals=float((cov & isq).sum() / max(1, isq.sum())),
        size_quantiles=q(loci.n_genes), span_quantiles=q(loci.span),
        size_hist={str(k): int(v) for k, v in zip(*np.unique(np.minimum(loci.n_genes, 21), return_counts=True))},
        syntons_with_hole=int((loci.span > loci.n_genes).sum()),
        query_syntons=int(len(qloc)), query_size_quantiles=q(qloc.n_genes), query_span_quantiles=q(qloc.span),
        genomes_with_syntons=int(loci.genome.nunique()),
        syntons_on_replicons_lt48=int((clen.loc[loci.contig_code].to_numpy() < 48).sum()))
    log("syntons", st["syntons"])

    # synton families (their table), counted
    if fam_path.exists():
        st["synton_families"] = dict(families=int(sf.family.nunique()), syntons=int(len(sf)),
                                     syntons_per_family_mean=float(len(sf) / max(1, sf.family.nunique())))

    # ---- paper rule (q < 1e-5 AND score > 80), only if it differs
    nq = st["associations"]["significant_score_gt80_and_q_ge_1e5"]
    st["paperq"] = dict(pairs_removed=nq)
    if nq > 0:
        bad = ab[sig & (ab.fdr_q_value >= 1e-5)][["A50", "B50"]]
        c = cl.set_index("gene").representative_50
        la, lb = c.loc[links.a].to_numpy(), c.loc[links.b].to_numpy()
        badset = set(zip(bad.A50, bad.B50))
        # a link row came from pair (A50 of a, B50 of b); it survives if that pair passes q < 1e-5
        keep = np.array([(x, y) not in badset for x, y in zip(la, lb)])
        G = nx.Graph()
        e = links[keep][["a", "b"]].dropna()
        G.add_edges_from(zip(e.a, e.b))
        rows = [(i, gname) for i, comp in enumerate(nx.connected_components(G)) for gname in comp]
        s2 = pd.DataFrame(rows, columns=["synton", "gene"])
        s2["gid"] = s2.gene.str.slice(1).astype(np.int64)
        l2 = intervals(s2, uni)
        l2.to_parquet(post / "loci_paperq.parquet", index=False)
        st["paperq"].update(links_removed=int((~keep).sum()), syntons=int(len(l2)),
                            genes_in_syntons=int(s2.gid.nunique()))
        log("paper q rule", st["paperq"])
    st["seconds"] = round(time.time() - t0, 1)
    json.dump(st, open(post / "stats.json", "w"), indent=1)
    log("wrote", post / "stats.json")


if __name__ == "__main__":
    main()
