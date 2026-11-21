#!/usr/bin/env python
"""Locus space, part 3 tables: clusters of the core-locus map against gold, census types, module labels, and the
dense uncalled clusters with observed domains (workstream ws_locusspace).

  lspace_report.py --rep REP [--threads T]
Single process (pandas + DuckDB with <= 4 threads); reads map/cores_clusters_<REP>.parquet.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lspace as LS  # noqa: E402
from lspace import log, dump, OUT, RD, PKG  # noqa: E402

MAP = OUT / "map"
CENSUS = RD / "cofold/s3/census"
SHARUR = RD / "ws_dpann/input/sharur.duckdb"
TOP = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="MEAN")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_ON_NODE", 2)))
    a = ap.parse_args()
    T0 = time.time()
    U = LS.Universe(vectors=False)
    C = pd.read_parquet(MAP / f"cores_clusters_{a.rep}.parquet")
    n = len(C)
    s, e = C.gid_s.to_numpy(np.int64), C.gid_e.to_numpy(np.int64)
    lens = e - s + 1
    gene = np.concatenate([np.arange(x, y + 1) for x, y in zip(s, e)])
    row = np.repeat(np.arange(n), lens)
    core_of = np.full(U.N, -1, np.int64)
    core_of[gene] = row
    # curated-caller membership (any caller, any member count) and annotation flags
    I = pd.read_parquet(PKG / "gold/instances.parquet", columns=["caller", "system_type", "system_subtype", "member_gids"])
    called = np.zeros(U.N, bool)
    called[np.concatenate([np.asarray(m, np.int64) for m in I.member_gids])] = True
    fl = pd.read_parquet(PKG / "annotation_flags.parquet", columns=["gid", "unannotated"])
    unann = np.zeros(U.N, bool)
    unann[fl.gid.to_numpy()] = fl.unannotated.to_numpy()
    C["called"] = np.bincount(row, weights=called[gene], minlength=n) > 0
    C["dark_frac"] = np.bincount(row, weights=unann[gene], minlength=n) / lens
    C["genome_code"] = U.gc[s]
    C["g075"] = U.groups["g075"][C.genome_code.to_numpy()]
    toks = U.tok[gene]
    fs = pd.Series([tuple(sorted(set(toks[a_:b_].tolist()))) for a_, b_ in zip(np.r_[0, np.cumsum(lens)[:-1]], np.cumsum(lens))])
    C["famset"] = pd.factorize(fs)[0]
    log(f"cores {n:,}; called {C.called.mean():.3f}; family sets {C.famset.nunique():,}")
    # per-cluster table
    gm = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome_code", "phylum"])
    phy = dict(zip(gm.genome_code, gm.phylum))
    C["phylum"] = C.genome_code.map(phy)
    G = C.groupby("cluster").agg(cores=("gid_s", "size"), genomes=("genome_code", "nunique"), groups=("g075", "nunique"),
                                 phyla=("phylum", "nunique"), famsets=("famset", "nunique"),
                                 modules=("module_id", "nunique"), called=("called", "max"),
                                 called_frac=("called", "mean"), dark=("dark_frac", "mean"),
                                 span_med=("span", "median"))
    res = dict(rep=a.rep, cores=int(n), clusters=int(len(G)),
               clusters_ge5=int((G.cores >= 5).sum()), clusters_ge10=int((G.cores >= 10).sum()),
               cores_in_ge10=float(G.cores[G.cores >= 10].sum() / n),
               size_q=np.quantile(G.cores, [.5, .9, .99, 1]).tolist(),
               singletons=int((G.cores == 1).sum()))
    # module labels vs locus-space clusters (same cores)
    mod_per_cl = G.modules
    cl_per_mod = C.groupby("module_id").cluster.nunique()
    res["modules"] = dict(n_modules=int(C.module_id.nunique()),
                          modules_per_cluster_q=np.quantile(mod_per_cl[G.cores >= 10], [.5, .9, .99]).tolist(),
                          clusters_per_module_q=np.quantile(cl_per_mod, [.5, .9, .99]).tolist(),
                          famsets_per_cluster_q=np.quantile(G.famsets[G.cores >= 10], [.5, .9, .99]).tolist())
    from sklearn.metrics import adjusted_mutual_info_score
    sub = C.sample(min(n, 200_000), random_state=17)
    res["modules"]["ami_cluster_vs_module_sample200k"] = float(adjusted_mutual_info_score(sub.cluster, sub.module_id))
    # gold distribution over clusters (all >= 2-member gold cores, both splits)
    B = pd.read_parquet(MAP / f"gold_cores_{a.rep}.parquet")
    B["cluster"] = C.cluster.to_numpy()[B.core_row.to_numpy()]
    B["module_id"] = C.module_id.to_numpy()[B.core_row.to_numpy()]
    gd = {}
    for caller, g in B.groupby("caller"):
        per_key = g.groupby("key").agg(n=("cluster", "size"), clusters=("cluster", "nunique"),
                                       modules=("module_id", "nunique"),
                                       top_share=("cluster", lambda v: v.value_counts().iloc[0] / len(v)),
                                       top_mod_share=("module_id", lambda v: v.value_counts().iloc[0] / len(v)))
        big = per_key[per_key.n >= 10]
        gd[caller] = dict(labels=int(len(per_key)), labels_ge10=int(len(big)),
                          clusters_per_label_median=float(big.clusters.median()) if len(big) else None,
                          modules_per_label_median=float(big.modules.median()) if len(big) else None,
                          top_cluster_share_median=float(big.top_share.median()) if len(big) else None,
                          top_module_share_median=float(big.top_mod_share.median()) if len(big) else None,
                          per_label=big.sort_values("n", ascending=False).head(25).round(3).reset_index().to_dict("records"))
    lab_per_cl = B.groupby("cluster").key.nunique()
    gd["labels_per_gold_cluster_q"] = np.quantile(lab_per_cl, [.5, .9, .99, 1]).tolist()
    gd["gold_clusters"] = int(len(lab_per_cl))
    res["gold"] = gd
    # census types
    Ci = pd.read_parquet(CENSUS / "census_instances.parquet", columns=["type_key", "gids"])
    Ct = pd.read_parquet(CENSUS / "census_types.parquet", columns=["type_key", "n_genomes", "unrec_class",
                                                                     "all_unrecognised", "most_status"])
    cores_i = []
    for gl in Ci.gids:
        c = core_of[np.asarray(gl, np.int64)]
        c = c[c >= 0]
        cores_i.append(Counter(c.tolist()).most_common(1)[0][0] if len(c) else -1)
    Ci["core"] = cores_i
    Ci["cluster"] = np.where(Ci.core >= 0, C.cluster.to_numpy()[np.maximum(Ci.core, 0)], -1)
    mapped = Ci[Ci.cluster >= 0]
    t_per_cl = mapped.groupby("cluster").type_key.nunique()
    cl_per_t = mapped.groupby("type_key").cluster.nunique()
    top_share_t = mapped.groupby("type_key").cluster.agg(lambda v: v.value_counts().iloc[0] / len(v))
    res["census"] = dict(instances=int(len(Ci)), mapped=int(len(mapped)), types=int(Ci.type_key.nunique()),
                         types_mapped=int(mapped.type_key.nunique()), clusters_with_census=int(len(t_per_cl)),
                         types_per_cluster_q=np.quantile(t_per_cl, [.5, .75, .9, .99, 1]).tolist(),
                         clusters_with_ge2_types=int((t_per_cl >= 2).sum()),
                         clusters_with_ge10_types=int((t_per_cl >= 10).sum()),
                         clusters_per_type_q=np.quantile(cl_per_t, [.5, .9, .99]).tolist(),
                         top_cluster_share_per_type_median=float(top_share_t.median()))
    unrec = set(Ct.type_key[Ct.all_unrecognised.fillna(False)])
    mu = mapped[mapped.type_key.isin(unrec)]
    res["census"]["unrecognised_types_mapped"] = int(mu.type_key.nunique())
    res["census"]["clusters_with_unrecognised_types"] = int(mu.cluster.nunique())
    G["census_types"] = t_per_cl.reindex(G.index).fillna(0).astype(int)
    G["census_unrec_types"] = mu.groupby("cluster").type_key.nunique().reindex(G.index).fillna(0).astype(int)
    # dense uncalled clusters
    dense = G[(G.cores >= 10) & (~G.called)].copy()
    res["uncalled_dense"] = dict(n=int(len(dense)), dark_ge50=int((dense.dark >= 0.5).sum()),
                                 groups_ge5=int((dense.groups >= 5).sum()),
                                 dark_ge50_groups_ge5=int(((dense.dark >= 0.5) & (dense.groups >= 5)).sum()))
    top_all = dense.sort_values(["groups", "famsets", "cores"], ascending=False).head(TOP)
    top_dark = dense[dense.dark >= 0.5].sort_values(["groups", "famsets", "cores"], ascending=False).head(TOP)
    # observed domains for the listed clusters (Pfam, KO, raw DefenseFinder / TXSScan profile hits)
    import duckdb
    want = sorted(set(top_all.index) | set(top_dark.index))
    sel = C[C.cluster.isin(want)]
    sel = sel.groupby("cluster", group_keys=False).apply(lambda g: g.sample(min(len(g), 300), random_state=17))
    gl = np.concatenate([np.arange(x, y + 1) for x, y in zip(sel.gid_s, sel.gid_e)])
    grow = np.repeat(sel.cluster.to_numpy(), (sel.gid_e - sel.gid_s + 1).to_numpy())
    gcore = np.repeat(np.arange(len(sel)), (sel.gid_e - sel.gid_s + 1).to_numpy())
    genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "protein_id", "length"])
    pid = genes.protein_id.to_numpy()[gl]
    lens_aa = genes.length.to_numpy()[gl]
    con = duckdb.connect(str(SHARUR), read_only=True, config={"threads": min(4, a.threads)})
    con.register("want", pd.DataFrame(dict(protein_id=np.unique(pid))))
    ann = con.execute("select a.protein_id, a.source, a.name from annotations a join want w using (protein_id) "
                      "where a.source in ('pfam', 'kofam', 'defensefinder', 'txsscan')").fetchdf()
    con.close()
    ann = ann.drop_duplicates()
    p2rows = {}
    for k, (p, cl_, co) in enumerate(zip(pid, grow, gcore)):
        p2rows.setdefault(p, []).append((cl_, co))
    dom = {}
    for p, src, name in zip(ann.protein_id, ann.source, ann.name):
        for cl_, co in p2rows.get(p, []):
            dom.setdefault(cl_, {}).setdefault((src, name), set()).add(co)
    ncore = sel.groupby("cluster").size()
    lens_cl = pd.DataFrame(dict(cl=grow, l=lens_aa)).groupby("cl").l.median()

    def describe(T):
        out = []
        for cl_, r in T.iterrows():
            d = dom.get(cl_, {})
            items = sorted(d.items(), key=lambda kv: -len(kv[1]))
            top = [f"{src}:{name} {len(v) / ncore[cl_]:.2f}" for (src, name), v in items[:6]]
            out.append(dict(cluster=int(cl_), cores=int(r.cores), genomes=int(r.genomes), groups=int(r.groups),
                            phyla=int(r.phyla), famsets=int(r.famsets), modules=int(r.modules),
                            span_med=float(r.span_med), dark=round(float(r.dark), 3),
                            gene_len_med_aa=float(lens_cl.get(cl_, np.nan)), census_types=int(r.census_types),
                            census_unrec_types=int(r.census_unrec_types),
                            observed=top if top else ["no Pfam / KO / profile hit on sampled cores"]))
        return out

    res["top_uncalled"] = describe(top_all)
    res["top_uncalled_dark"] = describe(top_dark)
    G.reset_index().to_parquet(MAP / f"cluster_table_{a.rep}.parquet", index=False)
    Ci[["type_key", "core", "cluster"]].to_parquet(MAP / f"census_to_cluster_{a.rep}.parquet", index=False)
    res["seconds"] = time.time() - T0
    dump(res, MAP / f"report_{a.rep}.json")
    log(f"report done {res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
