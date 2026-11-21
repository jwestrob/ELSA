#!/usr/bin/env python
"""Closer look at chosen families: taxonomic distribution with denominators, intactness on long contigs, flanks,
per-core-cluster evidence, and representative proteins of the unexplained core genes.
usage: closer.py SCRATCH FAMILY [FAMILY ...]   (writes data/closer_<fam>.txt and data/reps_<fam>.tsv)
"""
import sys
from collections import Counter

import duckdb
import numpy as np
import pandas as pd

W = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/large_loci_dpann"
DB = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"


def main(sp, fams):
    L = pd.read_parquet(sp + "/loci_annot.parquet")
    G = pd.read_parquet(W + "/data/family_genes.parquet")
    A = pd.read_parquet(sp + "/annot_all.parquet")
    CL = pd.read_csv(W + "/data/core_clusters.tsv", sep="\t")
    F = pd.read_csv(W + "/families.tsv", sep="\t")
    c = duckdb.connect()
    c.execute(f"ATTACH '{DB}' AS d (READ_ONLY)")
    bins = c.execute("SELECT bin_id, completeness, taxonomy FROM d.bins").df()
    tx = bins.taxonomy.str.split(";", expand=True)
    bins["phylum"], bins["order"] = tx[1], tx[3]
    ordn, phyn = bins.order.value_counts(), bins.phylum.value_counts()
    for f in fams:
        out = []
        fr = F[F.family == f].iloc[0]
        mem = L[L.family == f]
        out.append(f"family {f} [{fr.cls}] loci {len(mem)} genomes {mem.genome.nunique()} groups {mem.group.nunique()} "
                   f"orders {mem.order.nunique()} size median {mem['size'].median()}")
        go = mem.drop_duplicates("genome").order.value_counts()
        out.append("orders (genomes with locus / DPANN genomes in order): " +
                   "; ".join(f"{o} {n}/{ordn[o]}" for o, n in go.head(10).items()))
        gp = mem.drop_duplicates("genome").phylum.value_counts()
        out.append("phyla: " + "; ".join(f"{o} {n}/{phyn[o]}" for o, n in gp.items()))
        intact = (mem.flank_left >= 3) & (mem.flank_right >= 3)
        out.append(f"members with >= 3 flanking genes on both sides: {int(intact.sum())}/{len(mem)}; contig genes median "
                   f"{mem.contig_genes.median():.0f}; members on contigs >= 50 genes {int((mem.contig_genes >= 50).sum())}; "
                   f"completeness median {mem.completeness.median():.1f}")
        gg = G[G.family == f].merge(A[["protein_id", "aa", "hyd", "n_tm", "pfam", "ko", "cdd_desc", "cdd_e", "fam_hh_pfam",
                                       "fam_hh_pfam_prob", "fam_hh_pdb", "fam_hh_pdb_prob", "tags", "cur_systems"]],
                                    on="protein_id", how="left")
        core = CL[CL.family == f].sort_values("core_pos")
        # core presence on intact members vs edge members
        im = set(mem[intact].locus)
        for r in core.itertuples():
            sub = gg[gg.gene_cluster == r.cluster]
            pi = sub[sub.locus.isin(im)].locus.nunique() / max(len(im), 1)
            out.append(f" core {r.core_pos:>2} cl {int(r.cluster)} pres {r.presence:.2f} (intact {pi:.2f}) {r.median_aa} aa "
                       f"hyd {r.hyd:.2f} tm {r.n_tm:g} unann {r.unannotated_share:.2f} | {r.annotation[:60]} | "
                       f"CDD {plur(sub.cdd_desc)[:60]} | famHH {plur(sub.fam_hh_pfam)} / {plur(sub.fam_hh_pdb)[:50]} | "
                       f"slot {r.slot_label} {r.slot_tier}")
        # flanks: what sits within 3 genes outside the locus (plurality annotations)
        # representatives: the intact member with most core clusters
        cc = gg[gg.gene_cluster.isin(core.cluster)].groupby("locus").gene_cluster.nunique()
        cand = cc[cc.index.isin(im)].sort_values(ascending=False)
        reps = []
        for lid in cand.index[:3]:
            sub = gg[gg.locus == lid].sort_values("gid")
            out.append(f" rep locus {lid} {mem[mem.locus == lid].genome.iloc[0]} "
                       f"{mem[mem.locus == lid].order.iloc[0]}: " + " | ".join(
                           f"{p}:{a}aa:{(x if isinstance(x, str) else '-')[:22]}{'*' if k in set(core.cluster) else ''}"
                           for p, a, x, k in zip(sub.protein_id, sub.aa, sub.annot_label, sub.gene_cluster)))
            for r in sub[sub.gene_cluster.isin(core.cluster)].itertuples():
                reps.append(dict(family=f, locus=lid, protein_id=r.protein_id, cluster=int(r.gene_cluster), aa=r.aa,
                                 annot=r.annot_label))
        pd.DataFrame(reps).to_csv(W + f"/data/reps_{f}.tsv", sep="\t", index=False)
        open(W + f"/data/closer_{f}.txt", "w").write("\n".join(out) + "\n")
        print("\n".join(out))
        print()


def plur(s):
    v = [x for x in s if isinstance(x, str) and x]
    return Counter(v).most_common(1)[0][0] if v else "-"


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
