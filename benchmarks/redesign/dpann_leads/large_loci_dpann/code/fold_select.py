#!/usr/bin/env python
"""Pick representatives of the unexplained core genes (>= 50% of the cluster's genes without a row from any source) of
the closer-look families: two per cluster from different orders, intact member loci first, length nearest the cluster
median. Hydrophobic screen (ILVFMAWC >= 0.47 or TMHMM >= 3 helices -> not folded). Writes folds/in/reps.faa and
data/fold_reps.tsv.  usage: fold_select.py SCRATCH FAM [FAM ...]
"""
import sys

import duckdb
import pandas as pd

W = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/large_loci_dpann"
DB = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"


def main(sp, fams):
    L = pd.read_parquet(sp + "/loci_annot.parquet")
    G = pd.read_parquet(W + "/data/family_genes.parquet")
    A = pd.read_parquet(sp + "/annot_all.parquet", columns=["protein_id", "aa", "hyd", "n_tm"])
    CL = pd.read_csv(W + "/data/core_clusters.tsv", sep="\t")
    c = duckdb.connect()
    c.execute(f"ATTACH '{DB}' AS d (READ_ONLY)")
    rows = []
    for f in fams:
        mem = L[L.family == f].set_index("locus")
        intact = set(mem[(mem.flank_left >= 3) & (mem.flank_right >= 3)].index)
        dark = CL[(CL.family == f) & (CL.unannotated_share >= 0.5)]
        for r in dark.itertuples():
            sub = G[(G.family == f) & (G.gene_cluster == r.cluster)].merge(A, on="protein_id")
            sub["order"] = sub.locus.map(mem.order)
            sub["intact"] = sub.locus.isin(intact)
            sub["dev"] = (sub.aa - r.median_aa).abs()
            sub = sub.sort_values(["intact", "dev"], ascending=[False, True])
            picked, orders = [], set()
            for s in sub.itertuples():
                if s.order in orders:
                    continue
                picked.append(s)
                orders.add(s.order)
                if len(picked) == 2:
                    break
            for s in picked:
                memb = s.hyd >= 0.47 or (s.n_tm or 0) >= 3
                rows.append(dict(family=f, core_pos=r.core_pos, cluster=int(r.cluster), protein_id=s.protein_id,
                                 aa=s.aa, hyd=round(s.hyd, 3), n_tm=s.n_tm, order=s.order, intact=s.intact,
                                 fold=not memb, name=f"f{f}_c{r.core_pos}_{s.protein_id}"))
    R = pd.DataFrame(rows)
    c.register("R", R)
    seq = c.execute("SELECT protein_id, sequence FROM d.proteins JOIN R USING (protein_id)").df()
    R = R.merge(seq, on="protein_id")
    R.drop(columns=["sequence"]).to_csv(W + "/data/fold_reps.tsv", sep="\t", index=False)
    with open(W + "/folds/in/reps.faa", "w") as fh:
        for r in R[R.fold].itertuples():
            fh.write(f">{r.name}\n{r.sequence.rstrip('*')}\n")
    print(R.drop(columns=["sequence"]).to_string())
    print("to fold", int(R.fold.sum()), "residues", int(R[R.fold].aa.sum()))


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
