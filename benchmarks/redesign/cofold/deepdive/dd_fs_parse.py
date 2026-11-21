"""Parse the candidate-1 Foldseek and US-align results (local, seconds).

Foldseek tiers as in screen 2: strong = E <= 1e-3 and alignment TM-score >= 0.5; moderate = E <= 0.1, prob >= 0.9
and alignment TM-score >= 0.5. Query groups: A, B (12 cognate MSA models each), C (ABC MSA models), AB_concat
(complex as one chain), fused (DPANN and UniProt fused homologs). Reported per group: queries with >= 1 strong /
moderate hit in each database, and the recurring targets.
US-align: TM-score of each fused-homolog model against each AB-as-one-chain model, normalised by the fused model
(TM1, first structure) and by the AB model (TM2).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work/fs_c1"
COLS = "query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,prob,alntmscore,qtmscore,ttmscore,lddt,theader".split(",")


def tier(r):
    if r.evalue <= 1e-3 and r.alntmscore >= 0.5:
        return "strong"
    if r.evalue <= 0.1 and r.prob >= 0.9 and r.alntmscore >= 0.5:
        return "moderate"
    return "none"


def group(q):
    if q.startswith("AB_concat"):
        return "AB_concat"
    if "fused" in q:
        return "fused"
    if q.endswith("_C"):
        return "C"
    return q[-1]


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    out = []
    for db in ("pdb", "alphafold_swissprot"):
        p = W / f"{db}.m8"
        if not p.exists():
            print("missing", p)
            continue
        d = pd.read_csv(p, sep="\t", header=None, names=COLS, quoting=3)
        d["query"] = d["query"].str.replace(".pdb", "", regex=False)
        d["tier"] = d.apply(tier, axis=1)
        d["grp"] = d["query"].map(group)
        d["db"] = db
        out.append(d)
        for gname, x in d.groupby("grp"):
            nq = x["query"].nunique()
            st = x[x.tier == "strong"]
            mo = x[x.tier == "moderate"]
            best = x.sort_values("evalue").drop_duplicates("query")
            print(f"{db} {gname}: queries {nq}; with strong {st['query'].nunique()}, moderate-only "
                  f"{len(set(mo['query']) - set(st['query']))}; best E median {best.evalue.median():.2g}")
            rec = pd.concat([st, mo]).groupby("theader").agg(n_queries=("query", "nunique"), E_min=("evalue", "min"),
                                                            TM_max=("alntmscore", "max")).sort_values(
                "n_queries", ascending=False).head(6)
            if len(rec):
                print(rec.to_string())
    D = pd.concat(out)
    D.to_csv(HERE / "work/c1_foldseek_hits.tsv", sep="\t", index=False)
    mm = W / "pdb_multimer.m8"
    if mm.exists() and mm.stat().st_size:
        M = pd.read_csv(mm, sep="\t", header=None)
        print("multimer search rows", len(M))
        print(M.sort_values(M.columns[-1] if M.shape[1] > 10 else 0).head(10).to_string())
    us = W / "usalign/usalign_pairs.tsv"
    if us.exists():
        U = pd.read_csv(us, sep="\t", header=None)
        U = U.iloc[:, :12]
        U.columns = ["s1", "s2", "TM1", "TM2", "RMSD", "ID1", "ID2", "IDali", "L1", "L2", "Lali", "x"][:U.shape[1]]
        U["s1"] = U.s1.str.split("/").str[-1].str.replace(".pdb", "", regex=False).str.replace(":A", "", regex=False)
        U["s2"] = U.s2.str.split("/").str[-1].str.replace(".pdb", "", regex=False).str.replace(":A", "", regex=False)
        F = U[U.s1.str.contains("fused") & U.s2.str.startswith("AB_concat")]
        s = F.groupby("s1").agg(TM_fused_median=("TM1", "median"), TM_fused_max=("TM1", "max"),
                                TM_AB_median=("TM2", "median"), RMSD_median=("RMSD", "median"), Lali_median=("Lali", "median"))
        print(s.round(3).to_string())
        AB = U[U.s1.str.startswith("AB_concat") & U.s2.str.startswith("AB_concat")]
        print("AB-vs-AB (between instances) TM1 median", round(AB.TM1.median(), 3), "range", round(AB.TM1.min(), 3),
              round(AB.TM1.max(), 3))
        FF = U[U.s1.str.contains("fusedmsa|extfusedmsa") & U.s2.str.contains("fusedmsa|extfusedmsa")]
        print(FF[["s1", "s2", "TM1", "TM2", "RMSD", "Lali"]].round(3).to_string())
        s.to_csv(HERE / "work/c1_usalign_fused_vs_AB.tsv", sep="\t")


if __name__ == "__main__":
    main()
