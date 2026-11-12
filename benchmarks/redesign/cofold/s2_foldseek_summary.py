"""Foldseek summary per predicted chain and per complex (Part A and screen-2 hits).

Input: RES_DIR/{pdb,alphafold_swissprot}.m8 from s2_foldseek.sbatch (Foldseek 10, -e 10, format
query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,prob,alntmscore,qtmscore,ttmscore,lddt,theader).
Per chain and database: best hit by E-value; tier
  strong    E <= 1e-3 and alignment TM-score >= 0.5
  moderate  E <= 0.1, prob >= 0.9 and alignment TM-score >= 0.5
  none      otherwise
Per complex (chains of one fold): PDB entries (4-character code) hit by >= 2 of its chains at E <= 0.1
(the per-chain substitute for a multimer search; easy-multimersearch aborted in expandmultimer, job 78255).
Usage: python s2_foldseek_summary.py RES_DIR OUT_PREFIX
"""
import sys
from pathlib import Path

import pandas as pd

COLS = "query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,prob,alntmscore,qtmscore,ttmscore,lddt,theader".split(",")


def tier(r):
    if r.evalue <= 1e-3 and r.alntmscore >= 0.5:
        return "strong"
    if r.evalue <= 0.1 and r.prob >= 0.9 and r.alntmscore >= 0.5:
        return "moderate"
    return "none"


def main(res, prefix):
    res = Path(res)
    rows, allpdb = [], None
    for db in ("pdb", "alphafold_swissprot"):
        p = res / f"{db}.m8"
        m = pd.read_csv(p, sep="\t", header=None, names=COLS, quoting=3)
        m["query"] = m["query"].str.replace(".pdb", "", regex=False)
        m["fold_id"] = m["query"].str.rsplit("_", n=1).str[0]
        m["chain"] = m["query"].str.rsplit("_", n=1).str[1]
        if db == "pdb":
            allpdb = m
        b = m.sort_values(["evalue", "bits"], ascending=[True, False]).groupby("query").head(1).copy()
        b["tier"] = b.apply(tier, axis=1)
        b["db"] = db
        rows.append(b)
    best = pd.concat(rows)
    wide = best.pivot(index=["query", "fold_id", "chain"], columns="db",
                      values=["target", "evalue", "prob", "alntmscore", "qtmscore", "alnlen", "qlen", "tier", "theader"])
    wide.columns = [f"{b}_{a}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(f"{prefix}_chains.tsv", sep="\t", index=False)
    # complexes: PDB entries shared by >= 2 chains
    a = allpdb[allpdb.evalue <= 0.1].copy()
    a["entry"] = a.target.str[:4]
    co = []
    for fid, d in a.groupby("fold_id"):
        g = d.sort_values("evalue").groupby("entry").agg(chains=("chain", lambda s: "".join(sorted(set(s)))),
                                                         n=("chain", "nunique"), best_e=("evalue", "min"),
                                                         header=("theader", "first")).reset_index()
        g = g[g.n >= 2].sort_values("best_e")
        for _, r in g.head(3).iterrows():
            co.append(dict(fold_id=fid, entry=r.entry, chains=r.chains, best_e=r.best_e, header=r.header[:120]))
    pd.DataFrame(co).to_csv(f"{prefix}_complex_cohits.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(wide[["query", "pdb_tier", "pdb_evalue", "pdb_alntmscore", "pdb_theader", "alphafold_swissprot_tier",
                "alphafold_swissprot_evalue", "alphafold_swissprot_theader"]].to_string(index=False))
    print(pd.DataFrame(co).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
