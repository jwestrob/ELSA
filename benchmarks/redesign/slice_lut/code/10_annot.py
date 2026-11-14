#!/usr/bin/env python3
"""Direct Pfam / KOfam annotation of locus genes (workstream slice_lut).

The GlobDB r232 gene index does not follow the GTDB r232 Prodigal protein order (index check in
results/order/globdb_index_check.json), so GlobDB rows cannot be joined to these gene calls. The locus genes and their
flanks are annotated here with the same profile collections instead: Pfam 38.1 at the gathering threshold and KOfam
with the ko_list thresholds (score_type full -> full-sequence score, domain -> best domain score), as kofamscan does.

usage: 10_annot.py export RESULTS_DIR PROTEINS_FAA OUT_FAA [DEFINITIONS]   (default definitions: main)
       10_annot.py parse  PFAM_DOMTBL_GLOB KOFAM_DOMTBL_GLOB KO_LIST OUT_PARQUET
       10_annot.py mdh    MDH_DOMTBL_GLOB KO_LIST OUT_PARQUET
"""
import glob, sys
import pandas as pd


def read_domtbl(pattern):
    rows = []
    for fn in sorted(glob.glob(pattern)):
        for line in open(fn):
            if line.startswith("#"):
                continue
            f = line.split(None, 22)
            # hmmscan/hmmsearch domtblout: target, tacc, tlen, query, qacc, qlen, E, score, bias, #, of, cE, iE, dscore ...
            rows.append((f[0], f[1], f[3], f[4], float(f[6]), float(f[7]), float(f[13]), int(f[17]), int(f[18])))
    return pd.DataFrame(rows, columns=["t", "tacc", "q", "qacc", "evalue", "score", "dscore", "from", "to"])


def ko_pass(d, ko_list, key_col, gene_col):
    kl = pd.read_csv(ko_list, sep="\t", usecols=["knum", "threshold", "score_type", "definition"])
    kl = kl[kl.threshold != "-"].copy(); kl["threshold"] = kl.threshold.astype(float)
    best = d.groupby([gene_col, key_col]).agg(score=("score", "max"), dscore=("dscore", "max"), evalue=("evalue", "min")).reset_index()
    best = best.merge(kl, left_on=key_col, right_on="knum")
    ok = ((best.score_type == "full") & (best.score >= best.threshold)) | ((best.score_type == "domain") & (best.dscore >= best.threshold))
    return best[ok]


mode = sys.argv[1]
if mode == "export":
    R, FAA, OUT = sys.argv[2:5]
    defs = sys.argv[5].split(",") if len(sys.argv) > 5 else ["main"]
    lg = pd.read_csv(f"{R}/locus_genes.tsv", sep="\t", usecols=["definition", "gid"])
    need = set(lg[lg.definition.isin(defs)].gid.astype(int))
    n = 0
    with open(FAA) as fh, open(OUT, "w") as out:
        g = None
        for line in fh:
            if line[0] == ">":
                g = int(line[1:])
            elif g in need:
                out.write(f">{g}\n{line}"); n += 1
    print("exported", n, "of", len(need), flush=True)
elif mode == "parse":
    PF, KF, KL, OUT = sys.argv[2:6]
    p = read_domtbl(PF)  # hmmscan: t = model name, tacc = Pfam accession, q = gid
    p["acc"] = p.tacc.str.split(".").str[0]
    pf = (p.sort_values("from").groupby("q")
          .apply(lambda x: "; ".join(f"{a}:{n}({s}-{e})" for a, n, s, e in zip(x.acc, x.t, x["from"], x["to"])), include_groups=False)
          .rename("pfam"))
    k = read_domtbl(KF)  # hmmscan: t = KO, q = gid
    kp = ko_pass(k, KL, "t", "q")
    kf = (kp.sort_values("score", ascending=False).groupby("q")
          .apply(lambda x: "; ".join(f"{a}:{d}" for a, d in zip(x.knum, x.definition)), include_groups=False).rename("kofam"))
    a = pd.concat([pf, kf], axis=1).reset_index().rename(columns={"index": "gid", "q": "gid"})
    a["gid"] = a.gid.astype("int64")
    a.to_parquet(OUT, index=False)
    print("annotated genes", len(a), "pfam", int(a.pfam.notna().sum()), "kofam", int(a.kofam.notna().sum()), flush=True)
elif mode == "mdh":
    MF, KL, OUT = sys.argv[2:5]
    m = read_domtbl(MF)  # hmmsearch: t = gid, q = KO
    mp = ko_pass(m, KL, "q", "t")
    out = mp.rename(columns={"t": "gid", "q": "ko"})[["gid", "ko", "score", "dscore", "threshold", "score_type"]]
    out["gid"] = out.gid.astype("int64")
    out.to_parquet(OUT, index=False)
    print("MDH KO genes", out.ko.value_counts().to_dict(), flush=True)
