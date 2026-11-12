#!/usr/bin/env python
"""Pair recall of every token arm on native Spacedust pairs, split by the pair's pairwise
sequence identity and alignment coverage (from $S/work/aln_seq.tsv) and by the target
genome's stratum (rank relative to Streptomyces). Writes $RD/ws_tokens/recall_strata.tsv.

Identity bins use MMseqs seqid of the SW alignment; coverage = min(query cover, target cover).
"""
import argparse
import json
import os

import duckdb
import numpy as np
import pandas as pd

from common import NREF, S, TOK, UNIVERSE, WS, stamp

ID_BINS = [0, 0.3, 0.5, 0.7, 0.9, 1.01]
ID_LAB = ["<30%", "30-50%", "50-70%", "70-90%", ">=90%"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--arms", default="")
    a = ap.parse_args()
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    P = con.execute(f"""
        WITH sd AS (SELECT DISTINCT qkey, tkey FROM read_parquet('{S}/work/sd_pairs.parquet')),
        al AS (SELECT * FROM read_csv('{S}/work/aln_seq.tsv', delim='\t', header=false, all_varchar=true,
                    names=['qkey','tkey','bits','seqid','evalue','qstart','qend','qlen','tstart','tend','tlen'])),
        a2 AS (SELECT CAST(qkey AS BIGINT) qkey, CAST(tkey AS BIGINT) tkey,
                      TRY_CAST(seqid AS DOUBLE) seqid, TRY_CAST(evalue AS DOUBLE) evalue,
                      least((TRY_CAST(qend AS DOUBLE) - TRY_CAST(qstart AS DOUBLE) + 1) / TRY_CAST(qlen AS DOUBLE),
                            (TRY_CAST(tend AS DOUBLE) - TRY_CAST(tstart AS DOUBLE) + 1) / TRY_CAST(tlen AS DOUBLE)) AS acov
               FROM al)
        SELECT uq.gid AS g1, ut.gid AS g2, a2.seqid, a2.evalue, a2.acov, gm.stratum
        FROM sd JOIN a2 USING (qkey, tkey)
        JOIN read_parquet('{UNIVERSE}') uq ON uq.side = 'query' AND uq.mmseqs_key = sd.qkey
        JOIN read_parquet('{UNIVERSE}') ut ON ut.side = 'ref' AND ut.mmseqs_key = sd.tkey
        JOIN read_parquet('{UNIVERSE.parent}/genomes.parquet') gm ON gm.genome_code = ut.genome_code""").df()
    P["vis"] = P.evalue <= 1e-3
    P["id_bin"] = pd.cut(P.seqid, ID_BINS, labels=ID_LAB, right=False)
    P["cov_bin"] = np.where(P.acov >= 0.8, "cov>=0.8", "cov<0.8")
    stamp("pairs", len(P), "visible", int(P.vis.sum()))
    arms = a.arms.split(",") if a.arms else sorted(f.stem for f in TOK.glob("*.parquet"))
    rows = []
    g1, g2 = P.g1.to_numpy(), P.g2.to_numpy()
    for arm in arms:
        tok = con.execute(f"SELECT token FROM read_parquet('{TOK}/{arm}.parquet') ORDER BY gid").fetchnumpy()["token"]
        hit = (tok[g1] == tok[g2]) & (tok[g1] >= 0)
        P["hit"] = hit
        for key in (["id_bin"], ["cov_bin", "id_bin"], ["stratum"]):
            sub = P[P.vis] if key != ["stratum"] else P
            g = sub.groupby(key, observed=True).agg(n=("hit", "size"), shared=("hit", "sum")).reset_index()
            g["recall"] = g.shared / g.n
            g["split"] = "+".join(key)
            g["subset"] = "sd_vis" if key != ["stratum"] else "sd_all"
            g["arm"] = arm
            rows.append(g)
        stamp(arm)
    R = pd.concat(rows, ignore_index=True)
    R["level"] = R[["cov_bin", "id_bin", "stratum"]].astype(str).agg(lambda r: "|".join(x for x in r if x != "nan"), axis=1)
    R = R[["arm", "subset", "split", "level", "n", "shared", "recall"]]
    tmp = WS / f".recall_strata.tsv.tmp{os.getpid()}"
    R.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, WS / "recall_strata.tsv")
    stamp("rows", len(R))


if __name__ == "__main__":
    main()
