#!/usr/bin/env python3
"""Annotation proxy for lut-like loci across every GlobDB r232 genome (workstream slice_lut).

usage: 07_globdb_proxy.py WIDE_ROWS GLOBDB_TAX GTDB_SP_CLUSTERS OUT_DIR THREADS
Proxy genes (GlobDB Pfam, gathering threshold): A = PF13458 (lutA, Peripla_BP_6), B = PF10282 (lutB, Lactonase),
E = PF00005 (lutE, ABC_tran), F = PF01061 / PF12698 / PF12679 (lutF, ABC-2 membrane), H = PF00593 (lutH, TonB-dependent
receptor barrel), M = PF13202 (EF-hand_5, lanM-like). Runs of proxy genes on the GlobDB gene index with <= GAP intervening
genes (contig boundaries are absent from the annotation release). A run is
  ABEF   if it carries A, B, E and F (the lutA/B/E/F transporter module, also the P. putida PedA1A2BC set)
  ABEFH  ABEF plus H inside the run
Writes OUT_DIR/proxy_runs.parquet (every run with >= 2 distinct proxy letters), OUT_DIR/proxy_genomes.tsv (per genome:
dataset, GTDB-representative flag, taxonomy, best run class, letters), OUT_DIR/proxy_by_taxon.tsv, OUT_DIR/proxy.json.
"""
import json, os, sys, time
import numpy as np, pandas as pd, duckdb

WR, GTAX, SPC, OUT, T = sys.argv[1:6]
GAPS = (3, 5)
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
con = duckdb.connect()
con.execute(f"SET threads={T}; SET preserve_insertion_order=false")
LET = {"PF13458": "A", "PF10282": "B", "PF00005": "E", "PF01061": "F", "PF12698": "F", "PF12679": "F", "PF00593": "H", "PF13202": "M"}
case = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in LET.items())
con.execute(f"""CREATE TABLE p AS SELECT genome_id, gene_idx, string_agg(DISTINCT l, '' ORDER BY l) AS letters
    FROM (SELECT genome_id, gene_idx, CASE acc {case} END AS l FROM read_parquet('{WR}')
          WHERE source = 'Pfam' AND acc IN ({','.join(repr(k) for k in LET)}) AND gene_idx >= 0)
    GROUP BY 1, 2""")
P = con.execute("SELECT genome_id, gene_idx, letters FROM p ORDER BY genome_id, gene_idx").df()
print("proxy genes", len(P), "genomes", P.genome_id.nunique(), round(time.time() - t0), "s", flush=True)
gid = P.genome_id.values; idx = P.gene_idx.values.astype(np.int64); let = P.letters.values
runs = []
for GAP in GAPS:
    brk = np.r_[True, (gid[1:] != gid[:-1]) | (idx[1:] - idx[:-1] > GAP + 1)]
    st = np.flatnonzero(brk); en = np.r_[st[1:], len(P)]
    for s, e in zip(st, en):
        L = set("".join(let[s:e]))
        if len(L) < 2:
            continue
        cls = "ABEFH" if {"A", "B", "E", "F", "H"} <= L else ("ABEF" if {"A", "B", "E", "F"} <= L else "")
        runs.append((GAP, gid[s], int(idx[s]), int(idx[e - 1]), e - s, "".join(sorted(L)), cls, "M" in L))
R = pd.DataFrame(runs, columns=["gap", "genome_id", "first_idx", "last_idx", "n_genes", "letters", "cls", "has_M"])
R.to_parquet(f"{OUT}/proxy_runs.parquet", index=False)
con.register("R", R)
con.execute(f"""CREATE TABLE tax AS SELECT column0 AS genome_id, column1 AS tax FROM read_csv('{GTAX}', delim='\t', header=false, all_varchar=true)""")
con.execute(f"""CREATE TABLE reps AS SELECT split_part(substr("Representative genome", 4), '.', 1) AS genome_id, "Representative genome" AS gtdb_id
                FROM read_csv('{SPC}', delim='\t', header=true, all_varchar=true)""")
G = con.execute("""
  WITH b AS (SELECT genome_id, gap,
        max(CASE cls WHEN 'ABEFH' THEN 2 WHEN 'ABEF' THEN 1 ELSE 0 END) AS best,
        sum((cls = 'ABEFH')::INT) AS n_abefh, sum((cls <> '')::INT) AS n_abef_or_more
      FROM R GROUP BY 1, 2)
  SELECT t.genome_id, regexp_extract(t.genome_id, '^([A-Z]+)', 1) AS dataset, r.gtdb_id, r.gtdb_id IS NOT NULL AS gtdb_rep, t.tax,
         split_part(t.tax, ';', 1) AS domain, split_part(t.tax, ';', 2) AS phylum, split_part(t.tax, ';', 3) AS class,
         split_part(t.tax, ';', 4) AS ord, split_part(t.tax, ';', 5) AS family, split_part(t.tax, ';', 6) AS genus,
         coalesce(b3.best, 0) AS best_gap3, coalesce(b3.n_abefh, 0) AS n_abefh_gap3, coalesce(b3.n_abef_or_more, 0) AS n_abef_gap3,
         coalesce(b5.best, 0) AS best_gap5, coalesce(b5.n_abefh, 0) AS n_abefh_gap5, coalesce(b5.n_abef_or_more, 0) AS n_abef_gap5
  FROM tax t LEFT JOIN reps r USING (genome_id)
  LEFT JOIN b b3 ON b3.genome_id = t.genome_id AND b3.gap = 3 LEFT JOIN b b5 ON b5.genome_id = t.genome_id AND b5.gap = 5""").df()
G.to_csv(f"{OUT}/proxy_genomes.tsv", sep="\t", index=False)
con.register("G", G)
by = con.execute("""SELECT domain, phylum, class, ord, family, count(*) AS genomes, sum(gtdb_rep::INT) AS gtdb_reps,
        sum((best_gap3 >= 1)::INT) AS abef_gap3, sum((best_gap3 = 2)::INT) AS abefh_gap3,
        sum((best_gap5 >= 1)::INT) AS abef_gap5, sum((best_gap5 = 2)::INT) AS abefh_gap5,
        sum((best_gap5 >= 1 AND NOT gtdb_rep)::INT) AS abef_gap5_globdb_only
    FROM G GROUP BY ALL HAVING sum((best_gap5 >= 1)::INT) > 0 ORDER BY abef_gap5 DESC""").df()
by.to_csv(f"{OUT}/proxy_by_taxon.tsv", sep="\t", index=False)
res = dict(proxy_genes=int(len(P)), runs=int(len(R)),
           genomes_abef_gap3=int((G.best_gap3 >= 1).sum()), genomes_abefh_gap3=int((G.best_gap3 == 2).sum()),
           genomes_abef_gap5=int((G.best_gap5 >= 1).sum()), genomes_abefh_gap5=int((G.best_gap5 == 2).sum()),
           seconds=round(time.time() - t0, 1))
json.dump(res, open(f"{OUT}/proxy.json", "w"), indent=1)
print(json.dumps(res), flush=True)
print(by.head(40).to_string(), flush=True)
