#!/usr/bin/env python3
"""GlobDB r232 annotation rows for the lut slice (workstream slice_lut). One scan of the 41 GB Parquet release.

usage: 04_globdb_rows.py PARQUET_DIR GLOBDB_TAX OUT_DIR THREADS
Writes OUT_DIR/order_rows.parquet  every Pfam and KOfam row of every GlobDB genome in GTDB o__Rhizobiales
                                   (GTDB species representatives and GlobDB-only genomes)
       OUT_DIR/wide_rows.parquet   rows of the profiles that the AM1 lut proteins carry (Pfam GA hits of
                                   META1_1777-1788) and of methanol-dehydrogenase KOs, for every GlobDB genome
       OUT_DIR/genome_gene_counts.parquet  per genome: rows, max gene_idx, distinct annotated genes (all genomes)
       OUT_DIR/globdb_rows.json
"""
import json, os, sys, time
import duckdb

PQ, GTAX, OUT, TH = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
con = duckdb.connect()
con.execute(f"PRAGMA threads={TH}; SET preserve_insertion_order=false; PRAGMA temp_directory='{OUT}/duck_tmp'")
con.execute(f"""CREATE TABLE ord AS SELECT column0 AS genome_id, column1 AS tax FROM read_csv('{GTAX}', delim='\t',
                header=false, all_varchar=true) WHERE split_part(column1, ';', 4) = 'o__Rhizobiales'""")
PF = ["PF11684", "PF13458", "PF10282", "PF22494", "PF00005", "PF01061", "PF12698", "PF12679", "PF07715", "PF00593",
      "PF13202", "PF13727", "PF02397", "PF13499", "PF13405", "PF13833", "PF00036"]
KO = ["K23995", "K14028", "K14029", "K00114", "K00117", "K17067", "K00093", "K05889"]
src = f"read_parquet('{PQ}/*.parquet')"
con.execute(f"""CREATE TABLE r AS SELECT genome_id, gene_idx, gene_raw, source, accession, function, e_value,
       split_part(accession, '.', 1) AS acc, genome_id IN (SELECT genome_id FROM ord) AS in_order
    FROM {src}
    WHERE (source IN ('Pfam', 'KOfam') AND genome_id IN (SELECT genome_id FROM ord))
       OR (source = 'Pfam' AND split_part(accession, '.', 1) IN ({','.join(repr(x) for x in PF)}))
       OR (source = 'KOfam' AND accession IN ({','.join(repr(x) for x in KO)}))""")
print("scan done", round(time.time() - t0), "s", flush=True)
con.execute(f"COPY (SELECT * EXCLUDE (in_order, acc) FROM r WHERE in_order AND source IN ('Pfam','KOfam') ORDER BY genome_id, gene_idx) "
            f"TO '{OUT}/order_rows.parquet' (FORMAT parquet, COMPRESSION zstd)")
con.execute(f"""COPY (SELECT genome_id, gene_idx, gene_raw, source, acc, e_value FROM r
                WHERE (source = 'Pfam' AND acc IN ({','.join(repr(x) for x in PF)})) OR (source = 'KOfam' AND accession IN ({','.join(repr(x) for x in KO)}))
                ORDER BY genome_id, gene_idx) TO '{OUT}/wide_rows.parquet' (FORMAT parquet, COMPRESSION zstd)""")
con.execute(f"""COPY (SELECT genome_id, count(*) AS n_rows, max(gene_idx) AS max_gene_idx, min(gene_idx) AS min_gene_idx,
                count(DISTINCT gene_idx) AS n_genes_annotated, sum((gene_idx < 0)::INT) AS n_unparsed
                FROM {src} GROUP BY 1 ORDER BY 1) TO '{OUT}/genome_gene_counts.parquet' (FORMAT parquet)""")
res = dict(order_genomes=con.execute("SELECT count(*) FROM ord").fetchone()[0],
           order_rows=con.execute(f"SELECT count(*), count(DISTINCT genome_id) FROM '{OUT}/order_rows.parquet'").fetchone(),
           wide_rows=con.execute(f"SELECT count(*), count(DISTINCT genome_id) FROM '{OUT}/wide_rows.parquet'").fetchone(),
           genomes_counted=con.execute(f"SELECT count(*) FROM '{OUT}/genome_gene_counts.parquet'").fetchone()[0],
           pfam=PF, ko=KO, seconds=round(time.time() - t0, 1))
json.dump(res, open(f"{OUT}/globdb_rows.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str), flush=True)
