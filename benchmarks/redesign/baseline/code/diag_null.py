#!/usr/bin/env python
"""Where do the permuted-order (chance) blocks come from? (workstream ws_baseline)

Reading under test: on DPANN, chance collinear blocks arise between closely related genomes, where
nearly every gene carries a high-cosine anchor, so chance blocks have a higher mean anchor cosine
than observed blocks and a per-size cosine cut cannot separate them. Refuted if permuted blocks are
not enriched for same-species / anchor-dense genome pairs relative to observed blocks.
For each arm, observed blocks and permutation seed 17, by block size: share of blocks whose two
genomes share a GTDB species / genus, the median anchors shared by the genome pair, and the median
mean anchor cosine within and outside same-species pairs.
usage: diag_null.py --threads N --out FILE
"""
import argparse
import json
from pathlib import Path

import duckdb

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
B = RD / "baseline"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}")
    con.execute(f"CREATE TABLE g AS SELECT genome, species, genus FROM read_parquet('{RD}/ws_dpann/package/universe/genomes.parquet')")
    out = {}
    for arm in ("raw", "zca"):
        run = B / "runs" / arm
        con.execute(f"""CREATE OR REPLACE TABLE pairs AS
            SELECT CAST(query_genome AS VARCHAR) qg, CAST(target_genome AS VARCHAR) tg, count(*) n_anchors,
                   median(similarity) med_sim
            FROM read_parquet('{run}/anchors.parquet') GROUP BY 1, 2""")
        for label, f in (("observed", "blocks_checkpoint.parquet"), ("perm17", "perm_17_blocks.parquet")):
            q = f"""
            WITH b AS (
              SELECT least(n_anchors, 6) sz, chain_score / n_anchors m,
                     CAST(query_genome AS VARCHAR) qg, CAST(target_genome AS VARCHAR) tg
              FROM read_parquet('{run / f}')),
            j AS (
              SELECT b.*, (g1.species = g2.species AND g1.species IS NOT NULL AND g1.species <> '') same_sp,
                     (g1.genus = g2.genus AND g1.genus IS NOT NULL AND g1.genus <> '') same_gen,
                     p.n_anchors pair_anchors
              FROM b JOIN g g1 ON b.qg = g1.genome JOIN g g2 ON b.tg = g2.genome
                     LEFT JOIN pairs p ON (p.qg = b.qg AND p.tg = b.tg))
            SELECT sz, count(*) blocks, avg(same_sp::INT) same_species, avg(same_gen::INT) same_genus,
                   median(pair_anchors) median_pair_anchors, median(m) median_mean_cos,
                   median(m) FILTER (WHERE same_sp) mcos_same_species,
                   median(m) FILTER (WHERE NOT same_sp) mcos_other
            FROM j GROUP BY sz ORDER BY sz"""
            df = con.execute(q).df()
            out[f"{arm}/{label}"] = df.to_dict("records")
            print(arm, label)
            print(df.to_string(index=False), flush=True)
        df = con.execute("""SELECT quantile_cont(n_anchors, [0.5, 0.9, 0.99]) q, count(*) genome_pairs,
                            sum(n_anchors) anchors FROM pairs""").df()
        out[f"{arm}/pairs"] = df.to_dict("records")
        print(arm, "genome pairs", df.to_string(index=False), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
