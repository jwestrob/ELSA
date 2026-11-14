#!/usr/bin/env python
"""Inventory of the published Zenodo tables of Herve et al. 2026 (record 21099310). Descriptive counts only:
no method is run and nothing is scored. Gembase gene ids look like SPEC001.0523.00001.001C_02768
(species code + number . release MMYY . strain . replicon index + type . gene number).

usage (Slurm): zenodo_inventory.py --w $W --threads $SLURM_CPUS_ON_NODE
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import duckdb


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    Z = Path(a.w) / "zenodo"
    rec = json.load(open(Z / "record_21099310.json"))
    expect = {f["key"]: f["checksum"].split(":")[1] for f in rec["files"]}
    con = duckdb.connect()
    con.execute(f"set threads={a.threads}; set temp_directory='{Path(a.w) / 'tmp_duck'}'")
    out = dict(files={})
    for f in sorted(Z.glob("Supplementary_Table_*.csv")):
        m = md5(f)
        out["files"][f.name] = dict(bytes=f.stat().st_size, md5=m, md5_ok=(m == expect.get(f.name)))
        log(f.name, out["files"][f.name])
    st1 = f"read_csv('{Z}/Supplementary_Table_1_context_clusters.csv', header=true, all_varchar=true)"
    g = "regexp_extract({c}, '^([A-Z]{{4}}[0-9]{{3}}\\.[0-9]{{4}}\\.[0-9]{{5}})', 1)"
    r = con.execute(f"""
        select count(*) n_rows, count(distinct refseq) distinct_wp, count(distinct fam50) fam50,
               count(distinct context_cluster) context_clusters,
               count(distinct {g.format(c='context_cluster')}) genomes_in_context_ids,
               count(distinct {g.format(c='fam50')}) genomes_in_fam50_ids,
               count(distinct regexp_extract(context_cluster, '^([A-Z]{{4}}[0-9]{{3}})', 1)) species_codes,
               count(distinct regexp_extract(context_cluster, '^[^.]+\\.([0-9]{{4}})', 1)) release_codes
        from {st1}""").fetchone()
    out["ST1"] = dict(zip(["rows", "distinct_wp", "fam50", "context_clusters", "genome_codes_in_context_ids",
                           "genome_codes_in_fam50_ids", "species_codes", "release_codes"], [int(x) for x in r]))
    # singleton context clusters give an exact gembase gene id <-> WP pair (the representative is the only member)
    r = con.execute(f"""
        with c as (select context_cluster, count(*) n, any_value(refseq) wp from {st1} group by 1)
        select count(*) filter (where n = 1), count(distinct {g.format(c='context_cluster')}) filter (where n = 1)
        from c""").fetchone()
    out["ST1"]["singleton_context_clusters"] = int(r[0])
    out["ST1"]["genome_codes_with_singleton_contexts"] = int(r[1])
    log("ST1", out["ST1"])
    st2 = f"read_csv('{Z}/Supplementary_Table_2_family_associations.csv', header=true)"
    r = con.execute(f"""select count(*), count(*) filter (where sig_score > 80),
                               count(*) filter (where sig_score > 80 and fdr_q_value < 1e-5),
                               count(*) filter (where sig_score > 80 and fdr_q_value >= 1e-5),
                               min(k_obs), count(*) filter (where fdr_q_value < 1e-5) from {st2}""").fetchone()
    out["ST2"] = dict(zip(["rows", "score_gt80", "score_gt80_and_q_lt_1e5", "score_gt80_and_q_ge_1e5", "min_k_obs",
                           "q_lt_1e5"], [float(x) if isinstance(x, float) else int(x) for x in r]))
    log("ST2", out["ST2"])
    st3 = f"read_csv('{Z}/Supplementary_Table_3_syntons.csv', header=true, all_varchar=true)"
    r = con.execute(f"""with s as (select synton, count(*) n from {st3} group by 1)
                        select count(*), sum(n), median(n), quantile_cont(n, 0.9), max(n) from s""").fetchone()
    out["ST3"] = dict(zip(["syntons", "gene_rows", "median_genes", "q90_genes", "max_genes"], [float(x) for x in r]))
    out["ST3"]["distinct_wp"] = int(con.execute(f"select count(distinct refseq) from {st3}").fetchone()[0])
    log("ST3", out["ST3"])
    st4 = f"read_csv('{Z}/Supplementary_Table_4_synton_families.csv', header=true)"
    r = con.execute(f"select count(*), count(distinct family), count(distinct synton) from {st4}").fetchone()
    out["ST4"] = dict(rows=int(r[0]), families=int(r[1]), syntons=int(r[2]))
    st5 = f"read_csv('{Z}/Supplementary_Table_5_cre.csv', header=true)"
    out["ST5"] = dict(rows=int(con.execute(f"select count(*) from {st5}").fetchone()[0]))
    st7 = f"read_csv('{Z}/Supplementary_Table_7_cooccurrences.csv', header=true)"
    r = con.execute(f"select count(*), count(distinct subtype_a) from {st7}").fetchone()
    out["ST7"] = dict(rows=int(r[0]), subtypes_a=int(r[1]))
    log("ST4/5/7", out["ST4"], out["ST5"], out["ST7"])
    json.dump(out, open(Path(a.w) / "zenodo_inventory.json", "w"), indent=1)
    log("wrote", Path(a.w) / "zenodo_inventory.json")


if __name__ == "__main__":
    main()
