#!/usr/bin/env python3
"""Write O/README_annotation.txt for one collection of the CPU Pfam run.

Usage: gen_readme.py NAME O ARRAY_JOB FINISH_JOB SMOKE_JSON
Reads O/shards/SHARDS.json, $A/meta/meta_info.json and the smoke comparison JSON. Single process (login node).
"""
import datetime
import json
import os
import sys


def main():
    name, o, array_job, finish_job, smoke_json = sys.argv[1:6]
    a = os.path.dirname(os.path.abspath(o))
    info = json.load(open(f"{o}/shards/SHARDS.json"))
    meta = json.load(open(f"{a}/meta/meta_info.json"))
    sm = json.load(open(smoke_json))
    n, k = info["proteins"], info["shards"]
    lo, hi = info["proteins_per_shard_min"], info["proteins_per_shard_max"]
    per = sm["smoke3"]["per_shard_evalue_ratio"]
    assert sm["rows_identical"] and sm["smoke1"]["raw_tsv_bytes_identical"], "smoke comparison did not pass"
    e_lo = min(min(p["evalue_ratio_min"], p["i_evalue_ratio_min"]) for p in per)
    e_hi = max(max(p["evalue_ratio_max"], p["i_evalue_ratio_max"]) for p in per)
    c_lo = min(p["c_evalue_ratio_min"] for p in per)
    c_hi = max(p["c_evalue_ratio_max"] for p in per)
    c_med = sorted(p["c_evalue_ratio_median"] for p in per)[len(per) // 2]
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = f"""Whole-proteome Pfam annotation of {name}
Written {today}, when the CPU search array was submitted.

DATABASE  Pfam only: Aksha installed PFAM = Pfam 38.1, {meta['pfam_models']:,} models,
          {meta['pfam_hmm']} (sha256 {meta['pfam_hmm_sha256']}).
SEARCH    Aksha 0.2.3 (/home/jwestrob/.local/bin/aksha; tag v0.2.3, runtime 0.1.1), CPU mode,
          `aksha search --installed_hmms PFAM --cut_ga --prot_in <shard> --outdir <shard dir> --threads 24`
          (gathering thresholds; 24 CPUs per shard on -p standard --qos=standard_expand).
INPUT     {info['source']}
          {n:,} proteins, {info['residues']:,} residues, sha256 {info['source_sha256']}.
          Headers are the package gids 0..N-1; PFAM.parquet keeps each header string unchanged as protein_id
          (column gid holds the same value as int64).
SHARDS    {k} contiguous gid ranges of {lo:,}-{hi:,} proteins (shards/shards.tsv with per-shard sha256; the
          concatenated shards reproduce the input byte for byte). Slurm array {array_job} (at most 4 tasks running),
          finishing job {finish_job} (afterany).
OUTPUT    PFAM.parquet, one row per reported domain: protein_id, gid, accession, name, description, bitscore and
          evalue (full sequence), dom_bitscore, c_evalue, i_evalue, hmm_from, hmm_to, ali_from, ali_to, env_from,
          env_to; sorted by gid, env_from, accession. PFAM_summary.json holds the counts and per-shard times.
          STATUS is written last: DONE <UTC> rows <n> proteins_with_hit <n> shards <n> database PFAM only, or
          FAILED <reason>.

E-VALUES AND SHARD SIZE
With --cut_ga every row is selected by Pfam's gathering bit-score thresholds (sequence and domain), so the set of
Pfam rows (protein, model, coordinates, bit scores) is the same for any shard size. The reported E-values depend on
the search size: evalue and i_evalue scale with the number of target sequences in the search ({lo:,}-{hi:,} per
shard here, against {n:,} for one whole-collection search, so these two columns are about {k}x smaller than a
whole-collection search would print), and c_evalue also depends on the per-shard hit counts. Compare E-values
within a shard only; rank and filter on bit scores.
Smoke test, {sm['validation_rows']:,} rows of the previous agent's CPU validation table (4,000 testset3 proteins,
job 119815), run through this chain (shard, array, merge, STATUS; $A/cpu_smoke/): one shard reproduced the
validation table byte for byte; three shards of 1,333-1,334 proteins gave {sm['smoke3']['rows']:,} rows identical on
every column except the three E-value columns. There, evalue and i_evalue changed by the factor shard size / 4,000
(ratios {e_lo:.3f}-{e_hi:.3f}, expected 0.333) and c_evalue by factors {c_lo:.2f}-{c_hi:.2f} (median {c_med:.2f}).
"""
    with open(f"{o}/README_annotation.txt", "w") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
