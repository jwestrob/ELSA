Whole-proteome Pfam annotation of lut
Written 2026-11-21 05:11 UTC, when the CPU search array was submitted.

DATABASE  Pfam only: Aksha installed PFAM = Pfam 38.1, 27,481 models,
          /groups/banfield/users/jwestrob/.config/Astra/PFAM/Pfam-A.hmm (sha256 a78a42d6faf265b6bfca59e8f062d06fae6083ce2c6e335d7b381f20b82b7903).
SEARCH    Aksha 0.2.3 (/home/jwestrob/.local/bin/aksha; tag v0.2.3, runtime 0.1.1), CPU mode,
          `aksha search --installed_hmms PFAM --cut_ga --prot_in <shard> --outdir <shard dir> --threads 24`
          (gathering thresholds; 24 CPUs per shard on -p standard --qos=standard_expand).
INPUT     /groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/slice_lut/testset/package_k8/proteins_gid.faa
          4,507,042 proteins, 1,361,583,935 residues, sha256 1ca3d673d2df8450b2b5e813cf1e0e97dba093a1d80991073b41ec4e4b6e2571.
          Headers are the package gids 0..N-1; PFAM.parquet keeps each header string unchanged as protein_id
          (column gid holds the same value as int64).
SHARDS    12 contiguous gid ranges of 375,586-375,587 proteins (shards/shards.tsv with per-shard sha256; the
          concatenated shards reproduce the input byte for byte). Slurm array 119854 (at most 4 tasks running),
          finishing job 119855 (afterany).
OUTPUT    PFAM.parquet, one row per reported domain: protein_id, gid, accession, name, description, bitscore and
          evalue (full sequence), dom_bitscore, c_evalue, i_evalue, hmm_from, hmm_to, ali_from, ali_to, env_from,
          env_to; sorted by gid, env_from, accession. PFAM_summary.json holds the counts and per-shard times.
          STATUS is written last: DONE <UTC> rows <n> proteins_with_hit <n> shards <n> database PFAM only, or
          FAILED <reason>.

E-VALUES AND SHARD SIZE
With --cut_ga every row is selected by Pfam's gathering bit-score thresholds (sequence and domain), so the set of
Pfam rows (protein, model, coordinates, bit scores) is the same for any shard size. The reported E-values depend on
the search size: evalue and i_evalue scale with the number of target sequences in the search (375,586-375,587 per
shard here, against 4,507,042 for one whole-collection search, so these two columns are about 12x smaller than a
whole-collection search would print), and c_evalue also depends on the per-shard hit counts. Compare E-values
within a shard only; rank and filter on bit scores.
Smoke test, 6,872 rows of the previous agent's CPU validation table (4,000 testset3 proteins,
job 119815), run through this chain (shard, array, merge, STATUS; $A/cpu_smoke/): one shard reproduced the
validation table byte for byte; three shards of 1,333-1,334 proteins gave 6,872 rows identical on
every column except the three E-value columns. There, evalue and i_evalue changed by the factor shard size / 4,000
(ratios 0.331-0.336, expected 0.333) and c_evalue by factors 0.05-1.00 (median 0.50).
