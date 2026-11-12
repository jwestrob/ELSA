# DPANN benchmark package for the ELSA redesign

Workstream `ws_dpann`, built 11 November 2026 on biotite. Location:
`/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/ws_dpann/package/`.
Every file, its row count, size and SHA-256 is listed in `MANIFEST.tsv` / `MANIFEST.json`.
Stage receipts with every count quoted here are in `receipts/`.

## Source identity

| Input | Identity |
|---|---|
| Sharur DPANN database (canonical) | seal `805c93a69776eb746bb9f642d71bfd52055540968f3a1f8b40f6987f7fcc29e5` (2026-10-19); file SHA-256 `bdd4443500dc668cf974358e44a2824cd0f15691412863586363ca8f7793e175`, 13,408,677,888 B; byte copy at `../input/sharur.duckdb` (opened read-only) |
| March 2026 embeddings | `/groups/banfield/users/jwestrob/bin/Sharur/data/dpann/embeddings/protein_embeddings.h5`, SHA-256 `27b8e6fb6db2a6b1209d703e0d084ac744a69d66f11f5ab8c1ec029fc80a724a`, `facebook/esm2_t6_8M_UR50D`, 3,395,472 × 320 |
| March database (sequence of each embedded ID) | `/groups/banfield/users/jwestrob/bin/Sharur/data/dpann/sharur.duckdb`, SHA-256 `c7d8e563…` |
| Independent gene calls (strand check) | Prodigal 2.6.3 single-mode re-call of 14 Oct 2026, `/groups/banfield/users/jwestrob/dpann_reingest_2026-10-14/proteins/*.faa` (3,616,954 calls) |
| Taxonomy | GTDB r232 `ar53_taxonomy_r232.tsv`, SHA-256 `f20b8005…` |
| Sequence families | MMseqs2 release 18 `8cc5ce36…` |

## Contents

| Path | Rows | What |
|---|---|---|
| `universe/genes.parquet` | 3,585,303 | `gid, protein_id, genome, genome_code, contig, contig_code, position_index, strand, length, start, end` |
| `universe/genomes.parquet` | 3,564 | taxonomy (phylum to species, r232), `tax_source`, `tax_db`, completeness, contamination, order-trust flags and strand evidence |
| `universe/contigs.parquet` | 231,477 | `genome, contig, contig_code, n_genes, length_nt, first_gid` (contigs that carry genes) |
| `gold/instances.parquet` | 57,310 | one row per curated caller instance (below) |
| `gold/members.parquet` | 68,719 | `instance_id, caller, gid` |
| `annotation_flags.parquet` | 3,585,303 | per gid: `has_pfam, has_domain, has_orthology, unannotated, no_pfam_no_ko`, and `src_<source>` per annotation source |
| `tokens/mmseqs__seq__id30c80__dpann.parquet` (+ `.json`) | 3,585,303 | token arm `gid, token` (BRIEF section 5 format) |
| `embeddings/esm2_8m_march.npy` | 3,585,303 × 320 | March vectors aligned to gid, L2-normalized, NaN where no vector. **Defective, see below** |
| `embeddings/embedding_match.parquet` | 3,585,303 | `gid, genome, length, h5_row, match_kind, truncated_1022` |
| `embeddings/coverage_per_genome.tsv` | 3,564 | coverage per genome |
| `embeddings/control/` | | re-embedding control, pooling diagnosis, clean re-embedding timing benchmark |
| `qc/strand_evidence.parquet` | 3,585,303 | per gid: stop-codon matches to the Prodigal re-call and the implied strand |
| `scorer/score_dpann_loci.py`, `scorer/selftest_scorer.py`, `scorer/selftest.json` | | scorer, self-test, its result (PASSED) |
| `scorer/example_whole_contig/` | | scorer output for one whole-contig locus per contig |
| `code/` | | every build script and Slurm wrapper used |

## Gene universe

- `gid` = rank over (genome_code, contig_code, position_index), 0-based. Each contig is the
  contiguous gid range `[first_gid, first_gid + n_genes)`.
- `position_index` = rank of the gene on its (genome, contig) by (start, end, protein_id). It
  equals the stored Sharur `gene_index` order on all 231,477 contigs.
- `strand` = +1 / -1 (int8). Canonical values are `+` 1,796,332 and `-` 1,788,971.
- `length` = protein length in residues; `start`, `end` = 1-based nucleotide coordinates.

Quality checks (receipt `universe.json`, `strandqc.json`):

| Check | Result |
|---|---|
| duplicate (contig, position) | 0 |
| contigs with one gene | 6,572 (6,572 genes) |
| identical-coordinate models | 4 pairs (8 proteins; source-model variants differing at the initial residue) |
| synthetic `WP` contig / start <= 0 / gene outside its contig | 0 / 0 / 0 |
| strand vs Prodigal at a shared stop codon | 3,515,947 agree, 232 disagree (0.999934) on 3,516,179 matched genes; all 232 sit at contig edges where partial genes share a boundary coordinate |
| order-trusted genomes | 3,564 of 3,564 |

The seven RefSeq genomes that carried the synthetic `WP` contig in the March database
(GCF_002214165.1, GCF_003086415.1, GCF_009617975.1, GCF_016806735.1, GCF_023169545.1,
GCF_029207735.1, GCF_029232205.1) have real coordinates in the canonical database and agree with
Prodigal on strand for every matched gene.

## Gold loci

Curated, purpose-built system callers present in the live schema, with every instance exported
and the system type exactly as emitted:

| caller | source table | instances | system types | single-member | span (genes) median / 90th / max |
|---|---|---|---|---|---|
| DefenseFinder | `defense_systems` | 4,495 | 158 (`system_type`) | 2,023 | 2 / 4 / 13 |
| TXSScan | `secretion_systems` | 809 | 3 | 5 | 8 / 112 / 1,278 |
| CRISPRCasTyper (`cctyper-port 1.9.0`) | `crispr_cas_systems` | 52,006 | 24 (`prediction`; `status` kept: `cas_putative` 51,558, `cas` 250, `crispr_cas` 171, `crispr_cas_putative` 27) | 48,186 | 1 / 1 / 27 |

All 57,310 instances resolve: every member maps to a gid on one contig of the stated genome, and
the member lists equal `system_proteins`. Instance columns include `start_idx, end_idx` (gene-index
span, inclusive), `member_gids`, `member_positions`, `span_genes`, `n_members_resolved`,
`n_members_unannotated`, `n_members_no_pfam`, `n_span_unannotated` and the caller's own fields
(`system_subtype`, `status`, `extra` = DefenseFinder `activity` or the CCTyper caller string).

TXSScan admits scattered and multi-locus components: 72 of 809 instances have contiguous members.
Use `--max-span` to score compact instances only.

Present in the database but outside the gold set: 625 CRISPR arrays (`loci`, `crispr_array_types`;
nucleotide arrays without member proteins), the `bgc_loci` view (0 rows), and HydDB hydrogenase
classifications (2,228 proteins; a per-protein classifier).

Annotation flags: `has_domain` = any Pfam, DefenseFinder, TXSScan or HydDB profile hit;
`has_orthology` = any KOfam or KEGG row; `unannotated` = neither (1,408,443 proteins, 39.3 %);
`no_pfam_no_ko` matches the 2 Nov 2026 dark-protein definition.

## Token arm

`mmseqs easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0` on all 3,585,303 proteins (308 s on 64
threads): 674,172 tokens, every protein assigned, 461,906 singletons, 212,266 tokens with >= 2
members, 73,473 tokens in >= 5 genomes. The file lives here, outside `$RD/tokens/`, because that
directory is defined on the 4,257,531-row bacterial universe.

## Embeddings: the March vectors are defective

Coverage by exact identity (same protein ID, identical sequence): 3,392,868 proteins (94.63 %);
with identical sequences embedded under another ID, 3,410,980 (95.14 %); 3,452 of 3,564 genomes at
>= 80 %. Uncovered: 174,194 proteins added after March (median 87 aa) and 129 whose sequence
changed.

The vectors themselves were produced by Sharur commit `fc03c29`, which averaged the final hidden
state over **all positions of a padded batch of 16**, including BOS, EOS and padding. Re-running
the exact batches reproduces the h5 (median cosine 1.0000, 94 % >= 0.999 on 384 proteins; the
residue grows with the padded fraction, as expected from MPS versus CPU), while a clean
single-protein embedding of the same sequence sits at median cosine 0.93 from the stored vector,
the cosine of unrelated proteins in ESM2-8M space. Only the longest protein of each batch is clean.
The bacterial redesign vectors use residue-only means. **Do not use `esm2_8m_march.npy` for
similarity, tokens or anchors.** A clean re-embedding with the bacterial convention
(`code/reembed_esm2.py`, batch invariance checked to cosine 0.9999996) runs at 7,351 residues/s on
24 CPUs: about 800 CPU-hours for the 880 M residues of proteins <= 1,022 aa.
`code/reembed_shards.sbatch` (8 shards, 2 at a time) and `code/assemble_reembed.py` are ready and
have not been submitted.

## Scoring predicted loci

```bash
PY=/home/jwestrob/miniconda3/envs/elsa/bin/python
$PY scorer/score_dpann_loci.py --pkg <package dir> --loci my_loci.parquet --out results/ \
    --threads $SLURM_CPUS_ON_NODE [--callers DefenseFinder TXSScan] [--min-members 2] [--max-span 20] \
    [--include-untrusted] [--controls 200 --seed 17]
```

Input: `genome, contig, start, end, module_id` in gene-index coordinates (`position_index`, start
and end inclusive), parquet or TSV. Measures follow `evaluate` in
`$SCALE/code/evaluate_spacedust_queries.py`: per instance the single locus with the best gene-level
F1 (member overlap; tie-break coverage, smaller span, earlier start/end), its coverage and
boundary precision (member overlap / locus span), plus span precision (overlap with the
first-to-last member interval / locus span), `covered50` (some locus covers >= 50 % of members)
and `localized` (coverage and precision both >= 0.5). Control: the exact member pattern rigidly
translated to 200 uniform legal origins on its own contig, RNG seeded by
`sha256(f"{seed}:{instance_id}")`. Module consistency: share of a system type's instances whose
best locus carries the plurality module, with the same statistic on the controls. Output:
`instance_scores.parquet` and `summary.json` with `by_caller`, `by_type` (caller x system type)
and `by_size` (members 1, 2-3, 4-6, 7+), each with control means, 95 % control intervals, F1
minus control and a one-sided empirical p over draws.

Self-test (`scorer/selftest.json`): 32 of 32 checks pass on 57,310 instances. Whole-contig loci
give coverage 1, precision exactly k/n, F1 exactly 2k/(k+n), F1 minus control 0 and p = 1; gold
instances as loci give covered50 = 1, F1 = 1 on every contiguous instance and module consistency
1.0; no loci give zeros; the vectorized scorer equals a brute-force reference on 9,000 random
placements.

Whole-contig reference values (the trivial upper bound on coverage, `scorer/example_whole_contig/`):
mean best F1 0.119 (CRISPRCasTyper), 0.208 (DefenseFinder), 0.185 (TXSScan), each equal to its
control.

## Scope of the pooling defect across Sharur stores (follow-up, `embeddings/store_scope/`)

The padded-batch all-position mean was written by Sharur `fc03c29` and `4592775`; the
residue-masked mean starts at `8f89a02` (17 Jul 2026). Reconstruction tests (random 32-row
windows): omni_production (10 Mar, batch 16), dummy (31 Mar, batch 32) and spicy_lams (3 Apr,
batch 32, NCBI-protein store) reproduce at 100 % of sampled vectors >= 0.999 under the defective
pooling and at 3-7 % under a clean single-protein embedding, like DPANN. srvp_archaea (23 Oct)
is clean: single-protein residue means reproduce 100 %; its FAISS sidecar reads back equal to the
normalized h5 rows. srvp_bacteria_pb shares its writer and was not tested.
