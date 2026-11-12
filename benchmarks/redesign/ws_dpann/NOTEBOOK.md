# ws_dpann notebook

Append-only. Every entry carries a timestamp (local Mac time, MST). Corrections are new
entries that name the entry they correct.

Paths: local `benchmarks/redesign/ws_dpann/`; remote
`RD=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111`,
workstream directory `$RD/ws_dpann/`.

---

## 2026-11-11 04:09 MST — Orientation and input census

Read `docs/redesign/BRIEF.md` in full, the DPANN project context on biotite
(`obsidian-vault/projects/dpann-exploration/CONTEXT.md`), the Sharur `CLAUDE.md` on biotite
and the GTDB locations note.

Inputs found (verified with `ls`):

| Input | Location | State |
|---|---|---|
| Canonical DPANN DuckDB | Mac `~/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb`, 13,408,677,888 B, 19 Oct 2026 | Seal `805c93a69776eb746bb9f642d71bfd52055540968f3a1f8b40f6987f7fcc29e5`; seal records file SHA-256 `bdd44435…f7793e175`; 3,585,303 proteins, 3,564 bins, 236,099 contigs |
| March DuckDB on biotite | `/groups/banfield/users/jwestrob/bin/Sharur/data/dpann/sharur.duckdb`, 3,509,071,872 B, 31 Mar 2026 | Pre-repair: 3,395,557 proteins (with sequences), 2,326,143 contig rows, 4,660 defense and 163 secretion rows. This is the embedding-era protein set |
| March embeddings | `.../data/dpann/embeddings/protein_embeddings.h5` | `facebook/esm2_t6_8M_UR50D`, dim 320, 3,395,472 × 320 float32 plus `protein_ids` (S22); manifest created 2026-03-08 |
| Re-ingest trial | `data/dpann_reingest_2026-10-14/` (Mac) and `/groups/banfield/users/jwestrob/dpann_reingest_2026-10-14/` (biotite) | Prodigal single-mode re-call, 3,616,954 proteins; trial stage 07 only, not canonical. Biotite holds `proteins/*.faa` with Prodigal headers (start, end, strand) |
| 2 Nov sequence families | `/groups/banfield/users/jwestrob/dpann_dark_modules_2026-11-02/` | MMseqs2 easy-cluster 30 % / 80 % (`--cov-mode 0`) on **1,113,090 dark proteins only** (no Pfam, no KO, >= 80 aa) out of 3,585,303. Covers 31 % of proteins, so it cannot serve as the all-protein token arm |
| GTDB taxonomy | `/groups/banfield/projects/environmental/sr/srvp2020/gtdb/rp16_run_r232/ar53_taxonomy_r232.tsv` | r232, the release the bacterial universe uses |

**Decision: which database is "current".** The canonical database is the sealed 19 Oct file on
the Mac. The biotite copy under the Sharur tree is the March pre-repair state. The re-ingest
is a trial. The package is built from the sealed canonical file, copied byte-for-byte to
`$RD/ws_dpann/input/sharur.duckdb` (rsync from the Mac, started 04:05). Its SHA-256 is checked
on biotite against the seal before any query. The March biotite database is read only to
recover the sequence that was embedded for each March protein ID.

## 2026-11-11 04:09 MST — Predeclared usability criteria (written before any export)

The package is **usable** for scoring the recurrence engine when all of these hold:

1. **Gene universe integrity.** Every canonical protein appears exactly once; zero duplicate
   `(contig_code, position_index)`; `position_index` is 0..n-1 on every contig and is
   monotone in start coordinate.
2. **Trusted gene order for >= 95 % of genomes.** A genome is order-trusted when all its
   proteins carry real source coordinates (no synthetic `WP` contig, no start = 0 placeholder
   runs), positions are ranks of start coordinate, and strand is in {+1, -1} and agrees with
   an independent gene caller (the Prodigal re-call) on >= 98 % of the proteins that share
   a stop codon with a re-call. Strand check passes overall when the minus-strand fraction
   is within 0.45-0.55 and canonical-vs-Prodigal strand agreement on shared-stop matches is
   >= 99.5 %. The historical importer bug (27 % of -1 written as +1) would show as a
   minus fraction near 0.37 and agreement near 73 %.
3. **Embedding coverage >= 90 % of proteins overall** by exact identity (same protein ID and
   byte-identical sequence between the March embedded set and the canonical set), and
   >= 80 % in >= 95 % of genomes. A sampled re-embedding control (same model, CPU) must give
   cosine >= 0.999 on identical sequences, otherwise the identity rule is wrong.
4. **Gold loci.** Every exported caller instance resolves all member proteins to universe
   `gid`s on one contig; instances that fail are listed, never silently dropped.
5. **Token arm** covers 100 % of proteins with a token.
6. **Scorer self-test** returns exactly 1.0 coverage, precision and F1 on the gold instances
   themselves, module consistency 1.0 when each system type has its own module, and on
   whole-contig loci returns coverage 1.0 with precision equal to the analytically expected
   mean span ratio (instance length / contig length), computed independently.

**If a criterion fails.**
- (2) below 95 %: ship a per-genome `order_trusted` flag; the scorer excludes untrusted genomes
  by default and reports both views.
- (3) below 90 %: ship the NaN-padded matrix and the per-genome coverage table, and state in
  the report that an embedding arm on DPANN requires re-embedding the uncovered proteins
  (ESM2-8M on CPU is feasible on biotite); that re-embedding is a decision for the coordinator.
- (4) any unresolved instance: exported with `resolved = false` and excluded from scoring.
- (6) any deviation: fix the scorer before shipping.

## 2026-11-11 04:25 MST — Transfer, cluster preflight, code, job plan

- First rsync of the canonical DB ran at ~4.6 MB/s (projected 50 min); stopped and replaced by a
  zstd-3 stream (`zstd -3 -T4 | ssh 'zstd -d > sharur.duckdb.part'`, renamed on success), ~9 MB/s
  of decompressed bytes. Checksum against the seal runs in Slurm stage `verify`.
- Cluster preflight: read `/home/jwestrob/.claude/CLAUDE.md` and
  `/home/jwestrob/jwestrob/obsidian-vault/biotite-cluster-ops.md` in full (04:20). Headers:
  `-A banfield -p standard --qos=standard_expand -c 24` plus the exclude list; one
  `standard_expand_2 -c 64` job (MMseqs2); no `--mem`, `--time`, `-N`; threads from
  `$SLURM_CPUS_ON_NODE`; completion watcher polls `sacct` to a terminal state.
- March DB strand encoding observed on the login node (2 threads, < 2 s): `'1'` 926,698,
  `'-1'` 920,678, `'-'` 641,104, `'+'` 907,077. Mixed symbolic and numeric values; an importer
  that maps only `'+'/'-'` and defaults the rest to +1 turns 920,678 / 3,395,557 = 27.1 % of genes
  to +1, which is the historical bug. The build maps `{'+','1','+1'} -> +1`, `{'-','-1'} -> -1`,
  anything else -> 0 (counted, flags the genome).
- March DB `WP` contig: 7 RefSeq genomes (GCF_002214165.1 962, GCF_003086415.1 512,
  GCF_009617975.1 1,186, GCF_016806735.1 1,007, GCF_023169545.1 723, GCF_029207735.1 1,049,
  GCF_029232205.1 1,169 proteins), all with start = 0. Whether the canonical DB still carries
  them is checked in stage `universe`.
- Code (local `code/`, mirrored to `$RD/ws_dpann/code/`):
  `build_dpann_package.py` (stages verify, universe, prodigal, strandqc, taxonomy, embed, gold,
  fasta), `reembed_control.py`, `mmseqs_tokens.py`, `score_dpann_loci.py`, `selftest_scorer.py`,
  `make_manifest.py`; wrappers `build_a.sbatch` (verify universe prodigal strandqc taxonomy
  fasta), `build_b.sbatch` (embed + re-embedding control), `mmseqs.sbatch` (64 CPU),
  `gold.sbatch` (gold + scorer self-test).
- Definitions fixed before running:
  - `position_index` = rank of the gene on its (genome, contig) by (start, end, protein_id).
    `gid` = rank over (genome_code, contig_code, position_index), so every contig is a
    contiguous gid range.
  - Strand evidence: a canonical gene matches the Prodigal re-call at a stop codon when a
    Prodigal `+` call shares its `end` or a Prodigal `-` call shares its `start`; when exactly
    one orientation matches, that orientation is the implied strand.
  - Embedding identity: same protein ID and byte-identical sequence between the March database
    (sequence of the embedded ID) and the canonical database. A second class, `sequence_only`,
    takes the vector of an identical sequence embedded under another ID; the match table keeps
    the class so either rule can be applied.

## 2026-11-11 04:33 MST — Canonical copy landed; schema census; jobs submitted

- `$RD/ws_dpann/input/sharur.duckdb`: 13,408,677,888 B (equal to the sealed size); SHA-256 check
  runs as stage `verify` in job 76313.
- Live schema census of the canonical copy (login node, 2 threads, 1 GB cap, each query < 5 s):
  - `proteins.strand`: `'+'` 1,796,332, `'-'` 1,788,971. Canonical uses symbolic strand only.
  - No protein has `contig_id = 'WP'` (0 rows). The 7 RefSeq genomes are re-checked per genome in
    stage `universe`.
  - Curated, purpose-built callers with stored instances: `defense_systems` (DefenseFinder,
    4,495 instances, unique system_id), `secretion_systems` (TXSScan, 809), `crispr_cas_systems`
    (caller `cctyper-port 1.9.0`, 52,006 rows: `cas_putative` 51,558, `cas` 250, `crispr_cas` 171,
    `crispr_cas_putative` 27). Membership table `system_proteins` has sources
    `cctyper` (57,645 rows / 52,006 systems), `defensefinder_system` (7,746 / 4,495),
    `txsscan_system` (3,328 / 809).
  - Not locus callers: `loci` holds 625 CRISPR arrays (`locus_type = 'crispr'`, MinCED; no
    member proteins, `locus_proteins` empty); view `bgc_loci` = `loci WHERE locus_type = 'bgc'`,
    0 rows (no BGC caller output stored); `hydrogenase_classifications` (2,228) is a per-protein
    classifier. These are recorded in the census and excluded from gold loci.
  - `annotations.source`: pfam 4,218,124 rows / 2,140,312 proteins; kofam 1,238,969 / 1,077,802;
    defensefinder 367,830 / 145,088; txsscan 138,845 / 69,112; kegg 52,771 / 52,771;
    defensefinder_system 7,746; hyddb_subgroup 4,979 / 2,228; txsscan_system 3,328 / 3,246;
    hyddb 2,240 / 2,228.
  - Flag definitions: `has_domain` = any pfam, defensefinder, txsscan or hyddb profile hit;
    `has_orthology` = any kofam or kegg row; `unannotated` = neither; `no_pfam_no_ko` matches
    the 2 Nov dark-protein definition. Caller-membership rows (`*_system`) and `hyddb_subgroup`
    are not counted as domain evidence.
- Fix before submission: membership-source mapping in `gold` now maps the three
  `system_source` values explicitly (the first draft would have filed `cctyper` under TXSScan)
  and asserts there is no other value.
- Synthetic scorer test (`code/synthetic_scorer_test.py`, login node, < 2 s): 4 hand-computed
  instances, coverage/precision/F1/span precision and module shares exact. PASSED.
- Jobs (all `-c 24 standard_expand` except MMseqs2 `-c 64 standard_expand_2`):
  76313 `dpann_build_a` (verify universe prodigal strandqc taxonomy fasta);
  76314 `dpann_build_b` (embed + re-embedding control), afterok 76313;
  76315 `dpann_gold` (gold + scorer self-test), afterok 76313;
  76316 `dpann_mmseqs` (easy-cluster id 0.3, c 0.8, cov-mode 0; token arm), afterok 76313.

## 2026-11-11 04:45 MST — Job 76313 (build_a) COMPLETED in 1 min 44 s: universe, strand, taxonomy

Receipts: `$RD/ws_dpann/package/receipts/{verify,universe,prodigal,strandqc,taxonomy,fasta}.json`
(mirrored locally under `results/receipts/`).

**Seal.** File SHA-256 equals the sealed digest `bdd44435…f7793e175`; all 11 checked table row
counts equal the seal. Package source = seal `805c93a6…`.

**Gene universe.** 3,585,303 proteins, 3,564 genomes, 231,477 contigs with genes (236,099 contigs
in the assembly table; 4,622 carry no gene).

| Check | Result |
|---|---|
| duplicate `(contig_code, position_index)` | 0 |
| stored `gene_index` rank vs coordinate rank | 0 disagreements on 231,477 contigs |
| start <= 0, end < start, contig absent from assembly, gene beyond contig end | 0, 0, 0, 0 |
| contig named `WP` | 0 proteins |
| identical-coordinate groups | 4 groups, 8 proteins (the four source-model variant pairs noted in the 3 Oct repair) |
| contigs with 1 gene | 6,572 (6,572 genes, 0.18 %) |
| contigs with 2-4 / 5-9 / 10-49 / 50+ genes | 50,915 / 74,372 / 88,165 / 11,453 contigs holding 160,513 / 497,999 / 1,727,178 / 1,193,041 genes |
| strand values | `+` 1,796,332, `-` 1,788,971; minus fraction 0.4990; per-genome minus fraction 1st-99th percentile 0.418-0.588 |
| CDS length not 3 x aa length (+1) | 21,248; 17,668 at a contig edge (partial), 3,432 partial-flagged away from an edge, 22 flagged complete. Diagnostic only; order is unaffected |

**Strand against the October Prodigal re-call (independent caller).** 3,516,179 proteins (98.07 %)
share a stop codon with exactly one Prodigal orientation. Strand agrees for 3,515,947 and disagrees
for 232 (agreement 0.999934; disagreements symmetric, 116 `+`->`-` and 116 `-`->`+`). All 232 sit
at a contig edge (partial `01` 79, `10` 84, `11` 6, null 63), where an edge-truncated gene and an
opposite-strand edge-truncated call share a boundary coordinate without sharing a stop codon. So
the live strand values are correct; the historical 27 % importer error is absent (it would give
agreement near 0.73). Per-genome agreement minimum 0.9932, 1st percentile 0.9985; every genome has
>= 185 stop-matched proteins (minimum stop-match fraction 0.79).

**Seven RefSeq genomes.** The canonical database now carries real coordinates for all seven
(GCF_002214165.1 981 genes / 1 contig; GCF_003086415.1 529 / 20; GCF_009617975.1 1,207 / 1;
GCF_016806735.1 1,027 / 1; GCF_023169545.1 738 / 1; GCF_029207735.1 1,069 / 1;
GCF_029232205.1 1,188 / 1), minimum start >= 1, minus fraction 0.43-0.55, Prodigal strand
agreement 1.000 on 521-1,192 matched genes each. The `WP` placeholder seen in the March database
is gone; these genomes are order-trusted.

**Order trust: 3,564 / 3,564 genomes (100 %)** pass every rule (criterion 2 met; predeclared >= 95 %).

**Taxonomy.** All 3,564 genomes match a GTDB r232 accession exactly (`ar53_taxonomy_r232.tsv`,
SHA-256 `f20b8005…`). Phyla: Nanobdellota 1,738, Micrarchaeota 1,146, Aenigmatarchaeota 315,
Iainarchaeota 185, Altiarchaeota 63, Nanohalarchaeota 46, EX4484-52 44, Undinarchaeota 16,
Huberarchaeota 11; 41 orders. The database string (`bins.taxonomy`, from the March r220 metadata)
agrees at phylum for all 3,564 and differs at order for 65 genomes; r232 is used, the database
string is kept as `tax_db`.

**FASTA for MMseqs2.** `work/proteins_gid.faa`, 3,585,303 records with `>gid` headers, trailing
`*` removed, 0 empty, 0 internal `*`.

**Correction.** Job 76315 (gold, first attempt) FAILED in 3 s: ambiguous column `n_contigs`
(genome table joined into the instance table). Fixed by joining only `order_trusted`; resubmitted
as job 76327.

## 2026-11-11 04:55 MST — Jobs 76314 (embed + control) and 76327 (gold + self-test) COMPLETED

### Embedding alignment (stage `embed`, 75 s)

h5 `protein_embeddings.h5` SHA-256 `27b8e6fb…`, 3,395,472 × 320 float32, attrs and manifest both
`facebook/esm2_t6_8M_UR50D`, dim 320, created 2026-03-07/08. Every h5 ID exists in the March DB
(SHA-256 `c7d8e563…`); 2,405 h5 IDs are absent from the canonical DB (retired models).

| match class (canonical proteins) | proteins |
|---|---|
| `id_and_sequence` (same ID, identical sequence) | 3,392,868 (94.63 %) |
| `sequence_only` (identical sequence under another embedded ID) | 18,112 |
| `id_sequence_changed` (same ID, sequence changed since March) | 129 |
| `id_absent_from_h5` (model added after March) | 174,194 |

Coverage by ID and sequence 0.9463; with `sequence_only` 0.9514. Per genome (any class): median
0.976, 5th percentile 0.906, minimum 0.0054; 3,452 / 3,564 genomes (96.9 %) at >= 0.8 and 3,393 at
>= 0.9; no genome at 0. The 112 genomes below 0.8 hold 112,722 genes and 75,158 of the 174,323
uncovered proteins (Micrarchaeales 61, Parvarchaeales 26, Woesearchaeales 7, Pacearchaeales 5,
others 13); these are the genomes whose gene models were re-called after March. Uncovered proteins
are short (median 87 aa against 211 aa for covered ones). Raw vector norms 3.20-8.00, none zero;
38,728 covered proteins are longer than 1,022 aa.

Predeclared coverage criteria (>= 90 % proteins; >= 80 % coverage in >= 95 % of genomes) are met.

### Re-embedding control (`reembed_control.py`, same model, CPU, one protein per forward pass)

| stratum | n | median cosine to h5 (residue mean) | fraction >= 0.999 | median cosine (all-token mean) |
|---|---|---|---|---|
| id_and_sequence, <= 1,022 aa | 300 | 0.9217 (min 0.676) | 0.07 | 0.9244 |
| id_and_sequence, > 1,022 aa | 40 | 0.999999 (min 0.999998) | 1.00 | 1.0000 |
| sequence_only | 60 | 0.9239 | 0.02 | 0.9265 |
| id_sequence_changed | 60 | 0.8788 | 0.03 | 0.8816 |

**The predeclared control fails for proteins <= 1,022 aa.** The identity rule itself is correct:
proteins longer than 1,022 aa (truncated to the 1,024-token window, so the longest member of any
batch) reproduce the h5 vector at cosine 0.999998 with the all-token mean. Shorter proteins
reproduce at a median of 0.92, which is the cosine of unrelated proteins in ESM2-8M space
(BRIEF trap 4).

**Root cause (from code; exact test submitted as job 76330).** The March h5 was written by Sharur
commit `fc03c29` (`src/ingest/06_esm2_embeddings.py`, unchanged until `4592775` on 30 Mar 2026):
`tokenizer(batch, padding=True, truncation=True, max_length=1024)` then
`last_hidden_state.mean(dim=1)`, batch size 16 on MPS. That mean runs over BOS, EOS **and every
padding position of the batch**. Each vector therefore mixes in (batch max length - own length)
pad-position states, so it depends on which proteins shared the batch. Only the longest protein of
each batch is clean, which matches the > 1,022 aa stratum reproducing exactly. The bacterial ESM2
vectors of the redesign use residue-only means (specials and padding excluded) and skip > 1,022 aa,
so the March DPANN vectors are also on a different pooling convention.

### Gold loci (stage `gold`, 3.7 s)

Callers with stored instances: DefenseFinder (`defense_systems`), TXSScan (`secretion_systems`),
CRISPRCasTyper (`crispr_cas_systems`, `cctyper-port 1.9.0`). Membership from the `protein_ids`
lists equals `system_proteins` exactly for all three (7,746 / 3,328 / 57,645 rows; 0 rows on
either side unmatched). Every instance resolves: all members map to universe gids on one contig of
the stated genome, `genes_count` equals the member count, 0 contig mismatches.

| caller | instances | system types (as emitted) | single-member | span (genes) median / 90th / 99th / max | members median / max |
|---|---|---|---|---|---|
| DefenseFinder | 4,495 | 158 | 2,023 | 2 / 4 / 9 / 13 | 2 / 6 |
| TXSScan | 809 | 3 | 5 | 8 / 112 / 653 / 1,278 | 4 / 12 |
| CRISPRCasTyper | 52,006 (`cas_putative` 51,558, `cas` 250, `crispr_cas` 171, `crispr_cas_putative` 27) | 24 | 48,186 | 1 / 1 / 5 / 27 | 1 / 22 |

TXSScan instances are frequently scattered (MacSyFinder admits loner and multi-locus components):
only 72 of 809 have contiguous members, and the 90th-percentile span is 112 genes. The scorer has
`--max-span` to restrict scoring to compact instances. Not exported as gold (recorded in the
census): 625 CRISPR arrays (`loci`, `crispr_array_types`; nucleotide arrays without member
proteins), the empty `bgc_loci` view, and the per-protein HydDB classifier (2,228 proteins).

Per-protein flags (`annotation_flags.parquet`, 3,585,303 rows): has_pfam 2,140,312; has_domain
2,158,226; has_orthology 1,077,802; unannotated (no domain, no orthology) 1,408,443 (39.3 %).
Unannotated genes inside gold spans: DefenseFinder 1,023 / 9,489 span genes (0 members, by
construction of profile-based calls); TXSScan 14,319 / 37,283; CRISPRCasTyper 11,237 / 62,362
(9,179 of 57,645 members carry no Pfam/KO/DefenseFinder/TXSScan/HydDB hit).

### Scorer self-test (`selftest_scorer.py`): PASSED, 32/32 checks, 57,310 instances

- A, whole-contig loci: coverage 1 and covered50 for all; boundary precision = k/n and F1 =
  2k/(k+n) to < 1e-12; span precision = S/n; all 200 control draws identical to the observation;
  F1 minus control |delta| <= 5.6e-15; module plurality share equals the contig plurality for every
  type. Mean precision: CRISPRCasTyper 0.0705, DefenseFinder 0.1388, TXSScan 0.1155 (equal to the
  independently computed mean of k/n). Mean F1 0.119 / 0.208 / 0.185.
- B, gold instances as loci (one caller at a time): covered50 = 1 for all; F1 = 1 and precision = 1
  for every contiguous instance (49,550 / 3,798 / 72); F1 >= own-span F1 for all; module = own
  type for every isolated instance (52,006 / 4,359 / 809), plurality share 1.0. Mean F1 0.986 /
  0.954 / 0.565.
- C, no loci: all zero.
- D, 4,792 random loci, 1,500 sampled instances, observation + 5 control draws: vectorized scorer
  equals a pure-Python brute-force reference on all 9,000 placements (F1, coverage, precision,
  span precision, covered50, best module).
- Before the real run, `synthetic_scorer_test.py` (4 hand-computed instances) also passed.

## 2026-11-11 05:05 MST — March pooling defect confirmed exactly (job 76330); token arm done (job 76316)

### Diagnosis job 76330 (`diagnose_march_pooling.py`, 2 min)

Question: are the March vectors the `fc03c29` pooling (all-position mean over a padded batch of
16, h5 row order = processing order)? Criterion: re-running rows [16b, 16b+16) as one padded batch
reproduces each vector at cosine >= 0.999; alternatives must do worse.

Twelve random 32-row windows (384 proteins), CPU:

| variant | n | median cosine | min | fraction >= 0.999 |
|---|---|---|---|---|
| padded batch of 16, all-position mean | 384 | 1.0000 | 0.9967 | 0.940 |
| padded batch of 32 | 384 | 0.99985 | 0.910 | 0.633 |
| padded batch of 8 | 384 | 0.99947 | 0.885 | 0.557 |
| single protein, all-token mean | 192 | 0.9336 | 0.698 | 0.094 |
| single protein, residue mean | 192 | 0.9315 | 0.662 | 0.089 |

Batch-16 reconstruction reproduces the h5. Its residual shortfall grows with the padded fraction
of the window (median 1.0 in every bin; minimum 0.99982 at zero padding, 0.9991 at <= 50 %,
0.9984 at 50-80 %, 0.9975 at 80-90 %, 0.9967 above 90 %), as expected from MPS-versus-CPU
arithmetic on the pad states. **Conclusion: every March DPANN vector except the longest protein
of each batch of 16 carries pad-position states of its batch neighbours; the distortion relative
to a clean single-protein embedding is cosine ~0.93 (median), the level of unrelated proteins.**
Consequence for the package: `embeddings/esm2_8m_march.npy` is exported as specified (gid order,
L2-normalized, NaN for 174,323 uncovered proteins) and is flagged in the README as unfit for an
embedding token arm or any similarity work; it is a record of the March state.

### Token arm (job 76316, MMseqs2 18 `8cc5ce36…`, 64 threads, 308 s)

`package/tokens/mmseqs__seq__id30c80__dpann.parquet`: 3,585,303 rows (gid, token), every protein
assigned; 674,172 tokens; 461,906 singletons; 212,266 tokens with >= 2 members; token size median
1, 90th 5, 99th 58, max 4,358; tokens in >= 2 / >= 5 / >= 50 genomes: 206,368 / 73,473 / 7,506.
Kept in the package, outside `$RD/tokens/`, because `$RD/tokens/` is defined on the 4,257,531-row
bacterial universe and a 3,585,303-row file there would break that contract.

Stability against the 2 Nov dark clustering (same flags, MMseqs2 at `/shared/software/bin/mmseqs`,
dark proteins only): on the 1,113,090 shared proteins, 373,792 Nov-2 families against 374,555 new
tokens; co-family pairs recovered 0.748 (34,136,162 of 45,634,452), precision 0.743. Set-cover
clustering at 30 % / 80 % reassigns about a quarter of co-family pairs when the input set grows,
so family tokens are input-set dependent at this level.

### Next

Job 76333: CPU timing benchmark of a clean re-embedding (`reembed_esm2.py`, ESM2-8M, residue mean,
length-sorted batches, > 1,022 aa skipped, same convention as the bacterial arm) on 5,000 random
proteins, with a batch-invariance check. Job 76334: self-test rerun against the final scorer
(adds `--max-span`) plus a whole-contig CLI example.

## 2026-11-11 04:49 MST — Correction of entry timestamps; final jobs; package sealed by manifest

**Correction.** The headings "04:33", "04:45", "04:55" and "05:05" above were typed as estimates
without reading the clock; `date` now reads 04:48 MST. Those entries were written between about
04:30 and 04:45 MST. Authoritative times are the Slurm log stamps (biotite, PST = MST - 1 h):
76313 ended 03:31:16 PST, 76314 and 76327 by 03:33:29, 76316 03:36:28, 76330 03:37:10.
The self-test count "32/32" in the 04:55 entry was 31/31 for that run (job 76327); 32 checks
exist since the p-value check was added (below).

**Clean re-embedding benchmark (job 76333, 3 min 9 s).** `reembed_esm2.py`, ESM2-8M, residue mean,
length-sorted batches under a 16,384-token budget, > 1,022 aa skipped, 24 threads: 5,000 random
proteins (1,215,748 residues, 52 skipped) in 165.4 s = 7,351 residues/s = 30.2 proteins/s. Batch
invariance (same proteins one at a time): minimum cosine 0.9999996. Projection for all 880,237,761
residues of proteins <= 1,022 aa: 33.3 h on one 24-CPU job, 798 CPU-hours.
`code/reembed_shards.sbatch` (8 shards by gid % 8, %2 concurrency) and `assemble_reembed.py` are
written and NOT submitted: committing ~800 CPU-hours of the shared budget is the coordinator's call.

**Scorer change.** In the whole-contig example the flag `f1_exceeds_all_draws` read True although
observation and controls are identical (float noise of 6e-15 over 52,006 instances). Replaced by
`control_draws_ge_observed` and a one-sided empirical `p_f1_vs_control` with ties (within 1e-12)
counted against the observation; added self-test check `A_p_value_one`. Also added `--max-span`.

**Refresh (job 76349).** Stages `taxonomy` and `gold` rerun with the final build script
(integer count columns; per-source flags renamed `src_<source>` because `has_pfam` collided with
the per-source Pfam flag). Instance and flag counts unchanged (receipt `gold.json`). Self-test
against the final scorer (SHA-256 `51400ca5…`) and instances (`ed2132fe…`): **PASSED 32/32**.
Whole-contig example: mean best F1 0.1189 / 0.2076 / 0.1852 (CRISPRCasTyper / DefenseFinder /
TXSScan), F1 minus control 0, p = 1.0 (200 of 200 draws tie).

Receipts record the code hash of the version that ran each stage. Stages verify, universe,
prodigal, strandqc, fasta ran under `80707e07…`, embed under `bbdfc839…`, taxonomy and gold under
the final `6f89f4eb…`; the later edits touched only `st_gold` and `st_taxonomy`, so every output
was produced by stage code identical to the final file.

**Manifest (job 76351).** `package/MANIFEST.tsv` and `MANIFEST.json`: 49 files with rows, bytes
and SHA-256. Small files mirrored to local `results/`.

## 2026-11-11 04:53 MST — REPORT.md not written

The agent harness refused to let this subagent write `REPORT.md` ("subagents should return
findings as text"). The report content went to the coordinator in the final message; the package
`README.md` and this notebook hold the full record.

---

# Follow-up: scope of the March pooling defect across Sharur embedding stores

## 2026-11-11 17:13 MST — Question, census, code history, predeclared criteria

**Question (coordinator).** Which other Sharur embedding stores carry the padded-batch
all-position-mean defect found in the DPANN h5?

**Store census** (`ls /groups/banfield/users/jwestrob/bin/Sharur/data/*/embeddings/`; no `find`):

| dataset | file | date | manifest model | writer recorded in manifest |
|---|---|---|---|---|
| coronamine_boiler_100nm | (embeddings dir empty) | dir 31 Mar 2026 | - | - |
| dpann | protein_embeddings.h5 (3,395,472) | 8 Mar 2026 | esm2_t6_8M | no; identified as `fc03c29` (tested, job 76330) |
| omni_production | protein_embeddings.h5 (2,921,111; 1,831 genomes) | 10 Mar 2026 | esm2_t6_8M | no |
| dummy | protein_embeddings.h5 (10,102; 4 genomes) | 31 Mar 2026 | esm2_t6_8M | no |
| spicy_lams | protein_embeddings.h5 (10,683,624; 2,054 genomes) | 3 Apr 2026 | esm2_t6_8M | no |
| srvp_archaea | protein_embeddings.h5 (1,976,915) + 2 FAISS `ivf_flat_ip` + ids.sqlite | 23 Oct 2026 | esm2_t6_8M | no; run dir `srvp_research_2026-10-23/esm2/` holds the code snapshot and log |
| srvp_bacteria_pb | protein_embeddings.h5 (2,914,401) + 2 FAISS + ids.sqlite | 23 Oct 2026 | esm2_t6_8M | as srvp_archaea |

No `embeddings/` directory under example, reference, spicy_lams_alphaproteo_addon, srvp_joint.
No manifest records a code version.

**Correction to the premise: `4592775` (30 Mar 2026) did not fix the pooling.** Read from git
(Mac Sharur repo): at `4592775`, `_embed_batch` still calls
`tokenizer(..., padding=True, truncation=True, max_length=1024)` and
`outputs.last_hidden_state.mean(dim=1)`; it changed only the I/O (streaming HDF5 in processing
order, genome directories now `sorted()`), with batch size 16 on MPS, 32 on CUDA, 4 on CPU. The
residue-masked mean first appears in `8f89a02` (17 Jul 2026, "Harden Sharur's operator control
plane"), which moved streaming into `sharur/ingest/embedding_stream.py`. The only other commit
touching the script is `ef99580` (25 Sep 2026), which keeps the masked mean. `git grep
last_hidden_state` finds no other pooling site in the repo (HEAD `99a6306`). So every store
written before 17 Jul 2026 is a candidate, whichever of `fc03c29` or `4592775` wrote it; the
batch size depends on the device. The biotite checkout pulled main on 5 Apr 2026 (reflog), after
dummy (31 Mar) and spicy_lams (3 Apr) were written.

SR-VP writer identified: `esm2/scripts/esm2_full.sbatch` (job 21106, GPU) runs a code snapshot
`esm2/code/src/ingest/06_esm2_embeddings.py` whose lines 157-164 use
`return_special_tokens_mask=True`, `residue_mask = attention_mask & ~special_tokens_mask` and a
masked mean; the log reports `cuda`, batch size 32.

**Test (same as job 76330, generalized: `code/diagnose_store_pooling.py`).** Random 32-row windows
aligned to 32 (so aligned to 4, 8, 16 and 32). Sequences for the sampled IDs come from the
store's own stage03 FASTA (the embedder's input, trailing `*` stripped as both old writers did),
located through the manifest's genome order, with the dataset DuckDB (READ_ONLY) as fallback;
the source of each sequence is recorded. Variants: padded batches of 4, 8, 16, 32 with the
all-position mean (old writers); padded batch of 32 with the residue-masked mean (new writer);
single protein, all-token mean; single protein, residue mean. Windows: 12 for omni_production
(384 proteins, as for DPANN), 6 for dummy, spicy_lams and srvp_archaea (negative control).

**Predeclared criteria.**
- *Defect present*: some padded all-position variant reproduces >= 90 % of sampled vectors at
  cosine >= 0.999 and the single-protein residue mean reproduces < 50 %.
- *Clean*: the single-protein residue mean reproduces >= 95 % at >= 0.999.
- Anything else: unresolved, reported as such.
- Negative control (srvp_archaea) must come out clean; if it does not, the test is not trusted.
