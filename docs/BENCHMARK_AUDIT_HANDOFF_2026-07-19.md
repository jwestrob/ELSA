# ELSA Benchmark Audit Handoff

**Date:** 2026-07-19  
**Status:** Audit in progress; discovery and initial code tracing complete  
**Repository:** `/Users/jacob/Documents/Sandbox/elsa_test/ELSA`

## Objective

Audit every headline ELSA benchmark claim as an adversarial reproducibility
exercise:

1. Trace each number to source data, executable code, and a persisted result.
2. Recompute headline quantities independently.
3. Inspect cohort construction, denominators, leakage, pseudo-replication,
   baseline fairness, and metric incentives.
4. Run null and adversarial controls that expose oversized-block and
   result-volume artifacts.
5. Classify each claim as:
   - **VERIFIED**
   - **CONDITIONALLY TRUE**
   - **MISLEADING**
   - **UNSUPPORTED**
   - **REFUTED**
6. Produce a machine-readable result ledger and a concise scientific audit.

Jacob explicitly wants the audit to resist narrative momentum in either
direction. Preserve real positive results and state failures directly.

## Required startup context

Read:

- `AGENTS.md`
- `benchmarks/evaluation/CANONICAL_BENCHMARKS_NOPCA.md`
- `benchmarks/README.md`
- `BENCHMARKING_PLAN.md`
- `docs/RESULT_CONTRACT.md`
- `/Users/jacob/Documents/Obsidian Vault/projects/INDEX.md`
- `/Users/jacob/Documents/Obsidian Vault/agent-captures/2026-07-18_elsa-agent-result-contract-repair.md`

There is currently no indexed `projects/elsa-synteny/CONTEXT.md`. Relevant
durable context lives under the Sharur, DPANN, and Omnitrophota project notes.

## Hardware and execution constraints

- Use the `elsa` Conda environment:

  ```bash
  source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
  conda activate elsa
  ```

- One compute-heavy ELSA/FAISS/HNSW/model job at a time.
- One MPS process at a time.
- The work completed so far was read-only CPU analysis.
- No subagents were used.

## Dirty worktree boundary

The repository was already substantially dirty before this audit. Preserve all
existing modifications. The initial `git status --short --branch` included:

- Modified result-contract work in `elsa/`, `tests/`, `README.md`,
  `pyproject.toml`, package metadata, browser database files, caches, and zip
  artifacts.
- Untracked `AGENTS.md`, `.claude/`, `docs/RESULT_CONTRACT.md`,
  `elsa/results.py`, `tests/test_results.py`, benchmark scratch scripts, and
  browser redesign material.

This handoff is the only audit file added during the interrupted audit session.

## Canonical claim ledger

The March canonical report claims:

| Claim | Reported value |
|---|---:|
| E. coli strict operon recall | 98.9% (10,068/10,182) |
| E. coli any coverage | 99.2% (10,105/10,182) |
| Cross-species strict recall | 98.9% (10,073/10,182) |
| Search recall at 25, E. coli | 100% |
| Search recall at 50, cross-species | 100% |
| E. coli blocks | 20,117 |
| Cross-species blocks | 80,225 |
| Cross-genus fraction | 71.7% |
| ELSA vs MCScanX total blocks | 2.85× |
| ELSA vs MCScanX cross-genus blocks | 3.85× |
| ELSA vs MCScanX strict operon recall | 98.9% vs 80.3% |
| Clean anchor orthogroup precision | 97.1% (15,080/15,533) |
| All-anchor orthogroup precision | 74.7% (29,050/38,879) |
| Concave gap penalty block reduction | 59.3% with equivalent span recall |

The current audit has **not** accepted any biological-performance headline yet.
Basic file counts and several arithmetic summaries reproduce. Their
interpretation remains under audit.

## Confirmed findings

### 1. The benchmark documentation describes multiple incompatible runs

`benchmarks/README.md` and
`benchmarks/evaluation/CANONICAL_BENCHMARKS_NOPCA.md` differ in:

- Dataset composition: 21/5/4 versus 20/5/5
- Protein count: 142,952 versus 139,198
- Projection/model regime
- ELSA block counts: 76,954 versus 80,225
- MCScanX cross-genus counts: 14,186 versus 14,940
- Operon recall regimes
- Search recall values

Treat the March canonical report as the current claim ledger. Treat older
documents as historical provenance.

### 2. The “RegulonDB” ground truth is a hard-coded fallback list

The files:

- `benchmarks/operons/ecoli/OperonSet.txt`
- `benchmarks/operons/ecoli/GeneProductSet.txt`

contain the RegulonDB JavaScript application landing-page HTML, not dataset
tables.

`benchmarks/operons/ecoli/operons.tsv` matches the manually embedded fallback
list in `benchmarks/scripts/download_operons.py`. The source comments describe
it as classic or well-characterized textbook operons assembled from literature
and NCBI records. Its `evidence` column contains broad categories such as
`ribosomal`, `replication`, `division`, and `TA_system`; these are not
per-operon RegulonDB evidence records.

Therefore:

- “58 RegulonDB operons” is currently unsupported.
- “58 experimentally verified E. coli operons” is currently unsupported.
- A defensible description is “58 multi-gene loci from a manually curated
  fallback list” until a versioned RegulonDB export and evidence filter are
  incorporated.

The file contains 86 entries plus header; the builder drops single-gene rows,
leaving 58 multi-gene definitions.

### 3. The 10,182 denominator is pairwise pseudo-replication

`benchmarks/scripts/build_operon_ground_truth_v2.py` generates one row for
every genome pair sharing a detected locus.

Measured from `ecoli_operon_gt_v2.tsv`:

- Rows: 10,182
- Operon definitions represented: 58
- Genome pairs: 190, equal to `20 choose 2`
- Unique genome–operon loci: 1,105
- Rows per operon: median 190, range 10–190

The report calls the 10,182 rows “operon instances.” They are pairwise
comparisons of approximately 1,105 underlying locus copies. Their errors are
strongly dependent because every locus participates in up to 19 pair rows.

Future reporting should include:

- Per-locus recovery over the 1,105 unique genome–operon observations
- Per-operon macro-average over 58 definitions
- Per-genome macro-average over 20 genomes
- Pairwise results only as a secondary correlated analysis
- Clustered bootstrap intervals by operon and genome

### 4. Ground-truth construction accepts partial loci and ignores order/strand

`find_operon_in_genome()`:

- Requires only two named genes from an operon.
- Selects the nearby window with the greatest number of unique expected names.
- Ignores expected order.
- Ignores strand.
- Does not require the full operon definition.
- Uses a gap rule of at most three intervening genes.

Measured pair-row locus spans:

- Mean span: 3.99 genes
- Median span: 3 genes
- Range: 2–13 genes
- Mean matched-name count: 3.79

The next audit stage should quantify definition completeness, duplicate-name
windows, same/reverse order consistency, and the number of rows in which both
genomes contain the complete expected definition.

### 5. The ground-truth index implementation mixes sample-global and contig-local ranks

`match_gff_to_parquet_fast()` computes indices over the whole sample DataFrame.
ELSA block coordinates are contig-local ranks. `find_operon_in_genome()` stores
the sample-global indices as `gene_idx_start/end`.

Most benchmark loci appear to lie on the first chromosome, so the numerical
effect may be small. It must be measured across all 1,105 loci before accepting
the recall values. Multi-contig genomes range from one to seven contigs in the
canonical cross-species Parquet file.

### 6. “Strict operon recall” is positional span coverage

`benchmarks/scripts/evaluate_operon_recall.py` counts a hit when any block for
the genome pair spans at least 50% of the ground-truth interval on both sides.

The implementation:

- Does not require an anchor involving an operon gene.
- Does not require paired operon genes to correspond.
- Does not inspect anchor identities.
- Does not use `contig_a` or `contig_b` while selecting ELSA blocks.
- Allows 50% coverage, which can equal one gene for a two-gene interval.

The 98.9% number therefore means:

> For 98.9% of pair rows, an ELSA block from the same genome pair spans at
> least half of each stored interval.

It does not currently establish 98.9% operon recovery, ortholog
correspondence, or boundary accuracy.

A single whole-chromosome block per genome pair achieves 100% under this
metric. Span recall needs boundary precision, anchor correspondence, and null
controls.

The generated report also says “strict recall is low” while reporting 98.9%,
showing template narration detached from the result.

### 7. Search recall is locus-blind

`benchmarks/scripts/evaluate_search_recall.py` claims to check whether known
target operon copies are returned. Its actual hit criterion is:

```python
top_k_genomes.add(block.target_genome)
hits = len(target_genomes & top_k_genomes)
```

It never compares a returned block’s target contig or coordinates with the
known target operon interval.

The saved CSV makes the artifact visible:

- Most queries have 19 target genomes.
- At `k=10`, most report exactly 10/19.
- At `k=25`, most report exactly 19/19.

Thus the claimed 100% search recall measures target-genome representation in
the result list. It does not measure locus retrieval.

The corrected evaluator must:

1. Build an exact target-locus table per `(operon_id, genome)`.
2. Require matching target genome and contig.
3. Measure target interval overlap or anchor correspondence.
4. Deduplicate multiple blocks from the same target locus.
5. Report recall, average precision, reciprocal rank, and NDCG.
6. Evaluate all eligible query copies or a prespecified phylogenetically
   balanced sample, not the first nondeterministically generated row.

### 8. The 97.1% anchor-precision headline lacks reproducible provenance

Repository search found the exact values only in:

- `benchmarks/evaluation/CANONICAL_BENCHMARKS_NOPCA.md`
- README copies of the headline

No script or persisted row-level result reproduces:

- 15,080/15,533 clean anchors
- 29,050/38,879 all anchors

The report says “actual chained anchor pairs” were re-extracted, but the
canonical block CSV stores only ranges and anchor counts. It does not store
anchor identities.

The persisted file `benchmarks/evaluation/anchor_orthogroup_precision.csv`
reports a different positional-pair calculation:

| Method | Blocks | Pairs | Same-OG/all | Same-OG/OG-covered |
|---|---:|---:|---:|---:|
| ELSA | 76,954 | 1,510,346 | 47.95% | 48.90% |
| MCScanX | 28,196 | 1,385,558 | 90.04% | 90.34% |

That script zips every position across each block span. It also is not an
actual-anchor metric.

Until exact anchor identities are regenerated and persisted, classify the
97.1% and 74.7% claims as **UNSUPPORTED**.

Also use “same orthogroup rate,” not “ortholog precision”: membership in one
OrthoFinder orthogroup does not establish a one-to-one orthology relationship.

### 9. The internal-X explanation is biologically implausible and untraceable from genes.parquet

The canonical report states that approximately 48% of Prodigal translations
contain internal `X` residues. Canonical `genes.parquet` files do not retain
sequences, so this cannot be audited from the claimed canonical artifact.

Prodigal single-mode translation of ordinary complete Enterobacteriaceae
genomes should not yield an internal-X rate near 48%. Treat this as a likely
sequence ingestion, extraction, or denominator defect until recomputed from
the exact FASTA inputs.

Required checks:

- Parse every canonical `.faa` sequence.
- Count sequences with any `X`, internal `X`, terminal stop, other nonstandard
  residues, and partial-call flags.
- Stratify by genome and source.
- Verify that the FASTA sequences used for OrthoFinder are identical to those
  embedded by ELSA.

### 10. ELSA block counts reproduce, while their interpretation remains open

Canonical artifacts contain:

- E. coli genes: 90,589
- E. coli blocks: 20,117
- Cross-species genes: 139,198
- Cross-species blocks: 80,225

Reported block-size summaries reproduce from the saved JSON:

- E. coli median 15, mean 35.45, maximum 2,729
- Cross-species median 7, mean 18.28, maximum 2,729

These are output-volume and span summaries. They do not measure correctness.

### 11. “More blocks than MCScanX” is not an accuracy metric

ELSA uses `min_chain_size=2`. MCScanX’s reported configuration uses its
default minimum match size of five. Raw block-count ratios therefore compare
different minimum evidence thresholds and different segmentation rules.

The 2.85× and 3.85× ratios are arithmetic properties of the two outputs.
Their current interpretation as sensitivity or distant-homology advantage is
unsupported.

A fair comparison needs:

- Matched minimum anchor evidence, or a threshold curve
- Pair-level truth or calibrated nulls
- Boundary and correspondence metrics
- Redundancy/fragmentation accounting
- Runtime with complete preprocessing included

### 12. The canonical MCScanX provenance is internally inconsistent

Available files:

- January: `cross_species_v2.collinearity`
- March: `cross_species_v3_fixed.gff`
- No `cross_species_v3_fixed.collinearity`
- No `cross_species_v3_fixed.blast`

The March report appears to parse January collinearity with the March
coordinate file. This is valid only if every OrthoFinder internal ID maps to
the same biological protein in both runs.

The next session must compare:

- January `WorkingDirectory/SequenceIDs.txt`
- March canonical `WorkingDirectory/SequenceIDs.txt`
- Every internal ID referenced by `cross_species_v2.collinearity`
- Every ID and coordinate in `cross_species_v3_fixed.gff`

If mappings differ, all v3 MCScanX recall numbers are invalid.

### 13. The cross-genus fraction is cohort-composition dependent

The 30-genome cohort includes 20 Escherichia, five Salmonella, and five
Klebsiella genomes. Cross-genus genome pairs comprise 225/435 = 51.7% of all
possible pairs.

The reported 71.7% is a raw block fraction across an imbalanced pair design.
Report blocks per genome pair and aggregate at the pair level before
interpreting genus-level sensitivity.

### 14. The concave-gap result is a real computational observation with a weak endpoint

Saved results report:

- Hard gap 2: 19,057 blocks
- Concave scale 1: 7,748 blocks
- Reduction: 59.3%
- Similar positional-span recall

The block reduction is reproducible arithmetic. “Equivalent recall” uses the
defective span metric, so scientific equivalence remains untested.

## Canonical artifact hashes

These SHA-256 hashes were measured during the audit:

| Artifact | SHA-256 |
|---|---|
| `benchmarks/operons/ecoli/operons.tsv` | `abc3bf821632e72b0ac34aba7b2de43d3130ce7aa37d83b5f22f49bc063d7397` |
| `benchmarks/ground_truth/ecoli_operon_gt_v2.tsv` | `1962072bd44b4d0f8a02876cc5d517bfc14e6e21d1ee59fe588e1298d79ba6a0` |
| E. coli `genes.parquet` | `7061b3e40ef1edb9c49e7f560bab3a06cb630f2352be6c017a3b07d1be7c67d8` |
| E. coli blocks CSV | `1b225ea0419e4e8a1ff26f465ab0019e0cb2f9608a7de1841519a6f2e8f84873` |
| Cross-species `genes.parquet` | `86556b741eeaa315599b1964c8259ca275d2e064e5a2764b417aba1a599850b6` |
| Cross-species blocks CSV | `d6c43b58050748cf8c53f4d1bd931b836c0c8009660a9dcacf15f0445cfe55b7` |

The E. coli manifest records the same truncated genes checksum and canonical
configuration. The cross-species directory lacks an equivalent manifest.

## Highest-priority next analyses

### A. Ground-truth integrity table

Build one row per unique `(operon_id, genome)` locus with:

- Expected definition size
- Observed named genes
- Completeness fraction
- Same/reverse/scrambled order
- Strand coherence
- Contig-local rank
- Sample-global rank
- Coordinate mapping multiplicity
- Partial Prodigal status

Persist as:

`benchmarks/evaluation/audit/operon_locus_integrity.tsv`

### B. Corrected operon metrics

For each ELSA block and locus pair, compute:

- 50% and 100% span coverage
- Query and target boundary precision
- IoU
- Smallest containing block span
- Anchor density (`n_anchors / span`)
- Actual operon-gene anchor correspondence after anchor regeneration

Aggregate:

- Pair micro-average
- Per-locus macro-average
- Per-operon macro-average
- Per-genome macro-average
- Cluster bootstrap intervals

### C. Null controls

Run at least 100 deterministic permutations:

1. Shuffle target operon intervals within each genome pair.
2. Circularly shift target intervals within their contigs.
3. Preserve block spans and genome pairs while randomizing placement.
4. Whole-contig trivial baseline to demonstrate the span metric ceiling.

Report empirical p-values and the observed-minus-null effect.

### D. Corrected search benchmark

Run search on exact target loci and persist every returned block:

`query_id, target_genome, target_contig, target_start, target_end, rank, score`

Then calculate exact-locus recall, AP, MRR, NDCG, and false loci per query.
The existing search CSV is insufficient because it retained only genome-level
counts.

### E. Exact anchor regeneration

Re-run the canonical seed/chaining stages or add anchor persistence to the
pipeline. Store:

`block_id, query_gene_id, target_gene_id, similarity, query_rank, target_rank`

Validate same-orthogroup rate for:

- All anchors
- Reciprocal anchors
- PLM-only versus DIAMOND-supported anchors
- Identity bins
- Genus-pair bins
- Complete versus partial/nonstandard sequences

### F. MCScanX identity audit

Prove or refute compatibility between the January collinearity and March GFF.
Then rerun MCScanX from sealed inputs if needed.

## Suggested implementation

Add:

- `benchmarks/scripts/audit_canonical_benchmarks.py`
- `benchmarks/evaluation/audit/audit_results.json`
- `benchmarks/evaluation/audit/claim_ledger.tsv`
- `benchmarks/evaluation/BENCHMARK_AUDIT_2026-07-19.md`

The audit script should be read-only with respect to canonical inputs, accept
all paths explicitly, record complete hashes and versions, and fail closed on
missing provenance.

## Current plan state

1. Project context and artifact inventory — **complete**
2. Trace and independently recompute headline metrics — **in progress**
3. Audit cohort construction, leakage, definitions, and baselines — **in progress**
4. Run adversarial controls — **pending**
5. Write final audit and update durable project context — **pending**

## Provisional scientific verdict

ELSA’s chaining output and repaired result contract remain substantial
engineering contributions. The present benchmark suite does not yet establish
the headline biological performance claims.

The clearest current positive evidence is:

- The pipeline runs at the reported scale.
- Saved block counts and simple size statistics reproduce.
- ELSA constructs dense, long collinear chains in closely related
  Enterobacteriaceae.
- Concave gap scoring changes segmentation substantially while preserving the
  existing span endpoint.

The principal claims requiring re-evaluation are:

- 98.9% operon recall
- 100% search recall
- 97.1% ortholog precision
- PLM-specific distant-homology advantage
- Superiority inferred from greater block counts

The likely outcome is a smaller set of defensible, informative metrics and a
clear experiment showing whether PLM anchors add value over sequence-based
anchors under remote divergence.
