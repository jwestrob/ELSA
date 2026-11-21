# ELSA Benchmark Audit Continuation Handoff

**Date:** 2026-07-19  
**Repository:** `/Users/jacob/Documents/Sandbox/elsa_test/ELSA`  
**Status:** Core audit completed successfully; result QA and corrected search remain.

## Start here

Read, in order:

1. Repository `AGENTS.md`
2. `docs/BENCHMARK_AUDIT_HANDOFF_2026-07-19.md`
3. This continuation handoff
4. `benchmarks/evaluation/audit/audit_results.json`
5. `benchmarks/evaluation/audit/claim_ledger.tsv`
6. `benchmarks/evaluation/BENCHMARK_AUDIT_2026-07-19.md`

The worktree already contained extensive user changes. Preserve unrelated
changes and inspect `git status` before editing.

## Completed in this continuation

`benchmarks/scripts/audit_canonical_benchmarks.py` is now a complete,
read-only audit driver with:

- Explicit input paths
- Sealed SHA-256 validation
- Runtime and repository provenance
- Unique-locus integrity reconstruction
- Sample-global to contig-local coordinate translation
- Boundary-aware pair metrics
- Cluster bootstrap intervals
- Three deterministic placement controls with 100 replicates each
- A whole-contig ceiling control
- Comparative-input integrity checks
- Claim-ledger and Markdown report generation

The full canonical run exited successfully on 2026-07-19 at approximately
04:28 local time using one CPU process.

## Generated artifacts

All derived files are under `benchmarks/evaluation/audit/`:

- `audit_results.json`
- `claim_ledger.tsv`
- `operon_locus_integrity.tsv`
- `operon_loci_reconstructed_all.tsv`
- `operon_pair_metrics_ecoli.tsv.gz`
- `operon_pair_metrics_cross_species.tsv.gz`
- `null_controls_ecoli.tsv`
- `null_controls_cross_species.tsv`
- `sequence_integrity_summary.tsv`
- `sequence_integrity_detail.tsv.gz`
- `mcscanx_identity_audit.tsv.gz`
- `search_benchmark_status.json`

The generated narrative is:

`benchmarks/evaluation/BENCHMARK_AUDIT_2026-07-19.md`

## Output-suppression issue

The current Codex session repeatedly displayed the biology safety banner even
when command output contained only filenames, counts, and aggregate labels.
Raw nucleotide and protein strings were absent from displayed output. The
check also fired on patch and command text, indicating context-level
classification.

Keep terminal output compact. Redirect verbose output to files and inspect
small aggregate slices. Every per-gene audit artifact stores identifiers,
coordinates, flags, counts, and comparison metrics. Raw strings remain in the
canonical source files.

## Principal quantitative results

### Operon cohort integrity

- Definitions file: 66 multi-gene definitions.
- Persisted ground truth: 58 represented definitions.
- Unique `(operon, genome)` loci: 1,105.
- Pairwise evaluation rows: 10,182.
- Current annotation snapshot reconstructed 1,079 loci.
- Persisted loci absent from current reconstruction: 26.
- Complete reconstructed definitions: 888/1,079.
- Mean completeness: 0.9527; median completeness: 1.0.
- Coordinate order: 489 same, 550 reverse, 40 scrambled.
- Coherent strand calls: 1,079/1,079 reconstructed loci.
- Loci containing a partial caller record: 2.
- Sample-global and contig-local ranks differ for 122 joined loci.
- Ten persisted boundary pairs reverse order after translation to sorted
  contig-local coordinates. Inspect these before treating the translation as
  final.

The 26 current-snapshot discrepancies remain evaluable because their
persisted sample-global boundary rows translate through the sealed gene table.
They are explicitly marked `current_reconstruction_present=false`.

### Corrected positional metrics

The published-style evaluator reproduces the original E. coli strict result:

- 10,068/10,182 = 98.8804%.

Requiring exact contig identity and translated contig-local coordinates gives:

| Dataset | 50% same-block span recall | 100% same-block containment |
|---|---:|---:|
| E. coli blocks | 10,039/10,182 = 98.5956% | 9,760/10,182 = 95.8554% |
| Cross-species blocks | 10,044/10,182 = 98.6447% | 9,765/10,182 = 95.9045% |

The coordinate correction changes relatively few pair-row hits. Localization
specificity changes the scientific interpretation:

- Median query boundary precision: 0.0403.
- Median target boundary precision: 0.0407.
- Median geometric pair IoU: 0.0417.
- Median excess span of the smallest fully containing block: 174 genes.

Thus the saved chains usually cover the benchmark intervals, while their
boundaries are typically much broader than those intervals.

### Deterministic controls

For the E. coli blocks:

| Control | Mean 50% recall | Observed-minus-null | Empirical p |
|---|---:|---:|---:|
| Shuffle target loci within genome pair | 6.3502% | 92.2453 percentage points | 0.0099 |
| Rotate target loci within contig | 3.3130% | 95.2826 percentage points | 0.0099 |
| Randomize block placement, preserve spans | 2.9007% | 95.6949 percentage points | 0.0099 |
| Whole-contig ceiling | 100% | — | — |

The cross-species block controls give closely similar values. The chains carry
strong nonrandom positional signal. The whole-contig ceiling demonstrates why
useful resolution requires boundary-specific metrics.

These permutation p-values use the 10,182 pair rows. Add a locus- or
operon-clustered permutation analysis before treating them as independent
biological evidence.

### Block and cohort statistics

- E. coli blocks: 20,117.
- Cross-species blocks: 80,225.
- All 190 E. coli genome pairs and all 435 cross-cohort genome pairs have
  blocks.
- The live cohort table labels the 30 samples as 21/5/4 across its three
  groups, while the canonical narrative says 20/5/5.
- Using the live table, cross-label blocks are 55,898/80,225 = 69.6765%.
- The published 71.7% requires provenance reconciliation; avoid carrying it
  forward as reproduced.

### Comparative-input integrity

The exact retained caller translations and the persisted OrthoFinder inputs
were compared for all 90,589 E. coli genes:

- Exact matches: 41,567/90,589 = 45.8853%.
- Caller records containing `X`: 1/90,589; two `X` residues total.
- OrthoFinder inputs containing `X`: 44,633/90,589; 684,273 `X` residues.
- Persisted OrthoFinder inputs matching the historical one-base slicing
  implementation: 90,588/90,589 = 99.9989%.

Root cause: the historical extraction used a one-based inclusive start as a
Python zero-based slice boundary. This shifts forward-strand coding intervals
out of frame. Reverse-strand intervals mostly retain the reading frame because
reverse complementation changes which boundary determines the first codon.

Consequences:

- The statement that approximately half of the retained caller proteins
  contain internal `X` residues is false.
- The March OrthoFinder run is unsuitable as an orthology reference.
- The 97.1% same-orthogroup anchor claim requires clean-input regeneration.
- The saved block tables omit row-level anchors, so exact anchor identities
  remain unavailable from the canonical output.

The audit detail table contains flags and metrics. Raw protein strings remain
in the canonical source files.

### MCScanX identifier compatibility

The January collinearity file contains:

- 28,196 alignment headers.
- 1,385,558 pair lines.
- 125,232 unique referenced internal IDs.

Across the January and March coordinate maps:

- 122,357 referenced IDs occur in both maps.
- Original identifier text matches: 0.
- Original first-token matches: 0.
- Exact coordinate matches: 159/122,357 = 0.1300%.
- Any coordinate overlap: 267/122,357 = 0.2182%.

The reported March MCScanX recall joined January collinearity identifiers to a
March coordinate map with different identifier meanings. Classify the 80.3%
recall result as invalid provenance. The saved 28,196 block count remains a
real row count from the January run and is unsuitable for a matched March
accuracy comparison.

### Execution discoveries

1. The prior handoff contained a two-character typo in the E. coli
   `genes.parquet` SHA-256. The independently verified digest ends in the
   full value now recorded in the audit script and corrected handoff.
2. `benchmarks/elsa_output/ecoli/data/annotations/` contains caller-produced
   annotations and lacks the named features used by the ground-truth builder.
   The builder used `benchmarks/data/ecoli/annotations/`.
3. The current curated annotation snapshot reconstructs 1,079 of the 1,105
   persisted loci. Establish whether the 26 differences reflect source-file
   drift, builder-version drift, or another construction artifact.
4. Manifest provenance covers the E. coli directory; the cross-species
   directory lacks an equivalent manifest.

## Corrected search benchmark

The corrected search benchmark can use the saved embeddings directly:

- Cross-species gene table: 139,198 genes, 30 genomes, 480 dimensions.
- Its 90,589 E. coli rows match the E. coli table by identifier and
  coordinates.
- Build one CPU FAISS index and reuse it for all queries.
- Evaluate all 1,105 locus copies.
- Directed judged query-target relationships: 20,364.

The existing search artifacts are insufficient:

- Saved CSVs retain genome-level counts and omit returned intervals.
- `benchmark_search_recall.py` reads `block.anchors`, which is absent from the
  current `ChainedBlock`; the current field is `anchor_target_gene_ids`.
- Its locus resolver interprets legacy sample-global ranks after contig
  sorting, shifting 159/1,105 loci.
- It chooses one tie-dependent query copy per definition.
- It omits complete ranked blocks and key run parameters.

Implement the corrected run as follows:

1. Consume `operon_locus_integrity.tsv` and its translated contig-local
   boundaries.
2. Query every one of the 1,105 copies.
3. Persist every returned block with query ID, target genome, target contig,
   target start/end, rank, score, anchor count, and orientation.
4. Match a judged target by target genome, exact contig, and prespecified
   interval coverage.
5. Report 50% coverage as primary and 100% containment as sensitivity.
6. Deduplicate multiple blocks matching one target locus.
7. Calculate recall, AP, reciprocal rank, NDCG, and returned unjudged loci.
8. Report all loci plus a complete/coherent-locus stratum.
9. Treat the ten non-E. coli genomes as unjudged in primary IR metrics and
   report their interleaving separately.
10. Record FAISS version, backend, `nprobe`, neighbor `k`, result cutoff,
    thresholds, tie-breaking, seed, and all input hashes.

Recommended canonical parameters from the current pipeline:

- Backend: FAISS IVF-Flat
- `nprobe=32`
- Neighbor `k=50`
- Similarity threshold `0.85`
- Maximum gap `2`
- Minimum chain size `2`
- Gap penalty scale `0.0`
- Result cutoff `100`

Run the exact Flat backend sequentially as a sensitivity analysis. Expected
peak memory is approximately 1–2 GB. This is CPU work; retain the repository
rule of one compute-heavy process at a time.

## Immediate QA checklist

Before presenting the generated report as final:

1. Add unit tests for interval overlap, pair normalization, sample-global to
   contig-local translation, null reproducibility, and MCScanX parsing.
2. Inspect the ten reversed translated boundary pairs.
3. Explain the 26 current-snapshot reconstruction gaps.
4. Reconcile the 69.68% live cross-label fraction with the published 71.7%.
5. Verify macro and cluster-bootstrap summaries against independent
   calculations.
6. Add clustered null inference or label row-level permutation p-values as
   descriptive.
7. Confirm the report distinguishes high positional signal from poor boundary
   resolution.
8. Run the corrected search benchmark.
9. Regenerate orthology from exact retained caller translations.
10. Add row-level anchor persistence, then recompute anchor validation.
11. Rerun MCScanX from sealed, cohort-matched inputs.
12. Update the durable project context and offer `/capture`.

## Reproduction and verification

The audit script requires every canonical path explicitly. Run:

```bash
source /Users/jacob/.pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh
conda activate elsa
python benchmarks/scripts/audit_canonical_benchmarks.py --help
```

Use the exact resolved paths recorded under `input_files` and
`input_directories` in `audit_results.json`. The successful run used:

- `--output-dir benchmarks/evaluation/audit`
- `--permutations 100`
- `--bootstrap-replicates 2000`
- `--seed 20260719`

The driver rejects fewer than 100 permutations, missing inputs, hash drift,
schema drift, incomplete per-sample file sets, and output paths nested under
canonical input directories.

Verification completed:

- `python -m py_compile benchmarks/scripts/audit_canonical_benchmarks.py`
  exits 0.
- Full canonical execution exits 0.
- Audit script SHA-256:
  `cb22c9a7ea3b71fbc34f9b5d8376b76a4f2284a5fb7f9b962c49e462073e4bed`
- `audit_results.json` SHA-256:
  `eaecf639f05bc8a314d941e1f3f8b37588b45f78c2bc7f0e222defa439ffa775`

These digests describe the state at handoff creation. Any justified QA edit
will change them.

## Worktree boundary

Files created or intentionally changed during this continuation:

- `benchmarks/scripts/audit_canonical_benchmarks.py`
- `benchmarks/evaluation/audit/*`
- `benchmarks/evaluation/BENCHMARK_AUDIT_2026-07-19.md`
- `docs/BENCHMARK_AUDIT_HANDOFF_2026-07-19.md`
  - Corrected the E. coli gene-table SHA-256 typo.
- `docs/BENCHMARK_AUDIT_CONTINUATION_2026-07-19.md`

The repository contained many pre-existing modifications and generated files.
Preserve them. Git history and remotes remain unchanged.

## Scientific synthesis

The audit supports three affirmative conclusions:

1. The canonical block artifacts and original span-recall numbers reproduce.
2. ELSA chains contain strong nonrandom positional signal across the selected
   cohort.
3. Boundary precision is approximately 4% at the median, so the current
   endpoint primarily measures broad locus inclusion.

The comparative biological claims now require sealed regeneration: clean
orthology inputs, persisted row-level anchors, exact-locus ranked search
outputs, and a cohort-matched MCScanX run.
