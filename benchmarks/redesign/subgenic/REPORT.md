# Sub-genic synteny on DPANN: sizing measurement (ws_subgenic)

12 November 2026. CPU only, MMseqs2 release 18, no embeddings. Written by the orchestrator from
the workstream agent's hand-back. Gate fixed in `predeclared.json` at 2026-11-12T04:59Z, before
any measurement; `progress.log` is the timestamped run log.

## Verdict

The predeclared gate (b) is met, so the gate says "build"; gate (a) fails under its literal
control and passes under a shuffle control. The impact measurements point to a narrower and
cheaper target than residue-level sub-genic tokens:

- Composite (fused/split) genes are mostly absorbed by whole-gene tokens already, and gold
  instances containing them are recovered better than others.
- Fragments (truncated calls at contig edges) are the class that damages recovery. A rule that
  assigns a token by ≥ 80% coverage of the shorter protein would rejoin 60% of them.
- Split genes are mostly clade-wide two-gene states: evenly spread over genomes and lineages and
  unrelated to assembly quality.
- Giant proteins: 57% are tiled by two or more adjacent smaller genes in some other genome.

## Design

An all-against-all of the 2,547,528 unique sequences saturates `--max-seqs` for 98% of queries
(median 292, q90 2,047, q99 3,738 hits per query at E ≤ 1e-5 and target coverage ≥ 0.7; at least
1.6 billion hits). The search that ran:

| Setting | Value |
|---|---|
| Queries | All non-giant unique sequences of 356 random genomes (10%, seed 17; 353,941 genes) plus every member of the gold instances with ≥ 2 members and its ±3 neighbours (61,657 genes): 371,905 unique sequences |
| Targets | All 2,547,528 unique sequences |
| Search S | `-s 7.5 --max-seqs 8000 -e 1e-5 -a`, coverage applied afterwards; 4 shards of about 31 min; 342.4 M rows; 3,998 queries (1.1%) reached ≥ 0.8 × max-seqs |
| Search B | All 4,330 unique giants (≥ 2,046 aa; 5,034 genes) against all targets, `--max-seqs 20000` (largest hit list 8,459) |
| Search C | Giants against giants |
| Search D | Each giant cut into 200-aa windows and aligned to its parent with `--alt-ali` |

Gene fractions carry a genome-level bootstrap interval. `universe/genes.parquet` `length`
exceeds the FASTA length by one for 1,225,260 genes (stop counted); FASTA lengths were used.

## Gate (a): split-in-synteny composites, share of sample genes

| Quantity | Value |
|---|---|
| Involved in an adjacent split/fused relation | 29.6% [29.1, 30.2] |
| Fused (X) side | 25.1% |
| Split (Y) side | 7.8% |
| Literal control (same genome, pieces more than 3 genes apart) | 39.3%; excess −9.6% [−10.0, −9.3]; gate not met |
| Within-genome shuffle control | 4.4%; excess +25.3% [+24.8, +25.7] |
| Adjacent groups against the shuffle | 49.7× |
| Best adjacent pair co-oriented and in the order of the X regions | 91.3% (shuffle 31–37%); 92% at distance 1 |

The literal control counts multi-domain X genes whose separate domains have homologs scattered
over a genome, so it is not a chance null. The X-side 25% grows with collection size: a gene
counts if any of the 3,563 other genomes holds a split copy. The split side (7.8%) and the
fragments are the quantities that describe a genome.

## Gate (b): gold instances with ≥ 2 members (7,096; threshold 142)

- Affected: 5,006 (70.5%); gate met. Fragment members alone 530 (7.5%); split-side members alone
  1,037 (14.6%).
- Composite-containing instances recover better: poor-recovery (F1 < 0.5) Mantel–Haenszel odds
  ratio 0.53, stratified F1 difference +0.049 (p < 2e-4). The non-adjacent pseudo-control shows
  the same (odds ratio 0.48): a widespread-family effect.
- Fragment-containing instances recover worse: odds ratio 2.50 (2.17 stratified also on being
  within 3 genes of a contig end); F1 −0.088 (p < 2e-4).

## Token loss on the permissive base `e3c50s75`

Lost = singleton token, or no aligned partner in the same token.

| Gene class | Lost | Detail |
|---|---|---|
| Composite X genes | 0.1% | 85.1% share a token with a piece of their own split pair |
| Split Y genes | 1.9% | 40.9% share a token with the fused X |
| Fragments | 27.1% | 18.7% are singletons; 39.9% share a token with a longer partner |
| All other genes | 1.7% | |

## Fragments

Median 99 aa; partners 2.65× longer. Rate 4.2% of genes [3.9, 4.5]. Among genes with ≥ 2
partners, 39.2% of contig-edge genes are fragments against 1.7% of interior genes (23×). The
rate is 7.0% / 5.4% / 3.3% at completeness 50–70 / 70–90 / ≥ 90%, and per genome it tracks N50
(Spearman ρ −0.80, p 2e-81) and contig count (ρ 0.71). The top 10% of genomes hold 20.8% of
fragments.

## Split pairs (15,320 loci in sample genomes; 11,916 at distance 1)

| Gap between the pieces (nt) | Split pairs | All co-oriented neighbours |
|---|---|---|
| Overlap | 37.8% | 20.8% |
| 0–50 | 48.5% | 43.3% |
| 51–200 | 11.8% | 27.7% |
| > 200 | 2.0% | 8.1% |

- Same frame 17.9% (all neighbours 25.5%).
- Median joint coverage of the X homolog 0.94 (83.7% reach 0.8); median 15 aa of X left
  uncovered between the pieces, 40% leave ≤ 10 aa.
- "Broken-call-like" loci (collinear, adjacent, gap ≤ 50 nt, ≤ 10 aa of X uncovered): 27.3%.
- Concentration: per-genome rate 5.2% / 7.4% / 10.5% (q10 / q50 / q90); the top 10% of genomes
  hold 17.3%; unrelated to completeness, contamination, contig count or N50 (|ρ| ≤ 0.05);
  6.2–8.7% by order, 6.3–8.3% by phylum.
- Families with both forms (1,570 of 1,916 with a split form): the median family is split in 16%
  of its member genomes. Weighted by split loci, 59% lie in families where the split form is the
  majority and 9.8% where it is rare (≤ 10%). Within a GTDB order, 47.1% of loci are in families
  split in ≥ 80% of the order's member genomes and 11.9% in ≤ 20%. Broken-call-like loci show the
  same clade-wide pattern (median family split fraction 0.67). 81.7% of split token pairs recur
  in ≥ 2 sample genomes.

## Giant proteins (≥ 2,046 aa; 4,330 unique, 5,034 genes)

- 99.4% of giant genes have a local homolog (≥ 100 aligned aa, E ≤ 1e-5) in another genome.
- 97.5% of unique giants have another giant as a partner; 1.8% have smaller partners only.
- 2,471 (57%) are tiled by ≥ 2 adjacent smaller genes in another genome (median 56% of the
  giant's length): 1,104 tilings look like broken calls (collinear, adjacent, every gap ≤ 50 nt;
  790 with a frame shift at one or more junctions), 687 like separate genes (every gap > 50 nt),
  680 are mixed.
- Internal repeats (200-aa windows against the parent, E ≤ 1e-5): 68.6% of giants, median repeat
  fraction 0.29, median offset between copies 932 aa. Length control (1,000–2,045 aa) 30.6%,
  median 0; window-shuffled giants 0.4%.
- 59.1% of giant genes carry a Pfam hit (observation; no family or system assigned).

## What a repair step would recover (genome-wide = sample × 10.13)

- Merging broken-call-like split loci: 7,913 sample genes collapse into 3,726; about 80,200
  genes genome-wide (2.2%).
- Shorter-sequence coverage rule (alignment covers ≥ 80% of the shorter protein, E ≤ 1e-5): 8,948
  of 14,885 sample fragments join a family token, 2,786 of them current singletons; about 90,600
  genes genome-wide (2.5%).
- Gold instances touched: 574 of 7,096 (8.1%); 313 by split merges, 296 by fragment joins
  (DefenseFinder 253 / 2,472, CRISPRCasTyper 292 / 3,820, TXSScan 29 / 804).

## Open

- A nucleotide check across each split junction separates real fission from a systematic
  mis-call (programmed frameshift, recoded stop).
- The effect of the fragment rule and the split merge on locus F1 needs an engine run on
  repaired tokens (workstream `repair`).
- A future gate should use the shuffle null.

## Jobs and files

Jobs: 78148 (prep, failed on the length check), 78154 (prep and search benchmark), 78191
(searches D, C, B; S cancelled for the sharded rerun), 78284–78287 (search S), 78312–78314
(analysis), 78318, 78321 (interpretation).

Biotite, `redesign_20261111/subgenic/`: `hits/S_{0..3}.parquet` (4.4 GB), `hits/B.parquet`,
`hits/C.parquet`, `rep/aln.tsv`, `results/`, `tmp/comp.duckdb`, `tmp/frag.duckdb`, `db/uniqdb`,
`uniq.faa`, `gid2uid.parquet`, `uniq.parquet`, `lists/`.

Local, `benchmarks/redesign/subgenic/`: `predeclared.json`, `progress.log`, `results/` (JSON and
TSV, including `interp2.json`, `per_genome_rates.tsv`, `concentration_by_ord.tsv`), and the code
(`prep_unique.py`, `bench.sbatch`, `bench_summary.py`, `make_queries.py`, `make_segments.py`,
`search_full.sbatch`, `search_shard.sbatch`, `tsv2parquet.py`, `analyze.py`, `analyze.sbatch`,
`interp.py`, `interp.sbatch`, `interp2.py`, `interp2.sbatch`, `peek.py`).
