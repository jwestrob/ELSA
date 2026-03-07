# Figure Legends

## Figure 2: Operon detection benchmark and block quality comparison

**(a)** Operon recall for ELSA and MCScanX evaluated against 58 RegulonDB E. coli operons (10,182 pairwise instances across 20 genomes, 50% overlap threshold). Three metrics are shown: "any coverage" requires the operon to be covered in at least one genome of the pair; "independent" requires coverage in both genomes (by any blocks); "strict" requires a single block to cover both copies. ELSA achieves near-perfect recall across all metrics (99.3% any, 99.0% independent, 98.9% strict). MCScanX achieves 100.0% any coverage but lower independent (96.4%) and strict (80.3%) recall. Although MCScanX's strict recall appears respectable, 84.8% of those strict hits are "accidental span" — large blocks that happen to cover operons without true gene-by-gene correspondence (panel b). Panel (c) confirms that 59% of MCScanX blocks are internally fragmented.

**(b)** Distribution of region-level shared orthogroup rates for ELSA (n = 5,000 sampled blocks) and MCScanX (n = 27,372 cross-genome blocks) on the 30-genome Enterobacteriaceae dataset. For each block, all genes in the block span on both sides were assigned to OrthoFinder orthogroups; the shared rate is the fraction of genes belonging to orthogroups present on both sides (minimum of the two sides). ELSA blocks have a median shared rate of 0.90, indicating that blocks connect genuinely homologous gene neighborhoods. MCScanX blocks show a bimodal distribution with a median of 0.50, consistent with many blocks merging non-homologous regions.

**(c)** MCScanX block fragmentation analysis on the 30-genome Enterobacteriaceae dataset (n = 3,288 collinear blocks). Each MCScanX block was split at internal gaps exceeding 10 genes in either genome to count the number of independent sub-blocks within each reported block. 59% of blocks (1,953/3,288) contain two or more sub-blocks (mean across all blocks = 2.6, mean among fragmented blocks = 3.6, max = 18), indicating that MCScanX merges multiple distinct syntenic events into single blocks. Light bars: unfragmented blocks (1 sub-block); dark bars: fragmented blocks.

**(d)** Block size distributions for ELSA (n = 80,225; blue) and MCScanX (n = 27,372 cross-genome; yellow) on the 30-genome Enterobacteriaceae dataset (log scale). Block size is measured as anchor gene count for both methods. Both methods have a similar median block size, but ELSA identifies 2.9× more blocks. MCScanX has a substantially heavier right tail, reflecting the block-merging behavior shown in panel (c). Dashed vertical lines mark medians.

---

## Figure 3: Embedding similarity detects synteny beyond sequence homology

**(a)** Relationship between embedding cosine similarity and pairwise sequence identity for 1,500 cross-species gene pairs from the 30-genome Enterobacteriaceae dataset. Each hexagonal bin is colored by point density (viridis scale). The vertical dashed red line marks 30% sequence identity, below which BLAST sensitivity drops sharply ("twilight zone"). Gene pairs in the upper-left quadrant have high embedding similarity but low sequence identity — these represent conserved syntenic relationships that ELSA detects but sequence-based methods cannot.

**(b)** Cross-genus syntenic block counts for ELSA and MCScanX on the 30-genome Enterobacteriaceae dataset (21 E. coli, 5 Salmonella, 4 Klebsiella; NCBI-verified taxonomy). ELSA identifies 55,898 cross-genus blocks compared to 14,186 for MCScanX (3.9× more). Of ELSA's blocks, 8,069 have no detectable BLAST homology between anchor genes, representing cryptic synteny invisible to sequence alignment-based approaches. Blue: BLAST-visible ELSA blocks; purple: ELSA-only cryptic blocks; yellow: MCScanX blocks.

---

## Figure 4: Locus search retrieval performance

**(a)** Mean recall at k across 58 E. coli operons queried against the 30-genome Enterobacteriaceae index (139,198 genes, 480D raw embeddings, similarity threshold τ = 0.85). The solid line shows mean recall; the shaded region shows ±1 standard deviation. Recall climbs steeply from 5.9% at k = 1 to 55.8% at k = 10, then rapidly converges to 99.2% at k = 25 and 99.9% at k = 50 (57/58 operons at 100%). The query genome for each operon was selected as the genome with the most complete copy.

**(b)** Recall at k stratified by operon size: short (2–4 genes, n = 46), medium (5–7 genes, n = 8), and long (8–14 genes, n = 4). Search performance is consistent across operon sizes, with all categories converging to ~100% recall by k = 25.

**(c)** Cross-genus search hits for the top 20 operons (by total cross-genus hit count). Bars show the number of search results mapping to Salmonella (pink) and Klebsiella (green) genomes. Most operons retrieve hits from all 5 Salmonella genomes; Klebsiella representation varies by operon, reflecting genuine differences in gene content across genera.

---

## Supplementary Figure S1: Gap penalty configuration comparison

**(a)** Operon recall across three gap penalty configurations on the 20-genome E. coli dataset: hard gap with max_gap = 2 (default), hard gap with max_gap = 5, and concave gap penalty (minimap2-style, scale = 1.0, max_gap = 10). All three configurations achieve equivalent independent and any-coverage recall (~100%), with near-identical strict recall (98.5–98.8%).

**(b)** Block size distributions for each configuration (log-scale x-axis). The concave gap penalty produces fewer, larger blocks by tolerating variable-length gaps within syntenic regions, while the hard gap = 2 setting produces more, smaller blocks.

**(c)** Trade-off between total block count and strict recall. The concave gap penalty achieves equivalent recall with 59% fewer blocks than the default hard gap = 2 configuration, demonstrating that gap penalty choice affects block granularity without sacrificing detection sensitivity.

---

## Supplementary Figure S2: Similarity threshold calibration

**(a)** Effect of cosine similarity threshold (τ) on operon recall and block count for the 20-genome E. coli dataset. Left axis: strict, independent, and any-coverage recall versus τ (0.70–0.96). Right axis: total syntenic blocks versus τ. Recall is stable across a wide range of thresholds; the default τ = 0.85 (dashed line) balances sensitivity with specificity.

**(b)** Block and anchor counts versus τ for the 15-genome Borg dataset. No operon ground truth is available for Borg genomes; the dashed line marks τ = 0.70, which is required for these highly divergent extrachromosomal elements. Block counts increase steeply below τ = 0.80, reflecting the lower baseline similarity of Borg protein sequences.

---

## Supplementary Figure S3: Block size distributions across datasets

Block size distributions (log-scale x-axis) for four datasets analyzed with ELSA. **(a)** S. pneumoniae (6 genomes, 2,123 blocks, median 4 genes). **(b)** E. coli (20 genomes, 19,279 blocks, median 18 genes). **(c)** Enterobacteriaceae (30 genomes, 9,929 blocks, median 6 genes). **(d)** Borg genomes (15 genomes, 3,965 blocks, median 2 genes). Dashed lines indicate medians. Block sizes range from 2 genes (the configured minimum) to thousands of genes for closely related genome pairs. The Enterobacteriaceae dataset shows smaller median block size than E. coli alone because cross-genus blocks tend to be shorter, reflecting greater evolutionary divergence.

---

## Supplementary Figure S4: Negative control — cross-phylum embedding similarity

Distribution of pairwise embedding cosine similarities between 2,000 randomly sampled E. coli genes and 2,000 B. subtilis genes (4,000,000 pairs). The distribution is centered near zero (mean = 0.001), confirming that unrelated genomes from different phyla (Proteobacteria vs Firmicutes) produce near-random embedding similarities. Only 0.20% of pairs exceed the default threshold of τ = 0.80 (dashed red line), and these likely reflect deeply conserved housekeeping genes. This validates that ELSA's syntenic block detection is not driven by spurious embedding similarity between unrelated organisms.

---

## Figure 5: Model concordance — ELSA produces consistent results across PLMs

**(a)** Operon recall comparison between ESM2-t12 (480D, Meta) and ProtT5-XL-U50 (1024D, Rostlab) on the 30-genome Enterobacteriaceae dataset (10,182 pairwise instances across 58 E. coli operons, 20 genomes, 50% overlap threshold). Both models achieve near-identical recall: ESM2 98.9%/99.0%/99.3% vs ProtT5 98.0%/98.1%/98.9% (strict/independent/any coverage). The <1% difference demonstrates that ELSA's synteny detection is driven by the chaining algorithm, not the specific protein language model.

**(b)** Block count comparison for the same dataset. ESM2 identifies 80,225 total blocks (55,898 cross-genus) vs ProtT5's 76,947 (53,295 cross-genus) — within 5% across all categories. E. coli-only blocks are similarly concordant (20,117 vs 19,212).

**(c)** Block size distributions for ESM2 (blue, n = 80,225) and ProtT5 (magenta, n = 76,947) on log scale. Both distributions are nearly identical in shape with comparable medians (7 vs 8 anchor genes). Dashed vertical lines mark medians. The concordance across the full distribution — from 2-gene micro-blocks to 1,000+ gene mega-blocks — confirms that both PLMs capture equivalent syntenic signal.
