# ws_priorart: closest prior work and a novelty assessment for recurrence-first ELSA

Written 2026-11-11 by the priorart agent; saved by the coordinator from the agent's final
message because the harness blocked the agent's own REPORT.md write. Literature research
only. Search trail and dead ends: `NOTEBOOK.md` in this directory. Works marked verified
were opened at the link given. Works the agent could not open are in section 7 and marked
UNVERIFIED.

Design elements: **E1** a family token per protein, from quantized pLM embeddings, at
several resolutions; **E2** a similarity metric learned from synteny (contrastive,
same-genome negatives); **E3** token pairs within a few genes counted across thousands of
genomes, weighted by lineage; **E4** significance against a within-genome gene-order
shuffle, no pairwise step, linear cost; **E5** loci read from a per-genome support
landscape (level sets, modules sharing a phylogenetic profile, low-support islands between
conserved flanks); **E6** pairwise locus alignment on demand.

## 1. Verdict

The statistical core is prior art. Rogozin 2002, Junier & Rivoire 2016 and CSBFinder-S 2020
all count co-localized family pairs across many genomes, discount related genomes, test
against random gene order and read segments, using COG tokens. Hotspot and module detection
are published: Oliveira 2017, panRGP, Hochhauser 2023 and panModule. k-means tokens from pLM
embeddings are published in Bacformer (50,000 ESM-2 centroids).

New: (1) embedding tokens combined with a recurrence test and gene-order null at >= 10^3
genomes; (2) a synteny-supervised protein metric (E2); (3) a token-resolution sweep judged
by the null. Point 3 is a modest step.

Defensible framing: an embedding-tokenized, multi-resolution, lineage-weighted version of
the Rogozin / Junier-Rivoire / CSBFinder recurrence framework. It extends that framework to
proteins without COG or Pfam assignments, adds a synteny-trained metric, and is compared
head-to-head with Spacedust on Spacedust's benchmark. The measurable gains are:
- **Coverage:** the COG-based predecessors analyse only COG-assigned genes.
- **Scaling:** Spacedust needed 72 h on 256 cores for 1,308 genomes and is quadratic.

Confidence: high for E3-E6 and single-resolution E1. Moderate that E1 combined with E3/E4
is unpublished. Moderate-to-low that E2 is unpublished.

## 2. Closest prior works

**2.1 Junier & Rivoire 2016 (closest overall).**
- **Citation:** *PLoS ONE* 11:e0155740, doi:10.1371/journal.pone.0155740.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4873041. Companion: Junier, Frémont & Rivoire,
  arXiv:1801.07585, https://ar5iv.labs.arxiv.org/html/1801.07585.
- **Unit:** COG classes (4,764); genes without a COG are left out.
- **Neighbourhood:** distance between gene midpoints in bp, for every COG pair.
- **Statistics:** each genome is weighted 1 / (number of genomes within delta = 0.25),
  giving an effective count of 500 from 1,445 genomes. The null is uniform random gene
  positions, giving a binomial test with the minimum P over distance thresholds. FDR 0.005
  (P* about 4e-4) leaves about 36,000 significant pairs.
- **Outputs:** "synteny segments", maximal runs in which every gene pair is significant:
  740 in *E. coli*, 661 in *B. subtilis*. Validated against co-expression data.
- **Companion paper (2018):** 797 chromosomes, delta = 0.1.
- **Overlap:** E3, lineage weighting, analytic E4 with no pairwise genome comparison, E5
  segments.
- **Difference:** COG tokens rather than embedding tokens; distance in bp rather than a gene
  window; analytic null rather than an empirical shuffle; no learned metric, resolution
  sweep or hotspots.

**2.2 Rogozin et al. 2002.**
- **Citation:** *NAR* 30:2212, doi:10.1093/nar/30.10.2212.
  https://academic.oup.com/nar/article/30/10/2212/1102016
- **Unit and neighbourhood:** COGs; same-strand pairs separated by 0-2 genes, present in
  >= 3 of 31 genomes; pairs in closely related species counted once.
- **Null:** "random permutations of the gene order in each genome", 100 repeats. About 12
  pairs arise by chance against 1,505 observed.
- **Outputs:** maximal trails merged into 188 clusters; neighbourhoods covering 2,387 COGs.
- **Overlap:** the same window as the proposal and exactly the proposed null.
- **Difference:** 31 genomes, COG tokens, no calibrated per-pair P-values.

**2.3 CSBFinder-S (2020) / CSBFinder (2019).**
- **Citation:** *Bioinformatics* 36:i21, doi:10.1093/bioinformatics/btaa503 (PDF pages 1-6
  read). https://rcastoragev2.blob.core.windows.net/f393f428a0515380f646aba6a2f0e807/PMC7355258.pdf.
  CSBFinder 2019: *Bioinformatics* 35:1634. Tool: https://github.com/dinasv/CSBFinder
- **Unit and neighbourhood:** genomes as strings of COG/NOG IDs with strand; patterns found
  in >= q genomes with <= k inserted genes.
- **Statistics:** random-gene-order score, corrected for redundancy with
  m' = sum 1/|{S_j : D_ij >= delta}|, where D is the Jaccard index of gene-family content.
- **Scale and cost:** 1,485 genomes; O(nm + OCC). 133,823 blocks in 17 min (multi-threaded,
  q = 30, k = 10).
- **Useful finding:** 76.5% of blocks seen in >= 30 genomes have a single gene order.
- **Overlap:** E3, lineage weighting, E4, near-linear cost.
- **Difference:** annotation tokens; exact colinear patterns only.

**2.4 Spacedust (Nat Methods 2025): the direct baseline.**
- **Citation:** *Nat Methods* 22:2065, doi:10.1038/s41592-025-02816-x.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12510874/
- **Method:** Foldseek (3Di + AA) with MMseqs2 fallback, best hit per protein at P <= 1e-7.
  Pairwise all-versus-all genome comparison, greedy agglomerative clustering, clustering
  P-value x ordering P-value. Genomes deduplicated at 70% AAI only.
- **Scale:** 1,308 genomes, 4.2 M genes, 72 h on 256 cores. The authors state that runtime
  "scales quadratically".
- **Results:** 72,483 clusters covering 58% of genes. Recovers 95% of PADLOC v1.1.0 defence
  clusters. 207 BGCs: F1 0.61 (GECCO 0.43, ClusterFinder 0.44, DeepBGC 0.39). KEGG-module
  congruence about 50% for clusters seen in 2 genomes and over 80% for clusters seen in
  >= 50. The authors name hitchhiking as a limit.
- **Overlap:** the same collection and gold standard as BRIEF section 2.
- **Note:** BRIEF scores native Spacedust at F1 0.676 under ELSA's scoring; report both
  numbers.

**2.5 Bacformer (bioRxiv 2025.07.20.665723): E1 exists here.**
- **Link:** https://www.biorxiv.org/content/10.1101/2025.07.20.665723v1.full
- **Tokens:** k-means on ESM-2 embeddings with 50,000 centroids forms the protein-family
  vocabulary. NMI 0.88 against MMseqs2, 0.74 against eggNOG.
- **Model:** masked and next-family transformer trained on 1,314,344 MAGs (2.7 B proteins).
- **Outputs:** operons (ODB), STRING interactions, essentiality (DEG).
- **Difference:** no recurrence statistic, null, lineage weighting or explicit loci.

**2.6 Gecko 3 (NAR 2016).**
- **Citation:** *NAR* 44:9600, doi:10.1093/nar/gkw843.
  https://academic.oup.com/nar/article/44/20/9600/2606339
- **Method:** reference gene clusters with approximate occurrences elsewhere; order
  ignored. "Random gene order as the background model", Benjamini-Hochberg FDR.
- **Scale:** 678 genomes in under 40 min on a laptop.

**2.7 panModule and PPanGGOLiN/panRGP.**
- **Links:** https://www.biorxiv.org/content/10.1101/2021.12.06.471380.full and
  https://ppanggolin.readthedocs.io/en/dev/
- **Modules:** edges join variable families fewer than 4 genes apart whose co-occurrence
  Jaccard is >= 0.86; modules are connected components.
- **Spots:** groups of regions defined by conserved persistent flanking genes.
- **Benchmark:** EcoliRef, 165 curated modules in 12 genomes: MCC 0.63, precision 0.96,
  recall 0.51.
- **Statistics:** thresholds only, no significance test. Overlap: both forms of E5.

**2.8 Oliveira et al. 2017.**
- **Citation:** *Nat Commun* 8:841. https://pmc.ncbi.nlm.nih.gov/articles/PMC5635113/
- **Method:** spots between the same two core families; a hotspot exceeds the 95th
  percentile of a per-species simulation.
- **Results:** 932 genomes in 80 species; 1,841 hotspots holding about 60% of transferred
  genes.

**2.9 Hochhauser, Millman & Sorek 2023.**
- **Citation:** *PLoS Genet* 19:e1010694.
  https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1010694
- **Method:** 1,351 *E. coli* genomes, DefenseFinder 1.2.0 and PADLOC 1.1.0 calls, 41
  hotspots mapped by flanking core genes.
- **Result:** 90.6% of defence systems in finished genomes lie in those 41 hotspots.

**2.10 Rodríguez del Río et al. 2024 (Nature 626:377).**
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10849945
- **Method:** MMseqs2 families from about 150k environmental genomes plus about 20k
  references; +/-3-gene window; conservation = fraction of family members with a neighbour
  in a KEGG pathway. Thresholds benchmarked on 108,823 KEGG-annotated clusters.
- **Statistics:** none; no significance test.

**2.11 Miller, Stern & Burstein 2022.**
- **Citation:** *Nat Commun* 13:5731. https://pmc.ncbi.nlm.nih.gov/articles/PMC9523054/
- **Method:** KO and MMseqs2 family tokens; word2vec with window 5 over 360 M genes.
- **Results:** 98.6% of known defence words recovered; 40,247 families predicted
  defence-associated. No null.

**2.12 STRING neighbourhood channel.**
- **Links:** Snel et al. 2000, *NAR* 28:3442,
  https://academic.oup.com/nar/article/28/18/3442/2903795; scoring help page
  https://string-db.org/help/scores/
- **Method:** runs of same-strand genes with gaps under 300 bp; bidirectional-best-hit
  orthology; counts over phylogenetically distant genomes reported separately.
- **Shuffle chance:** 0.02 for 2 species, < 0.002 for 3, < 0.0005 for 4 or more.

**2.13 gLM, gLM2 and Gaia.**
- **gLM:** bioRxiv 2023.04.07.536042 v3,
  https://www.biorxiv.org/content/10.1101/2023.04.07.536042v3.full. ESM2 embeddings over
  7.3 M subcontigs; an attention head tracks operons (rho = 0.44).
- **gLM2/OMG:** https://www.biorxiv.org/content/10.1101/2024.08.14.607850v1.full
- **Gaia:** https://www.biorxiv.org/content/10.1101/2024.11.19.624387v1.full. InfoNCE
  training on 2.3 M AlphaFold DB structural clusters, so its supervision is structure.
- Jacob is a coauthor of gLM2 and Gaia.

**2.14 MCScanX.**
- **Citation:** *NAR* 40:e49. https://academic.oup.com/nar/article/40/7/e49/1202057
- **Method:** top-5 BLASTP hits as anchors, dynamic-programming chaining (match 50, gap -1,
  < 25 intervening genes), block E-value. This is the lineage ELSA v2 inherits.

**2.15 Raghupathy & Durand 2009.**
- **Citation:** *MBE* 26:957 (page 1 read).
  https://rcastoragev2.blob.core.windows.net/2fbff68c3cfa5ce62ce615a6bcd2df9e/PMC2668827.pdf
- **Point:** random gene order is the standard null, and ignoring family size overestimates
  significance. Coarse tokens behave as large families.

## 3. Other families checked (verified unless marked)

- **Collinearity:** ntSynt (*BMC Biol* 23:367; GitHub bcgsc/ntSynt), minimizer macrosynteny.
- **Pangenome graphs:** PanGraph (PMC10327495) works only on closely related genomes.
  Panaroo (bioRxiv 2020.01.28.922989) uses context to split paralogs and to merge families
  that share a neighbour.
- **Multi-resolution families:** PIRATE (bioRxiv 598391) clusters across identity
  thresholds. GeCoViz (*NAR* 50:W352) uses eggNOG groups at user-chosen taxonomic levels
  across 12,221 genomes, with no statistics.
- **Context tools:** EFI-GNT (+/-10 ORFs, co-occurrence >= 20%), NetSyn (community
  detection, no null) and FlaGs are all query-centred.
- **Targeted cluster search:** zol/fai, CORASON and MultiGeneBlast (abstract only). cblaster
  details are UNVERIFIED.
- **Presence/absence association:** Coinfinder uses presence/absence with lineage
  correction and no position.
- **Embedding search:** structure-supervised: PLMSearch, TM-Vec (journal version
  UNVERIFIED), ProtTucker (CATH). Untrained: pLM-BLAST. DHR: positive pairs UNVERIFIED.
  Dickson & Mofrad (GO labels). Littmann 2021: DBSCAN on embeddings to split FunFams.
- **Embeddings plus synteny or context:** SAFPred trains nothing on synteny.
  DefensePredictor (abstract only). FGBERT's contrastive positives are mutated copies of the
  same gene.
- **Discrete tokens:** Foldseek 3Di (VQ-VAE trained for conservation), ESM3 (codebook
  4,096), ProSST (k-means details UNVERIFIED), TEA alphabet (2025).
- **Synteny-aware orthology:** Proteinortho PoFF.
- **Scale reference:** Aplakidou et al. 2025, 6.59 M families, no neighbourhood null.

## 4. Element status

| Element | Status | Source |
|---|---|---|
| E1 pLM tokens | Exists (one resolution) | Bacformer |
| E1 multi-resolution | Exists for sequence clusters and OGs | PIRATE, GeCoViz |
| E2 synteny-trained metric | No prior found | - |
| E3 pair counting | Exists | Rogozin, Junier & Rivoire, STRING, Rodríguez del Río |
| E3 lineage weighting | Exists | Junier & Rivoire, CSBFinder-S, Rogozin, STRING |
| E4 order null, linear cost | Exists | Rogozin, Gecko 3, CSBFinder(-S), Junier & Rivoire |
| E5 segments, modules, hotspots | Exists | Junier & Rivoire; panModule; Oliveira, panRGP, Hochhauser |
| E6 | Exists | MCScanX, Spacedust, fai |
| E1 combined with E3/E4 at >= 10^3 genomes | No prior found | - |

## 5. What reviewers will ask for

**Comparisons**
1. Spacedust head-to-head: 207 BGCs, PADLOC recovery, KEGG congruence versus support, and
   core-hours.
2. A token ablation inside one engine: COG/eggNOG, MMseqs2, Foldseek clusters and embedding
   tokens. This is the only clean separation from Junier & Rivoire and CSBFinder-S.
3. CSBFinder-S and Gecko 3 run on the same tokens.
4. MCScanX and the current ELSA chainer on identical anchors.
5. panModule and PPanGGOLiN on EcoliRef.
6. The fraction of genes that get a usable token versus the fraction with a COG, per phylum.

**Controls**
- Calibration separately for frequent and rare tokens at each resolution (Raghupathy &
  Durand).
- Sensitivity to the lineage-weight parameter delta, plus a count of independent clades per
  module. Descent passes a gene-order shuffle.
- A stated rule for tandem paralogs and self-pairs.
- Hitchhiking: functional precision as a function of support.
- E2 circularity: hold out genomes and phyla, label held-out pairs independently (MMseqs2,
  Foldseek or TM-align), and compare against raw ESM cosine and against TM-Vec, PLMSearch,
  ProtTucker and Gaia.
- Fit the token model on held-out genomes.

**Datasets**
- **Spacedust collection:** http://wwwuser.gwdguser.de/~compbiol/spacedust/ holds
  `KEGG_70.tar.gz` and the workflow script. The BGC labels are from the Cimermancic 2014 and
  Hannigan 2019 supplements.
- **Defence-system callers:** PADLOC (now v2.0.0) and DefenseFinder (GitHub).
- **Hotspots:** Hochhauser's 41 hotspots and Oliveira's 80-species tables.
- **Modules:** panModule's EcoliRef.
- **Operons:** ODB v4 (operondb.jp, title only verified) and RegulonDB.
- **BGCs:** MIBiG (title only verified).
- **Families and segments:** Rodríguez del Río's 108,823 KEGG-annotated clusters; Junier &
  Rivoire's segment files.
- **Embeddings:** DGEB (Jacob is first author; task list UNVERIFIED).

## 6. Naming and framing risks

**Terms already taken:** "synteny segments" (Junier & Rivoire); "colinear syntenic block"
(CSBFinder); "conserved gene cluster" (Spacedust); "reference gene cluster" (Gecko 3);
"connected gene neighborhoods" and "über-operon" (Rogozin); "spot", "hotspot", "coldspot"
(Oliveira); "spots of insertion" and "RGP" (panRGP); "module" (panModule); "GNN" (EFI-GNT);
"protein family vocabulary" (Bacformer); "3Di", "structure tokens", "TEA"; "CRISPRicity"
(UNVERIFIED).

**Claims that would be overreach:**
- First recurrence-based, alignment-free or reference-free method.
- First lineage weighting.
- First linear-time method, or first without pairwise comparison.
- A new shuffle null.
- First hotspot detection from conserved flanks.
- First pLM family tokens.
- "Significant" as evidence of selection. A shuffle measures recurrence beyond gene-order
  chance.
- "Distant homology" without structural confirmation.
- "A new architecture" as the headline.

## 7. Not verified

- Shmakov 2018/2019 (CRISPRicity and the neighbourhood protocol): PMC, PNAS and OmicsDI
  blocked.
- Doron 2018 (scoring of 14,083 Pfams in > 45k genomes): known from search snippets only;
  PMC blocked, PDF connection reset.
- Makarova 2011: PMC and ASM blocked.
- Pellegrini 1999: blocked. The PDF that opened is Marcotte 1999 instead.
- Hoberman, Sankoff & Durand 2005: figshare 403.
- GENESPACE: page failed to load; omitted.
- Training pairs for DHR and TEA; k-means details for ProSST.
- Journal versions of gLM, TM-Vec, Panaroo and Foldseek: publisher pages required login;
  preprints opened instead.
- Spacedust supplement formulas.
- cblaster methods; MIBiG and ODB sizes; DGEB task list.

The most likely thing this review missed is a 2026 preprint in the genomic-language-model
space.
