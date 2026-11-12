# ws_priorart notebook (append-only)

Workstream: closest prior work and an honest novelty assessment for the
recurrence-first ELSA redesign (BRIEF.md section 3). Literature and tool research
only; no computation.

## 2026-11-11 04:08 MST — scope and predeclared criteria

Question: how much of the proposed design already exists?
Design elements, labelled for tracking:
- E1 discrete family tokens from quantized pLM embeddings (k-means or graph
  communities), several resolutions
- E2 similarity metric learned from synteny (contrastive projection on anchor
  pairs from conserved blocks, same-genome negatives)
- E3 counting token pairs within a few genes across thousands of genomes,
  support weighted by independent lineages
- E4 significance from recurrence against a within-genome gene-order shuffle;
  no pairwise significance step; linear cost
- E5 loci/modules read from a per-genome support landscape: level sets,
  segments with a shared phylogenetic profile, low-support islands between
  conserved flanks (hotspots)
- E6 pairwise locus alignment on demand

Predeclared criterion for the verdict:
- An element counts as "already exists" when a primary paper or official tool
  documentation that I opened describes it (same operation, any unit).
- The design counts as "novel as a combination" only if no single opened work
  combines E1 (embedding-derived tokens) with E3+E4 (window co-occurrence
  counting with a gene-order null) at multi-thousand-genome scale.
- "Novel" claims about any single element require that targeted searches for
  it return nothing; searches and dead ends are logged below.

Tools: WebSearch (standard) and WebFetch. Every cited work is opened; anything
not opened is marked unverified in REPORT.md.

## 2026-11-11 04:20 MST — first search round (verified sources)

Searches run (WebSearch, standard): "Spacedust de novo discovery...", "Rodríguez
del Río 2024 Nature ... genomic context", "CSBFinder colinear syntenic blocks",
"contrastive learning protein embeddings genomic context synteny supervision
metric", "Junier Rivoire 2016 ...", "Shmakov Koonin ... Nature Protocols",
"panModule ...", "Oliveira Touchon Cury Rocha 2017 ...".

Opened and read:
- Spacedust, Nat Methods 2025 (pmc.ncbi.nlm.nih.gov/articles/PMC12510874). Same
  1,308-genome collection and 207-BGC benchmark as BRIEF section 2. Pairwise
  all-vs-all; clustering P-value x ordering P-value; genomes deduplicated at 70% AAI
  only; authors state runtime scales quadratically.
- Rodríguez del Río 2024 Nature (PMC10849945): MMseqs2 families, +/-3-gene window,
  conservation = fraction of family members with a pathway neighbour; thresholds
  benchmarked on 108,823 KEGG-annotated clusters; no significance test.
- CSBFinder-S, Bioinformatics 2020 (PDF from PMC7355258, pages 1-6 read): COG
  string tokens, exact discovery of colinear blocks with quorum q and k insertions,
  O(nm + OCC); ranking score assumes randomized gene order, corrected for genome
  redundancy with Jaccard of gene-family content (delta threshold), following
  Junier & Rivoire 2016. 1,485 genomes.
- CSBFinder GitHub README (github.com/dinasv/CSBFinder).
- Junier & Rivoire 2016 PLoS ONE (PMC4873041): VERY CLOSE to E3+E4+E5. COG
  tokens, 1,445 genomes, every genome weighted inversely to the number of genomes
  within phylogenetic distance 0.25 (effective M' = 500), null = uniform random
  gene positions, binomial P per COG pair, FDR cut, "synteny segments" = maximal
  runs of genes in which every pair is significant.
- panModule bioRxiv 2021: PPanGGOLiN families, edges for families < t=4 genes
  apart, Jaccard co-occurrence >= 0.86, connected components; EcoliRef benchmark
  (12 genomes, 165 curated modules, Touchon 2009). No significance test.
- Oliveira et al. 2017 Nat Commun (PMC5635113): spots = intervals between the same
  two core families; hotspot = more HT genes than the 95th percentile of a
  per-species simulation; 932 genomes, 80 species; 1,841 hotspots.

Dead end: PMC6938587 (Shmakov 2019 Nat Protoc) returned a reCAPTCHA page; retry
via another host. PMC7355258 HTML also returned reCAPTCHA; the PDF mirror worked.

## 2026-11-11 04:45 MST — second round: genomic LMs, context tools, collinearity

Searches: "protein language model embeddings clustered into tokens gene
neighborhood co-occurrence" (extended mode; returned nothing that combines
pLM-derived tokens with neighbourhood co-occurrence statistics), Bacformer,
Miller 2022, gLM, gLM2/OMG, Gaia, Rogozin 2002, STRING scores, PanGraph,
MCScanX, EFI-GNT, NetSyn, zol/fai, FlaGs, GeCoViz, cblaster, CORASON,
MultiGeneBlast.

Opened and read:
- Bacformer bioRxiv 2025 (10.1101/2025.07.20.665723): ** k-means on ESM-2
  embeddings, 50,000 centroids = protein-family vocabulary ** (NMI 0.88 vs MMseqs2,
  0.74 vs eggNOG). Tokens are prediction targets for a masked-family transformer
  trained on 1,314,344 MAGs / 2.7 B proteins. So element E1 (k-means tokens from pLM
  embeddings) exists, at one resolution, used for a context model.
- Miller, Stern, Burstein 2022 Nat Commun (PMC9523054): KO/MMseqs2/CD-HIT family
  tokens, word2vec window 5 over 360 M genes; no null model.
- gLM, Hwang et al. (bioRxiv 10.1101/2023.04.07.536042 v3; journal Nat Commun 15,
  2880, 2024 — nature.com redirected to an auth page, journal version not opened):
  ESM2 per-gene embeddings, 15-30-gene windows, 7.3 M MGnify subcontigs; operon
  signal in attention (RegulonDB 817 operons).
- gLM2/OMG bioRxiv 2024 (10.1101/2024.08.14.607850): amino-acid + BPE nucleotide
  tokens; Jacob West-Roberts is a coauthor.
- Rogozin et al. 2002 NAR 30:2212 (academic.oup.com): ** COG pairs separated by
  0-2 genes, same strand, in >= 3 of 31 genomes; closely related species counted
  once; null = random permutation of gene order in each genome (100 Monte Carlo
  repeats: 12 chance pairs vs 1,505 observed); neighbourhoods = merged maximal
  trails in the pair graph. ** This is E3+E4 (window pair counting, lineage
  collapse, within-genome order shuffle) on 31 genomes in 2002.
- STRING help page (string-db.org/help/scores): neighbourhood and co-occurrence
  channels; co-occurrence "accounts for the relatedness of species"; no formula.
- PanGraph Microb Genom 2023 (PMC10327495): nucleotide pancontigs, closely related
  genomes only (fails at ~26% core divergence).
- PPanGGOLiN docs: panRGP spots = RGPs grouped by conserved persistent flanks.
- MCScanX NAR 2012 (academic.oup.com): top-5 BLASTP hits, DP chaining, match 50,
  gap -1, < 25 intervening genes, block E-value.
- EFI-GNT tool page: +/-10 ORFs, Pfam neighbours, co-occurrence >= 20% default.
- NetSyn bioRxiv 2023: MMseqs2 families, window 11, synteny score, community
  detection on target network; no null.
- zol/fai GitHub; FlaGs Bioinformatics 2021 (PDF pages 1-2 read); GeCoViz NAR
  2022 (OUP): eggNOG OGs, 12,221 proGenomes genomes, % of genomes, no statistics;
  CORASON (bioRxiv 445270); MultiGeneBlast (eprints.gla.ac.uk/80717, abstract only).

Dead ends: PMC pages for Shmakov 2019, gLM, MCScanX intermittently return a
browser-check page; nature.com redirects to an auth page. Used OUP and bioRxiv
copies instead.

## 2026-11-11 05:10 MST — third round: embedding search, tokens, metric learning, defence

Opened and read:
- PLMSearch Nat Commun 2024 (PMC10981738): ESM-1b embeddings + SS-predictor
  trained on TM-align TM-scores (SCOPe40-train, CATHS40). Supervision = structure.
- TM-Vec bioRxiv 2022 (10.1101/2022.07.25.501437v2; journal Nat Biotechnol 42:975
  per search result, not opened): twin network on ProtT5, regresses TM-score.
- pLM-BLAST Bioinformatics 39(10) btad579 (OUP): untrained; cosine of per-residue
  ProtT5 embeddings as substitution matrix; ECOD benchmark.
- DHR: Gerstein lab page confirms citation Nat Biotechnol 43:983-995 (2025);
  GitHub README lacks training details. Positive-pair definition NOT verified.
- Foldseek bioRxiv v2: 3Di = 20-state VQ-VAE trained so states are conserved
  across structurally aligned residue pairs (SCOPe, TM >= 0.6).
- ProSST NeurIPS 2024 abstract: pre-trained clustering model quantizes local
  structure vectors into tokens (k-means and vocabulary 2048 from search snippet
  only; abstract page did not state them).
- ESM3 bioRxiv 10.1101/2024.07.01.600583: VQ-VAE structure tokens, codebook 4096.
- TEA alphabet bioRxiv 2025.11.27.690975 (Pantolini ... Durairaj): contrastive
  learning maps pLM embeddings to a 20-letter alphabet for MMseqs2 search.
  Positive-pair definition not on the abstract page.
- SAP bioRxiv 2023 (Urhan ... Abeel): ESM-1b embeddings + operon database built
  from GTDB r202 by proximity; NO model trained on synteny; nearest-neighbour
  GO transfer.
- Proteinortho PoFF PLoS ONE 2014: synteny (FFAdj-MCS) as a filter to resolve
  co-orthologs.
- Panaroo bioRxiv 2020 abstract page; graph-neighbourhood paralog splitting from
  search snippet of Genome Biology 21:180.
- Hochhauser, Millman, Sorek 2023 PLoS Genet (journals.plos.org): 1,351 E. coli
  genomes, DefenseFinder 1.2.0 + PADLOC 1.1.0, 41 hotspots mapped by flanking
  core genes (>80% of genomes); 90.6% of defence systems in finished genomes lie
  in the 41 hotspots.

Targeted searches for E2 (synteny/genomic context as supervision for a protein
similarity metric), two queries in extended mode:
"gene neighborhood contrastive learning protein language model fine-tuning
co-localized genes positive pairs remote homology" and "synteny conserved gene
order used to train protein embedding metric learning siamese positional
orthologs training pairs". Result: no work found that trains a protein
similarity metric on synteny-derived positive pairs. Closest: Bacformer adds a
contrastive loss between genome-context embeddings and the protein's own
sequence embedding (self-retrieval, not homolog pairs); Gaia fine-tunes gLM2
contrastively on AlphaFoldDB structural clusters (structure supervision). This
is an absence-of-evidence finding; two searches are not exhaustive.

Dead ends: europepmc.org article pages 403; Europe PMC REST 500/502; PMC
Makarova 2011, Doron 2018 browser-check; Doron 2018 Weizmann PDF ECONNRESET twice;
ASM journals 403.

## 2026-11-11 05:40 MST — fourth round: statistics lineage, orthology, recent atlases

Opened and read:
- Raghupathy & Durand 2009 MBE 26:957 (PDF page 1, PMC2668827 mirror): "Random
  gene order is the most widely used null hypothesis for gene cluster
  statistics"; ignoring gene-family size overestimates significance.
- Gecko 3, Winter et al. 2016 NAR 44:9600 (OUP): STRING COG/NOG families, reference
  gene clusters, "random gene order as the background model", per-genome
  probabilities combined with quorum, BH-FDR; 678 genomes, < 40 min on a laptop.
- Coinfinder bioRxiv 859371: presence/absence only (no position), D statistic for
  lineage dependence, binomial exact test, 534 genomes.
- STRING 2000, Snel et al. NAR 28:3442 (OUP): runs (same strand, < 300 bp gaps),
  BBH orthology, separate counts over phylogenetically distant genomes, shuffle
  chance probability 0.02 / < 0.002 / < 0.0005 for 2 / 3 / 4+ species.
- Marcotte et al. 1999 Science 285:751 (Rosetta Stone; PDF from Pellegrini lab;
  the 751.pdf link is this paper, not the PNAS phylogenetic-profile paper).
  Pellegrini et al. 1999 PNAS phylogenetic profiles: PMC and pnas.org both
  blocked -> UNVERIFIED (citation from search result only).
- SAFPred, Urhan et al. 2024 Bioinformatics 40(6) btae328: confirms no model
  trained on synteny; nearest-neighbour over 406,293 GTDB-derived regions.
- Dickson & Mofrad 2024 Bioinformatics btae445: fine-tunes pLM on GO labels (BCE).
- ProtTucker, Heinzinger et al. 2022 NAR Genom Bioinform lqac043: triplet loss on
  ProtT5, positives from CATH hierarchy.
- Gaia bioRxiv 2024.11.19.624387 (Jha, Kravitz, West-Roberts, ... Hwang): gLM2
  projection trained with InfoNCE on 2.3 M AFDB structural clusters; 85 M
  centroids from 131,744 genomes; context benchmark = >7/10 neighbours homologous.
  Jacob West-Roberts is a coauthor.
- FGBERT arXiv 2402.16901v2: ESM-2 vectors per gene; contrastive positives are
  mutated copies of the same gene, negatives from other EC clusters.
- "Quadrupling the protein family space" bioRxiv 10.64898/2025.12.21.695754:
  MMseqs2 Linclust families (6.59 M families >= 25 members); neighbourhood
  analysis = top-5 Pfam per scaffold; no null.
- PIRATE (search result only: GigaScience 8 giz119; multiple identity thresholds)
  -> not opened, mark UNVERIFIED if cited.
- ntSynt GitHub (BMC Biology 23:367, 2025): minimizer-graph macrosynteny.
- PanRGP: only via PPanGGOLiN docs (Bazin 2020 Bioinformatics btaa792).
- Spacedust data dir http://wwwuser.gwdguser.de/~compbiol/spacedust/ holds
  KEGG_70.tar.gz and run_spacedust_main_workflow.sh.
- ODB v4 at operondb.jp (title only verified).

Extended-mode search "conserved gene neighbourhoods at GTDB scale co-localization
gene families null model shuffled gene order significance 2024 2025" and
"atlas of conserved gene neighborhoods across metagenomes ..." found no 2024-2026
work that runs a gene-order-shuffle recurrence test at GTDB scale; Spacedust
remains the newest large-scale de novo method, and it is pairwise.

## 2026-11-11 06:05 MST — fifth round: last checks before writing

Opened and read:
- Junier, Fremont, Rivoire 2018 (arXiv 1801.07585 via ar5iv): 797 STRING
  chromosomes, COG pairs, strain weights = 1 / #strains within delta = 0.1
  (effective ~553), synteny frequency f_ij within 100 kb; "synteny segments";
  no formal null for synteny.
- Littmann et al. 2021 Bioinformatics 37:3449 (PDF page 1): DBSCAN on PB-Tucker
  embeddings sub-clusters CATH FunFams (embedding-derived families).
- DefensePredictor bioRxiv 2025.01.08.631726 (abstract only): pLM-embedding
  classifier, 69 E. coli strains, 45 new systems validated.
- PIRATE bioRxiv 598391 (abstract): clustering over a range of identity
  thresholds; linear scaling.
- Panaroo bioRxiv full text: nodes = gene clusters, edges = contig adjacency;
  context used to split paralogs and to collapse families sharing a neighbour at
  70% identity.
- DGEB bioRxiv 2024.07.10.602933: Jacob West-Roberts first author; task list not
  on the abstract page.
- DefenseFinder and PADLOC GitHub READMEs (citations; PADLOC now v2.0.0).
- MIBiG home page: title only.

Dead ends: Shmakov 2018/2019 (PMC browser check x3, PNAS 403, OmicsDI truncated);
Doron 2018 (PMC x2, Weizmann PDF ECONNRESET x2); Makarova 2011 (PMC, ASM 403);
GENESPACE eLife (page failed to load); Springer/BMC pages redirect to auth.
These are marked UNVERIFIED in REPORT.md.

Verdict reached before writing (criterion from 04:08 entry): E3, E4 (random /
shuffled gene-order null), lineage weighting, segments, hotspots and
co-occurrence modules all exist in opened primary sources; E1 exists (Bacformer
k-means on ESM-2). No opened work combines E1 with E3+E4 at >= 10^3 genomes.
No opened work trains a protein similarity metric on synteny-derived pairs (E2).
The design therefore counts as a new combination with one candidate new
component (E2), and the statistical framework counts as prior art.

## 2026-11-11 06:30 MST — REPORT.md write blocked

The Write tool refused REPORT.md ("Subagents should return findings as text, not
write report files"). The full report text was returned to the coordinator in
the final message for saving at
benchmarks/redesign/ws_priorart/REPORT.md. No workaround was attempted.
