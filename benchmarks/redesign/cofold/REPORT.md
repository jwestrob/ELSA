# Co-fold screen of dark conserved DPANN blocks (ws_cofold, screen 1)

12 November 2026. Engine v1.5 loci on the permissive MMseqs2 base (`results_dpann_v5`), DPANN
package of 11 November. Folding: ESMFold2 `esmfold2-2026-05` through the Biohub batch API,
endpoint `fold_all_atom`, hetero-complex client `biohub_cofold.py`. Written by the orchestrator
from the workstream agent's hand-back; numbers are the agent's, spot-checked against
`class_distributions.tsv`, `folds.tsv` and `dossier_top.json`.

## Result

Engine-called unannotated gene pairs co-fold at the rate of curated system mates and far above
matched controls. Eight of 150 folded candidates met the predeclared bar in both distant
replicates; two were removed by predeclared refutation rules; six remain.

| Class | n | ipTM median | ipTM q90 | ipTM ≥ 0.75 | Pass criteria 1–3 |
|---|---|---|---|---|---|
| Cognate (engine-called dark pair) | 300 | 0.197 | 0.787 | 39 | 27 (9.0%) |
| Non-cognate (length-matched partner swap) | 300 | 0.127 | 0.353 | 1 | 0 |
| Uncalled neighbour (adjacent dark pair the engine did not call) | 297 | 0.121 | 0.521 | 5 | 4 (1.35%) |
| Positive (mates inside curated gold systems) | 150 | 0.476 | 0.840 | 43 | 19 (12.7%) |

- Cognate against non-cognate: Mann–Whitney p = 9e-9; criteria 1–3, Fisher p = 8e-9.
- Cognate against uncalled neighbours: Mann–Whitney p = 1e-7; Fisher odds ratio 7.2, p = 2.2e-5.
- The bar is strict: of 52 curated system types with two folded instances, 3 pass criteria 1–3 in
  both (5.8%); the candidates pass at 8 / 150 (5.3%).
- Uncalled controls were matched on the same contig for 60, the same genome for 171 and the
  same order for 66 (DPANN contigs are short).

## Predeclaration

`predeclared_criteria.json`, written 05:10:29 UTC, 12 s before the screen batch; one payload
test had been folded.

A hit satisfies all four in both replicates:
1. ipTM ≥ 0.75.
2. Interface PAE ≤ 5 Å, averaged over contacts (CB–CB ≤ 8 Å).
3. At least 10 inter-chain contacts and interface pLDDT ≥ 70.
4. Cognate ipTM minus matched non-cognate ipTM ≥ 0.30 (replicate 1 swaps chain B, replicate 2
   swaps chain A).

Refutations:
- Promiscuity: the kept chain scores ipTM ≥ 0.6 with a non-cognate partner.
- Disorder zipping: ≥ 50% of a chain's interface residues have pLDDT < 60 in that chain's
  monomer fold, or the chain's pLDDT in the complex is < 60.
- Split gene: fused homologs (one protein hit by both chains in collinear, non-overlapping
  segments) occur in ≥ 25% as many genomes as the block, or in a replicate's genus.
- Generic adjacency: uncalled-neighbour pairs pass as often as cognate pairs.

MSA step: a hit is supported if ipTM stays ≥ 0.75 in both replicates with the MSA; below 0.5
marks a single-sequence artefact.

## Candidates

A gene is "dark" when it has no row in any annotation source of the Sharur DPANN database
(seal `805c93a6…`, 19 October 2026): `annotations` sources pfam, kofam, kegg, defensefinder,
defensefinder_system, txsscan, txsscan_system, hyddb, hyddb_subgroup; `system_proteins`
(including cctyper); derived atoms `_locus` and `_validation`; and membership of a gold
instance. `bgc_loci` and `locus_proteins` are empty and dbCAN is absent from the canonical
database. 1,398,957 genes (39.0%) are dark. A family (engine token) is dark when ≥ 98% of its
members are dark.

| Step | Count |
|---|---|
| Adjacent links (dark genes, dark families, significant token pair) | 77,928 |
| Runs of linked genes | 54,182 (132,110 genes) |
| Block instances of 2–4 genes | 110,680 |
| Block types | 14,309 |
| Types in ≥ 3 genomes | 8,546 |
| Pass length 60–700 aa (sum ≤ 1,400) | 6,545 |
| Pass hydrophobic fraction ≤ 0.45 | 4,924 |
| Pass low complexity ≤ 0.145 | 5,037 |
| Pass copies per genome ≤ 1.5 | 7,726 |
| Pass geometry (same strand ≥ 0.8, adjacency ≥ 0.5, intergenic ≤ 150 nt) | 6,347 |
| Pass recurrence (≥ 5 genomes, ≥ 2 orders) | 1,110 |
| Pass every filter | 358 (302 pairs, 47 triples, 9 four-gene blocks) |
| Non-redundant (each family used once) | 189, all pairs; top 150 folded |

Every instance of the top 150 lies inside an engine `conserved` locus. The two replicates come
from different phyla for 83 candidates and different orders for 67; median identity between
them is about 39%.

## The eight predeclared hits

| Rank (tokens) | Lengths (aa) | Genomes / orders / phyla | ipTM r1 / r2 | Non-cognate r1 / r2 | Outcome |
|---|---|---|---|---|---|
| 27 (100239-199393) | 104 / 168 | 15 / 4 / 2 | 0.912 / 0.886 | 0.34 / 0.15 | Survives. No annotated homolog (44 homologs of A, 28 of B). |
| 13 (221251-351854) | 186 / 181 | 22 / 7 / 5 | 0.917 / 0.892 | 0.11 / 0.10 | Survives. 16 of 38 homologs of A carry Pfam UPF0175. |
| 67 (229034-423021) | 95 / 86 | 7 / 3 / 3 | 0.915 / 0.906 | 0.09 / 0.13 | Survives. Homologs carry Pfam ParD_antitoxin (A) and HigB/ParE_toxin (B). |
| 38 (23083-68571) | 81 / 101 | 18 / 3 / 2 | 0.882 / 0.870 | 0.08 / 0.33 | Survives. 180 of 393 homologs of B carry Pfam MazE_antitoxin. |
| 106 (27009-27010) | 80 / 68 | 7 / 2 / 1 | 0.926 / 0.802 | 0.07 / 0.12 | Survives, narrow. Fully dark; 20-nt overlap in all 7 instances. |
| 71 (171614-322104) | 108 / 180 | 5 / 3 / 2 | 0.753 / 0.886 | 0.11 / 0.13 | Survives, marginal and narrow. No homologs outside the block. |
| 6 (322564-326340) | 116 / 88 | 54 / 9 / 3 | 0.910 / 0.902 | 0.23 / 0.10 | Flagged by the zipping rule (replicate 2, 50.0% of B's interface is low-confidence in the monomer). Homologs carry Pfam ParE_toxin and NTP_transf_2. |
| 14 (48264-404027) | 181 / 228 | 67 / 6 / 2 | 0.914 / 0.927 | 0.22 / 0.11 | Refuted as a split gene: fused homologs of 408–425 aa in 18 of 67 genomes (27%). |

- Interface PAE of every hit lies between 1.0 and 4.8 Å.
- The Pfam families named for ranks 6, 13, 38 and 67 are observed on homologs of the candidate
  genes; no curated caller emitted a system call on the candidates. UNVERIFIED as system
  assignments.
- The zipping rule also fires on 3 of the 19 curated positive pairs that pass criteria 1–3, so
  rank 6 matches a pattern seen in curated mates where one partner folds on binding.
- Promiscuity fired only on non-hits (ranks 53, 25, 84).

Near misses:
- Rank 9 (114055-318532; 89 / 220 aa; 18 genomes, 8 orders, 5 phyla; homologs dark): ipTM
  0.849 / 0.814 with interface PAE 8.4 / 6.3 Å (fails criterion 2); with the MSA ipTM 0.900 /
  0.901 and PAE 4.4 / 2.0 Å.
- Rank 1 (81718-393054; 59–60 / 69–70 aa; 249 genomes, 21 orders, 7 phyla; homologs dark): ipTM
  0.856 / 0.708 (with MSA 0.790 / 0.731), non-cognate 0.17 / 0.10; 164 of 268 instances overlap
  by the 4-nt ATGA motif; fused 133–139 aa homologs in 3 genomes (1.2%).

## MSA refold (21 candidates × 2 replicates, 210 items)

- Genome-paired MSAs built from the ELSA block instances (mafft), 5 to 358 rows.
- All eight hits stay ≥ 0.75 with the MSA (lowest: rank 71 replicate 1, 0.752).
- Median change from the screen is −0.001 and paired minus shuffled pairing is 0.000, so the MSA
  shows robustness and adds no coevolution evidence; the API's pairing behaviour is unverified.
- Non-cognate MSA folds reach ≥ 0.75 in 0 of 42.
- Rank 13 as a three-gene block with family 164787 (384 aa): triple ipTM 0.875 (replicate 1;
  A–C 88 contacts, PAE 3.3 Å; B–C PAE 0.9 Å) and 0.651 (replicate 2; A–C PAE 9.9 Å); controls
  with chain A swapped 0.354 / 0.345.
- Protenix v2 inputs for the top 10 are staged (`protenix_top10.json`); nothing submitted.

## Orchestrator's checks (12 November)

- Recomputed the per-class table from `folds.tsv`; it matches.
- Ranks 27 and 106, overlapping reading frames. In the block alignments chain B begins with its
  own methionine in all 15 (rank 27) and all 7 (rank 106) instances, and the conserved
  N-terminal region of B follows directly. In rank 27 the overlap is 14 nt in one replicate and
  32 nt in the other, tracking the variable C-terminal tail of A. This is the pattern of two
  translationally coupled genes. A nucleotide-level check for a shifted-frame fusion is still
  open.
- Rank 27 chain B carries a conserved N-terminal motif (hydrophobic pair, then D-T-S-A-[LF]-I-S-L)
  in all 15 instances. Observation only; a profile and structure search is the test of whether
  this protein resembles a known fold.
- Small, tightly coupled, widely distributed two-gene blocks are the profile of toxin–antitoxin
  modules, and three or four of the surviving hits have homologs carrying such Pfam families.
  Screen 2 therefore adds a remote-homology triage before folding and extends to blocks of three
  and four genes and longer proteins.

## RPS-BLAST against CDD (orchestrator, 12 November; asked for by Jacob)

`rpsblast` against `/shared/db/cdd/latest/blast/Cdd` (database of 3 April 2023), E ≤ 1, 20
chains: both replicates of ranks 27, 9, 1 and 13 (pair and three-gene block). Hits:
`work/cdd_check_hits.tsv`; biotite `cofold/cdd_check/`.

| Candidate | Chain | Best reproducible CDD hit |
|---|---|---|
| Rank 27 | B (164–168 aa) | cd09879 PIN_VapC_AF0591-like, E 7.9e-07 and 2.4e-05, residues 7–116 |
| Rank 27 | A (98–107 aa) | cd22231 RHH_NikR_HicB-like, E 0.044, residues 4–34, one replicate only |
| Rank 13 | pair chain B (179–182 aa) | COG2405 (predicted nucleic-acid-binding protein with a PIN domain), E 0.004 and 0.029, residues 81–181 |
| Rank 13 | pair chain A (180–190 aa) | none reproducible (homologs carry Pfam UPF0175) |
| Rank 13 | third gene (375–391 aa) | none reproducible |
| Rank 9 | both chains | none reproducible (all E ≥ 0.18, different models per replicate) |
| Rank 1 | both chains | none reproducible |

Rank 27 has the domain pattern of a PIN-domain nuclease with a small partner, the architecture
of a common class of two-gene toxin–antitoxin modules; UNVERIFIED as a system (no curated
caller emitted it). Rank 13 shows a marginal, reproducible PIN-containing-COG hit on one chain.
Ranks 9 and 1 have no CDD similarity at E ≤ 0.1 that recurs across replicates. Protenix
confirmation was dropped at Jacob's direction.

## Open items

1. Nucleotide check of the shifted-frame overlaps of ranks 27 and 106.
2. "Dark" means dark in this database: 85 of 150 candidates have annotated homologs at ≥ 50%
   query coverage. Profile and structure searches against external databases have not been run.
3. Protenix v2 on the staged inputs.
4. The four high-scoring uncalled pairs (for example `unc_071_r2`, ipTM 0.868).
5. Only rank 13 was folded as a three-gene block.
6. `universe/genes.parquet` `length` includes the stop codon for 1.23 M genes (+1 aa); fold
   tables use true sequence lengths.

## Resources

Biohub: 1,309 items on 12 November UTC. Slurm: job 78168 (MMseqs2 split-gene search, 24 CPUs,
1 min 12 s). No GPU work.

## Files

`benchmarks/redesign/cofold/`: code (`build_gene_flags.py`, `extract_sharur_flags.py`,
`composition.py`, `build_candidates.py`, `select_folds.py`, `make_jobs.py`,
`extract_seqs_biotite.py`, `biohub_cofold.py`, `splitgene_search.sbatch`, `split_gene_check.py`,
`analyze_folds.py`, `build_paired_msa.py`, `analyze_msa.py`, `dossier.py`, `final_tables.py`);
tables (`candidates.tsv`, `folds.tsv`, `class_distributions.tsv`, `per_candidate_folds.tsv`,
`fold_summary.json`, `dossier_top.tsv`, `dossier_top.json`, `protenix_top10.json`, `msa_step.tsv`,
`msa_step_by_candidate.tsv`, `split_gene_check.tsv`, `zipping_calibration.tsv`,
`control_homology.tsv`, `replicate_identity.tsv`, `candidates_summary.json`); `work/` (920 MB,
git-ignored: structures, PAE, alignments). Biotite:
`/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/`
(`seqs/`, `split/`, `protenix/`, `structures/`).
