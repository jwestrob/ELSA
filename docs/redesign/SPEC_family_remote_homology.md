# Spec: family-level remote-homology annotation for Sharur

12 November 2026. From the ELSA redesign session, at Jacob's request, for the Sharur agent.
Everything here ran on biotite on the DPANN dataset during the ELSA co-folding triage; paths
and commands are the ones that worked. Reference implementation:
`ELSA/benchmarks/redesign/cofold/s2_triage_remote.py` with `s2_triage*.sbatch`; outputs under
`/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/s2/`.

## Why

Sharur's "unannotated" status reflects Pfam at gathering thresholds (plus KOfam, KEGG and the
system callers). In the ELSA triage of 148 DPANN families from conserved unannotated blocks:

| Search | Families flagged as resembling a characterised family (of 62) |
|---|---|
| HHsearch of the family profile against PDB100 | 54 |
| RPS-BLAST against CDD (per protein) | 18 |
| HHsearch of the family profile against Pfam 39 | 8 |
| hmmsearch Pfam 39 at gathering thresholds (the current rule) | 4 |

47 of the 62 had no Pfam gathering-threshold hit. Nine of the ten strongest "unannotated"
co-folding blocks were recognisable this way.

## Two additions

1. **CDD per protein** (cheap, fits the existing per-protein annotation ingest).
2. **A family tier**: cluster proteins into families, build one profile per family, compare it
   with Pfam and PDB profiles (HMM–HMM). This runs once per family.

## Prerequisite: protein families (yes, clustering comes first)

The family tier needs a protein-to-family table. Recommended: make the clustering a standard
ingest product. ELSA uses the same families as its tokens, so one clustering serves both.

```bash
MM=/groups/banfield/users/jwestrob/bin/mmseqs_18/mmseqs/bin/mmseqs    # MMseqs2 release 18
$MM createdb proteins.faa seqdb
$MM cluster seqdb clu tmp -s 7.5 -e 1e-3 --min-seq-id 0 -c 0.5 --cov-mode 0 --cluster-mode 0 --threads $SLURM_CPUS_ON_NODE
$MM createtsv seqdb seqdb clu clu.tsv
```

- DPANN: 3,585,303 proteins give 430,471 families; 278,361 are singletons, 152,110 have two or
  more members, the largest has 7,675.
- These are ELSA's default ("permissive") parameters: E ≤ 1e-3, 50% bidirectional coverage,
  sensitivity 7.5, no identity floor.
- Any job that runs MMseqs2 needs the bad-node exclude line from `biotite-cluster-ops.md`.
- Known weakness (ELSA sub-genic study): truncated gene calls at contig edges (4.2% of DPANN
  genes) fall out of their family 27% of the time under bidirectional coverage. A repair rule is
  under test in ELSA; the family table should carry a clustering-run identifier so it can be
  replaced.

## Family tier, per family

Inputs: the family's member sequences, up to 250 unique sequences, representative first.

```bash
HH=/shared/software/hhsuite/latest/bin                                   # HH-suite 3.1
UNIREF=/shared/db/uniclust/30_2020_06/UniRef30_2020_06
PDB=/shared/db/rosettafold/rosettafold2/pdb100_2021Mar03/pdb100_2021Mar03
PFAMHH=/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/cofold/s2/db/pfam39/pfam39

mafft --auto --thread 1 --quiet fam.fa > fam.aln                         # skip for a singleton
$HH/hhblits  -i fam.aln -M first -d $UNIREF -n 2 -e 0.001 -cpu 4 -o uniref.hhr -oa3m enr.a3m -v 0
$HH/hhsearch -i enr.a3m -d $PFAMHH -o pfam.hhr -cpu 4 -v 0
$HH/hhsearch -i enr.a3m -d $PDB    -o pdb.hhr  -cpu 4 -v 0
```

- The triage also searched the family-only alignment (no UniRef enrichment) against both
  databases; keep it as an option, it is cheap.
- Parse each `.hhr`: hit accession and name, probability, E-value, aligned columns, query range,
  template range.
- Rule used in ELSA: "resembles a characterised family" = probability ≥ 90% over ≥ 30 aligned
  columns. Store every hit with probability ≥ 50% so the threshold can be changed later, and
  keep hits to families of unknown function (DUF, UPF) distinguishable from characterised ones.

### Databases

| Database | Path | Note |
|---|---|---|
| UniRef30 (HH-suite) | `/shared/db/uniclust/30_2020_06/UniRef30_2020_06` | June 2020; a newer build would find more |
| PDB100 (HH-suite) | `/shared/db/rosettafold/rosettafold2/pdb100_2021Mar03/pdb100_2021Mar03` | March 2021, the copy inside the RoseTTAFold2 data; a current PDB70/PDB100 build is worth installing |
| Pfam 39.0 (HH-suite) | `…/cofold/s2/db/pfam39/pfam39` (`pfam39_hhm.ffdata`, `.ffindex`) | Built during the triage; 53,275 HMMs; copy it to a stable shared location |
| Pfam 39.0 (HMMER3) | `/shared/db/pfam/39.0/Pfam-A.hmm` | For hmmsearch only |

Building the Pfam HH database: the HMMER3 models crash HH-suite 3.1 `hhsearch`, so it was built
from `Pfam-A.seed` 39.0: one alignment per family, `hhmake -i fam.a2m -o fam.hhm -name
"<ACC> <ID>; <DE>" -M 50`, then `ffindex_build -s -f list pfam39_hhm.ffdata pfam39_hhm.ffindex`.
`-M a2m` fails for 88% of seed alignments (they mark no insert columns); `-M 50` works.
Self-test that passed: seed sequences of PF00004, PF01850 and PF02604 each rank their own family
first at probability 98.8–99.9%.

## CDD, per protein

```bash
rpsblast -query proteins.faa -db /shared/db/cdd/latest/blast/Cdd -evalue 1 -num_threads $SLURM_CPUS_ON_NODE \
    -outfmt "6 qseqid sseqid qlen qstart qend evalue bitscore stitle"
```

CDD v3.20 (3 April 2023). ELSA's rule for a family: the same model or superfamily at E ≤ 0.01 in
two or more members. For ingest, store hits at E ≤ 0.01 on unique sequences and expand to
proteins.

## Suggested tables

- `protein_families(protein_id, family_id, is_representative, clustering_run)`
- `family_stats(family_id, clustering_run, n_members, n_genomes, rep_protein_id, rep_length)`
- `family_annotations(family_id, clustering_run, source, db_version, msa_kind, hit_accession,
  hit_name, probability, evalue, aligned_cols, q_start, q_end, t_start, t_end)` with `source` in
  `hhsearch_pfam`, `hhsearch_pdb`, `hhblits_uniref` and `msa_kind` in `enriched`, `family_only`
- CDD hits in the existing per-protein `annotations` table with `source = 'cdd'`

Provenance: these rows are observed similarities. They belong beside Pfam hits, outside the
curated system tables, and consumers should read them as "resembles", never as a system call.

## Cost and scheduling

- Measured: 173 families through every search above (HHblits, four HHsearch runs, CDD,
  hmmsearch) took 82 minutes on 24 CPUs (job 78254), about 11 CPU-minutes per family as an upper
  bound. HHblits dominates.
- Extrapolated: the 152,110 DPANN families with two or more members would cost about 29,000
  CPU-hours, roughly 2.4 days at the 500-CPU limit. Restricting to families found in three or
  more genomes, or to families without a Pfam gathering-threshold hit, cuts this; Sharur can
  count those subsets directly.
- Run as 24-CPU Slurm jobs (`-A banfield -p standard --qos=standard_expand -c 24`, the exclude
  line, no `--mem`, no `--time`), six workers of four threads each, families sharded by index,
  one output directory per family so reruns skip finished families.
- CDD: 20 short proteins took 10 s on 2 threads including database load; unmeasured at scale.

## Validation set

Expected results are in `ELSA/benchmarks/redesign/cofold/partA_triage.tsv` and
`screen2_tables/triage_family.tsv` (family = ELSA token of the `e3c50s75` clustering). Checks:

| ELSA family (token) | Expected best Pfam HH hit |
|---|---|
| 199393 | DUF3368 "PIN domain", probability 99.8% |
| 229034 | ParD_antitoxin, 98.4% |
| 423021 | ParE_toxin, 98.5% |
| 107147 | Lon_C, 99.9% (PDB 4fw9, 100%) |
| 205870 | Pkinase, 99.9% |
| 81718 and 393054 | nothing at ≥ 90% (best 89% and 66%) |

## Open choices for Jacob

1. Scope: all families with two or more members, or only those without a gathering-threshold
   Pfam hit.
2. Whether to install newer UniRef30 and PDB HH-suite databases before the first full run.
3. Whether Sharur's clustering parameters should equal ELSA's permissive setting (so ELSA reads
   Sharur's families as its tokens) or be a separate, stricter clustering.
