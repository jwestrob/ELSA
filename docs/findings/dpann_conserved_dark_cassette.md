# A conserved tandem triplication of a dark hydrophobic protein, in half of DPANN

**Status: observed, unnamed.** No curated caller fires on any member, so nothing
here is a functional claim. The curated callers populated in
`data/dpann/sharur.duckdb` are DefenseFinder, TXSScan and HydDB; `bgc_loci` and
`locus_proteins` are empty.

## How it was found

ELSA run `results_v6` (3,564 DPANN genomes, 3.4M genes, soft-ZCA α=0.1,
threshold 0.30, hnsw ef=256). Clusters were ranked by breadth and by how little
of their content carries a Pfam hit — the case a domain scan cannot reach and
synteny can. 117 clusters have ≥50 genomes, ≥3 slots and ≥90% dark content; 97
of them carry no curated system call at all.

`benchmarks/scripts/find_conserved_dark_loci.py` reproduces the ranking.

## cluster:1043

| property | value |
|---|---|
| genomes | **1,668** of 3,564 (47%) |
| phyla | 6 (85% Nanobdellota) |
| slots | 3 |
| proteins | 5,561 |
| median length | 188 aa (q10 164, q90 240) |

**Annotation coverage, all sources:** pfam 1%, kofam 0%, kegg 0%,
defensefinder 0%.

### It is a genuine cassette, not a clustering artifact

Over 1,097 loci carrying all three slots:

- genes strictly consecutive on the contig — **96.5%**
- all three on the same strand — **94.8%**
- slot order along the contig — (0,1,2) 75%, reversed (2,1,0) 12%

### Each slot is one protein family — and the three are paralogs

The three slots are homologous to each other, so this is a **tandem triplication
of a single family** rather than three unrelated ones:

| comparison | members hit | median identity |
|---|---|---|
| slot 0 vs slot 1 | 88.7% | 32.4% |
| slot 0 vs slot 2 | 86.1% | 40.7% |
| slot 1 vs slot 2 | 96.7% | 31.5% |

Members are strongly hydrophobic — mean Kyte-Doolittle +0.88, +0.95 and +1.08
by slot, with 84-96% of members above +0.5. That composition is consistent with
integral membrane proteins; it is a compositional observation and no curated
caller has assigned a family, a fold or a topology.

### Slot coherence

Strict clustering splits the slots into hundreds of groups (362 at 30% identity
/ 70% coverage for slot 0), which alone would suggest heterogeneity. Sensitive
search says otherwise — each slot is a single transitively-connected homology
component:

| slot | proteins | largest component |
|---|---|---|
| 0 | 2,444 | 2,228 (**91.2%**) |
| 1 | 1,617 | 1,532 (**94.7%**) |
| 2 | 1,500 | 1,411 (**94.1%**) |

The gap between those two readings is the point: these are families too
divergent for identity-threshold clustering and coherent under sensitive search,
which is the regime embedding-based retrieval is meant to occupy.

## What this is not

A tandem triplication of one uncharacterized hydrophobic family, in fixed order
and orientation across half a phylum-spanning collection, is a strong and
specific pattern — and the pattern is all that is established. There is no
evidence here for any function, no curated call to borrow a name from, and no
topology beyond an amino-acid composition consistent with membrane residence.

An earlier revision of this document called the three slots distinct families.
That was wrong: they are paralogs at 31-41% median identity, which changes the
reading from a three-component assembly to a repeat expansion.

## Checks worth doing before building on it

- Contamination and mobility: GC and coding density against the host contig,
  and proximity to mobile-element or tRNA features.
- Structure prediction on a representative from each slot, which would give
  fold-level hints where sequence gives none.
- Whether the 12% reversed-order loci differ from the 75% in any other respect.
- Copy number: whether genomes carry two, three or more tandem copies, and
  whether the count tracks taxonomy — the loci table already shows a 4-gene
  variant (0,0,1,2) in 4% of cases.
- The same ranking on Omnitrophota, to see whether the cassette is DPANN-specific.

## Other candidates from the same ranking

| cluster | genomes | slots | dark | dominant phylum |
|---|---|---|---|---|
| 15585 | 886 | 8 | 99.1% | Micrarchaeota 97% |
| 26 | 890 | 3 | 99.2% | Nanobdellota 100% |
| 24952 | 644 | 8 | 92.3% | Micrarchaeota 100% |
| 25065 | 509 | 3 | 99.6% | Micrarchaeota 100% |

`cluster:15585` is the larger structure — an eight-slot neighbourhood in 886
Micrarchaeota genomes at 99.1% dark, with 502 sequence families across its slots.
