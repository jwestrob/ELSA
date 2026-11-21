# A conserved two-gene unit in Nanobdellota pairing a dark protein family with a cell-wall-active one

**Status: observed, unnamed.** No curated caller (DefenseFinder, TXSScan, HydDB)
fires on any member. Nothing here asserts a mechanism.

## The observation

`cluster:3913` in the DPANN v6 run is a two-slot syntenic unit across **669
genomes**, of which **641 are Nanobdellota** (37% of the 1,738 in the
collection), with smaller numbers in Aenigmatarchaeota and Nanohalarchaeota.

| slot | proteins | dominant family | wall-active Pfam |
|---|---:|---|---:|
| 0 | 1,149 | family #0, 821 members (68%) | 0.4% |
| 1 | 744 | family #1, 269 members (34%) | **7.1%** |

Slot 0 holds a large, annotation-dark protein family: median 775 aa, q90 2,055
aa, longest 5,637 aa, with a signal-peptide-like N-terminus on 87% of members
and low body hydrophobicity — the profile of a secreted or surface protein
rather than a membrane one. No internal repeats (k-mer repeat fraction 0.000),
so its architecture differs from the repeat-expanded Omnitrophota giants.

Slot 1 holds a family in which Pfam detects a cell-wall-active domain — `SLT` and
relatives — on a minority of members.

## Why this is one family and not co-clustering

The first reading was that the whole cluster was a wall-active family Pfam
mostly misses. That is wrong, and the test says so: **none** of the 58
wall-domain proteins fall in slot 0's 821-member family. The two slots hold
unrelated families.

Within slot 1 the picture is different. Sensitive search (mmseqs, e<1e-3, 40%
coverage, s=7.5) puts **50 of the 53** wall-domain proteins in a *single*
homology family of 269 members, where **19.7% carry the detected domain**. So
slot 1 is one family whose divergent members fall below HMM sensitivity, rather
than a mixture.

That distinction matters: 19.7% detection inside one homologous family is
evidence about that family; 3% across a heterogeneous cluster is not.

## Co-occurrence

Of the 57 loci where a wall-active domain is detected, **56 (98.2%)** also carry
the slot-0 dark family. The pairing is essentially invariant where it can be
seen.

## Enrichment against background

Wall-active Pfam domains cover 0.100% of DPANN proteins overall, and their rate
is highest in Nanobdellota:

| phylum | rate |
|---|---:|
| Nanobdellota | **0.134%** |
| Aenigmatarchaeota | 0.110% |
| Micrarchaeota | 0.056% |
| Iainarchaeota | 0.058% |

Within the candidate clusters the domains concentrate 12–31x over that
background: `cluster:3913` 3.06% (30.8x), `cluster:5379` 3.07% (30.9x),
`cluster:1096` 2.19% (22.0x), `cluster:207` 1.25% (12.6x).

`cluster:1043` — the tandem membrane-protein triplication — comes back at
**0.00%**, which is the negative control this comparison needed.

## What is and is not established

Established: a two-gene arrangement conserved across hundreds of Nanobdellota
genomes, pairing a large secreted-type dark family with a family carrying
cell-wall-active domain hits, at 12–31x enrichment over background, in a phylum
whose members lack a peptidoglycan wall themselves.

Not established: any function. The domain is detected on 19.7% of slot 1's
family and inferred by homology for the rest — an inference, not a domain call.
No curated system caller fires. Whether the arrangement acts on a host envelope
is a hypothesis this data motivates and does not test.

## Next

- Fold a representative of each slot; slot 0's size and secreted profile make
  its fold the informative one.
- Check whether slot 0's family carries a recognisable fold at all, given no
  sequence-level annotation reaches it.
- Test whether the 3913 / 5379 / 1096 / 207 dark families are related to each
  other, which would make this one system rather than four.
- Establish whether slot 1's family is monophyletic with characterised lytic
  transglycosylases or convergent.
