#!/usr/bin/env python
"""galt step 7 (question 6): what the embedding lists see that alignment lists and profiles miss.
Inputs: data/voters/supporters.tsv and lists.tsv (g06, exact reconstruction of the slot supporters; Z supporter counts
match the stored supporting_genomes in 76/76), data/census_proteins.tsv, data/profile_scores.tsv.
Pairwise identity: Smith-Waterman (Biopython PairwiseAligner, BLOSUM62, gap open 11, extend 1); identity = identical
positions / aligned columns without gaps; coverage = aligned columns / length of the shorter sequence.
Per labeled gene: Z supporters, Z voters (supporters carrying the label PF01087 as a Pfam row), identity to voters
(median, max) and the voter's group, nearest named DPANN family member (the 1,245 named members outside the labeled set;
alignments covering >= 70% of the shorter sequence) by identity,
first named family member in the Z list and in the A list (rank), A supporters carrying the label.
Writes data/q6_identity.tsv."""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
C = pd.read_csv(H / "data/census_proteins.tsv", sep="\t")
S = pd.read_csv(H / "data/voters/supporters.tsv", sep="\t")
Lz = pd.read_csv(H / "data/voters/lists.tsv", sep="\t")
PS = pd.read_csv(H / "data/profile_scores.tsv", sep="\t").set_index("sequence_id")
c = duckdb.connect()
c.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
c.execute("SET threads=1")
ids = pd.DataFrame({"protein_id": sorted(set(S.partner) | set(Lz.partner) | set(C.protein_id))})
c.register("q", ids)
seq = dict(c.execute("SELECT p.protein_id, p.sequence FROM d.proteins p JOIN q USING(protein_id)").fetchall())
pfr = set(c.execute("SELECT DISTINCT a.protein_id FROM d.annotations a JOIN q USING(protein_id) WHERE a.source='pfam' AND a.accession='PF01087'").fetchdf().protein_id)
named = set(C.protein_id[C.named])
lab = set(C.protein_id[C.lab76])
al = Align.PairwiseAligner(mode="local", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                           open_gap_score=-11, extend_gap_score=-1)


def ident(a, b):
    a, b = a.rstrip("*"), b.rstrip("*")
    aln = al.align(a, b)[0]
    idn = cols = 0
    for (s1, e1), (s2, e2) in zip(*aln.aligned):
        x, y = a[s1:e1], b[s2:e2]
        idn += sum(u == v for u, v in zip(x, y)); cols += len(x)
    return (idn / cols if cols else 0.0), cols / min(len(a), len(b))


named_list = sorted(named)
# tier P = Pfam PF01087 / PF02744 or KOfam K00965 row; tier C = CDD-only (cd00608 / COG1085)
tierP = set(C.protein_id[C.fl.fillna("").str.contains("pfam:PF01087|pfam:PF02744|kofam:K00965")])
rows = []
for q in C.protein_id[C.lab76]:
    r = C.set_index("protein_id").loc[q]
    sz = S[(S["query"] == q) & (S.arm == "Z")]
    sa = S[(S["query"] == q) & (S.arm == "A")]
    vz = sz[sz.partner.isin(pfr)]
    va = sa[sa.partner.isin(pfr)]
    iv = [ident(seq[q], seq[p]) for p in vz.partner]
    cand = [(ident(seq[q], seq[p]), p) for p in named_list if p != q and p not in lab]
    cand = [x for x in cand if x[0][1] >= 0.7] or cand
    best_named = max(cand, key=lambda t: t[0][0])
    candP = [x for x in cand if x[1] in tierP]
    bestP = max(candP, key=lambda t: t[0][0]) if candP else ((np.nan, np.nan), "")
    zl = Lz[(Lz["query"] == q) & (Lz.arm == "Z")]
    alst = Lz[(Lz["query"] == q) & (Lz.arm == "A")]
    zfam = zl[zl.partner.isin(named | lab)]
    afam = alst[alst.partner.isin(named | lab)]
    rows.append(dict(protein_id=q, group=r.group, order=r.order, length=r.length,
                     z_supporters=len(sz), z_voters=len(vz), votes_for_stored=r.votes_for,
                     voter_ident_median=round(float(np.median([x[0] for x in iv])), 3) if iv else np.nan,
                     voter_ident_max=round(max(x[0] for x in iv), 3) if iv else np.nan,
                     voter_cov_median=round(float(np.median([x[1] for x in iv])), 2) if iv else np.nan,
                     voters_lab76=int(vz.partner.isin(lab).sum()), voters_named_unlabeled=int((vz.partner.isin(named) & ~vz.partner.isin(lab)).sum()),
                     nearest_named=best_named[1], nearest_named_ident=round(best_named[0][0], 3), nearest_named_cov=round(best_named[0][1], 2),
                     nearest_named_group=C.set_index("protein_id").group.get(best_named[1], ""),
                     nearest_named_tier=("P" if best_named[1] in tierP else "C"),
                     nearest_tierP=bestP[1], nearest_tierP_ident=round(bestP[0][0], 3), nearest_tierP_cov=round(bestP[0][1], 2),
                     zlist_named_unlab=int((zl.partner.isin(named) & ~zl.partner.isin(lab)).sum()), zlist_lab76=int(zl.partner.isin(lab).sum()),
                     alist_named_unlab=int((alst.partner.isin(named) & ~alst.partner.isin(lab)).sum()), alist_lab76=int(alst.partner.isin(lab).sum()),
                     alist_first_named_rank=int(alst[alst.partner.isin(named) & ~alst.partner.isin(lab)]["rank"].min()) if (alst.partner.isin(named) & ~alst.partner.isin(lab)).any() else -1,
                     zlist_first_named_rank=int(zl[zl.partner.isin(named) & ~zl.partner.isin(lab)]["rank"].min()) if (zl.partner.isin(named) & ~zl.partner.isin(lab)).any() else -1,
                     a_supp_lab76=int(sa.partner.isin(lab).sum()),
                     zlist_len=len(zl), zlist_family_members=len(zfam), zlist_first_family_rank=int(zfam["rank"].min()) if len(zfam) else -1,
                     zlist_first_family_z=float(zfam.z.iloc[0]) if len(zfam) else np.nan,
                     alist_len=len(alst), alist_family_members=len(afam), alist_first_family_rank=int(afam["rank"].min()) if len(afam) else -1,
                     a_supporters=len(sa), a_voters=len(va),
                     bits_PF01087=PS.bits_GalP_UDP_transf.get(q), bits_PF02744=PS.bits_GalP_UDP_tr_C.get(q),
                     bits_PTHR42763=PS.bits_PTHR42763.get(q), bits_PTHR11943=PS.bits_PTHR11943.get(q), bits_PTHR38420=PS.bits_PTHR38420.get(q)))
T = pd.DataFrame(rows)
T.to_csv(H / "data/q6_identity.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
cols = ["z_supporters", "z_voters", "voter_ident_median", "voter_ident_max", "voter_cov_median", "nearest_named_ident", "nearest_named_cov", "nearest_tierP_ident", "nearest_tierP_cov",
        "zlist_named_unlab", "zlist_lab76", "zlist_first_named_rank", "alist_len", "alist_named_unlab", "alist_lab76", "alist_first_named_rank",
        "a_supporters", "a_supp_lab76", "a_voters"]
print(T.groupby("group")[cols].median().round(3).T)
print(T[T.group == "lead15"][["protein_id", "order"] + cols + ["voters_lab76", "voters_named_unlabeled", "nearest_named_group"]].to_string(index=False))
