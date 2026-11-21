#!/usr/bin/env python
"""lut_loci e01 (biotite login node): the EF-hand_5-labeled, K22619-partner genes handed over by leads_lut.

Inputs: data/ef_gids.txt (32 gids: 31 main-population T2b/rest + 1 contradicted, from leads_lut/data/population.parquet).
Per gene: pass row and tier; supporters (pass_port supporters/part_*.parquet) with carries_label; for every supporter: token,
Pfam domains (annot_gpu GA), length, KOfam partner KO (annot_gpu/lut/KOFAM_partners.parquet), whether it is a lanM-family
gene (rescue3 token 358533 or an n01 window gene with gold component M) and whether it lies in a lut window.
Context per gene: own token, +-10 neighbors (Pfam), nearest main lut instance on the same contig (genes), nearest K23995 gene
on the same contig, genome carries a lanM-token gene / an EF-hand_5 (PF13202) protein / other EF-hand Pfam proteins.
Writes data/ef_genes.tsv, data/ef_supporters.tsv, data/ef_neighbors.tsv, data/ef_genome_efhands.tsv.
"""
import glob

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PK = RD + "/slice_lut/testset/package_k8"
PP = RD + "/dpann_leads/pass_port/out/lut"
EF_PF = {"PF13202": "EF-hand_5", "PF00036": "EF-hand_1", "PF13405": "EF-hand_6", "PF13499": "EF-hand_7",
         "PF13833": "EF-hand_8", "PF14658": "EF-hand_9", "PF12763": "EF-hand_4"}
gids = [int(x) for x in open("data/ef_gids.txt") if x.strip()]
gn = pd.read_parquet(PK + "/universe/genes.parquet",
                     columns=["gid", "genome", "genome_code", "contig_code", "position_index", "strand", "length"])
gm = pd.read_parquet(PK + "/universe/genomes.parquet", columns=["genome_code", "genus", "family", "lut_status", "layer"])
gn = gn.merge(gm, on="genome_code", how="left")
tok = pd.read_parquet(RD + "/long_loci/frozen/tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet").token.to_numpy()
gn["token"] = tok
pf = pd.read_parquet(RD + "/annot_gpu/lut/PFAM.parquet", columns=["gid", "accession", "name", "ali_from", "ali_to"])
pf["acc"] = pf.accession.str.split(".").str[0]
pfs = pf.sort_values(["gid", "ali_from"]).groupby("gid").name.apply(";".join)
ko = pd.read_parquet(RD + "/annot_gpu/lut/KOFAM_partners.parquet")
kocol = "ko" if "ko" in ko.columns else [c for c in ko.columns if c.lower().startswith("ko")][0]
idcol = "gid" if "gid" in ko.columns else "protein_id"
ko["gid_"] = ko[idcol].astype(np.int64)
kos = ko.groupby("gid_")[kocol].apply(lambda s: ";".join(sorted(set(s))))
W = pd.read_parquet("data/window_genes.parquet")
win = set(W.gid)
compM = set(W[W.component == "M"].gid)
lanm_tok = 358533
ps = pd.read_parquet(PP + "/lut_slot_rescue_pass.parquet")
ti = pd.read_parquet(PP + "/tiers.parquet")[["protein_id", "tier"]]
ps = ps.merge(ti, on="protein_id")
G = ps[ps.gid.isin(gids)].copy()
# supporters
sup = []
for f in sorted(glob.glob(PP + "/supporters/part_*.parquet")):
    d = pd.read_parquet(f)
    d = d[d.gid.isin(gids)]
    for r in d.itertuples():
        for rank, (s, c) in enumerate(zip(r.supporters, r.carries_label)):
            sup.append((r.gid, rank, int(s), bool(c)))
S = pd.DataFrame(sup, columns=["gid", "rank", "supporter", "carries_label"])
gi = gn.set_index("gid")
S["sup_token"] = tok[S.supporter]
S["sup_len"] = gi.length.reindex(S.supporter).to_numpy()
S["sup_genus"] = gi.genus.reindex(S.supporter).to_numpy()
S["sup_pfam"] = S.supporter.map(pfs)
S["sup_ko"] = S.supporter.map(kos)
S["sup_lanM_token"] = S.sup_token == lanm_tok
S["sup_lut_component_M"] = S.supporter.isin(compM)
S["sup_in_lut_window"] = S.supporter.isin(win)
S.to_csv("data/ef_supporters.tsv", sep="\t", index=False)
# context
I = pd.read_parquet(PK + "/gold/lut_instances.parquet")
I = I[I.definition == "main"]
mdh = pd.read_parquet("data/mdh_pkg.parquet")
k23 = mdh[mdh.ko == "K23995"].gid.to_numpy()
rows, nb = [], []
for g in gids:
    r = gi.loc[g]
    same = gn[(gn.contig_code == r.contig_code)]
    ii = I[I.contig_code == r.contig_code]
    if len(ii):
        dist = np.minimum(np.abs(ii.start_idx - r.position_index), np.abs(ii.end_idx - r.position_index))
        inside = ((ii.start_idx <= r.position_index) & (ii.end_idx >= r.position_index)).any()
        dlut = 0 if inside else int(dist.min())
        arch = ii.iloc[int(np.argmin(dist))].architecture
    else:
        dlut, arch = -1, ""
    kk = [x for x in k23 if gi.loc[x].contig_code == r.contig_code] if len(k23) else []
    dk = int(min(abs(gi.loc[x].position_index - r.position_index) for x in kk)) if kk else -1
    gen = gn[gn.genome_code == r.genome_code]
    gpf = pf[pf.gid.isin(gen.gid)]
    rows.append(dict(gid=g, genome=r.genome, genus=r.genus, family=r.family, lut_status=r.lut_status, length=int(r.length),
                     token=int(r.token), strand=int(r.strand), dist_to_lut_locus_genes=dlut, nearest_lut_arch=arch,
                     dist_to_K23995_genes=dk, in_lut_window=g in win,
                     genome_lanM_token_genes=int((gen.token == lanm_tok).sum()),
                     genome_lanM_in_lut_locus=int(gen.gid.isin(compM).sum()),
                     genome_EF5_proteins=int(gpf[gpf.acc == "PF13202"].gid.nunique()),
                     genome_otherEF_proteins=int(gpf[gpf.acc.isin(set(EF_PF) - {"PF13202"})].gid.nunique()),
                     token_members=int((tok == r.token).sum())))
    for d in range(-10, 11):
        x = same[same.position_index == r.position_index + d]
        if len(x):
            xg = int(x.gid.iloc[0])
            nb.append(dict(gid=g, off=d * int(r.strand), neighbor=xg, strand_rel=int(x.strand.iloc[0] * r.strand),
                           length=int(x.length.iloc[0]), token=int(tok[xg]), pfam=pfs.get(xg, ""),
                           lut_member=xg in set(np.concatenate(I.member_gids.to_numpy())), lanM_token=tok[xg] == lanm_tok))
C = pd.DataFrame(rows)
G = G.merge(C, on="gid", suffixes=("", "_ctx"))
G.to_csv("data/ef_genes.tsv", sep="\t", index=False)
pd.DataFrame(nb).to_csv("data/ef_neighbors.tsv", sep="\t", index=False)
# EF-hand proteins of the 32 genomes (for sequence comparison)
gcs = set(gi.loc[gids].genome_code)
E = pf[pf.acc.isin(EF_PF) & pf.gid.isin(gn[gn.genome_code.isin(gcs)].gid)]
E = E.groupby("gid").agg(efpf=("name", lambda s: ";".join(s)), n_ef=("name", "size")).reset_index()
E["genome_code"] = gi.genome_code.reindex(E.gid).to_numpy()
E["length"] = gi.length.reindex(E.gid).to_numpy()
E["token"] = tok[E.gid]
E["lanM_token"] = E.token == lanm_tok
E["lut_component_M"] = E.gid.isin(compM)
E["all_pfam"] = E.gid.map(pfs)
E.to_csv("data/ef_genome_efhands.tsv", sep="\t", index=False)
print(len(G), "genes;", len(S), "supporter rows;", int(S.carries_label.sum()), "label carriers;", len(E), "EF-hand proteins in",
      len(gcs), "genomes")
