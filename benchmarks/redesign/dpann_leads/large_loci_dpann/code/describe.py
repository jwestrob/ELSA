#!/usr/bin/env python
"""large_loci_dpann, local step: validation, family description, classification, ranking, top-20 maps
(rules in predeclared.json). Reads the biotite outputs (out/), the per-gene annotation export (annot_export.py),
the gene universe, genome groups, Sharur bins (read-only), the slot-rescue pass and verification tiers, KEGG KO->module.

usage: describe.py SCRATCH_DIR  (holds out/*.parquet, run.json, genes.parquet, annot_all.parquet, groups parquet)
writes families.tsv, top20_maps.txt, data/*.tsv, data/validation.json in the workstream folder
"""
import json
import sys
from collections import Counter

import duckdb
import numpy as np
import pandas as pd

W = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/large_loci_dpann"
R = "/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign"
DB = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
FAM = "family"
MOB = {"mobile_element", "transposase", "integrase", "phage_related"}
HK = {"ribosomal_protein", "rna_polymerase", "atp_synthesis"}
HK2 = {"translation", "translation_factor", "trna_synthetase", "replication", "dna_polymerase", "rna_processing",
       "trna_modification", "rrna_modification", "sec_pathway", "signal_recognition", "chaperone", "cell_division",
       "divisome", "topoisomerase", "transcription"}   # amendment_3 (post hoc)
PROC = {"carbohydrate": {"carbohydrate_active", "glycosyltransferase"}, "transport": {"transporter"},
        "secretion": {"secretion_component", "secretion_system", "pilus"}, "defense": {"defense_component"},
        "energy": {"electron_transport", "energy_metabolism"}, "cofactor": {"cofactor_biosynthesis"},
        "amino_acid": {"amino_acid_biosynthesis"}, "lipid": {"lipid_metabolism"},
        "nucleotide": {"nucleotide_metabolism"}, "dna": {"dna_repair", "replication"}, "translation": {"translation"},
        "surface": {"cell_surface", "adhesin"}, "toxin_antitoxin": {"toxin_domain", "antitoxin_domain", "toxin"}}


def plur(vals):
    v = [x for x in vals if isinstance(x, str) and x]
    if not v:
        return "", 0.0
    k, c = Counter(v).most_common(1)[0]
    return k, c / len(vals)


def tagshare(tags, S):
    return np.mean([bool(set(t.split(";")) & S) if isinstance(t, str) and t else False for t in tags])


def lis_len(seq):
    import bisect
    tails = []
    for x in seq:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def main(sp):
    out = sp + "/out"
    L = pd.read_parquet(out + "/loci.parquet").drop(columns=["fam_f33", "fam_f25", "fam_f50", "degree_f33"])
    L2 = pd.read_parquet(sp + "/out2/loci2.parquet")[["locus", "family", "degree_v2"]]
    L = L.merge(L2, on="locus")
    Gn = pd.read_parquet(sp + "/out2/genes2.parquet")
    U = pd.read_parquet(sp + "/genes.parquet")
    A = pd.read_parquet(sp + "/annot_all.parquet").drop(columns=["strand"])
    grp = pd.read_parquet(sp + "/emb_hom_average_t0.75.parquet")[["genome_code", "group"]]
    c = duckdb.connect()
    c.execute(f"ATTACH '{DB}' AS d (READ_ONLY)")
    bins = c.execute("SELECT bin_id, completeness, contamination, taxonomy FROM d.bins").df()
    tx = bins.taxonomy.str.split(";", expand=True)
    bins["phylum"], bins["order"], bins["tfamily"], bins["genus"] = tx[1], tx[3], tx[4], tx[5]
    # slot labels and tiers
    sl = pd.read_parquet(R + "/slot_rescue/pass/dpann_slot_rescue_pass.parquet",
                         columns=["protein_id", "proposed_label_name", "proposed_label_positional_homology", "vote_share",
                                  "alignment_lists_same_label"])
    tiers = pd.read_parquet(R + "/slot_rescue/verify/verification_per_protein.parquet", columns=["protein_id", "tier"])
    sl = sl.merge(tiers, on="protein_id", how="left")
    kk = pd.read_csv(R + "/dpann_leads/gapscreen/data/kegg_ko.tsv", sep="\t", keep_default_na=False)
    ko_mod = {r.ko: set(r.modules.split(",")) - {""} for r in kk.itertuples()}
    mods = pd.read_csv(R + "/dpann_leads/gapscreen/data/kegg_modules.tsv", sep="\t")
    mod_name = dict(zip(mods.module, mods.name))

    # ---------------- per-gene table for genes in large loci
    U = U[["gid", "protein_id", "genome", "genome_code", "contig_code", "position_index", "strand", "length"]]
    G = Gn.merge(U.drop(columns=["protein_id"]), on="gid").merge(A, on="protein_id", how="left")
    G = G.merge(sl, on="protein_id", how="left")
    G["unannot_any"] = (~G.any_annot_row.fillna(False)) & G.cdd_acc.isna()
    G["unannot_core3"] = G.pfam.isna() & G.ko.isna()
    G["is_ribo"] = G.pfam.fillna("").str.contains("(?:^|;)Ribosomal_", regex=True)
    hjc = G.pfam.fillna("x").map(lambda s: set(s.split(";")) <= {"Hjc", "Endonuc_Holl"})   # amendment_1
    G.loc[hjc, "tags"] = G.loc[hjc, "tags"].fillna("").map(
        lambda t: ";".join(x for x in t.split(";") if x not in MOB))
    G["is_mob"] = [bool(set(t.split(";")) & MOB) if isinstance(t, str) else False for t in G.tags]
    G["has_cur"] = G.cur_systems.notna()
    G["annot_label"] = np.where(G.pfam.notna(), G.pfam,
                       np.where(G.ko.notna(), "KO:" + G.ko.fillna("") + " " + G.ko_name.fillna(""),
                       np.where(G.cdd_acc.notna(), "CDD:" + G.cdd_desc.fillna("").str.slice(0, 60),
                       np.where(G.defensefinder.notna(), "DF:" + G.defensefinder.fillna(""),
                       np.where(G.txsscan.notna(), "TXSS:" + G.txsscan.fillna(""), "")))))
    G = G.sort_values("gid").reset_index(drop=True)

    # ---------------- per-locus table
    L = L.merge(grp, on="genome_code", suffixes=("", "_g"))
    gen = U.drop_duplicates("genome_code")[["genome_code", "genome"]]
    L = L.merge(gen, on="genome_code").merge(bins, left_on="genome", right_on="bin_id", how="left")
    L["edge"] = (L.flank_left < 3) | (L.flank_right < 3)
    agg = G.groupby("locus").agg(n_ribo=("is_ribo", "sum"), n_cur=("has_cur", "sum"), n_mob=("is_mob", "sum"),
                                 n_unannot=("unannot_any", "sum"))
    L = L.merge(agg, left_on="locus", right_index=True, how="left")
    cur_types = G[G.has_cur].assign(t=G.cur_systems.str.split(";")).explode("t").groupby("locus").t.agg(
        lambda x: ";".join(sorted(set(x))))
    L["cur_types"] = L.locus.map(cur_types).fillna("")
    L.to_parquet(sp + "/loci_annot.parquet", index=False)

    # ---------------- family descriptions (primary rule)
    L = L[L[FAM] >= 0].copy()
    G = G[G.locus.isin(L.locus)].merge(L[["locus", FAM]], on="locus")
    rows, cl_rows, maps = [], [], {}
    Lgrp = dict(tuple(L.groupby(FAM)))
    for f, gg in G.groupby(FAM):
        mem = Lgrp[f]
        nm = len(mem)
        pres = gg.groupby("gene_cluster").locus.nunique() / nm
        core = pres[pres >= 0.5].index.to_numpy()
        # orientation and order
        byloc = {lid: sub for lid, sub in gg.groupby("locus")}
        seqs = {}
        for lid, sub in byloc.items():
            sub = sub.sort_values("gid")
            s0 = sub[sub.gene_cluster.isin(core)].drop_duplicates("gene_cluster")
            seqs[lid] = (s0.gene_cluster.to_numpy(), (s0.gid.to_numpy() - sub.gid.min()) / max(len(sub) - 1, 1),
                         s0.strand.to_numpy())
        ref = max(seqs, key=lambda k: len(seqs[k][0])) if seqs else None
        rrank = {cc: i for i, cc in enumerate(seqs[ref][0])} if ref is not None else {}
        pos_or, orient = {cc: [] for cc in core}, {}
        for lid, (cs, ps, st) in seqs.items():
            rk = [rrank[x] for x in cs if x in rrank]
            fw, rv = lis_len(rk), lis_len(rk[::-1])
            o = 1 if fw >= rv else -1
            orient[lid] = o
            for cc, p in zip(cs, ps):
                pos_or[cc].append(p if o == 1 else 1 - p)
        cons = sorted(core, key=lambda cc: np.median(pos_or[cc]) if pos_or[cc] else 2)
        crank = {cc: i for i, cc in enumerate(cons)}
        oc = []
        for lid, (cs, ps, st) in seqs.items():
            if len(cs) >= 2:
                rk = [crank[x] for x in cs]
                oc.append(max(lis_len(rk), lis_len(rk[::-1])) / len(cs))
        order_cons = float(np.median(oc)) if oc else np.nan
        # per core cluster summaries
        cov_n, cov_pre, mob_core, proc_core, ann_core = 0, 0, False, Counter(), 0
        bcat = Counter()
        modcount = Counter()
        cl_info = []
        for cc in cons:
            sub = gg[gg.gene_cluster == cc]
            lab, lsh = plur(sub.annot_label.tolist())
            pf, pfs = plur(sub.pfam.tolist())
            ko, kos = plur(sub.ko.tolist())
            for m in ko_mod.get(ko, ()):
                modcount[m] += 1
            cl_info.append((cc, sub, lab, lsh, ko))
        hk_mods = {m for m, n_ in modcount.items() if n_ >= 3}
        basis = Counter()
        for cc, sub, lab, lsh, ko in cl_info:
            cur_share = float(sub.has_cur.mean())
            tg = sub.tags.tolist()
            hk = {t for t in HK if tagshare(tg, {t}) >= 0.5}
            km = ko_mod.get(ko, set()) & hk_mods
            covered_pre = cur_share >= 0.5 or bool(hk) or bool(km)
            hk2 = {t for t in HK2 if tagshare(tg, {t}) >= 0.5}
            prot = float(sub.pfam.fillna("").str.contains("Proteasome").mean()) >= 0.5
            covered = covered_pre or bool(hk2) or prot
            if cur_share >= 0.5:
                ct, _ = plur([x.split(";")[0] for x in sub.cur_systems.dropna()])
                basis[ct] += 1
                bcat["curated"] += 1
            elif hk:
                basis[sorted(hk)[0]] += 1
                bcat["textbook"] += 1
            elif km:
                m = sorted(km)[0]
                basis[f"{m} {mod_name.get(m, '')}"] += 1
                bcat["textbook"] += 1
            elif hk2 or prot:
                basis["housekeeping:" + (sorted(hk2)[0] if hk2 else "proteasome")] += 1
                bcat["housekeeping"] += 1
            cov_n += covered
            cov_pre += covered_pre
            if tagshare(tg, MOB) >= 0.5:
                mob_core = True
            annotated = float((~sub.unannot_any).mean()) >= 0.5
            ann_core += annotated
            if annotated:
                for pn, S in PROC.items():
                    if tagshare(tg, S) >= 0.5:
                        proc_core[pn] += 1
            # map line
            pos = crank[cc]
            o = np.array([orient[l] for l in sub.locus])
            rel_strand = sub.strand.to_numpy() * o
            sl_lab, _ = plur(sub.proposed_label_name.tolist())
            tier, _ = plur(sub[sub.proposed_label_name == sl_lab].tier.tolist()) if sl_lab else ("", 0)
            cl_rows.append(dict(family=f, core_pos=pos + 1, cluster=cc, presence=round(pres[cc], 3),
                                strand="+" if np.mean(rel_strand > 0) >= 0.5 else "-",
                                strand_agree=round(max(np.mean(rel_strand > 0), np.mean(rel_strand < 0)), 2),
                                median_aa=int(sub.length.median()), annotation=lab or "none",
                                annotation_share=round(lsh, 2), unannotated_share=round(float(sub.unannot_any.mean()), 2),
                                slot_label=sl_lab, slot_tier=tier, covered=covered,
                                cur_share=round(cur_share, 2), hyd=round(float(sub.hyd.median()), 3),
                                n_tm=float(sub.n_tm.median()), ko=ko, cdd=plur(sub.cdd_desc.tolist())[0][:70],
                                fam_hh=plur(sub.fam_hh_pfam.tolist())[0], tags=plur(sub.tags.tolist())[0][:80]))
        ncore = len(cons)
        unann = float(gg.unannot_any.mean())
        mob_share = float(gg.is_mob.mean())
        cov_share = cov_n / ncore if ncore else 0.0
        top_proc = proc_core.most_common(1)[0] if proc_core else ("", 0)
        coh = top_proc[1] / ann_core if ann_core else 0.0
        def classify(cv, kfrac):
            if ncore and cv / ncore >= kfrac:
                return "KNOWN"
            if cv >= 2 and (ncore - cv) >= 3:
                return "KNOWN-EXTENDED"
            if mob_share >= 0.10 or mob_core:
                return "MOBILE"
            if unann >= 0.5:
                return "DARK"
            if coh >= 0.5:
                return "COHERENT-UNNAMED"
            return "MIXED"
        cls, cls_pre = classify(cov_n, 0.5), classify(cov_pre, 2 / 3)
        basis_cat = bcat.most_common(1)[0][0] if bcat else ""
        curt = Counter(t for x in mem.cur_types for t in x.split(";") if t)
        domc = curt.most_common(1)[0] if curt else ("", 0)
        rows.append(dict(
            family=int(f), cls=cls, cls_predeclared=cls_pre, basis_cat=basis_cat, basis=";".join(f"{k}:{v}" for k, v in basis.most_common(3)),
            loci=nm, genomes=mem.genome.nunique(), groups=mem.group.nunique(), phyla=mem.phylum.nunique(),
            orders=mem.order.nunique(), top_phylum=plur(mem.phylum.tolist())[0], top_order=plur(mem.order.tolist())[0],
            phyla_list=";".join(f"{k}:{v}" for k, v in Counter(mem.phylum).most_common(4)),
            size_median=float(mem["size"].median()), size_min=int(mem["size"].min()), size_max=int(mem["size"].max()),
            gene_clusters=int(pres.size), core=ncore, core_presence_mean=round(float(pres[core].mean()), 3) if ncore else 0,
            variable=int((pres < 0.5).sum()), order_cons=round(order_cons, 3),
            core_covered=cov_n, core_annotated=ann_core, top_process=top_proc[0], process_coherence=round(coh, 3),
            unannotated_share=round(unann, 3), no_pfam_ko_share=round(float(gg.unannot_core3.mean()), 3),
            mobile_gene_share=round(mob_share, 3), mobile_loci_share=round(float((mem.n_mob > 0).mean()), 3),
            curated_loci_share=round(float((mem.cur_types != "").mean()), 3), curated_top=domc[0],
            curated_top_share=round(domc[1] / nm, 3),
            ribo_loci_share=round(float((mem.n_ribo >= 5).mean()), 3),
            hk_gene_share=round(tagshare(gg.tags.tolist(), HK), 3),
            edge_rate=round(float(mem.edge.mean()), 3), contig_genes_median=float(mem.contig_genes.median()),
            completeness_median=round(float(mem.completeness.median()), 1),
            contamination_median=round(float(mem.contamination.median()), 2),
            root_L4b_share=round(float(mem.root_in_L4b.mean()), 3),
            nested_ge10_share=round(float((mem.n_nested_ge10 > 0).mean()), 3),
            pfam_top=";".join(k for k, _ in Counter(x for x in gg.pfam.dropna() for x in x.split(";")).most_common(8)),
            example_locus=f"{mem.iloc[0].genome}:{gg[gg.locus == mem.iloc[0].locus].protein_id.iloc[0]}"
                          f"..{gg[gg.locus == mem.iloc[0].locus].protein_id.iloc[-1]}"))
    F = pd.DataFrame(rows)
    F["size_bin"] = pd.cut(F.size_median, [9, 14.5, 24.5, 1e9], labels=["10-14", "15-24", ">=25"])
    CL = pd.DataFrame(cl_rows)
    # ranking
    el = (F.cls != "KNOWN") & (F.groups >= 3) & (F.orders >= 2) & (F.loci >= 5) & (F.core >= 3)
    E = F[el].copy()
    E["ann_core_share"] = np.where(E.core > 0, E.core_annotated / E.core.clip(lower=1), 0)
    comp = ["groups", "orders", "size_median", "core_presence_mean", "order_cons", "ann_core_share"]
    E["score"] = E[comp].rank(pct=True).mean(axis=1)
    E = E.sort_values(["score", "groups"], ascending=False)
    F = F.merge(E[["family", "score"]], on="family", how="left")
    F["rank"] = F.score.rank(ascending=False, method="first")
    F = F.sort_values(["rank", "groups"], ascending=[True, False])
    F.to_csv(W + "/families.tsv", sep="\t", index=False)
    CL.to_csv(W + "/data/core_clusters.tsv", sep="\t", index=False)
    G[["gid", "protein_id", "genome", "locus", FAM, "gene_cluster", "length", "strand", "annot_label", "unannot_any",
       "proposed_label_name", "tier"]].to_parquet(W + "/data/family_genes.parquet", index=False)
    print(F.groupby(["cls", "size_bin"], observed=False).size().unstack())
    print(F[F.groups >= 3].groupby(["cls", "size_bin"], observed=False).size().unstack())
    print(E.head(25)[["family", "cls", "loci", "groups", "orders", "size_median", "core", "order_cons",
                      "unannotated_share", "top_process", "curated_top", "score"]].to_string())


if __name__ == "__main__":
    main(sys.argv[1])
