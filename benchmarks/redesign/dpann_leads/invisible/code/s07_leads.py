#!/usr/bin/env python
"""Step 7: context for every gene that passes the assay in an informative family (leads.tsv, one row per passing gene).
Per gene: residues matched, TM, pLDDT, clade, related labeled genes (same function: all labeled genes, the novel
population, its 90% cluster), whether the genome has the function otherwise by any source (other proteins: KO in KOSET
incl. KOs sharing the EC, i.e. non-homologous enzymes for the same reaction; Pfam in PFSET; CDD placement by Pfam model,
KEGG-linked COG or enzyme name in the CDD name / first sentence of its description), KEGG module steps present in the genome, +-5 gene neighbors, completeness vs the clade,
criterion A (ILVFMAWC, length, SUBUNIT comment of the curated relative). Writes leads.tsv, data/lead_neighbors.tsv."""
import json, re
from pathlib import Path
import duckdb
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/invisible")
G = H / "../gapscreen/data"
O = pd.read_csv(H / "data/readout_full.tsv", sep="\t")
RES = pd.read_csv(H / "data/residues_full.tsv", sep="\t", keep_default_na=False)
st = json.load(open(H / "data/stats.json"))
inf = set(st["informative_families"])
PA = pd.read_parquet(H / "data/population_amended.parquet")
GF = pd.read_parquet(G / "genes_function.parquet", columns=["protein_id", "func", "tier", "genome", "order"])
RC = pd.read_csv(H / "data/candidate_ranking.tsv", sep="\t")
REFS = pd.read_csv(H / "data/references_testable.tsv", sep="\t").set_index("func")
S = pd.read_csv(H / "data/sets_per_function.tsv", sep="\t", keep_default_na=False)
NM = json.load(open(H / "data/function_names.json"))
K = pd.read_csv(G / "kegg_ko.tsv", sep="\t", keep_default_na=False).set_index("ko")
MODS = pd.read_csv(G / "kegg_modules.tsv", sep="\t", keep_default_na=False)
norm = lambda s: " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip() + " "
c = duckdb.connect()
c.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
c.execute("SET threads=4")
P = O[(O.role == "candidate") & (O.pair == "own") & (O.PASS == True) & O.func.isin(inf)].copy()  # noqa: E712
P = P.merge(PA, on=["protein_id", "func"], how="left")
bins = c.execute("SELECT bin_id, completeness, taxonomy FROM d.bins").fetchdf().set_index("bin_id")
bins["order"] = bins.taxonomy.str.extract(r"o__([^;]*)")[0]
bins["genus"] = bins.taxonomy.str.extract(r"g__([^;]*)")[0]
rows, nb_rows = [], []
for r in P.itertuples():
    f, gid = r.func, r.protein_id
    srow = S[(S.func == f) & (S.label == r.proposed_label_positional_homology)].iloc[0]
    ks, ps, es = set(srow.koset.split()), set(srow.pfset.split()), set(srow.ecset.split())
    cogs = {cg for k in ks for cg in NM["cog"].get(k, [])}
    names = {norm(re.sub(r"\[EC:[^\]]*\]", "", K.name.get(k, ""))) for k in ks if k in K.index}
    for e in es:
        names |= {norm(x) for x in NM["ec_names"].get(e, [])}
    names = {n for n in names if len(n.strip()) >= 12 and len(n.split()) >= 2}
    b = r.genome
    gb = c.execute(f"SELECT protein_id, contig_id, gene_index, sequence FROM d.proteins WHERE bin_id = '{b}'").fetchdf()
    c.register("gb", gb[["protein_id"]])
    ann = c.execute("SELECT a.protein_id, a.source, a.accession, a.name FROM d.annotations a JOIN gb USING(protein_id)").fetchdf()
    cdd = c.execute("SELECT r.protein_id, r.accession, r.name nm, r.description FROM d.protein_resemblance r JOIN gb USING(protein_id)").fetchdf()
    other = []
    for x in ann[ann.protein_id != gid].itertuples():
        if x.source in ("kofam", "kegg") and x.accession in ks:
            other.append(f"{x.protein_id}:{x.accession}")
        elif x.source == "pfam" and x.accession in ps:
            other.append(f"{x.protein_id}:{x.accession}")
    for x in cdd[cdd.protein_id != gid].itertuples():
        txt = norm(f"{x.nm} {re.split(r'[.;:] ', str(x.description))[0][:200]}")   # name + first sentence (context only)
        if (x.accession.startswith("pfam") and "PF" + x.accession[4:] in ps) or x.accession in cogs or any(n in txt for n in names):
            other.append(f"{x.protein_id}:{x.accession}")
    gko = set(ann[ann.source.isin(["kofam", "kegg"])].accession)
    mods = []
    for m in MODS.itertuples():
        kk = set(re.findall(r"K\d{5}", m.definition))
        if f in kk:
            pres = sorted(kk & gko)
            mods.append(f"{m.module} {len(kk & gko)}/{len(kk)} ({' '.join(pres)})")
    me = gb[gb.protein_id == gid].iloc[0]
    nb = gb[(gb.contig_id == me.contig_id) & ((gb.gene_index - me.gene_index).abs() <= 5) & (gb.protein_id != gid)].sort_values("gene_index")
    nbtxt = []
    for x in nb.itertuples():
        a = ann[ann.protein_id == x.protein_id]
        lab = ",".join(sorted(set(a[a.source == "pfam"].name.dropna())))[:40] or ",".join(sorted(set(a[a.source == "kofam"].accession)))[:40]
        if not lab:
            cc = cdd[cdd.protein_id == x.protein_id]
            lab = ("cdd:" + ",".join(cc.nm.astype(str))[:40]) if len(cc) else "-"
        nbtxt.append(f"{x.gene_index - me.gene_index:+d}:{lab}")
        nb_rows.append(dict(lead=gid, offset=x.gene_index - me.gene_index, neighbor=x.protein_id, annotation=lab))
    rr = RES[(RES.pair_id == r.pair_id)]
    cat = rr[rr.feature == "Active site"]
    cats = " ".join(f"{x.ref_residue}{x.uniprot_pos}->{x.model_residue}{x.model_pos}" for x in cat.itertuples())
    rel_all = GF[GF.func == f]
    rel_nov = PA[(PA.func == f) & ~PA.placed_any & PA.tier.isin(["T2", "T2b", "rest"])]
    order = bins.loc[b, "order"]
    clade_g = int((bins.order == order).sum())
    seq = me.sequence.rstrip("*")
    j = json.load(open(H / f"ref/{REFS.loc[f, 'ref_acc']}.json"))
    sub = " ".join(t["value"] for cm in j.get("comments", []) if cm.get("commentType") == "SUBUNIT" for t in cm.get("texts", []))
    rows.append(dict(lead=gid, family_label=r.proposed_label_name, function=f"{f} {K.name.get(f, '')[:60]}", clade=order,
                     genus=bins.loc[b, "genus"], genome=b, genes=int(rel_all[rel_all.order == order].shape[0]),
                     genomes=int(rel_all[rel_all.order == order].genome.nunique()), clade_genomes=clade_g,
                     median_length=len(seq), tier_T1=0, tier=r.tier, support_median=round(r.support, 3),
                     votes=f"{r.votes_for}/{r.supporting_genomes}", catalytic_residues_ok=cats, binding=r.binding,
                     structure_TM_to_reference=round(r.TM_ref, 3), TM_by_model=round(r.TM_model, 3), site_plddt=r.site_plddt,
                     mean_plddt=round(r.mean_plddt, 1), reference=f"{REFS.loc[f, 'ref_acc']} {REFS.loc[f, 'pdb_chain']}",
                     function_absent_otherwise=len(other) == 0, function_other_sources=";".join(other)[:200],
                     embedding_only_share=1.0 if r.list_type == "embedding_only" else 0.0, list_type=r.list_type, func_source=r.func_source,
                     named_by_amendment=bool(r.placed_any), amendment_detail=r.nov_detail2 if r.placed_any else "",
                     related_labeled_all_tiers=f"{len(rel_all)} genes / {rel_all.genome.nunique()} genomes",
                     related_novel=f"{len(rel_nov)} genes / {rel_nov.genome.nunique()} genomes",
                     cluster90_size=int(RC[(RC.protein_id == gid)].cluster_size.iloc[0]),
                     completeness=round(float(bins.loc[b, "completeness"]), 1),
                     clade_completeness_median=round(float(bins[bins.order == order].completeness.median()), 1),
                     kegg_modules=" | ".join(mods), neighbors=" ".join(nbtxt), ilvfmawc=round(sum(ch in "ILVFMAWC" for ch in seq) / len(seq), 3),
                     relative_subunit=sub[:160], verdict="PASS", one_line_reason=""))
L = pd.DataFrame(rows)
L.to_csv(H / "leads.tsv", sep="\t", index=False)
pd.DataFrame(nb_rows).to_csv(H / "data/lead_neighbors.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(L[["lead", "function", "clade", "TM_by_model", "structure_TM_to_reference", "catalytic_residues_ok", "binding", "site_plddt",
         "function_absent_otherwise", "list_type", "named_by_amendment", "completeness"]].to_string(index=False))
