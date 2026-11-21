#!/usr/bin/env python
"""Workstream invisible, step 1: the profile-invisible enzyme population and the novelty filter.

Funnel (counts written to data/funnel_s01.json):
  labeled genes (pass table) -> no own weak profile hit to the proposed family (tiers T2, T2b, rest; `contradicted` is
  carried as a side population because its genes DO have weak profile hits, to other label families, sister families
  included) -> gene-level function from the gap screen (genes_function.parquet; partner-derived preferred, label-level
  fallback kept and flagged) -> function is an enzyme (KO with an EC) -> 100-600 aa and ILVFMAWC < 0.45 -> novelty filter.

Novelty filter: a gene is dropped when any annotation source in the Sharur DPANN database places it in the family or
its function. For function f (KO) with label L:
  KOSET(f)   = f plus every KO sharing a complete EC number with f (KEGG ko list, gap screen snapshot); plus L if L is a KO.
  PFSET(f,L) = L if L is a Pfam, plus every Pfam carried by >= 30% of the DPANN kofam carriers of f (>= 3 carriers).
  Sources:
   own        annotations rows of the gene itself: KO in KOSET (kofam / kegg), Pfam in PFSET (pfam), any KO with an EC of f.
   cdd        protein_resemblance (CDD best hits per region): CDD accession pfamNNNNN with PFNNNNN in PFSET; or a CDD model
              whose DPANN hits with a KO carry a KOSET KO in >= 50% (n >= 5), or whose DPANN hits with a Pfam carry a PFSET
              Pfam in >= 50% (n >= 5); or a CDD description containing an EC number of f.
   hhs_pfam   family_annotations hhsearch_pfam on the gene's protein family (permissive-e3c50s75): hit in PFSET, prob >= 50.
   hhb_pdb    family_annotations hhblits_pdb, prob >= 50, hit chain mapped by SIFTS to an EC of f or a Pfam in PFSET.
   famstats   family_stats of any of the gene's protein families (5 clustering runs): dominant KO in KOSET with
              dominant_ko_fraction >= 0.5, or dominant Pfam architecture containing a PFSET Pfam name with
              pfam_fraction * dominant_pfam_architecture_fraction >= 0.5.
Writes data/population.parquet (one row per gene that reaches the composition step, with novelty flags),
data/novel.parquet (survivors), data/funnel_s01.json, data/sets_per_function.tsv."""
import gzip, json, re
from pathlib import Path
import duckdb
import pandas as pd

R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign")
H = R / "dpann_leads/invisible"
DB = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
c = duckdb.connect()
c.execute(f"ATTACH '{DB}' AS d (READ_ONLY)")
c.execute("SET threads=4")

P = pd.read_parquet(R / "slot_rescue/pass/dpann_slot_rescue_pass.parquet", columns=["protein_id"])
V = pd.read_parquet(R / "slot_rescue/verify/verification_per_protein.parquet", columns=["protein_id", "tier"])
G = pd.read_parquet(R / "dpann_leads/gapscreen/data/genes_function.parquet")
K = pd.read_csv(R / "dpann_leads/gapscreen/data/kegg_ko.tsv", sep="\t", keep_default_na=False)
fun = {}
TV = V.set_index("protein_id").tier
fun["labeled_genes"] = len(P)
inv = {"T2", "T2b", "rest"}
fun["tier_counts_all"] = V.tier.value_counts().to_dict()
fun["no_own_hit_T2_T2b_rest"] = int(V.tier.isin(inv).sum())
fun["contradicted_side_population"] = int((V.tier == "contradicted").sum())
g = G[G.tier.isin(inv | {"contradicted"})].copy()
step = lambda df, key: {t: int((df.tier == t).sum()) for t in ["T2", "T2b", "rest", "contradicted"]}
fun["with_gene_function"] = step(g, "")
fun["with_gene_function_by_source"] = g[g.tier.isin(inv)].func_source.value_counts().to_dict()
g = g[g.enzyme]
fun["enzyme"] = step(g, "")
g = g[(g.length >= 100) & (g.length <= 600) & (g.ilvfmawc < 0.45)].copy()
fun["len100_600_soluble"] = step(g, "")
print(json.dumps(fun, indent=1))

# ---- per-function sets
ec_of = {r.ko: [e for e in r.ec.split() if e and "-" not in e] for r in K.itertuples()}
ko_of_ec = {}
for k, es in ec_of.items():
    for e in es:
        ko_of_ec.setdefault(e, set()).add(k)
funcs = sorted(g.func.unique())
c.register("fk", pd.DataFrame({"ko": funcs}))
carr = c.execute("""
  WITH car AS (SELECT DISTINCT a.protein_id, a.accession ko FROM d.annotations a JOIN fk ON a.accession = fk.ko
               WHERE a.source IN ('kofam','kegg'))
  SELECT car.ko, count(DISTINCT car.protein_id) n_car, p.accession pf, any_value(p.name) pfname, count(DISTINCT p.protein_id) n_pf
  FROM car LEFT JOIN d.annotations p ON p.protein_id = car.protein_id AND p.source = 'pfam'
  GROUP BY car.ko, p.accession""").fetchdf()
pfname = dict(c.execute("SELECT accession, any_value(name) FROM d.annotations WHERE source='pfam' GROUP BY 1").fetchall())
KOSET, PFSET, ECSET = {}, {}, {}
for f in funcs:
    s = {f}
    for e in ec_of.get(f, []):
        s |= ko_of_ec.get(e, set())
    KOSET[f] = s
    ECSET[f] = set(ec_of.get(f, []))
    sub = carr[(carr.ko == f) & carr.pf.notna()]
    ncar = int(carr[carr.ko == f].n_car.max()) if (carr.ko == f).any() else 0
    PFSET[f] = set(sub[(sub.n_pf >= 0.3 * ncar) & (ncar >= 3)].pf)
rows = []
for r in g[["func", "proposed_label_positional_homology", "proposed_label_kind"]].drop_duplicates().itertuples(index=False):
    f, L, kind = r
    ks, ps = set(KOSET[f]), set(PFSET[f])
    if kind == "pfam":
        ps.add(L)
    else:
        ks.add(L)
    rows.append(dict(func=f, label=L, koset=" ".join(sorted(ks)), pfset=" ".join(sorted(ps)), ecset=" ".join(sorted(ECSET[f]))))
S = pd.DataFrame(rows)
S.to_csv(H / "data/sets_per_function.tsv", sep="\t", index=False)
SK = {(r.func, r.label): set(r.koset.split()) for r in S.itertuples()}
SP = {(r.func, r.label): set(r.pfset.split()) for r in S.itertuples()}
SE = {(r.func, r.label): set(r.ecset.split()) for r in S.itertuples()}

c.register("gg", g[["protein_id"]])
# own annotations
own = c.execute("SELECT a.protein_id, a.source, a.accession FROM d.annotations a JOIN gg USING(protein_id)").fetchdf()
# CDD hits and the CDD -> KO / Pfam association over all of DPANN
cdd = c.execute("SELECT r.protein_id, r.accession, r.name nm, r.description, r.evalue FROM d.protein_resemblance r JOIN gg USING(protein_id)").fetchdf()
c.register("cm", pd.DataFrame({"acc": sorted(cdd.accession.unique())}))
cdd_ko = c.execute("""SELECT r.accession acc, a.accession ko, count(DISTINCT r.protein_id) n FROM d.protein_resemblance r
  JOIN cm ON r.accession = cm.acc JOIN d.annotations a ON a.protein_id = r.protein_id AND a.source IN ('kofam','kegg') GROUP BY 1,2""").fetchdf()
cdd_kon = c.execute("""SELECT r.accession acc, count(DISTINCT r.protein_id) n FROM d.protein_resemblance r JOIN cm ON r.accession = cm.acc
  WHERE EXISTS (SELECT 1 FROM d.annotations a WHERE a.protein_id = r.protein_id AND a.source IN ('kofam','kegg')) GROUP BY 1""").fetchdf()
cdd_pf = c.execute("""SELECT r.accession acc, a.accession pf, count(DISTINCT r.protein_id) n FROM d.protein_resemblance r
  JOIN cm ON r.accession = cm.acc JOIN d.annotations a ON a.protein_id = r.protein_id AND a.source = 'pfam' GROUP BY 1,2""").fetchdf()
cdd_pfn = c.execute("""SELECT r.accession acc, count(DISTINCT r.protein_id) n FROM d.protein_resemblance r JOIN cm ON r.accession = cm.acc
  WHERE EXISTS (SELECT 1 FROM d.annotations a WHERE a.protein_id = r.protein_id AND a.source = 'pfam') GROUP BY 1""").fetchdf()
kon = dict(zip(cdd_kon.acc, cdd_kon.n)); pfn = dict(zip(cdd_pfn.acc, cdd_pfn.n))
cko = cdd_ko.groupby("acc").apply(lambda d: dict(zip(d.ko, d.n))).to_dict()
cpf = cdd_pf.groupby("acc").apply(lambda d: dict(zip(d.pf, d.n))).to_dict()
# family annotations (HHsearch Pfam, HHblits PDB100) through permissive-e3c50s75 families
fam = c.execute("""SELECT pf.protein_id, fa.source, fa.hit_accession, fa.probability FROM d.protein_families pf JOIN gg USING(protein_id)
  JOIN d.family_annotations fa ON fa.family_id = pf.family_id AND fa.clustering_run = pf.clustering_run
  WHERE pf.clustering_run = 'permissive-e3c50s75' AND fa.probability >= 50""").fetchdf()
fs = c.execute("""SELECT pf.protein_id, s.clustering_run, s.dominant_ko, s.dominant_ko_fraction, s.dominant_pfam_architecture,
  s.pfam_fraction, s.dominant_pfam_architecture_fraction FROM d.protein_families pf JOIN gg USING(protein_id)
  JOIN d.family_stats s ON s.family_id = pf.family_id AND s.clustering_run = pf.clustering_run""").fetchdf()
sec = pd.read_csv(H / "ref/sifts/pdb_chain_enzyme.tsv", sep="\t", comment="#", dtype=str)
spf = pd.read_csv(H / "ref/sifts/pdb_chain_pfam.tsv", sep="\t", comment="#", dtype=str)
chain_ec = sec.groupby(sec.PDB.str.lower() + "_" + sec.CHAIN).EC_NUMBER.apply(set).to_dict()
chain_pf = spf.groupby(spf.PDB.str.lower() + "_" + spf.CHAIN).PFAM_ID.apply(set).to_dict()

own_g = own.groupby("protein_id"); cdd_g = cdd.groupby("protein_id"); fam_g = fam.groupby("protein_id"); fs_g = fs.groupby("protein_id")
grp = lambda gb, k: gb.get_group(k) if k in gb.groups else None
flags = []
for r in g.itertuples(index=False):
    key = (r.func, r.proposed_label_positional_homology)
    ks, ps, es = SK[key], SP[key], SE[key]
    psn = {pfname.get(p, p) for p in ps}
    fl = dict(protein_id=r.protein_id, nov_own=False, nov_cdd=False, nov_hhs_pfam=False, nov_hhb_pdb=False, nov_famstats=False,
              nov_detail="")
    det = []
    o = grp(own_g, r.protein_id)
    if o is not None:
        for x in o.itertuples():
            acc = x.accession
            hit = (x.source in ("kofam", "kegg") and (acc in ks or set(ec_of.get(acc, [])) & es)) or (x.source == "pfam" and acc in ps)
            if hit:
                fl["nov_own"] = True; det.append(f"own:{x.source}:{acc}")
    o = grp(cdd_g, r.protein_id)
    if o is not None:
        for x in o.itertuples():
            a = x.accession
            why = None
            if a.startswith("pfam") and "PF" + a[4:] in ps:
                why = "pfam_model"
            elif kon.get(a, 0) >= 5 and sum(v for k2, v in cko.get(a, {}).items() if k2 in ks) >= 0.5 * kon[a]:
                why = "ko_assoc"
            elif pfn.get(a, 0) >= 5 and max([v for p2, v in cpf.get(a, {}).items() if p2 in ps] + [0]) >= 0.5 * pfn[a]:
                why = "pf_assoc"
            elif any(re.search(r"(?<![\d.])" + re.escape(e) + r"(?![\d])", str(x.description)) for e in es):
                why = "ec_text"
            if why:
                fl["nov_cdd"] = True; det.append(f"cdd:{a}:{why}")
    o = grp(fam_g, r.protein_id)
    if o is not None:
        for x in o.itertuples():
            if x.source == "hhsearch_pfam" and x.hit_accession.split(".")[0] in ps:
                fl["nov_hhs_pfam"] = True; det.append(f"hhs:{x.hit_accession}:{x.probability:.0f}")
            elif x.source == "hhblits_pdb":
                ch = x.hit_accession.lower()
                if (chain_ec.get(ch, set()) & es) or (chain_pf.get(ch, set()) & ps):
                    fl["nov_hhb_pdb"] = True; det.append(f"pdb:{x.hit_accession}:{x.probability:.0f}")
    o = grp(fs_g, r.protein_id)
    if o is not None:
        for x in o.itertuples():
            if x.dominant_ko in ks and (x.dominant_ko_fraction or 0) >= 0.5:
                fl["nov_famstats"] = True; det.append(f"fs:{x.clustering_run}:{x.dominant_ko}")
            elif x.dominant_pfam_architecture and (x.pfam_fraction or 0) * (x.dominant_pfam_architecture_fraction or 0) >= 0.5 \
                    and set(s.strip() for s in str(x.dominant_pfam_architecture).split(" - ")) & psn:
                fl["nov_famstats"] = True; det.append(f"fs:{x.clustering_run}:arch")
    fl["nov_detail"] = ";".join(sorted(set(det)))[:500]
    flags.append(fl)
F = pd.DataFrame(flags)
pop = g.merge(F, on="protein_id")
nov_cols = ["nov_own", "nov_cdd", "nov_hhs_pfam", "nov_hhb_pdb", "nov_famstats"]
pop["placed_any"] = pop[nov_cols].any(axis=1)
pop["n_cdd_hits"] = pop.protein_id.map(cdd.groupby("protein_id").size()).fillna(0).astype(int)
pop["list_type"] = pop.alignment_lists_same_label.map({"yes": "both", "no": "lists_disagree", "no call": "embedding_only",
                                                       "no list": "no_alignment_list"})
pop["support"] = pop.votes_for / pop.supporting_genomes
pop.to_parquet(H / "data/population.parquet", index=False)
fun["novelty_placed_by_source"] = {k: {t: int(pop[(pop.tier == t)][k].sum()) for t in ["T2", "T2b", "rest", "contradicted"]} for k in nov_cols}
fun["novelty_placed_any"] = {t: int(pop[(pop.tier == t)].placed_any.sum()) for t in ["T2", "T2b", "rest", "contradicted"]}
nv = pop[~pop.placed_any]
fun["novel_unannotated"] = step(nv, "")
inv_nv = nv[nv.tier.isin(inv)]
fun["novel_main_genes"] = int(len(inv_nv))
fun["novel_main_functions"] = int(inv_nv.func.nunique())
fun["novel_main_by_list_type"] = inv_nv.list_type.value_counts().to_dict()
fun["novel_main_tier_x_list_type"] = {f"{a}|{b}": int(n) for (a, b), n in inv_nv.groupby(["tier", "list_type"]).size().items()}
fun["novel_main_by_func_source"] = inv_nv.func_source.value_counts().to_dict()
nv.to_parquet(H / "data/novel.parquet", index=False)
json.dump(fun, open(H / "data/funnel_s01.json", "w"), indent=1)
print(json.dumps({k: fun[k] for k in list(fun)[6:]}, indent=1))
