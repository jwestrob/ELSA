#!/usr/bin/env python
"""Workstream invisible, step 1b: amendment of the novelty filter (written after the first full-run folds were requested,
before any full-run comparison was scored). The first pass recognized CDD placements only through Pfam-derived models, an
EC string or the DPANN KO / Pfam association of the model. It missed CDD models that name the function in words with few
annotated DPANN hits (COG1085 'Galactose-1-phosphate uridylyltransferase' hits 47 of the 64 GalP_UDP_transf-labeled novel
genes, E down to 3.6e-35). Added sources:
  cdd_cog   CDD COG model listed in the KEGG KO DBLINKS (COG) of any KO in KOSET (KEGG REST, by KO accession).
  cdd_name  CDD model whose name + description (first 400 characters, normalized) contains an enzyme name of the function:
            KEGG KO name, or a recommended / alternative name (>= 12 characters, >= 2 words) of a reviewed UniProt entry with
            an EC of the function (UniProt REST, by EC).
  pdb_name  HHblits PDB100 hit (probability >= 50) on the gene's protein family whose description contains such a name.
Writes data/population_amended.parquet, data/novel_amended.parquet, data/funnel_amended.json, data/function_names.json."""
import json, re, time, urllib.parse, urllib.request
from pathlib import Path
import duckdb
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/invisible")
pop = pd.read_parquet(H / "data/population.parquet")
S = pd.read_csv(H / "data/sets_per_function.tsv", sep="\t", keep_default_na=False)
K = pd.read_csv(H / "../gapscreen/data/kegg_ko.tsv", sep="\t", keep_default_na=False).set_index("ko")
norm = lambda s: " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip() + " "


def get(url):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-urllib"}), timeout=60) as r:
                return r.read().decode()
        except Exception as e:
            if "404" in str(e):
                return ""
            time.sleep(2 + 3 * i)
    return ""


cache = H / "data/function_names.json"
NM = json.load(open(cache)) if cache.exists() else {"cog": {}, "ec_names": {}}
kos = sorted(set(" ".join(S.koset).split()))
for k in kos:
    if k in NM["cog"]:
        continue
    t = get(f"https://rest.kegg.jp/get/{k}")
    cogs = re.findall(r"COG\d{4}", t.split("DBLINKS", 1)[1].split("\nGENES")[0]) if "DBLINKS" in t else []
    NM["cog"][k] = sorted(set(cogs))
    time.sleep(0.15)
ecs = sorted(set(" ".join(S.ecset).split()))
for e in ecs:
    if e in NM["ec_names"]:
        continue
    t = get("https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
        dict(query=f"ec:{e} AND reviewed:true", fields="protein_name", format="tsv", size=500)))
    names = set()
    for line in t.splitlines()[1:]:
        parts = [line.split(" (")[0]] + re.findall(r"\(([^()]+)\)", line)
        for p in parts:
            p = p.strip()
            if p.startswith("EC ") or len(p) < 12 or len(p.split()) < 2:
                continue
            names.add(p)
    NM["ec_names"][e] = sorted(names)
json.dump(NM, open(cache, "w"), indent=0)

c = duckdb.connect()
c.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
c.register("gg", pop[["protein_id"]])
cdd = c.execute("SELECT r.protein_id, r.accession, r.name nm, r.description FROM d.protein_resemblance r JOIN gg USING(protein_id)").fetchdf()
pdb = c.execute("""SELECT pf.protein_id, fa.hit_accession, fa.hit_description, fa.probability FROM d.protein_families pf JOIN gg USING(protein_id)
  JOIN d.family_annotations fa ON fa.family_id = pf.family_id AND fa.clustering_run = pf.clustering_run
  WHERE pf.clustering_run = 'permissive-e3c50s75' AND fa.source = 'hhblits_pdb' AND fa.probability >= 50""").fetchdf()
cdd_g, pdb_g = cdd.groupby("protein_id"), pdb.groupby("protein_id")
names_of, cogs_of = {}, {}
for r in S.itertuples():
    ks, es = r.koset.split(), r.ecset.split()
    nms = {norm(re.sub(r"\[EC:[^\]]*\]", "", K.name.get(k, ""))) for k in ks if k in K.index}
    for e in es:
        nms |= {norm(x) for x in NM["ec_names"].get(e, [])}
    names_of[(r.func, r.label)] = {n for n in nms if len(n.strip()) >= 12 and len(n.split()) >= 2}
    cogs_of[(r.func, r.label)] = {cg for k in ks for cg in NM["cog"].get(k, [])}
fl = []
for r in pop.itertuples(index=False):
    key = (r.func, r.proposed_label_positional_homology)
    nms, cogs = names_of[key], cogs_of[key]
    rec = dict(protein_id=r.protein_id, nov_cdd_cog=False, nov_cdd_name=False, nov_pdb_name=False, nov_detail2="")
    det = []
    if r.protein_id in cdd_g.groups:
        for x in cdd_g.get_group(r.protein_id).itertuples():
            if x.accession in cogs:
                rec["nov_cdd_cog"] = True; det.append(f"cog:{x.accession}")
            txt = norm(f"{x.nm} {str(x.description)[:400]}")
            hit = next((n for n in nms if n in txt), None)
            if hit:
                rec["nov_cdd_name"] = True; det.append(f"cddname:{x.accession}:{hit.strip()[:40]}")
    if r.protein_id in pdb_g.groups:
        for x in pdb_g.get_group(r.protein_id).itertuples():
            txt = norm(x.hit_description)
            hit = next((n for n in nms if n in txt), None)
            if hit:
                rec["nov_pdb_name"] = True; det.append(f"pdbname:{x.hit_accession}:{hit.strip()[:40]}")
    rec["nov_detail2"] = ";".join(sorted(set(det)))[:400]
    fl.append(rec)
A = pop.merge(pd.DataFrame(fl), on="protein_id")
A["placed_any_v1"] = A.placed_any
A["placed_any"] = A.placed_any_v1 | A.nov_cdd_cog | A.nov_cdd_name | A.nov_pdb_name
A.to_parquet(H / "data/population_amended.parquet", index=False)
nv = A[~A.placed_any]
nv.to_parquet(H / "data/novel_amended.parquet", index=False)
tiers = ["T2", "T2b", "rest", "contradicted"]
fun = json.load(open(H / "data/funnel_s01.json"))
fun["amendment_placed_by_new_source"] = {k: {t: int(A[A.tier == t][k].sum()) for t in tiers} for k in ["nov_cdd_cog", "nov_cdd_name", "nov_pdb_name"]}
fun["amendment_newly_placed"] = {t: int((A[A.tier == t].placed_any & ~A[A.tier == t].placed_any_v1).sum()) for t in tiers}
fun["novel_unannotated_amended"] = {t: int((nv.tier == t).sum()) for t in tiers}
m = nv[nv.tier.isin(["T2", "T2b", "rest"])]
fun["novel_main_genes_amended"] = int(len(m))
fun["novel_main_functions_amended"] = int(m.func.nunique())
fun["novel_main_by_list_type_amended"] = m.list_type.value_counts().to_dict()
fun["novel_main_tier_x_list_type_amended"] = {f"{a}|{b}": int(n) for (a, b), n in m.groupby(["tier", "list_type"]).size().items()}
fun["novel_main_by_func_source_amended"] = m.func_source.value_counts().to_dict()
json.dump(fun, open(H / "data/funnel_amended.json", "w"), indent=1)
print(json.dumps({k: v for k, v in fun.items() if "amend" in k}, indent=1))
print(m.groupby("func").size().sort_values(ascending=False).head(25).to_string())
