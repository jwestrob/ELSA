#!/usr/bin/env python
"""Leads pipeline step 2-3 (local): population and novelty split for one collection.
usage: L2_population.py COLLECTION
Inputs (copied from the remote leads folder by L0_sync.py): data/genes_function.parquet, data/pfset.tsv, data/funnel_r1.json,
data/novelty_{pfam,kofam,cdd}.tsv. KEGG snapshot: dpann_leads/gapscreen/data/kegg_ko.tsv.
Population: tiers T2, T2b, rest (T1c and contradicted carried as side populations, reported, not assayed), STRONG partner
function with an EC, 100-600 aa, ILVFMAWC < 0.45.
For function f with label L: KOSET = f + KOs sharing a complete EC with f; PFSET = L + Pfams carried by >= 30% of the
collection's KOfam carriers of f (partner KOfam table). A gene is 'already named' when
  own_pfam   a relaxed Aksha Pfam-A hit (E <= 0.01, no GA) is in PFSET;
  own_kofam  a relaxed Aksha KOfam hit (E <= 0.01, no ko_list threshold) is in KOSET;
  cdd        a CDD best-per-region hit (rpsblast E <= 0.01) is a Pfam-derived model in PFSET, a COG listed in the KEGG
             DBLINKS of a KOSET KO, carries an enzyme name of f in its name + description (first 400 characters: KEGG KO
             names, reviewed UniProt names for the EC; >= 12 characters, >= 2 words), or an EC number of f.
These are the DPANN sources available here; the DPANN family-level sources (HHsearch / HHblits family annotations,
family_stats) and the DPANN-wide CDD association rule have no counterpart in this collection.
Writes data/population.parquet, data/sets_per_function.tsv, data/function_names.json, funnel.tsv."""
import json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd

C = sys.argv[1]
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CFG = json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
D = H / "data"
K = pd.read_csv(R / "benchmarks/redesign/dpann_leads/gapscreen/data/kegg_ko.tsv", sep="\t", keep_default_na=False).set_index("ko")
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


F = pd.read_parquet(D / "genes_function.parquet")
F["protein_id"] = F.protein_id.astype(str)
tiers = ["T2", "T2b", "rest", "T1c", "contradicted"]
fun = []
step = lambda name, d: fun.append(dict(step=name, **{t: int((d.tier == t).sum()) for t in tiers},
                                       main_T2_T2b_rest=int(d.tier.isin(["T2", "T2b", "rest"]).sum())))
step("labeled genes", F)
F["has_carrier_ko"] = F.n_carriers_ko2 > 0
step("label carriers with a KOfam KO", F[F.has_carrier_ko])
step("STRONG partner-derived function", F[F.strong_partner])
step("STRONG enzyme (KO with EC)", F[F.strong_partner & F.enzyme])
pop = F[F.strong_partner & F.enzyme & F.seq_len.between(100, 600) & (F.ilvfmawc < 0.45) & F.tier.isin(tiers)].copy()
step("100-600 aa and ILVFMAWC < 0.45", pop)
pop["func"] = pop.ko_top
ec_of = {k: [e for e in str(r.ec).split() if e and "-" not in e] for k, r in K.iterrows()}
ko_of_ec = {}
for k, es in ec_of.items():
    for e in es:
        ko_of_ec.setdefault(e, set()).add(k)
PS = pd.read_csv(D / "pfset.tsv", sep="\t")
pfset_ko = PS.groupby("ko").pf.apply(set).to_dict()
rows = []
for (f, L), _ in pop.groupby(["func", "proposed_label_positional_homology"]):
    ks = {f} | {k2 for e in ec_of.get(f, []) for k2 in ko_of_ec.get(e, set())}
    ps = {L} | pfset_ko.get(f, set())
    rows.append(dict(func=f, label=L, koset=" ".join(sorted(ks)), pfset=" ".join(sorted(ps)), ecset=" ".join(sorted(ec_of.get(f, [])))))
S = pd.DataFrame(rows)
S.to_csv(D / "sets_per_function.tsv", sep="\t", index=False)
# names and COGs (cache; DPANN cache reused for entries it already holds)
cache = D / "function_names.json"
NM = json.load(open(cache)) if cache.exists() else {"cog": {}, "ec_names": {}}
dp = R / "benchmarks/redesign/dpann_leads/invisible/data/function_names.json"
if dp.exists():
    d0 = json.load(open(dp))
    for k in ("cog", "ec_names"):
        for a, b in d0[k].items():
            NM[k].setdefault(a, b)
for k in sorted(set(" ".join(S.koset).split())):
    if k not in NM["cog"]:
        t = get(f"https://rest.kegg.jp/get/{k}")
        NM["cog"][k] = sorted(set(re.findall(r"COG\d{4}", t.split("DBLINKS", 1)[1].split("\nGENES")[0]))) if "DBLINKS" in t else []
        time.sleep(0.15)
for e in sorted(set(" ".join(S.ecset).split())):
    if e not in NM["ec_names"]:
        t = get("https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
            dict(query=f"ec:{e} AND reviewed:true", fields="protein_name", format="tsv", size=500)))
        names = set()
        for line in t.splitlines()[1:]:
            for p in [line.split(" (")[0]] + re.findall(r"\(([^()]+)\)", line):
                p = p.strip()
                if not p.startswith("EC ") and len(p) >= 12 and len(p.split()) >= 2:
                    names.add(p)
        NM["ec_names"][e] = sorted(names)
json.dump(NM, open(cache, "w"), indent=0)
SK = {(r.func, r.label): set(r.koset.split()) for r in S.itertuples()}
SP = {(r.func, r.label): set(r.pfset.split()) for r in S.itertuples()}
SE = {(r.func, r.label): set(r.ecset.split()) for r in S.itertuples()}
NAMES, COGS = {}, {}
for key, ks in SK.items():
    nms = {norm(re.sub(r"\[EC:[^\]]*\]", "", K.name.get(k, ""))) for k in ks if k in K.index}
    for e in SE[key]:
        nms |= {norm(x) for x in NM["ec_names"].get(e, [])}
    NAMES[key] = {n for n in nms if len(n.strip()) >= 12 and len(n.split()) >= 2}
    COGS[key] = {cg for k in ks for cg in NM["cog"].get(k, [])}
rd = lambda f: pd.read_csv(D / f, sep="\t", dtype={"gid": str}, keep_default_na=False)
NP, NK, NC = rd("novelty_pfam.tsv"), rd("novelty_kofam.tsv"), rd("novelty_cdd.tsv")
gp, gk, gc = NP.groupby("gid"), NK.groupby("gid"), NC.groupby("gid")
fl = []
for r in pop.itertuples(index=False):
    g = str(r.gid); key = (r.func, r.proposed_label_positional_homology)
    ks, ps, es, nms, cogs = SK[key], SP[key], SE[key], NAMES[key], COGS[key]
    rec = dict(gid=int(r.gid), nov_own_pfam=False, nov_own_kofam=False, nov_cdd=False, nov_detail="",
               any_pfam_relaxed=g in gp.groups, any_kofam_relaxed=g in gk.groups, n_cdd=len(gc.get_group(g)) if g in gc.groups else 0)
    det = []
    if g in gp.groups:
        for x in gp.get_group(g).itertuples():
            if x.pf in ps:
                rec["nov_own_pfam"] = True; det.append(f"pfam:{x.pfname}:{float(x.evalue):.1e}")
    if g in gk.groups:
        for x in gk.get_group(g).itertuples():
            if x.ko in ks:
                rec["nov_own_kofam"] = True; det.append(f"kofam:{x.ko}:{float(x.evalue):.1e}")
    if g in gc.groups:
        for x in gc.get_group(g).itertuples():
            a = x.accession
            txt = norm(f"{x.name} {str(x.description)[:400]}")
            why = ("pfam_model" if a.startswith("pfam") and "PF" + a[4:] in ps else "cog" if a in cogs else
                   "name" if any(n in txt for n in nms) else
                   "ec_text" if any(re.search(r"(?<![\d.])" + re.escape(e) + r"(?![\d])", str(x.description)) for e in es) else None)
            if why:
                rec["nov_cdd"] = True; det.append(f"cdd:{a}:{why}")
    rec["nov_detail"] = ";".join(det)[:400]
    fl.append(rec)
pop = pop.merge(pd.DataFrame(fl), on="gid")
pop["named"] = pop.nov_own_pfam | pop.nov_own_kofam | pop.nov_cdd
for k, lab in [("nov_own_pfam", "named by own relaxed Pfam hit (PFSET)"), ("nov_own_kofam", "named by own relaxed KOfam hit (KOSET)"),
               ("nov_cdd", "named by CDD (model, COG, name, EC)"), ("named", "already named (any source)")]:
    step(lab, pop[pop[k]])
step("unannotated profile-invisible", pop[~pop.named])
pop["list_type"] = "no alignment lists"
pop.to_parquet(D / "population.parquet", index=False)
FT = pd.DataFrame(fun)
FT.to_csv(H / "funnel.tsv", sep="\t", index=False)
pd.set_option("display.width", 200)
print(FT.to_string(index=False))
m = pop[pop.tier.isin(["T2", "T2b", "rest"])]
print("main population functions:", m.func.nunique(), "| unannotated:", (~m.named).sum(), "genes,", m[~m.named].func.nunique(), "functions")
print(m.groupby("func").agg(n=("gid", "size"), unannot=("named", lambda s: int((~s).sum())), label=("proposed_label_name", "first")).sort_values("n", ascending=False).head(40).to_string())
