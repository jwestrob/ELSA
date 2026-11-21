#!/usr/bin/env python
"""Leads pipeline step 4b (local): candidates, positive controls, negative-A references and negative-B proteins, following
assay_rule.json (the DPANN rule file, unchanged) and dpann_leads/invisible/code/s03_select.py.
usage: L4_select.py COLLECTION pilot|full
  pilot: the 6 testable families with the most unannotated candidates, 6 unannotated candidates each.
  full : every testable family; per family and novelty stratum (unannotated, already named) one gene per 90%-identity
         cluster (cd-hit), round-robin over order x tier (seed 17), cap 40 (unannotated) and 10 (already named, scored as
         the second stratum); pilot genes keep their place; families found uninformative at the pilot keep their pilot genes.
  The DPANN ordering 'embedding-only first' has no counterpart (no alignment lists).
  lut change (coordinator: dense sample of one order, sample within each family across genera): the round-robin buckets
  are GENERA (genome -> genus from data/genomes_tax.tsv) instead of order x tier, so the first picks of a family come
  from distinct genera; positives likewise take one per genus first.
Positives: proteins with the family KO in the partner KOfam table and a family Pfam at GA (remote r4_pools.py positives);
2 / 3 / 4 per family by candidate clusters (< 10 / 10-29 / >= 30); one per order first; seed 17.
Negative B: random collection proteins with no Pfam row and no label, length within 15% of each candidate (remote
r4_pools.py negpool), without replacement inside a family; across families an already-chosen pool protein is preferred
(all pool members are equally random draws), so fewer new folds are needed.
Writes data/selection_<mode>.tsv, data/candidate_ranking.tsv, data/references_testable.tsv,
FOLD_DIR/in/<mode>_new.faa (proteins without a model yet)."""
import json, random, subprocess, sys
from pathlib import Path
import pandas as pd

C, MODE = sys.argv[1], sys.argv[2]
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CODE = Path(__file__).resolve().parent
CFG = json.load(open(CODE / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
FD = Path(CFG["fold_dir"]) if CFG["fold_dir"] else H / "folds"
RS = CFG["remote"].replace("RD/", RD + "/")
B = "jwestrob@biotite.berkeley.edu"
PY = "/home/jwestrob/miniconda3/envs/elsa/bin/python"
for d in (FD / "in", FD / "seqs", FD / "models"):
    d.mkdir(parents=True, exist_ok=True)
TMP = FD / "cdhit"; TMP.mkdir(exist_ok=True)
CDHIT = "/Users/jacob/Documents/Software/cdhit/cd-hit"


def ssh(cmd):
    return subprocess.run(["ssh", B, cmd], check=True, capture_output=True, text=True).stdout


def pull(remote, local):
    subprocess.run(["rsync", "-a", f"{B}:{remote}", str(local)], check=True)


def read_faa(p):
    d, n = {}, None
    for l in open(p):
        if l.startswith(">"):
            n = l[1:].split()[0]; d[n] = []
        else:
            d[n].append(l.strip())
    return {k: "".join(v) for k, v in d.items()}


REFS = pd.read_csv(H / "data/references.tsv", sep="\t")
REFS = REFS[REFS.status == "ok"].copy()
pop = pd.read_parquet(H / "data/population.parquet")
pop = pop[pop.tier.isin(["T2", "T2b", "rest"]) & pop.func.isin(REFS.func)].copy()
pop["protein_id"] = pop.gid.astype(str)
if not (FD / "seqs/candidates.faa").exists():
    pull(f"{RS}/novelty/candidates.faa", FD / "seqs/candidates.faa")
SEQ = read_faa(FD / "seqs/candidates.faa")
pop["sequence"] = pop.protein_id.map(SEQ)
pop["stratum"] = pop.named.map({True: "named", False: "unannotated"})
GT = pd.read_csv(H / "data/genomes_tax.tsv", sep="\t", keep_default_na=False)
genus_of = dict(zip(GT.genome, GT.genus))
pop["genus"] = pop.genome.map(genus_of).fillna("NA")

rows = []
for (f, st), d in pop.groupby(["func", "stratum"]):
    fa = TMP / f"{f}_{st}.faa"
    fa.write_text("".join(f">{r.protein_id}\n{r.sequence}\n" for r in d.itertuples()))
    subprocess.run([CDHIT, "-i", str(fa), "-o", str(TMP / f"{f}_{st}.cd"), "-c", "0.9", "-n", "5", "-T", "4", "-M", "2000", "-d", "0"],
                   capture_output=True, check=True)
    clu, cur = {}, None
    for l in open(TMP / f"{f}_{st}.cd.clstr"):
        if l.startswith(">Cluster"):
            cur = int(l.split()[1])
        else:
            clu[l.split(">")[1].split("...")[0]] = cur
    d = d.assign(cluster=d.protein_id.map(clu))
    reps = []
    for k, m in d.groupby("cluster"):
        r0 = m.sample(frac=1, random_state=17).iloc[0]
        reps.append(dict(protein_id=r0.protein_id, cluster=k, cluster_size=len(m), stratum_key=str(r0.genus)))
    reps = pd.DataFrame(reps)
    rng = random.Random(17)
    buckets = {s: list(g.sample(frac=1, random_state=17).protein_id) for s, g in reps.groupby("stratum_key")}
    keys = sorted(buckets); rng.shuffle(keys)
    order = []
    while any(buckets.values()):
        for k in keys:
            if buckets[k]:
                order.append(buckets[k].pop(0))
    rank = {p: i for i, p in enumerate(order)}
    for r in reps.itertuples():
        rows.append(dict(func=f, stratum=st, protein_id=r.protein_id, rank=rank[r.protein_id], cluster_size=r.cluster_size,
                         n_clusters=len(reps)))
RC = pd.DataFrame(rows).merge(pop.drop(columns=["stratum"]), on=["func", "protein_id"])
RC.drop(columns=["sequence"]).to_csv(H / "data/candidate_ranking.tsv", sep="\t", index=False)
nun = RC[RC.stratum == "unannotated"].groupby("func").size().rename("n_clusters_unannotated")
REFS = REFS.merge(nun, left_on="func", right_index=True, how="left").fillna({"n_clusters_unannotated": 0})
REFS = REFS.sort_values(["n_clusters_unannotated", "genes", "func"], ascending=[False, False, True])
if MODE == "pilot":
    fams = list(REFS.func[:6])
    sel = RC[RC.func.isin(fams) & (RC.stratum == "unannotated") & (RC["rank"] < 6)].copy()
else:
    fams = list(REFS.func)
    sel = RC[RC.func.isin(fams) & (((RC.stratum == "unannotated") & (RC["rank"] < 40)) | ((RC.stratum == "named") & (RC["rank"] < 10)))].copy()
    unin = json.load(open(H / "data/pilot_gate.json"))["uninformative"]
    sel = sel[~sel.func.isin(unin) | ((sel.stratum == "unannotated") & (sel["rank"] < 6))]
sel["role"] = "candidate"

# negative-A reference
ec3 = lambda s: {".".join(e.split(".")[:3]) for e in str(s).split(";")}
negA = {}
for r in REFS.itertuples():
    others = REFS[(REFS.func != r.func) & REFS.ecs.apply(lambda e: not (ec3(e) & ec3(r.ecs)))]
    negA[r.func] = others.iloc[(others.ref_fam_len - r.ref_fam_len).abs().argsort(kind="stable")].iloc[0].func if len(others) else ""
REFS["negA_func"] = REFS.func.map(negA)

# positives
fam_pf = pd.DataFrame([dict(func=r.func, pfams=r.label if r.ref_has_label_pfam else r.pf_match) for r in REFS.itertuples()])
fam_pf.to_csv(H / "data/families_pfam.tsv", sep="\t", index=False)
subprocess.run(["rsync", "-a", str(H / "data/families_pfam.tsv"), f"{B}:{RS}/data/"], check=True)
if not (H / "data/positives_pool.tsv").exists() or MODE == "pilot":
    ssh(f"cd {RS} && {PY} code/r4_pools.py positives --collection {C} --families {RS}/data/families_pfam.tsv")
    pull(f"{RS}/data/positives_pool.tsv", H / "data/")
    pull(f"{RS}/data/positives_pool.faa", FD / "seqs/")
PP = pd.read_csv(H / "data/positives_pool.tsv", sep="\t").sort_values(["func", "gid"])
PP["genus"] = PP.genome.map(genus_of).fillna("NA")      # lut: one positive per genus first
PSEQ = read_faa(FD / "seqs/positives_pool.faa")
pos_rows = []
PIL = pd.read_csv(H / "data/selection_pilot.tsv", sep="\t") if MODE == "full" else None
for r in REFS[REFS.func.isin(fams)].itertuples():
    if MODE == "full" and r.func in set(PIL.func):
        for x in PIL[(PIL.func == r.func) & (PIL.role == "positive")].itertuples():
            pos_rows.append(dict(func=r.func, protein_id=str(x.protein_id), role="positive", length=x.length, order=x.order))
        continue
    nc = int(min(r.n_clusters_unannotated, 40))
    npos = 2 if nc < 10 else 3 if nc < 30 else 4
    P = PP[PP.func == r.func].sample(frac=1, random_state=17)
    picked, seen = [], set()
    for x in P.itertuples():
        if x.genus not in seen:
            picked.append(x); seen.add(x.genus)
        if len(picked) == npos:
            break
    for x in P.itertuples():
        if len(picked) >= npos:
            break
        if x.gid not in [y.gid for y in picked]:
            picked.append(x)
    for x in picked:
        pos_rows.append(dict(func=r.func, protein_id=str(x.gid), role="positive", length=x.length, order=x.order))
POS = pd.DataFrame(pos_rows, columns=["func", "protein_id", "role", "length", "order"])

# negative-B
if not (H / "data/negpool.tsv").exists():
    ssh(f"cd {RS} && {PY} code/r4_pools.py negpool --collection {C} --n 4000")
    pull(f"{RS}/data/negpool.tsv", H / "data/")
    pull(f"{RS}/data/negpool.faa", FD / "seqs/")
pool = pd.read_csv(H / "data/negpool.tsv", sep="\t").sort_values("gid")
pool["protein_id"] = pool.gid.astype(str)
NSEQ = read_faa(FD / "seqs/negpool.faa")
pil_neg = {(r.func, str(r.matched_to)): str(r.protein_id) for r in PIL[PIL.role == "negB"].itertuples()} if MODE == "full" else {}
used_global = set(pil_neg.values())
neg_rows = []
for f in fams:
    used = set()
    for x in sel[sel.func == f].sort_values(["stratum", "rank"], ascending=[False, True]).itertuples():
        if (f, x.protein_id) in pil_neg:
            y = pil_neg[(f, x.protein_id)]
        else:
            ok = pool[~pool.protein_id.isin(used) & ((pool.length - x.length).abs() <= 0.15 * x.length)].copy()
            if ok.empty:
                ok = pool[~pool.protein_id.isin(used)].copy()
            ok["fresh"] = ~ok.protein_id.isin(used_global)
            ok["d"] = (ok.length - x.length).abs()
            y = ok.sort_values(["fresh", "d", "gid"]).iloc[0].protein_id
        used.add(y); used_global.add(y)
        neg_rows.append(dict(func=f, protein_id=y, role="negB", length=len(NSEQ[y]), matched_to=x.protein_id))
NEG = pd.DataFrame(neg_rows)
out = pd.concat([sel[["func", "protein_id", "role", "stratum", "length", "rank", "tier", "order", "support"]], POS, NEG], ignore_index=True)
out = out.merge(REFS[["func", "ref_file", "negA_func"]], on="func")
out["fold_path"] = [str(FD / f"models/{p}.pdb") for p in out.protein_id]
out.to_csv(H / f"data/selection_{MODE}.tsv", sep="\t", index=False)
REFS.to_csv(H / "data/references_testable.tsv", sep="\t", index=False)
allseq = {**SEQ, **PSEQ, **NSEQ}
need = out[~out.fold_path.apply(lambda p: Path(p).exists())].drop_duplicates("protein_id")
(FD / f"in/{MODE}_new.faa").write_text("".join(f">{p}\n{allseq[p]}\n" for p in need.protein_id))
print(out.groupby(["func", "role"]).size().unstack(fill_value=0).to_string())
print("new folds needed:", len(need), "| distinct negB proteins:", NEG.protein_id.nunique())
print(REFS[["func", "label_name", "genes", "n_clusters_unannotated", "negA_func", "ref_fam_len"]].to_string())
