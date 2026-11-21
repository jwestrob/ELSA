#!/usr/bin/env python
"""Workstream invisible, step 3: choose candidates, positive controls, negative-A references and negative-B proteins
(rules in assay_rule.json). usage: s03_select.py [pilot|full]
  pilot: the 6 testable families with the most candidates, first 6 candidates each.
  full : every testable family, up to 40 candidates each (pilot genes keep their place).
Writes data/selection_<mode>.tsv (protein_id, role, func, ref, length, ...), folds_in/<mode>_new.faa (proteins that still
need a fold; check-B folds and earlier folds of this workstream are reused)."""
import random, subprocess, sys, tempfile
from pathlib import Path
import duckdb
import pandas as pd

MODE = sys.argv[1] if len(sys.argv) > 1 else "pilot"
H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/invisible")
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign")
CB = R / "slot_rescue/verify/B/folds"
TMP = Path("/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad/invisible")
TMP.mkdir(parents=True, exist_ok=True)
c = duckdb.connect()
c.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
c.execute("SET threads=4")
SOL = set("ILVFMAWC")
comp = lambda s: sum(ch in SOL for ch in s) / max(1, len(s))

REFS = pd.read_csv(H / "data/references.tsv", sep="\t")
REFS = REFS[REFS.status == "ok"].copy()
nv = pd.read_parquet(H / "data/novel.parquet")
nv = nv[nv.tier.isin(["T2", "T2b", "rest"]) & nv.func.isin(REFS.func)].copy()
c.register("ids", nv[["protein_id"]])
seq = dict(c.execute("SELECT p.protein_id, p.sequence FROM d.proteins p JOIN ids USING(protein_id)").fetchall())
nv["sequence"] = nv.protein_id.map(seq).str.rstrip("*")
labeled_all = set(pd.read_parquet(R / "slot_rescue/pass/dpann_slot_rescue_pass.parquet", columns=["protein_id"]).protein_id)

# ---- candidates: cd-hit 90% per family, one per cluster, embedding-only first, then round-robin over order x tier
cand_rows = []
for f, d in nv.groupby("func"):
    fa = TMP / f"{f}.faa"
    fa.write_text("".join(f">{r.protein_id}\n{r.sequence}\n" for r in d.itertuples()))
    subprocess.run(["/Users/jacob/Documents/Software/cdhit/cd-hit", "-i", str(fa), "-o", str(TMP / f"{f}.cd"), "-c", "0.9", "-n", "5",
                    "-T", "4", "-M", "2000", "-d", "0"], capture_output=True, check=True)
    clu, cur = {}, None
    for l in open(TMP / f"{f}.cd.clstr"):
        if l.startswith(">Cluster"):
            cur = int(l.split()[1])
        else:
            clu[l.split(">")[1].split("...")[0]] = cur
    d = d.assign(cluster=d.protein_id.map(clu))
    rng = random.Random(17)
    reps = []
    for k, m in d.groupby("cluster"):
        m = m.sample(frac=1, random_state=17)
        m = m.sort_values("list_type", key=lambda s: s != "embedding_only", kind="stable")
        r0 = m.iloc[0]
        reps.append(dict(protein_id=r0.protein_id, cluster=k, cluster_size=len(m), emb=r0.list_type == "embedding_only",
                         stratum=f"{r0.order}|{r0.tier}"))
    reps = pd.DataFrame(reps)
    order = []
    for emb in (True, False):
        sub = reps[reps.emb == emb]
        buckets = {s: list(g.sample(frac=1, random_state=17).protein_id) for s, g in sub.groupby("stratum")}
        keys = sorted(buckets); rng.shuffle(keys)
        while any(buckets.values()):
            for k in keys:
                if buckets[k]:
                    order.append(buckets[k].pop(0))
    rank = {p: i for i, p in enumerate(order)}
    reps["rank"] = reps.protein_id.map(rank)
    for r in reps.itertuples():
        cand_rows.append(dict(func=f, protein_id=r.protein_id, rank=r.rank, cluster_size=r.cluster_size, n_clusters=len(reps)))
C = pd.DataFrame(cand_rows).merge(nv, on=["func", "protein_id"])
C.to_csv(H / "data/candidate_ranking.tsv", sep="\t", index=False,
         columns=["func", "protein_id", "rank", "cluster_size", "n_clusters", "tier", "list_type", "order", "length", "support",
                  "func_source", "proposed_label_name"])
fam_n = C.groupby("func").size().rename("n_clusters")
REFS = REFS.merge(fam_n, left_on="func", right_index=True)
REFS = REFS.sort_values(["genes", "func"], ascending=[False, True])
if MODE == "pilot":
    fams = list(REFS.func[:6]); cap = 6
else:
    fams = list(REFS.func); cap = 40
sel = C[C.func.isin(fams) & (C["rank"] < cap)].copy()
if MODE == "full":  # families found uninformative at the pilot keep only their pilot genes (assay_rule: drop)
    import json
    unin = json.load(open(H / "data/pilot_gate.json"))["uninformative"]
    sel = sel[~sel.func.isin(unin) | (sel["rank"] < 6)]
sel["role"] = "candidate"
n_cand = sel.groupby("func").size()

# ---- negative-A reference: closest family-region length, EC differs at the first three levels
ec3 = lambda s: {".".join(e.split(".")[:3]) for e in str(s).split(";")}
negA = {}
for r in REFS.itertuples():
    others = REFS[(REFS.func != r.func) & REFS.ecs.apply(lambda e: not (ec3(e) & ec3(r.ecs)))]
    o = others.iloc[(others.ref_fam_len - r.ref_fam_len).abs().argsort()].iloc[0]
    negA[r.func] = o.func
REFS["negA_func"] = REFS.func.map(negA)

# ---- positive controls
pos_rows = []
for r in REFS[REFS.func.isin(fams)].itertuples():
    pf = [r.label] if r.ref_has_label_pfam else str(r.pf_match).split(",")
    q = f"""SELECT p.protein_id, p.sequence, p.sequence_length, b.taxonomy FROM d.proteins p JOIN d.bins b ON b.bin_id = p.bin_id
      WHERE EXISTS (SELECT 1 FROM d.annotations a WHERE a.protein_id = p.protein_id AND a.source IN ('kofam','kegg') AND a.accession = '{r.func}')
        AND EXISTS (SELECT 1 FROM d.annotations a WHERE a.protein_id = p.protein_id AND a.source = 'pfam' AND a.accession IN ({",".join(f"'{x}'" for x in pf)}))
        AND p.sequence_length BETWEEN 100 AND 601 ORDER BY p.protein_id"""
    P = c.execute(q).fetchdf()
    P["sequence"] = P.sequence.str.rstrip("*")
    P = P[P.sequence.str.len().between(100, 600) & (P.sequence.apply(comp) < 0.45) & ~P.protein_id.isin(labeled_all)]
    P["order"] = P.taxonomy.str.extract(r"o__([^;]+)")[0].fillna("NA")
    nc = C[C.func == r.func].shape[0]
    nc_eff = min(nc, 40)
    npos = 2 if nc_eff < 10 else 3 if nc_eff < 30 else 4
    P = P.sample(frac=1, random_state=17)
    picked, seen = [], set()
    for x in P.itertuples():
        if x.order not in seen:
            picked.append(x); seen.add(x.order)
        if len(picked) == npos:
            break
    for x in P.itertuples():
        if len(picked) >= npos:
            break
        if x not in picked:
            picked.append(x)
    for x in picked:
        pos_rows.append(dict(func=r.func, protein_id=x.protein_id, role="positive", length=len(x.sequence), order=x.order,
                             sequence=x.sequence, pool_size=len(P)))
POS = pd.DataFrame(pos_rows)
PIL = pd.read_csv(H / "data/selection_pilot.tsv", sep="\t") if MODE == "full" else None
if MODE == "full":  # pilot families keep the positives and negative-B proteins that were folded and scored at the pilot
    pil_pos = PIL[PIL.role == "positive"].copy()
    pil_seq = dict(c.execute("SELECT protein_id, sequence FROM d.proteins WHERE protein_id IN (" +
                             ",".join(f"'{x}'" for x in PIL.protein_id) + ")").fetchall())
    pil_pos["sequence"] = pil_pos.protein_id.map(pil_seq).str.rstrip("*")
    POS = pd.concat([POS[~POS.func.isin(pil_pos.func)], pil_pos[["func", "protein_id", "role", "length", "order", "sequence"]]],
                    ignore_index=True)
    pil_neg = {(r.func, r.matched_to): r.protein_id for r in PIL[PIL.role == "negB"].itertuples()}

# ---- negative-B proteins
cb = pd.read_csv(R / "slot_rescue/verify/B/checkB_pairs.tsv", sep="\t", dtype=str)
c.register("cbids", cb[["protein_id"]].drop_duplicates())
pool = c.execute("""SELECT p.protein_id, p.sequence FROM d.proteins p JOIN cbids USING(protein_id)
  WHERE NOT EXISTS (SELECT 1 FROM d.annotations a WHERE a.protein_id = p.protein_id AND a.source = 'pfam') ORDER BY p.protein_id""").fetchdf()
pool["sequence"] = pool.sequence.str.rstrip("*")
pool = pool[pool.sequence.str.len().between(100, 600) & (pool.sequence.apply(comp) < 0.45)]
pool = pool[pool.protein_id.apply(lambda p: (CB / f"{p}.pdb").exists())]
pl = pd.read_parquet(R / "slot_rescue/pass/dpann_slot_rescue_pass.parquet", columns=["protein_id", "proposed_label_positional_homology"])
lab_of = dict(zip(pl.protein_id, pl.proposed_label_positional_homology))
pool["label"] = pool.protein_id.map(lab_of)
pool["length"] = pool.sequence.str.len()
pool["source"] = "checkB"
prev = H / "data/negB_new_pool.tsv"
if prev.exists():
    pool = pd.concat([pool, pd.read_csv(prev, sep="\t")], ignore_index=True)
neg_rows, new_rows = [], []
rng = random.Random(17)
for f in fams:
    rr = REFS[REFS.func == f].iloc[0]
    kos = {f}
    used = set()
    for x in sel[sel.func == f].sort_values("rank").itertuples():
        if MODE == "full" and (f, x.protein_id) in pil_neg:
            y = pool[pool.protein_id == pil_neg[(f, x.protein_id)]].iloc[0]
            used.add(y.protein_id)
            neg_rows.append(dict(func=f, protein_id=y.protein_id, role="negB", length=y.length, matched_to=x.protein_id, source=y.source,
                                 sequence=y.sequence))
            continue
        ok = pool[(pool.label != rr.label) & ~pool.protein_id.isin(used) & ((pool.length - x.length).abs() <= 0.15 * x.length)]
        if len(ok):
            y = ok.iloc[(ok.length - x.length).abs().argsort(kind="stable")].iloc[0]
            used.add(y.protein_id)
            neg_rows.append(dict(func=f, protein_id=y.protein_id, role="negB", length=y.length, matched_to=x.protein_id, source=y.source,
                                 sequence=y.sequence))
            continue
        # new random protein: no Pfam row, not labeled with this label, no KO of the family, length within 10%
        lo, hi = int(x.length * 0.9), int(x.length * 1.1)
        cands = c.execute(f"""SELECT * FROM (SELECT p.protein_id, p.sequence FROM d.proteins p WHERE p.sequence_length BETWEEN {lo} AND {hi + 1}
            AND NOT EXISTS (SELECT 1 FROM d.annotations a WHERE a.protein_id = p.protein_id AND (a.source = 'pfam' OR a.accession = '{f}'))
            ORDER BY p.protein_id) USING SAMPLE reservoir(400 ROWS) REPEATABLE (17)""").fetchdf()
        cands["sequence"] = cands.sequence.str.rstrip("*")
        cands = cands[cands.sequence.str.len().between(max(100, lo), min(600, hi)) & (cands.sequence.apply(comp) < 0.45)
                      & (cands.protein_id.map(lab_of).fillna("") != rr.label) & ~cands.protein_id.isin(used)]
        y = cands.iloc[rng.randrange(len(cands))]
        used.add(y.protein_id)
        row = dict(protein_id=y.protein_id, sequence=y.sequence, label=lab_of.get(y.protein_id, ""), length=len(y.sequence), source="new")
        new_rows.append(row)
        pool = pd.concat([pool, pd.DataFrame([row])], ignore_index=True)
        neg_rows.append(dict(func=f, protein_id=y.protein_id, role="negB", length=row["length"], matched_to=x.protein_id, source="new",
                             sequence=y.sequence))
NEG = pd.DataFrame(neg_rows)
if new_rows:
    allnew = pool[pool.source == "new"][["protein_id", "sequence", "label", "length", "source"]].drop_duplicates("protein_id")
    allnew.to_csv(prev, sep="\t", index=False)

out = pd.concat([sel[["func", "protein_id", "role", "length", "rank", "tier", "list_type", "order", "support", "sequence"]],
                 POS[["func", "protein_id", "role", "length", "order", "sequence"]] if len(POS) else None,
                 NEG[["func", "protein_id", "role", "length", "matched_to", "source", "sequence"]]], ignore_index=True)
out = out.merge(REFS[["func", "ref_file", "negA_func"]], on="func")
out["fold_path"] = [str(CB / f"{p}.pdb") if (CB / f"{p}.pdb").exists() else str(H / f"folds/{p}.pdb") for p in out.protein_id]
out.drop(columns="sequence").to_csv(H / f"data/selection_{MODE}.tsv", sep="\t", index=False)
need = out[~out.fold_path.apply(lambda p: Path(p).exists())].drop_duplicates("protein_id")
(H / f"folds_in/{MODE}_new.faa").write_text("".join(f">{r.protein_id}\n{r.sequence}\n" for r in need.itertuples()))
REFS.to_csv(H / "data/references_testable.tsv", sep="\t", index=False)
print(out.groupby(["func", "role"]).size().unstack(fill_value=0))
print("new folds needed:", len(need), "| negB from checkB:", (NEG.source == "checkB").sum() if len(NEG) else 0, "new:", len(new_rows))
print(REFS[["func", "label_name", "genes", "n_clusters", "negA_func", "ref_fam_len"]].to_string())
