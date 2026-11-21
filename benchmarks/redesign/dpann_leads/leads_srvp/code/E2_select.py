#!/usr/bin/env python
"""Extended tier step 2 (local; rule: assay_rule_extended.json, sha256 54fed579...c5e9, unchanged). Written for leads_srvp
after E1_define.py (copied from leads_testset3); the testset3 workstream had no selection / scoring code for this tier yet.
usage: E2_select.py COLLECTION

1. Reference member: the first of the (up to 2) top-identity definition members whose ESMFold2 model (FOLD_DIR/models/<gid>.pdb),
   trimmed to its label-Pfam envelope +-10 (E1 region) and renumbered from 1, has mean pLDDT >= 70. The trimmed model is an
   SR-VP structure: it is written to FOLD_DIR/ext_refs/<KO>_ref.pdb (git-ignored), never into the repository.
   Site positions of that member (E1 site.tsv, invariant + curated) -> trimmed index. A family keeps its place only with
   >= 3 site positions on the chosen member.
2. Candidates: population genes (tiers T2, T2b, rest) of the in-scope families; per family and novelty stratum one gene per
   90%-identity cluster (cd-hit -c 0.9), round-robin over order x tier (seed 17), cap 40 unannotated and 5 already named
   (the L4_select ranking code, applied to these families).
3. Positives: the held-out representatives (E1 members_split.tsv, set 'heldout').
4. Negative A: each candidate against the reference member and site of a different in-scope family: closest trimmed-reference
   length among families whose EC differs at the first three levels.
5. Negative B: per family one length-matched (+-15%) protein per candidate from the negative pool (r4_pools.py negpool), without
   replacement inside the family; pool proteins already folded (fixed assay or earlier families) preferred.
Writes data/ext/{reference_choice.tsv, site_ref.tsv, selection.tsv, candidate_ranking.tsv}, FOLD_DIR/in/ext_new.faa."""
import json, random, subprocess, sys
from pathlib import Path
import pandas as pd

C = sys.argv[1]
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CODE = Path(__file__).resolve().parent
CFG = json.load(open(CODE / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
FD = Path(CFG["fold_dir"]) if CFG["fold_dir"] else H / "folds"
E = H / "data/ext"
(FD / "ext_refs").mkdir(parents=True, exist_ok=True)
TMP = FD / "cdhit_ext"; TMP.mkdir(exist_ok=True)
CDHIT = "/Users/jacob/Documents/Software/cdhit/cd-hit"


def read_faa(p):
    d, n = {}, None
    for l in open(p):
        if l.startswith(">"):
            n = l[1:].split()[0]; d[n] = []
        else:
            d[n].append(l.strip())
    return {k: "".join(v) for k, v in d.items()}


def residues_of(pdb):
    """[(resnum, [ATOM lines], CA pLDDT)] in file order."""
    out, cur = [], None
    for l in open(pdb):
        if not l.startswith("ATOM"):
            continue
        rn = int(l[22:26])
        if cur is None or cur[0] != rn:
            cur = [rn, [], None]; out.append(cur)
        cur[1].append(l.rstrip("\n"))
        if l[12:16].strip() == "CA":
            cur[2] = float(l[60:66])
    return out


FAM = pd.read_csv(E / "families.tsv", sep="\t")
SITE = pd.read_csv(E / "site.tsv", sep="\t", keep_default_na=False)
SPLIT = pd.read_csv(E / "members_split.tsv", sep="\t")
FAM = FAM[FAM.in_scope == True].copy()  # noqa: E712
PLAN = {p["func"]: p["strata"].split(",") for p in json.load(open(E / "budget_plan.json"))["plan"]}   # E2_plan.py (fold budget)
FAM = FAM[FAM.func.isin(PLAN)].copy()
choice, site_rows = [], []
for r in FAM.itertuples():
    picked = None
    for k in (0, 1):
        g = getattr(r, f"ref{k}", None)
        if g is None or pd.isna(g):
            continue
        g = int(g)
        p = FD / f"models/{g}.pdb"
        if not p.exists():
            choice.append(dict(func=r.func, ref_rank=k, ref_member=g, status="no_model")); continue
        lo, hi = [int(x) for x in getattr(r, f"ref{k}_region").split("-")]
        res = [x for x in residues_of(p) if lo <= x[0] <= hi]
        pl = sum(x[2] for x in res) / max(1, len(res))
        st = dict(func=r.func, ref_rank=k, ref_member=g, region=f"{lo}-{hi}", trimmed_len=len(res), trimmed_mean_plddt=round(pl, 1))
        if pl < 70:
            choice.append({**st, "status": "plddt_below_70"}); continue
        s = SITE[(SITE.func == r.func) & (SITE.ref_rank == k)]
        if len(s) < 3:
            choice.append({**st, "status": "fewer_than_3_site_positions"}); continue
        lines = []
        for i, (rn, atoms, _) in enumerate(res):
            for a in atoms:
                lines.append(a[:22] + "%4d " % (i + 1) + a[27:])
        out = FD / f"ext_refs/{r.func}_ref.pdb"
        out.write_text("\n".join(lines) + "\nEND\n")
        for x in s.itertuples():
            site_rows.append(dict(func=r.func, ref_member=g, member_pos=int(x.member_pos), trimmed_index=int(x.member_pos) - lo,  # 0-based
                                  residue=x.residue, kind=x.kind, note=x.note))
        choice.append({**st, "status": "chosen", "ref_file": str(out), "n_site": len(s), "n_curated": int((s.kind == "curated").sum())})
        picked = g
        break
CH = pd.DataFrame(choice)
CH.to_csv(E / "reference_choice.tsv", sep="\t", index=False)
SR = pd.DataFrame(site_rows)
SR.to_csv(E / "site_ref.tsv", sep="\t", index=False)
REFS = CH[CH.status == "chosen"].merge(FAM[["func", "label", "label_name", "ecs", "genes", "unannotated"]], on="func")
print(CH.to_string(index=False))

# candidates (L4_select ranking, caps 40 / 5)
pop = pd.read_parquet(H / "data/population.parquet")
pop = pop[pop.tier.isin(["T2", "T2b", "rest"]) & pop.func.isin(REFS.func)].copy()
pop["protein_id"] = pop.gid.astype(str)
SEQ = read_faa(FD / "seqs/candidates.faa")
pop["sequence"] = pop.protein_id.map(SEQ)
pop["stratum"] = pop.named.map({True: "named", False: "unannotated"})
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
        reps.append(dict(protein_id=r0.protein_id, cluster=k, cluster_size=len(m), stratum_key=f"{r0.order}|{r0.tier}"))
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
        rows.append(dict(func=f, stratum=st, protein_id=r.protein_id, rank=rank[r.protein_id], cluster_size=r.cluster_size, n_clusters=len(reps)))
RC = pd.DataFrame(rows).merge(pop.drop(columns=["stratum"]), on=["func", "protein_id"])
RC.drop(columns=["sequence"]).to_csv(E / "candidate_ranking.tsv", sep="\t", index=False)
sel = RC[((RC.stratum == "unannotated") & (RC["rank"] < 40)) | ((RC.stratum == "named") & (RC["rank"] < 5))].copy()
sel = sel[[st in PLAN[f] for f, st in zip(sel.func, sel.stratum)]].copy()      # strata the budget plan admits
sel["role"] = "candidate"

# negative-A reference
ec3 = lambda s: {".".join(e.split(".")[:3]) for e in str(s).split(";") if e}
negA = {}
for r in REFS.itertuples():
    others = REFS[(REFS.func != r.func) & REFS.ecs.apply(lambda e: not (ec3(e) & ec3(r.ecs)))]
    negA[r.func] = others.iloc[(others.trimmed_len - r.trimmed_len).abs().argsort(kind="stable")].iloc[0].func if len(others) else ""
REFS["negA_func"] = REFS.func.map(negA)
REFS.to_csv(E / "references_ext.tsv", sep="\t", index=False)

# positives: held-out representatives
MSEQ = read_faa(FD / "seqs/ext_members.faa")
MEM = pd.read_csv(E / "ext_members.tsv", sep="\t")
pos = SPLIT[(SPLIT.set == "heldout") & SPLIT.func.isin(REFS.func)].merge(MEM[["func", "gid", "length", "order"]], on=["func", "gid"], how="left")
POS = pd.DataFrame(dict(func=pos.func, protein_id=pos.gid.astype(str), role="positive", length=pos.length, order=pos.order))

# negative B
pool = pd.read_csv(H / "data/negpool.tsv", sep="\t").sort_values("gid")
pool["protein_id"] = pool.gid.astype(str)
NSEQ = read_faa(FD / "seqs/negpool.faa")
folded = {p.stem for p in (FD / "models").glob("*.pdb")}
used_global = set()
neg_rows = []
for f in REFS.func:
    used = set()
    for x in sel[sel.func == f].sort_values(["stratum", "rank"], ascending=[False, True]).itertuples():
        ok = pool[~pool.protein_id.isin(used) & ((pool.length - x.seq_len).abs() <= 0.15 * x.seq_len)].copy()
        if ok.empty:
            ok = pool[~pool.protein_id.isin(used)].copy()
        ok["fresh"] = ~(ok.protein_id.isin(folded) | ok.protein_id.isin(used_global))
        ok["d"] = (ok.length - x.seq_len).abs()
        y = ok.sort_values(["fresh", "d", "gid"]).iloc[0].protein_id
        used.add(y); used_global.add(y)
        neg_rows.append(dict(func=f, protein_id=y, role="negB", length=len(NSEQ[y]), matched_to=x.protein_id))
NEG = pd.DataFrame(neg_rows)
out = pd.concat([sel[["func", "protein_id", "role", "stratum", "length", "rank", "tier", "order", "support"]], POS, NEG], ignore_index=True)
out = out.merge(REFS[["func", "ref_file", "negA_func"]], on="func")
out["fold_path"] = [str(FD / f"models/{p}.pdb") for p in out.protein_id]
out.to_csv(E / "selection.tsv", sep="\t", index=False)
allseq = {**SEQ, **MSEQ, **NSEQ}
need = out[~out.fold_path.apply(lambda p: Path(p).exists())].drop_duplicates("protein_id")
(FD / "in/ext_new.faa").write_text("".join(f">{p}\n{allseq[p]}\n" for p in need.protein_id))
print(out.groupby(["func", "role"]).size().unstack(fill_value=0).to_string())
print("new folds needed:", len(need))
print(REFS[["func", "label_name", "ref_member", "trimmed_len", "trimmed_mean_plddt", "n_site", "n_curated", "negA_func"]].to_string(index=False))
