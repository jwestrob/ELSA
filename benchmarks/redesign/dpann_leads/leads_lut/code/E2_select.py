#!/usr/bin/env python
"""Extended tier step 2 (local; rule: assay_rule_extended.json). Run after the reference-member models of
FOLD_DIR/in/ext_refs.faa are in FOLD_DIR/models/.
usage: E2_select.py COLLECTION
Reference member: rank-0 member unless its model, trimmed to its label envelope +-10, has mean pLDDT < 70 (then rank 1).
Trimmed references go to FOLD_DIR/ext_refs/ (outside the repository for private collections). Site table -> data/ext/
ref_features.tsv (DPANN ref_features layout: func, feature, uniprot_pos = reference-member position, residue, ligand = note,
chain_index = 0-based index in the trimmed reference). Candidates: per in-scope family and stratum one gene per 90% cluster,
round-robin over order x tier (seed 17), cap 40 unannotated / 5 already named. Positives: held-out representatives.
lut copy: the round-robin buckets are genera (data/genomes_tax.tsv), as in the lut L4_select.py (coordinator: dense sample of
one order, sample across genera); families, unannotated cap and named cap from data/ext/budget_rule.json (E0_budget.py).
Negative A: closest trimmed-reference length among other in-scope families with a different EC at three levels.
Negative B: length-matched (+-15%) negative-pool proteins, without replacement inside a family, already-folded preferred.
Writes data/ext/{references_ext.tsv, ref_features.tsv, selection_ext.tsv, candidate_ranking.tsv}, FOLD_DIR/in/ext_new.faa."""
import json, random, subprocess, sys
from pathlib import Path
import pandas as pd

C = sys.argv[1]
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CFG = json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
FD = Path(CFG["fold_dir"]) if CFG["fold_dir"] else H / "folds"
E = H / "data/ext"
XR = FD / "ext_refs"; XR.mkdir(parents=True, exist_ok=True)
CDHIT = "/Users/jacob/Documents/Software/cdhit/cd-hit"


def read_faa(p):
    d, n = {}, None
    for l in open(p):
        if l.startswith(">"):
            n = l[1:].split()[0]; d[n] = []
        else:
            d[n].append(l.strip())
    return {k: "".join(v) for k, v in d.items()}


def residues(p):
    out = []
    for l in open(p):
        if l.startswith("ATOM"):
            k = int(l[22:26])
            if not out or out[-1][0] != k:
                out.append((k, []))
            out[-1][1].append(l)
    return out


F = pd.read_csv(E / "families.tsv", sep="\t")
F = F[F.in_scope == True].copy()  # noqa: E712
BR = json.load(open(E / "budget_rule.json"))            # lut budget rule (E0_budget.py), fixed before any extended fold
F = F[F.func.isin(BR["families"])].copy()
CAP_U, CAP_N = int(BR["unannotated_cap"]), int(BR["named_cap"])
S = pd.read_csv(E / "site.tsv", sep="\t", keep_default_na=False)
MS = pd.read_csv(E / "members_split.tsv", sep="\t")
MEM = pd.read_csv(E / "ext_members.tsv", sep="\t")
MSEQ = read_faa(FD / "seqs/ext_members.faa")
ref_rows, feat_rows = [], []
for r in F.itertuples():
    chosen = None
    for k in (0, 1):
        g = getattr(r, f"ref{k}", None)
        if g is None or pd.isna(g):
            continue
        g = int(g)
        mp = FD / f"models/{g}.pdb"
        if not mp.exists():
            continue
        lo, hi = [int(x) for x in str(getattr(r, f"ref{k}_region")).split("-")]
        res = [x for x in residues(mp) if lo <= x[0] <= hi]
        pl = [float(l[60:66]) for _, ls in res for l in ls if l[12:16].strip() == "CA"]
        mean_pl = sum(pl) / max(1, len(pl))
        if mean_pl >= 70 or k == 1:
            chosen = (k, g, lo, hi, res, mean_pl)
            if mean_pl >= 70:
                break
    if not chosen:
        ref_rows.append(dict(func=r.func, status="no_reference_model")); continue
    k, g, lo, hi, res, mean_pl = chosen
    lines = []
    for i, (num, ls) in enumerate(res):
        for l in ls:
            lines.append(l[:22] + "%4d " % (i + 1) + l[27:])
    out = XR / f"ext_{r.func}_ref.pdb"
    out.write_text("\n".join(lines) + "\nEND\n")
    status = "ok" if mean_pl >= 70 else "reference_plddt_below_70"
    s = S[(S.func == r.func) & (S.ref_rank == k)]
    for x in s.itertuples():
        feat_rows.append(dict(func=r.func, ref_acc=f"member {g}", ref_chain=out.name, feature="Active site", kind=x.kind,
                              uniprot_pos=x.member_pos, residue=x.residue, ligand=x.note, chain_index=x.member_pos - lo,
                              chain_residue=MSEQ[str(g)][x.member_pos - 1]))
    exp = H / f"ref/ext_{r.func}_exp.pdb"
    ref_rows.append(dict(func=r.func, status=status, label=r.label, label_name=r.label_name, ecs=r.ecs, ref_member=g, ref_rank=k,
                         ref_file=str(out), ref_fam_len=len(res), ref_region_plddt=round(mean_pl, 1), n_site=len(s),
                         n_curated=int((s.kind == "curated").sum()), curated_record=r.curated_record, exp_file=str(exp) if exp.exists() else "",
                         exp_chain=r.exp_chain, n_members=r.n_members, n_reps=r.n_reps))
RX = pd.DataFrame(ref_rows)
RX = RX[RX.status == "ok"].copy()
FX = pd.DataFrame(feat_rows)
# one row per site position: a position that is both invariant and curated counts once, kind 'curated+invariant'
FX = FX.groupby(["func", "chain_index"], as_index=False).agg(
    ref_acc=("ref_acc", "first"), ref_chain=("ref_chain", "first"), feature=("feature", "first"),
    kind=("kind", lambda s_: "+".join(sorted(set(s_)))), uniprot_pos=("uniprot_pos", "first"), residue=("residue", "first"),
    ligand=("ligand", lambda s_: "; ".join(x for x in s_ if x)), chain_residue=("chain_residue", "first"))
FX.to_csv(E / "ref_features.tsv", sep="\t", index=False)
nsite = FX.groupby("func").size().to_dict(); ncur = FX[FX.kind.str.contains("curated")].groupby("func").size().to_dict()
for rr in ref_rows:
    if rr.get("status") in ("ok", "reference_plddt_below_70"):
        rr["n_site"] = nsite.get(rr["func"], 0); rr["n_curated"] = ncur.get(rr["func"], 0)
# negative-A reference
ec3 = lambda s_: {".".join(e.split(".")[:3]) for e in str(s_).split(";")}
RX["negA_func"] = [(lambda o: o.iloc[(o.ref_fam_len - r.ref_fam_len).abs().argsort(kind="stable")].iloc[0].func if len(o) else "")(
    RX[(RX.func != r.func) & RX.ecs.apply(lambda e: not (ec3(e) & ec3(r.ecs)))]) for r in RX.itertuples()]
RX.to_csv(E / "references_ext.tsv", sep="\t", index=False)
# candidates
pop = pd.read_parquet(H / "data/population.parquet")
pop = pop[pop.tier.isin(["T2", "T2b", "rest"]) & pop.func.isin(RX.func)].copy()
pop["protein_id"] = pop.gid.astype(str)
SEQ = read_faa(FD / "seqs/candidates.faa")
pop["stratum"] = pop.named.map({True: "named", False: "unannotated"})
GT = pd.read_csv(H / "data/genomes_tax.tsv", sep="\t", keep_default_na=False)   # lut change: round-robin over genera
genus_of = dict(zip(GT.genome, GT.genus))
TMP = FD / "cdhit"; TMP.mkdir(exist_ok=True)
rows = []
for (f, st), d in pop.groupby(["func", "stratum"]):
    fa = TMP / f"ext_{f}_{st}.faa"
    fa.write_text("".join(f">{p}\n{SEQ[p]}\n" for p in d.protein_id))
    subprocess.run([CDHIT, "-i", str(fa), "-o", str(TMP / f"ext_{f}_{st}.cd"), "-c", "0.9", "-n", "5", "-T", "4", "-M", "2000", "-d", "0"],
                   capture_output=True, check=True)
    clu, cur = {}, None
    for l in open(TMP / f"ext_{f}_{st}.cd.clstr"):
        if l.startswith(">Cluster"):
            cur = int(l.split()[1])
        else:
            clu[l.split(">")[1].split("...")[0]] = cur
    d = d.assign(cluster=d.protein_id.map(clu))
    reps = []
    for k, m in d.groupby("cluster"):
        r0 = m.sample(frac=1, random_state=17).iloc[0]
        reps.append(dict(protein_id=r0.protein_id, cluster_size=len(m), key=str(genus_of.get(r0.genome, "NA"))))  # lut: genus buckets
    reps = pd.DataFrame(reps)
    rng = random.Random(17)
    buckets = {s_: list(g.sample(frac=1, random_state=17).protein_id) for s_, g in reps.groupby("key")}
    keys = sorted(buckets); rng.shuffle(keys)
    order = []
    while any(buckets.values()):
        for k in keys:
            if buckets[k]:
                order.append(buckets[k].pop(0))
    rank = {p: i for i, p in enumerate(order)}
    for x in reps.itertuples():
        rows.append(dict(func=f, stratum=st, protein_id=x.protein_id, rank=rank[x.protein_id], cluster_size=x.cluster_size))
RC = pd.DataFrame(rows).merge(pop.drop(columns=["stratum"]), on=["func", "protein_id"])
RC.to_csv(E / "candidate_ranking.tsv", sep="\t", index=False)
sel = RC[((RC.stratum == "unannotated") & (RC["rank"] < CAP_U)) | ((RC.stratum == "named") & (RC["rank"] < CAP_N))].copy()
sel["role"] = "candidate"
pos = MS[(MS.set == "heldout") & MS.func.isin(RX.func)].copy()
pos["protein_id"] = pos.gid.astype(str); pos["role"] = "positive"
pos["length"] = pos.protein_id.map(lambda p: len(MSEQ[p]))
pos["order"] = pos.gid.map(MEM.drop_duplicates("gid").set_index("gid").order)
# negative B
pool = pd.read_csv(H / "data/negpool.tsv", sep="\t").sort_values("gid")
pool["protein_id"] = pool.gid.astype(str)
NSEQ = read_faa(FD / "seqs/negpool.faa")
# pool proteins already chosen as negative B by the fixed assay (already folded) are preferred
folded = set()
for m in ("pilot", "full"):
    fp = H / f"data/selection_{m}.tsv"
    if fp.exists():
        t = pd.read_csv(fp, sep="\t", dtype={"protein_id": str})
        folded |= set(t[t.role == "negB"].protein_id)
used_global = set()
neg = []
for f in RX.func:
    used = set()
    for x in sel[sel.func == f].sort_values(["stratum", "rank"], ascending=[False, True]).itertuples():
        ok = pool[~pool.protein_id.isin(used) & ((pool.length - x.length).abs() <= 0.15 * x.length)].copy()
        if ok.empty:
            ok = pool[~pool.protein_id.isin(used)].copy()
        ok["fresh"] = ~(ok.protein_id.isin(used_global) | ok.protein_id.isin(folded))
        ok["d"] = (ok.length - x.length).abs()
        y = ok.sort_values(["fresh", "d", "gid"]).iloc[0].protein_id
        used.add(y); used_global.add(y)
        neg.append(dict(func=f, protein_id=y, role="negB", length=len(NSEQ[y]), matched_to=x.protein_id))
NEG = pd.DataFrame(neg)
out = pd.concat([sel[["func", "protein_id", "role", "stratum", "length", "rank", "tier", "order", "support"]],
                 pos[["func", "protein_id", "role", "length", "order"]], NEG], ignore_index=True)
out = out.merge(RX[["func", "ref_file", "negA_func"]], on="func")
out["fold_path"] = [str(FD / f"models/{p}.pdb") for p in out.protein_id]
out.to_csv(E / "selection_ext.tsv", sep="\t", index=False)
allseq = {**SEQ, **MSEQ, **NSEQ}
need = out[~out.fold_path.apply(lambda p: Path(p).exists())].drop_duplicates("protein_id")
(FD / "in/ext_new.faa").write_text("".join(f">{p}\n{allseq[p]}\n" for p in need.protein_id))
print(out.groupby(["func", "role"]).size().unstack(fill_value=0).to_string())
print("families with a reference:", len(RX), "| new folds needed:", len(need), "| distinct negB:", NEG.protein_id.nunique())
