#!/usr/bin/env python
"""Extended tier, budget plan (local; written and run after E1_define.py, before any extended-tier fold or score).
The rule (assay_rule_extended.json) caps the tier at the folds remaining from 600 but sets no order when the in-scope
families need more. srvp order, fixed here before folding:
  P1  in-scope families holding >= 1 unannotated population gene (tiers T2, T2b, rest): 2 reference members, all held-out
      positives, the unannotated candidates (one per 90% cluster, cap 40) and one negative-B fold per candidate;
      families taken in descending order of unannotated genes (ties: more representatives first, then KO).
  P2  the already-named stratum (cap 5 per family) of P1 families, with its negative-B folds, family by family in the
      same order.
  P3  in-scope families without unannotated genes, by named-gene count, whole families (refs + held-out + named + negB).
A family enters only when its whole cost fits the remaining budget (reserve: 12 folds for co-folds and replacements).
Fold cost estimate per family = 2 + held-out + candidates + candidates (negB; pool proteins already folded reduce it later).
usage: E2_plan.py COLLECTION FOLDS_USED   writes data/ext/budget_plan.json and FOLD_DIR/in/ext_refs_plan.faa"""
import json, subprocess, sys
from pathlib import Path
import pandas as pd

C, USED = sys.argv[1], int(sys.argv[2])
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CFG = json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
FD = Path(CFG["fold_dir"]) if CFG["fold_dir"] else H / "folds"
E = H / "data/ext"
CDHIT = "/Users/jacob/Documents/Software/cdhit/cd-hit"
TMP = FD / "cdhit_plan"; TMP.mkdir(exist_ok=True)
F = pd.read_csv(E / "families.tsv", sep="\t")
F = F[F.in_scope == True].copy()  # noqa: E712
# E1 ran on the reference table before the multi-EC fix; families that the corrected lookup made testable by the fixed rule
# ('ok') or length-incompatible leave the extended scope, as the rule's scope sentence requires
REFNOW = pd.read_csv(H / "data/references.tsv", sep="\t").set_index("func").status
moved = sorted(f for f in F.func if REFNOW.get(f, "") in ("ok", "ok_but_length_incompatible"))
F = F[~F.func.isin(moved)].copy()
print("left the extended scope after the EC fix:", moved)
SPLIT = pd.read_csv(E / "members_split.tsv", sep="\t")
nheld = SPLIT[SPLIT.set == "heldout"].groupby("func").size()
pop = pd.read_parquet(H / "data/population.parquet")
pop = pop[pop.tier.isin(["T2", "T2b", "rest"]) & pop.func.isin(F.func)]
seq = {}
n = None
for l in open(FD / "seqs/candidates.faa"):
    if l.startswith(">"):
        n = l[1:].strip(); seq[n] = ""
    else:
        seq[n] += l.strip()


def nclusters(ids):
    if len(ids) <= 1:
        return len(ids)
    fa = TMP / "x.faa"
    fa.write_text("".join(f">{g}\n{seq[str(g)]}\n" for g in ids))
    subprocess.run([CDHIT, "-i", str(fa), "-o", str(TMP / "x.cd"), "-c", "0.9", "-n", "5", "-T", "4", "-M", "2000", "-d", "0"],
                   capture_output=True, check=True)
    return sum(1 for l in open(TMP / "x.cd") if l.startswith(">"))


rows = []
for r in F.itertuples():
    d = pop[pop.func == r.func]
    nu = nclusters(list(d[~d.named].gid)); nn = min(5, nclusters(list(d[d.named].gid)))
    rows.append(dict(func=r.func, label_name=r.label_name, n_reps=r.n_reps, heldout=int(nheld.get(r.func, 0)), unannot_genes=int((~d.named).sum()),
                     unannot_clusters=min(40, nu), named_clusters=nn))
T = pd.DataFrame(rows)
budget = 600 - USED - 12
plan, left = [], budget
P1 = T[T.unannot_genes > 0].sort_values(["unannot_genes", "n_reps", "func"], ascending=[False, False, True])
for r in P1.itertuples():
    cost = 2 + r.heldout + 2 * r.unannot_clusters
    if cost <= left:
        plan.append(dict(func=r.func, tier="P1", cost=cost, strata="unannotated")); left -= cost
for r in P1.itertuples():
    if r.func in {p["func"] for p in plan} and r.named_clusters:
        cost = 2 * r.named_clusters
        if cost <= left:
            for p in plan:
                if p["func"] == r.func:
                    p["strata"] = "unannotated,named"; p["cost"] += cost
            left -= cost
for r in T[T.unannot_genes == 0].sort_values(["named_clusters", "n_reps", "func"], ascending=[False, False, True]).itertuples():
    cost = 2 + r.heldout + 2 * r.named_clusters
    if r.named_clusters and cost <= left:
        plan.append(dict(func=r.func, tier="P3", cost=cost, strata="named")); left -= cost
PL = pd.DataFrame(plan)
out = dict(written_utc="before any extended-tier fold or comparison", folds_used_before=USED, reserve=12, budget=budget,
           planned_cost=int(PL.cost.sum()) if len(PL) else 0, left=left, in_scope_families=len(T),
           in_scope_with_unannotated=int((T.unannot_genes > 0).sum()), unannotated_genes_in_scope=int(T.unannot_genes.sum()),
           plan=plan, not_planned=sorted(set(T.func) - set(PL.func if len(PL) else [])), family_costs=rows)
json.dump(out, open(E / "budget_plan.json", "w"), indent=1)
mem = {}
for l in open(FD / "seqs/ext_members.faa"):
    if l.startswith(">"):
        n = l[1:].strip(); mem[n] = ""
    else:
        mem[n] += l.strip()
ids = [int(g) for f in (PL.func if len(PL) else []) for g in F.set_index("func").loc[f, ["ref0", "ref1"]] if pd.notna(g)]
(FD / "in/ext_refs_plan.faa").write_text("".join(f">{g}\n{mem[str(g)]}\n" for g in ids))
print(T.to_string(index=False))
print({k: v for k, v in out.items() if k not in ("plan", "family_costs")})
print(PL.to_string(index=False) if len(PL) else "empty plan")
print("reference members to fold:", len(ids))
