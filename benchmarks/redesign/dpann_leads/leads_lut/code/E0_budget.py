#!/usr/bin/env python
"""Extended tier, lut budget rule (written and run after E1_define.py and before any extended-tier fold, selection or score).
The rule (assay_rule_extended.json) allows 'the folds remaining from the collection's 600'. The rule's caps over all 139
in-scope families exceed them, so this script fixes, from counts only, which part of the tier is run:
  1. families: in-scope families (data/ext/families.tsv) with >= 1 unannotated population gene (T2/T2b/rest, not named);
     families without unannotated genes cannot yield an unannotated pass and are listed as 'not assayed (budget)'.
  2. already-named stratum: no folds (the rule's cap of 5 is a maximum).
  3. unannotated candidates: one per 90%-identity cluster (cd-hit -c 0.9 -n 5, as E2_select.py), cap per family = the
     largest value in 40, 35, ..., 5 whose total cost fits budget - reserve.
  cost = rank-0 reference members (1 per family) + held-out positives (min(5, max(3, round(0.2 x n_reps)))) + capped
         unannotated candidate clusters + new negative-B folds (E2's choice simulated: one +-15% length-matched pool protein
         per candidate, without replacement inside a family, already-folded pool proteins first).
  reserve = 30 folds (rank-1 reference fallbacks, criterion-A co-folds).
usage: E0_budget.py COLLECTION      writes data/ext/budget_rule.json"""
import json, subprocess, sys
from pathlib import Path
import pandas as pd

C = sys.argv[1]
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA")
CFG = json.load(open(Path(__file__).resolve().parent / "collections_leads.json"))[C]
H = Path(CFG["local_out"].replace("R/", str(R) + "/"))
FD = Path(CFG["fold_dir"]) if CFG["fold_dir"] else H / "folds"
E = H / "data/ext"
CDHIT = "/Users/jacob/Documents/Software/cdhit/cd-hit"
TOTAL, RESERVE = 600, 30


def read_faa(p):
    d, n = {}, None
    for l in open(p):
        if l.startswith(">"):
            n = l[1:].split()[0]; d[n] = []
        else:
            d[n].append(l.strip())
    return {k: "".join(v) for k, v in d.items()}


used = len(list((FD / "models").glob("*.pdb")))
budget = TOTAL - used
F = pd.read_csv(E / "families.tsv", sep="\t")
F = F[F.in_scope == True]  # noqa: E712
pop = pd.read_parquet(H / "data/population.parquet")
pop = pop[pop.tier.isin(["T2", "T2b", "rest"]) & ~pop.named & pop.func.isin(F.func)]
fams = sorted(set(pop.func))
SEQ = read_faa(FD / "seqs/candidates.faa")
TMP = FD / "cdhit"; TMP.mkdir(exist_ok=True)
ncl, lens = {}, {}
for f in fams:
    d = pop[pop.func == f]
    fa = TMP / f"budget_{f}.faa"
    fa.write_text("".join(f">{g}\n{SEQ[str(g)]}\n" for g in d.gid))
    subprocess.run([CDHIT, "-i", str(fa), "-o", str(TMP / f"budget_{f}.cd"), "-c", "0.9", "-n", "5", "-T", "4", "-M", "2000", "-d", "0"],
                   capture_output=True, check=True)
    reps = list(read_faa(TMP / f"budget_{f}.cd"))
    ncl[f] = len(reps)
    lens[f] = sorted(len(SEQ[g]) for g in reps)
nrep = F.set_index("func").n_reps
npos = {f: min(5, max(3, int(round(0.2 * nrep[f])))) for f in fams}
pool = pd.read_csv(H / "data/negpool.tsv", sep="\t")
folded = {p.stem for p in (FD / "models").glob("*.pdb")}
pool["folded"] = pool.gid.astype(str).isin(folded)


def neg_new(f, cap):
    """new negative-B folds for family f at cap (lengths of the first `cap` clusters approximated by all cluster lengths)."""
    usedf, new = set(), 0
    for L in lens[f][:cap]:
        ok = pool[~pool.gid.isin(usedf) & ((pool.length - L).abs() <= 0.15 * L)].copy()
        ok["d"] = (ok.length - L).abs()
        y = ok.sort_values(["folded", "d"], ascending=[False, True]).iloc[0]
        usedf.add(y.gid)
        if not y.folded:
            new += 1
    return new


plan = None
for cap in range(40, 0, -5):
    cost = {f: 1 + npos[f] + min(ncl[f], cap) + neg_new(f, min(ncl[f], cap)) for f in fams}
    tot = sum(cost.values())
    print(f"cap {cap}: {tot} folds (budget {budget}, reserve {RESERVE})")
    if tot <= budget - RESERVE:
        plan = dict(cap=cap, total=tot, cost=cost)
        break
rule = dict(written_utc="2026-11-21, after E1_define.py, before any extended-tier fold, selection or comparison",
            folds_used_before=used, budget=budget, reserve=RESERVE, families=fams, n_families=len(fams),
            unannotated_cap=plan["cap"] if plan else 0, named_cap=0, expected_folds=plan["total"] if plan else 0,
            clusters_unannotated=ncl, positives=npos, cost_per_family=plan["cost"] if plan else {},
            not_assayed_budget=sorted(set(F.func) - set(fams)))
json.dump(rule, open(E / "budget_rule.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rule.items() if k not in ("cost_per_family", "not_assayed_budget")}, indent=1))
