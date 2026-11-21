#!/usr/bin/env python
"""ws_showcase2 fold analysis: per-interface metrics and the predeclared calls (predeclared.json; the census screen's
rules, predeclared_criteria_screen3.json, unchanged).

Per fold and chain pair (i, j): pair_iptm = max(M[i][j], M[j][i]) of pair_chains_iptm (equals the complex ipTM for two
chains); contacts (CB-CB <= 8 A, CA for Gly), interface PAE (mean symmetrised PAE over the i-j contact pairs) and
interface pLDDT (mean pLDDT of the i and j residues in an i-j contact), from biohub_cofold.py's per-pair metrics.
Pass in a replicate: pair_iptm >= 0.75, contacts >= 10, interface PAE <= 5 A, interface pLDDT >= 70.
Interface hit: passes in both replicates and both margins >= 0.30 (rep 1: cognate minus the swap control that
replaces j; rep 2: cognate minus the control that replaces i); controls from the screen (sc2s_*) or the follow-up
(sc2f_*). For k = 2 this is screen-1 criteria 1-4. Promiscuity: the kept chain reaches pair_iptm >= 0.6 with >= 10
contacts against the swapped-in chain. Full-block co-fold (k >= 3): the interface hits connect all k chains.
Usage: python sc2_analyze.py FOLD_DIR [FOLD_DIR ...]
Writes analysis/: interfaces.tsv, calls.tsv, blocks.tsv, fold_level.tsv, class.tsv, class_tests.json, need_controls.tsv.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

HERE = Path(__file__).resolve().parent.parent
W = HERE / "work"
A = HERE / "analysis"
IPTM, PAE, NCON, PLDDT, DELTA, PROM = 0.75, 5.0, 10, 70.0, 0.30, 0.6
AB = "ABCDEFGH"


def load_metrics(dirs):
    rows, mono = [], {}
    for d in map(Path, dirs):
        p = d / "metrics.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            m = json.loads(line)
            if len(m["lengths"]) < 2:
                mono[m["fold_id"]] = m
                continue
            M = np.array(m["pair_chains_iptm"], dtype=float)
            n = len(m["lengths"])
            for a, b in combinations(range(n), 2):
                pp = m["pairs"][f"{AB[a]}{AB[b]}"]
                rows.append(dict(fold_id=m["fold_id"], i=a, j=b, pair_iptm=float(max(M[a, b], M[b, a])),
                                 contacts=pp["contacts"], iface_pae=pp["iface_pae"], iface_plddt=pp.get("iface_plddt"),
                                 plddt_i=m["chain_plddt"][a], plddt_j=m["chain_plddt"][b],
                                 complex_iptm=m["iptm"], complex_ptm=m["ptm"]))
    it = pd.DataFrame(rows).drop_duplicates(["fold_id", "i", "j"], keep="last")
    return it, mono


def passes(df):
    return ((df.pair_iptm >= IPTM) & (df.contacts >= NCON) & (df.iface_pae.fillna(99) <= PAE)
            & (df.iface_plddt.fillna(0) >= PLDDT))


def main(dirs):
    A.mkdir(exist_ok=True)
    man = pd.concat([pd.read_csv(p, sep="\t") for p in sorted(W.glob("manifest*.tsv"))], ignore_index=True)
    man = man.drop_duplicates("fold_id", keep="last")
    it, mono = load_metrics(dirs)
    it["pass"] = passes(it)
    it = it.merge(man[["fold_id", "cls", "rank", "k", "rep", "slot", "total_aa"]], on="fold_id", how="left")
    assert it.cls.notna().all(), it[it.cls.isna()].fold_id.unique()[:5]
    it.to_csv(A / "interfaces.tsv", sep="\t", index=False)
    ix = it.set_index(["fold_id", "i", "j"])
    print("folds:", it.drop_duplicates("fold_id").groupby("cls").size().to_dict(), "monomers:", len(mono))
    calls, need = [], []
    cog = man[man.cls == "cognate"]
    for rank, cg in cog.groupby("rank"):
        k = int(cg.k.iloc[0])
        for a, b in combinations(range(k), 2):
            r = dict(rank=int(rank), k=k, i=a, j=b, adjacent=b - a == 1, total_aa=int(cg.total_aa.max()))
            complete = True
            for rep in (1, 2):
                fid = f"sc2c_{int(rank):03d}_r{rep}"
                if (fid, a, b) not in ix.index:
                    complete = False
                    continue
                x = ix.loc[(fid, a, b)]
                for c in ("pair_iptm", "contacts", "iface_pae", "iface_plddt", "plddt_i", "plddt_j"):
                    r[f"r{rep}_{c}"] = x[c]
                r[f"r{rep}_pass"] = bool(x["pass"])
                s = b if rep == 1 else a
                y = None
                for cid in (f"sc2s_{int(rank):03d}_r{rep}_s{s}", f"sc2f_{int(rank):03d}_r{rep}_s{s}"):
                    if (cid, a, b) in ix.index:
                        y = ix.loc[(cid, a, b)]
                        r[f"r{rep}_ctrl_id"] = cid
                        break
                if y is not None:
                    r[f"r{rep}_ctrl_iptm"] = y["pair_iptm"]
                    r[f"r{rep}_ctrl_contacts"] = y["contacts"]
                    r[f"r{rep}_delta"] = x["pair_iptm"] - y["pair_iptm"]
                    r[f"r{rep}_promiscuous"] = bool(y["pair_iptm"] >= PROM and y["contacts"] >= NCON)
                else:
                    complete = False
            r["pass_both"] = bool(r.get("r1_pass") and r.get("r2_pass"))
            r["controls_complete"] = complete
            r["delta_both"] = bool(complete and r["r1_delta"] >= DELTA and r["r2_delta"] >= DELTA)
            r["interface_hit"] = r["pass_both"] and r["delta_both"]
            r["promiscuous"] = bool(r.get("r1_promiscuous") or r.get("r2_promiscuous"))
            r["min_pair_iptm"] = np.nanmin([r.get("r1_pair_iptm", np.nan), r.get("r2_pair_iptm", np.nan)])
            calls.append(r)
            if r["pass_both"]:
                for rep in (1, 2):
                    if f"r{rep}_ctrl_iptm" not in r:
                        need.append(dict(rank=int(rank), rep=rep, slot=b if rep == 1 else a, i=a, j=b))
    calls = pd.DataFrame(calls)
    calls.to_csv(A / "calls.tsv", sep="\t", index=False)
    pd.DataFrame(need, columns=["rank", "rep", "slot", "i", "j"]).drop_duplicates(["rank", "rep", "slot"]) \
        .to_csv(A / "need_controls.tsv", sep="\t", index=False)
    blocks = []
    for rank, c in calls.groupby("rank"):
        k = int(c.k.iloc[0])
        hit = c[c.interface_hit]
        parent = list(range(k))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in zip(hit.i, hit.j):
            parent[find(a)] = find(b)
        full = len(hit) > 0 and len({find(x) for x in range(k)}) == 1
        cx = {}
        for rep in (1, 2):
            z = it[it.fold_id == f"sc2c_{int(rank):03d}_r{rep}"]
            cx[rep] = float(z.complex_iptm.iloc[0]) if len(z) else np.nan
        blocks.append(dict(rank=int(rank), k=k, total_aa=int(c.total_aa.iloc[0]), interfaces=len(c),
                           interfaces_pass_r1=int(c.get("r1_pass", pd.Series(dtype=bool)).eq(True).sum()),
                           interfaces_pass_r2=int(c.get("r2_pass", pd.Series(dtype=bool)).eq(True).sum()),
                           interfaces_pass_both=int(c.pass_both.sum()), interface_hits=len(hit),
                           hit_pairs=";".join(f"{AB[a]}{AB[b]}" for a, b in zip(hit.i, hit.j)),
                           pass_both_pairs=";".join(f"{AB[a]}{AB[b]}" for a, b in zip(c[c.pass_both].i, c[c.pass_both].j)),
                           controls_pending=int((c.pass_both & ~c.controls_complete).sum()),
                           interface_hit_block=len(hit) > 0, full_block_cofold=bool(full) if k > 2 else len(hit) > 0,
                           promiscuous_hit=bool(hit.promiscuous.any()) if len(hit) else False,
                           complex_iptm_r1=cx[1], complex_iptm_r2=cx[2],
                           best_min_pair_iptm=float(c.min_pair_iptm.max()) if c.min_pair_iptm.notna().any() else np.nan))
    bl = pd.DataFrame(blocks).sort_values(["interface_hits", "interfaces_pass_both", "best_min_pair_iptm"], ascending=False)
    bl.to_csv(A / "blocks.tsv", sep="\t", index=False)
    # class level, per fold (swap controls count only interfaces of the replaced chain; follow-up controls excluded)
    it2 = it[(it.cls != "swap") | (it.i == it.slot) | (it.j == it.slot)]
    it2 = it2[~it2.fold_id.str.startswith("sc2f_")]
    fl = it2.groupby("fold_id").agg(cls=("cls", "first"), k=("k", "first"), total_aa=("total_aa", "first"),
                                   any_pass=("pass", "any"), max_pair_iptm=("pair_iptm", "max")).reset_index()
    fl.to_csv(A / "fold_level.tsv", sep="\t", index=False)
    rows = []
    for c_, d in fl.groupby("cls"):
        rows.append(dict(cls=c_, n=len(d), any_pass=int(d.any_pass.sum()), frac=round(float(d.any_pass.mean()), 4),
                         max_pair_iptm_med=round(float(d.max_pair_iptm.median()), 3),
                         max_pair_iptm_q90=round(float(d.max_pair_iptm.quantile(0.9)), 3),
                         ge075=int((d.max_pair_iptm >= 0.75).sum())))
    pd.DataFrame(rows).to_csv(A / "class.tsv", sep="\t", index=False)
    tests = {}
    a, o = fl[fl.cls == "cognate"], fl[fl.cls == "swap"]
    if len(a) and len(o):
        t = [[int(a.any_pass.sum()), int((~a.any_pass).sum())], [int(o.any_pass.sum()), int((~o.any_pass).sum())]]
        tests["cognate_vs_swap"] = dict(table=t, fisher_or=float(fisher_exact(t).statistic), fisher_p=float(fisher_exact(t).pvalue),
                                        mw_p=float(mannwhitneyu(a.max_pair_iptm, o.max_pair_iptm).pvalue),
                                        med_cog=float(a.max_pair_iptm.median()), med_swap=float(o.max_pair_iptm.median()))
    json.dump(tests, open(A / "class_tests.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string())
    print(json.dumps(tests, indent=1))
    print(bl.head(30).to_string())
    print("interfaces passing both replicates:", int(calls.pass_both.sum()), "; blocks with one:",
          int((bl.interfaces_pass_both > 0).sum()), "; controls needed:", len(need))


if __name__ == "__main__":
    main(sys.argv[1:])
