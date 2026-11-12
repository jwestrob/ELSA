"""Screen 3 analysis: per-interface metrics of every fold and the predeclared calls
(predeclared_criteria_screen3.json; pair rule of screen 1 and per-interface rule of screen 2, unchanged).

Per fold and chain pair (i, j): pair_iptm = max(M[i][j], M[j][i]) of pair_chains_iptm (equals the response ipTM
for two chains); contacts (CB-CB <= 8 A, CA for Gly), interface PAE (mean symmetrised PAE over i-j contact
pairs) and interface pLDDT (mean pLDDT of the i and j residues in an i-j contact), all from biohub_cofold's
per-pair metrics computed at full precision on arrival.
Pass in a replicate: pair_iptm >= 0.75, contacts >= 10, iface PAE <= 5 A, iface pLDDT >= 70.
Interface hit: passes in both replicates and both margins >= 0.30 (rep 1: cognate minus the control that
replaces j; rep 2: cognate minus the control that replaces i). Controls come from the screen (non3_*) or from
the follow-up that completes the margins (non3f_*). For k = 2 this is screen-1 criteria 1-4.
Usage: python s3_analyze.py FOLD_DIR [FOLD_DIR ...]
Writes work/s3/: interfaces.tsv, s3_calls.tsv, s3_blocks.tsv, s3_fold_level.tsv, s3_class.tsv, s3_class_tests.json,
need_controls.tsv (interfaces passing both replicates with a missing margin control).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

HERE = Path(__file__).resolve().parent
W3 = HERE / "work" / "s3"
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
                                 pc_ij=float(M[a, b]), pc_ji=float(M[b, a]), contacts=pp["contacts"],
                                 iface_pae=pp["iface_pae"], iface_plddt=pp.get("iface_plddt"),
                                 plddt_i=m["chain_plddt"][a], plddt_j=m["chain_plddt"][b],
                                 complex_iptm=m["iptm"], complex_ptm=m["ptm"], fold_dir=str(d)))
    it = pd.DataFrame(rows).drop_duplicates(["fold_id", "i", "j"], keep="last")
    return it, mono


def passes(df):
    return ((df.pair_iptm >= IPTM) & (df.contacts >= NCON) & (df.iface_pae.fillna(99) <= PAE)
            & (df.iface_plddt.fillna(0) >= PLDDT))


def main(dirs):
    man = pd.concat([pd.read_csv(p, sep="\t") for p in sorted(W3.glob("fold_manifest3*.tsv"))], ignore_index=True)
    man = man.drop_duplicates("fold_id", keep="last")
    it, mono = load_metrics(dirs)
    it["pass"] = passes(it)
    it = it.merge(man[["fold_id", "cls", "s3_rank", "k", "rep", "slot", "type_key", "total_aa", "unrec_class"]],
                  on="fold_id", how="left")
    assert it.cls.notna().all(), it[it.cls.isna()].fold_id.unique()[:5]
    it.to_csv(W3 / "interfaces.tsv", sep="\t", index=False)
    ix = it.set_index(["fold_id", "i", "j"])
    print("folds:", it.drop_duplicates("fold_id").groupby("cls").size().to_dict(), "monomers:", len(mono))

    # ---- per candidate interface ---------------------------------------------------------------------
    calls, need = [], []
    cog = man[man.cls == "cognate"]
    for rank, cg in cog.groupby("s3_rank"):
        k = int(cg.k.iloc[0])
        for a, b in combinations(range(k), 2):
            r = dict(s3_rank=int(rank), k=k, type_key=cg.type_key.iloc[0], unrec_class=cg.unrec_class.iloc[0], i=a, j=b,
                     adjacent=b - a == 1, total_aa=int(cg.total_aa.max()))
            complete = True
            for rep in (1, 2):
                fid = f"cog3_{int(rank):03d}_r{rep}"
                if (fid, a, b) not in ix.index:
                    complete = False
                    continue
                x = ix.loc[(fid, a, b)]
                for c in ("pair_iptm", "contacts", "iface_pae", "iface_plddt", "plddt_i", "plddt_j"):
                    r[f"r{rep}_{c}"] = x[c]
                r[f"r{rep}_pass"] = bool(x["pass"])
                s = b if rep == 1 else a
                y = None
                for cid in (f"non3_{int(rank):03d}_r{rep}_s{s}", f"non3f_{int(rank):03d}_r{rep}_s{s}"):
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
            # screen-2 wording: any control of the interface (any replicate, the replaced chain is i or j)
            pr = []
            for cid in man[(man.cls == "noncognate") & (man.s3_rank == rank) & man.slot.isin([a, b])].fold_id:
                if (cid, a, b) in ix.index:
                    y = ix.loc[(cid, a, b)]
                    if y["pair_iptm"] >= PROM and y["contacts"] >= NCON:
                        pr.append(f"{cid}:{y['pair_iptm']:.2f}")
            r["promiscuous_any_control"] = ";".join(pr)
            r["min_pair_iptm"] = np.nanmin([r.get("r1_pair_iptm", np.nan), r.get("r2_pair_iptm", np.nan)])
            calls.append(r)
            if r["pass_both"]:
                for rep in (1, 2):
                    if f"r{rep}_ctrl_iptm" not in r:
                        need.append(dict(s3_rank=int(rank), rep=rep, slot=b if rep == 1 else a, i=a, j=b))
    calls = pd.DataFrame(calls)
    calls.to_csv(W3 / "s3_calls.tsv", sep="\t", index=False)
    pd.DataFrame(need).drop_duplicates(["s3_rank", "rep", "slot"]).to_csv(W3 / "need_controls.tsv", sep="\t", index=False)

    # ---- per block -------------------------------------------------------------------------------------
    blocks = []
    for rank, c in calls.groupby("s3_rank"):
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
            z = it[it.fold_id == f"cog3_{int(rank):03d}_r{rep}"]
            cx[rep] = float(z.complex_iptm.iloc[0]) if len(z) else np.nan
        best = c.loc[c.min_pair_iptm.idxmax()] if c.min_pair_iptm.notna().any() else None
        blocks.append(dict(s3_rank=int(rank), k=k, type_key=c.type_key.iloc[0], unrec_class=c.unrec_class.iloc[0],
                           total_aa=int(c.total_aa.iloc[0]), interfaces=len(c),
                           interfaces_pass_r1=int(c.r1_pass.eq(True).sum()) if "r1_pass" in c else 0,
                           interfaces_pass_r2=int(c.r2_pass.eq(True).sum()) if "r2_pass" in c else 0,
                           interfaces_pass_both=int(c.pass_both.sum()), interface_hits=len(hit),
                           hit_pairs=";".join(f"{AB[a]}{AB[b]}" for a, b in zip(hit.i, hit.j)),
                           pass_both_pairs=";".join(f"{AB[a]}{AB[b]}" for a, b in zip(c[c.pass_both].i, c[c.pass_both].j)),
                           controls_pending=int((c.pass_both & ~c.controls_complete).sum()),
                           interface_hit_block=len(hit) > 0, full_block_cofold=bool(full),
                           promiscuous_hit=bool(hit.promiscuous.any()) if len(hit) else False,
                           complex_iptm_r1=cx[1], complex_iptm_r2=cx[2],
                           best_pair=(f"{AB[int(best.i)]}{AB[int(best.j)]}" if best is not None else ""),
                           best_min_pair_iptm=float(c.min_pair_iptm.max()) if c.min_pair_iptm.notna().any() else np.nan))
    bl = pd.DataFrame(blocks).sort_values(["full_block_cofold", "interface_hits", "interfaces_pass_both", "best_min_pair_iptm"],
                                          ascending=False)
    bl.to_csv(W3 / "s3_blocks.tsv", sep="\t", index=False)

    # ---- class level, per fold -------------------------------------------------------------------------
    it2 = it[(it.cls != "noncognate") | (it.i == it.slot) | (it.j == it.slot)]
    it2 = it2[~it2.fold_id.str.startswith("non3f_")]
    fl = it2.groupby("fold_id").agg(cls=("cls", "first"), k=("k", "first"), total_aa=("total_aa", "first"),
                                   unrec_class=("unrec_class", "first"), any_pass=("pass", "any"),
                                   max_pair_iptm=("pair_iptm", "max"), complex_iptm=("complex_iptm", "first")).reset_index()
    fl["shape"] = np.where(fl.k == 2, "pair", "multi")
    fl.to_csv(W3 / "s3_fold_level.tsv", sep="\t", index=False)
    rows = []
    for keys, d in list(fl.groupby(["cls", "shape"])) + [((c_, "all"), d) for c_, d in fl.groupby("cls")]:
        rows.append(dict(cls=keys[0], shape=keys[1], n=len(d), any_pass=int(d.any_pass.sum()),
                         frac=round(float(d.any_pass.mean()), 4), max_pair_iptm_med=round(float(d.max_pair_iptm.median()), 3),
                         max_pair_iptm_q90=round(float(d.max_pair_iptm.quantile(0.9)), 3),
                         ge075=int((d.max_pair_iptm >= 0.75).sum())))
    cogf = fl[fl.cls == "cognate"]
    for uc, d in cogf.groupby("unrec_class"):
        rows.append(dict(cls=f"cognate:{uc}", shape="all", n=len(d), any_pass=int(d.any_pass.sum()),
                         frac=round(float(d.any_pass.mean()), 4), max_pair_iptm_med=round(float(d.max_pair_iptm.median()), 3),
                         max_pair_iptm_q90=round(float(d.max_pair_iptm.quantile(0.9)), 3), ge075=int((d.max_pair_iptm >= 0.75).sum())))
    cl = pd.DataFrame(rows)
    cl.to_csv(W3 / "s3_class.tsv", sep="\t", index=False)
    # predeclared split: without the blocks that have a chain the earlier enriched triage classed characterised
    etc = man.drop_duplicates("s3_rank").set_index("s3_rank").get("earlier_triage_characterised")
    fl["earlier_char"] = fl.fold_id.map(man.set_index("fold_id").s3_rank).map(etc).fillna(False).astype(bool) \
        if etc is not None else False
    tests = {}
    for subset, flx in (("pool", fl), ("excl_earlier_char", fl[~fl.earlier_char])):
        for shape in ("pair", "multi", "all"):
            a = flx[(flx.cls == "cognate") & ((flx["shape"] == shape) | (shape == "all"))]
            for other in ("uncalled", "noncognate", "positive"):
                o = flx[(flx.cls == other) & ((flx["shape"] == shape) | (shape == "all"))]
                if len(a) and len(o):
                    t = [[int(a.any_pass.sum()), int((~a.any_pass).sum())], [int(o.any_pass.sum()), int((~o.any_pass).sum())]]
                    fe = fisher_exact(t)
                    mw = mannwhitneyu(a.max_pair_iptm, o.max_pair_iptm)
                    key = f"{shape}:cognate_vs_{other}" if subset == "pool" else f"{subset}:{shape}:cognate_vs_{other}"
                    tests[key] = dict(table=t, fisher_or=float(fe.statistic), fisher_p=float(fe.pvalue),
                                      mw_p_max_pair_iptm=float(mw.pvalue), med_cog=float(a.max_pair_iptm.median()),
                                      med_other=float(o.max_pair_iptm.median()))
    fl.to_csv(W3 / "s3_fold_level.tsv", sep="\t", index=False)
    json.dump(tests, open(W3 / "s3_class_tests.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(cl.to_string())
    print(json.dumps(tests, indent=1))
    print(bl.head(40).to_string())
    print("interfaces passing both replicates:", int(calls.pass_both.sum()), "; blocks with one:",
          int((bl.interfaces_pass_both > 0).sum()), "; controls needed:", len(need))


if __name__ == "__main__":
    main(sys.argv[1:])
