"""Screen 2, step 5: per-interface metrics of every fold and the predeclared calls
(predeclared_criteria_screen2.json, 'multi_chain' and 'pair_blocks_longpair').

Per fold and chain pair (i, j): pair_iptm = max(M[i][j], M[j][i]) of pair_chains_iptm; contacts (CB-CB <= 8 A,
CA for Gly) and interface pLDDT recomputed from the returned structure (B-factor = pLDDT); interface PAE
from biohub_cofold's per-pair value (mean symmetrised PAE over the i-j contact pairs); the contact count is
checked against biohub_cofold's.
Usage: python s2_analyze_folds.py FOLD_DIR [FOLD_DIR ...]
Writes work/s2/interfaces.tsv (every fold x chain pair), work/s2/s2_calls.tsv (per candidate interface),
work/s2/s2_blocks.tsv (per candidate), work/s2/s2_class.tsv (class-level rates) and prints them.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

HERE = Path(__file__).resolve().parent
W2 = HERE / "work" / "s2"
CR = json.load(open(HERE / "predeclared_criteria_screen2.json"))
IPTM, PAE, NCON, PLDDT, DELTA = 0.75, 5.0, 10, 70.0, 0.30


def read_pdb(path):
    ch, rep, bf = [], [], []
    cur = None
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        key = (line[21], int(line[22:26]))
        nm = line[12:16].strip()
        rn = line[17:20]
        if key != cur:
            cur = key
            ch.append(line[21])
            rep.append(None)
            bf.append(float(line[60:66]))
            ca = None
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        if nm == "CA":
            ca = xyz
            if rep[-1] is None:
                rep[-1] = xyz
        if (nm == "CB" and rn != "GLY") or (nm == "CA" and rn == "GLY"):
            rep[-1] = xyz
    return np.array(ch), np.array(rep, dtype=float), np.array(bf)


def fold_interfaces(m, pdb):
    ch, xyz, pl = read_pdb(pdb)
    ids = sorted(set(ch), key=lambda c: "ABCDEFGH".index(c))
    M = np.array(m["pair_chains_iptm"], dtype=float)
    d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    rows = []
    for a, b in combinations(range(len(ids)), 2):
        ia, ib = ch == ids[a], ch == ids[b]
        c = d[np.ix_(ia, ib)] <= 8.0
        n = int(c.sum())
        res = np.r_[pl[ia][c.any(1)], pl[ib][c.any(0)]]
        pp = (m.get("pairs") or {}).get(f"{ids[a]}{ids[b]}", {})
        # PDB coordinates are rounded to 1e-3 A: pairs at the 8 A boundary can differ by one or two; the
        # full-precision count from biohub_cofold is authoritative, pLDDT uses the PDB contact set
        assert abs(pp.get("contacts", n) - n) <= 3, f"{m['fold_id']} {ids[a]}{ids[b]} contacts {pp.get('contacts')} vs {n}"
        n = pp.get("contacts", n)
        rows.append(dict(fold_id=m["fold_id"], i=a, j=b, pair_iptm=float(max(M[a, b], M[b, a])), pc_ij=float(M[a, b]),
                         pc_ji=float(M[b, a]), contacts=n, iface_pae=pp.get("iface_pae"),
                         iface_plddt=float(res.mean()) if n else np.nan,
                         plddt_i=float(pl[ia].mean()), plddt_j=float(pl[ib].mean()), complex_iptm=m["iptm"],
                         complex_ptm=m["ptm"]))
    return rows


def passes(r):
    return (r.pair_iptm >= IPTM) and (r.contacts >= NCON) and (r.iface_pae is not None and not pd.isna(r.iface_pae)
                                                               and r.iface_pae <= PAE) and (r.iface_plddt >= PLDDT)


LEN_BINS = [0, 200, 300, 450, 800, 10 ** 6]
LEN_LABELS = ["<200", "200-300", "300-450", "450-800", ">800"]


def calls_for(it, ix, man, mode):
    folded = set(it[it["mode"] == mode].base_id)
    calls = []
    cog = man[man.cls == "cognate"]
    for rank, cg in cog.groupby("s2_rank"):
        if not any(f in folded for f in cg.fold_id):
            continue
        k = int(cg.k.iloc[0])
        for a, b in combinations(range(k), 2):
            r = dict(mode=mode, s2_rank=rank, pool=cg.pool.iloc[0], k=k, type_key=cg.type_key.iloc[0], i=a, j=b,
                     adjacent=b - a == 1, total_aa=int(cg.total_aa.max()))
            ok = True
            for rep in (1, 2):
                fid = f"cog2_{rank:03d}_r{rep}"
                if (mode, fid, a, b) not in ix.index:
                    ok = False
                    continue
                x = ix.loc[(mode, fid, a, b)]
                for c in ("pair_iptm", "contacts", "iface_pae", "iface_plddt"):
                    r[f"r{rep}_{c}"] = x[c]
                r[f"r{rep}_pass"] = bool(x["pass"])
                s = b if rep == 1 else a          # control: rep 1 swaps j, rep 2 swaps i
                cid = f"non2_{rank:03d}_r{rep}_s{s}"
                if (mode, cid, a, b) in ix.index:
                    y = ix.loc[(mode, cid, a, b)]
                    r[f"r{rep}_ctrl_iptm"] = y["pair_iptm"]
                    r[f"r{rep}_ctrl_contacts"] = y["contacts"]
                    r[f"r{rep}_delta"] = x["pair_iptm"] - y["pair_iptm"]
                    r[f"r{rep}_promiscuous"] = bool(y["pair_iptm"] >= 0.6 and y["contacts"] >= NCON)
                else:
                    ok = False
            r["complete"] = ok
            r["pass_both"] = bool(r.get("r1_pass") and r.get("r2_pass"))
            r["delta_both"] = bool(ok and r["r1_delta"] >= DELTA and r["r2_delta"] >= DELTA)
            r["interface_hit"] = r["pass_both"] and r["delta_both"]
            r["promiscuous"] = bool(r.get("r1_promiscuous") or r.get("r2_promiscuous"))
            calls.append(r)
    return calls


def block_table(calls, it):
    blocks = []
    for (mode, rank), c in calls.groupby(["mode", "s2_rank"]):
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
            z = it[(it["mode"] == mode) & (it.base_id == f"cog2_{rank:03d}_r{rep}")]
            cx[rep] = float(z.complex_iptm.iloc[0]) if len(z) else np.nan
        mm = np.minimum(c.get("r1_pair_iptm", pd.Series(np.nan, index=c.index)),
                        c.get("r2_pair_iptm", pd.Series(np.nan, index=c.index)))
        best = c.loc[mm.idxmax()] if mm.notna().any() else None
        blocks.append(dict(mode=mode, s2_rank=rank, pool=c.pool.iloc[0], k=k, type_key=c.type_key.iloc[0],
                           total_aa=int(c.total_aa.iloc[0]), interfaces=len(c),
                           interfaces_pass_both=int(c.pass_both.sum()), interface_hits=len(hit),
                           hit_pairs=";".join(f"{'ABCDEFGH'[a]}{'ABCDEFGH'[b]}" for a, b in zip(hit.i, hit.j)),
                           interface_hit_block=len(hit) > 0, full_block_cofold=bool(full),
                           promiscuous_hit=bool(hit.promiscuous.any()) if len(hit) else False,
                           complex_iptm_r1=cx[1], complex_iptm_r2=cx[2],
                           best_pair=(f"{'ABCDEFGH'[int(best.i)]}{'ABCDEFGH'[int(best.j)]}" if best is not None else ""),
                           best_min_pair_iptm=float(mm.max()) if mm.notna().any() else np.nan,
                           best_ctrl=(f"{best.get('r1_ctrl_iptm', np.nan):.2f}/{best.get('r2_ctrl_iptm', np.nan):.2f}"
                                      if best is not None else "")))
    return pd.DataFrame(blocks)


def main(dirs):
    man = pd.read_csv(W2 / "fold_manifest2.tsv", sep="\t")
    rows = []
    for d in map(Path, dirs):
        if not (d / "metrics.jsonl").exists():
            continue
        for line in open(d / "metrics.jsonl"):
            m = json.loads(line)
            if len(m["lengths"]) < 2:
                continue
            rows.extend(fold_interfaces(m, d / "pdb" / f"{m['fold_id']}.pdb"))
    it = pd.DataFrame(rows).drop_duplicates(["fold_id", "i", "j"], keep="last")
    it["pass"] = it.apply(passes, axis=1)
    it["mode"] = np.where(it.fold_id.str.startswith("msa2_"), "msa", "single")
    it["base_id"] = it.fold_id.str.replace("^msa2_", "", regex=True)
    it = it.merge(man[["fold_id", "cls", "s2_rank", "pool", "k", "rep", "slot", "type_key", "total_aa"]]
                  .rename(columns={"fold_id": "base_id"}), on="base_id", how="left")
    it["adjacent"] = (it.j - it.i) == 1
    it["len_class"] = pd.cut(it.total_aa, LEN_BINS, right=False, labels=LEN_LABELS)
    it.to_csv(W2 / "interfaces.tsv", sep="\t", index=False)
    print("folds:", it.drop_duplicates("fold_id").groupby(["mode", "cls"]).size().to_dict())

    ix = it.set_index(["mode", "base_id", "i", "j"])
    calls = pd.DataFrame(calls_for(it, ix, man, "single") + calls_for(it, ix, man, "msa"))
    calls.to_csv(W2 / "s2_calls.tsv", sep="\t", index=False)
    bl = block_table(calls, it) if len(calls) else pd.DataFrame()
    if len(bl):
        bl = bl.sort_values(["full_block_cofold", "interface_hits", "best_min_pair_iptm"], ascending=False)
    bl.to_csv(W2 / "s2_blocks.tsv", sep="\t", index=False)

    # ---- class level, per fold: >= 1 interface meeting the per-replicate pass rule -----------------------
    it2 = it[(it.cls != "noncognate") | (it.i == it.slot) | (it.j == it.slot)]
    fl = it2.groupby("fold_id").agg(mode=("mode", "first"), cls=("cls", "first"), k=("k", "first"),
                                   total_aa=("total_aa", "first"), len_class=("len_class", "first"),
                                   any_pass=("pass", "any"), max_pair_iptm=("pair_iptm", "max"),
                                   complex_iptm=("complex_iptm", "first")).reset_index()
    fl["shape"] = np.where(fl.k == 2, "pair", "multi")
    fl.to_csv(W2 / "s2_fold_level.tsv", sep="\t", index=False)
    rows = []
    for keys, d in fl.groupby(["mode", "cls", "shape"]):
        rows.append(dict(zip(("mode", "cls", "shape"), keys), n=len(d), any_pass=int(d.any_pass.sum()),
                         frac=round(float(d.any_pass.mean()), 4), max_pair_iptm_med=round(float(d.max_pair_iptm.median()), 3)))
    cl = pd.DataFrame(rows)
    lc = (fl.groupby(["mode", "cls", "len_class"], observed=True)
            .agg(n=("any_pass", "size"), any_pass=("any_pass", "sum")).reset_index())
    lc["frac"] = (lc.any_pass / lc.n).round(3)
    lc.to_csv(W2 / "s2_length_classes.tsv", sep="\t", index=False)
    tests = {}
    for mode in ("single", "msa"):
        for shape in ("pair", "multi"):
            a = fl[(fl["mode"] == mode) & (fl.cls == "cognate") & (fl["shape"] == shape)]
            for other in ("uncalled", "positive", "noncognate"):
                o = fl[(fl["mode"] == mode) & (fl.cls == other) & (fl["shape"] == shape)]
                if len(a) and len(o):
                    t = [[int(a.any_pass.sum()), int((~a.any_pass).sum())], [int(o.any_pass.sum()), int((~o.any_pass).sum())]]
                    fe = fisher_exact(t)
                    mw = mannwhitneyu(a.max_pair_iptm, o.max_pair_iptm)
                    tests[f"{mode}:{shape}:cognate_vs_{other}"] = dict(table=t, fisher_or=float(fe.statistic),
                                                                      fisher_p=float(fe.pvalue), mw_p_max_pair_iptm=float(mw.pvalue))
    cl.to_csv(W2 / "s2_class.tsv", sep="\t", index=False)
    json.dump(tests, open(W2 / "s2_class_tests.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(cl.to_string())
    print(lc.pivot_table(index=["mode", "cls"], columns="len_class", values="frac", observed=True).to_string())
    print(json.dumps(tests, indent=1))
    if len(bl):
        print(bl.head(40).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
