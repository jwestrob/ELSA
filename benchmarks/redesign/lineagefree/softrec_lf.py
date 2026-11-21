#!/usr/bin/env python
"""lineagefree: the token-free round-2 configuration (softrec2) with a data-derived lineage unit.

usage
  softrec_lf.py run --lineage-map FILE --tag TAG --threads T [--refcheck]
      One process, lists kept in memory between the passes and the validated edges:
        1. universe with the lineage unit from FILE (genome_code -> group); no taxonomy column is read
        2. zca10 lists of genes outside the gene's own group, depth 1,000 (round 2's variant K1000_fnone)
        3. observed + 3 gene-order null passes (softrec2.run_pass2: L = other groups with a supporting pair, G, E,
           rank-weighted group sums W)
        4. rw1000_e50 rule selection (softrec2.build_candidate: rules W >= t & L >= 2, FDR <= 0.01, top 12 by observed,
           most genes covered)
        5. validated edges (softrec2.extract_edges) -> Leiden r2000 (igraph modularity, cosine weights >= 0, seed 17)
        6. module tokens = Leiden family of the gene's unique sequence; engine pair statistics with the SAME unit
           (L = distinct groups); Louvain res 0.3 (readings2.modules); readings R1 and DH (readings2.readings)
        7. full-package scoring; paired genome bootstrap against the genus-based round-2 selected outputs
           ($RD/softrec/r2/scores/all/selected__rw1000_e50__l2000__0.3__{R1,DH}.parquet)
      --refcheck (genus map): assert that L, G, E, W equal round 2's stats at K1000, sidx / edges / Leiden families
      equal round 2's candidate files, and per-instance F1 equals the stored round-2 scores.
      Writes $RD/lineagefree/softrec/<TAG>/ (cand arrays, fam_l2000.npy, result.json) and
      $RD/lineagefree/scores/sr_<TAG>__{R1,DH}.parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
sys.path.insert(0, str(RD / "softrec/code"))
import softrec as SR  # noqa: E402  (read-only imports of the round-2 code)
import softrec2 as S2  # noqa: E402
import readings2 as RG  # noqa: E402

LF = RD / "lineagefree"
R2 = RD / "softrec/r2"
CAND = "rw1000_e50"
SPEC = dict(kind="rankw", K=1000, w="e50")
W = 3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    os.replace(tmp, path)


def lineage_codes(fn, ng):
    lm = pd.read_parquet(fn, columns=["genome_code", "group"]).sort_values("genome_code").reset_index(drop=True)
    assert np.array_equal(lm.genome_code.to_numpy(), np.arange(ng))
    return pd.factorize(lm.group)[0].astype(np.int32)


class LFUniverse(SR.Universe):
    """softrec.Universe with the lineage unit as an input (reads genome, genome_code, n_genes only)."""

    def __init__(self, lineage_map):
        g = pd.read_parquet(SR.PKG / "universe/genes.parquet",
                            columns=["gid", "genome", "genome_code", "contig", "contig_code", "position_index"])
        assert (g.gid.to_numpy() == np.arange(len(g))).all()
        self.n = n = len(g)
        self.genome_name = g.genome.to_numpy()
        self.contig_name = g.contig.to_numpy()
        self.pos = g.position_index.to_numpy(np.int64)
        self.gc = g.genome_code.to_numpy(np.int32)
        cc = g.contig_code.to_numpy(np.int64)
        assert (np.diff(self.gc) >= 0).all() and (np.diff(cc) >= 0).all()
        brk = np.flatnonzero(np.diff(cc)) + 1
        self.c_start = np.r_[0, brk].astype(np.int64)
        self.c_end = np.r_[brk, n].astype(np.int64)
        self.cid = np.repeat(np.arange(len(self.c_start)), self.c_end - self.c_start).astype(np.int32)
        gm = pd.read_parquet(SR.PKG / "universe/genomes.parquet", columns=["genome", "genome_code", "n_genes"])
        gm = gm.sort_values("genome_code").reset_index(drop=True)
        assert np.array_equal(gm.genome_code.to_numpy(), np.arange(len(gm)))
        self.ng = len(gm)
        self.n_g = gm.n_genes.to_numpy(np.float64)
        self.lin_of_genome = lineage_codes(lineage_map, self.ng)
        self.nl = int(self.lin_of_genome.max()) + 1
        self.lin = self.lin_of_genome[self.gc]
        gchange = np.flatnonzero(np.diff(self.gc)) + 1
        self.g_start = np.r_[0, gchange].astype(np.int64)
        self.g_end = np.r_[gchange, n].astype(np.int64)
        self.genes_in_lin = np.bincount(self.lin, minlength=self.nl).astype(np.int64)
        self.gm = gm


def leiden(ea, eb, ew, U, res=2000.0, seed=17):
    import random
    import igraph as ig
    t0 = time.time()
    G = ig.Graph(n=U, edges=np.column_stack([ea, eb]), directed=False)
    t1 = time.time()
    wt = np.maximum(ew.astype(np.float64), 0.0)
    random.seed(seed)
    ig.set_random_number_generator(random)
    part = G.community_leiden(objective_function="modularity", weights=wt, resolution=res, n_iterations=-1)
    memb = np.asarray(part.membership, dtype=np.int32)
    return memb, round(t1 - t0, 1), round(time.time() - t1, 1)


def cmd_run(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    out = LF / "softrec" / a.tag
    cd = out / "cand"
    cd.mkdir(parents=True, exist_ok=True)
    (LF / "scores").mkdir(parents=True, exist_ok=True)
    res = dict(tag=a.tag, lineage_map=str(a.lineage_map), timing={}, refcheck={})
    u = LFUniverse(a.lineage_map)
    res["lineages"] = int(u.nl)
    nd, off, left, d = u.instances(W)
    n_inst = len(left)
    # ---- lists, passes, rule ------------------------------------------------------------------------------------
    t1 = time.time()
    L = SR.Lists(u, S2.ListArgs((1000,)))
    assert L.variants == ["K1000_fnone"], L.variants
    res["timing"]["lists"] = round(time.time() - t1, 1)
    res["lists"] = L.info
    ident = np.arange(u.n, dtype=np.int64)
    t1 = time.time()
    obs = S2.run_pass2(u, L, ident, ident, nd, off, n_inst, a.threads, "observed")
    res["timing"]["observed_pass"] = round(time.time() - t1, 1)
    nulls = []
    for r in range(3):
        t1 = time.time()
        ga, so = SR.null_order(u, r)
        nulls.append(S2.run_pass2(u, L, ga, so, nd, off, n_inst, a.threads, f"null {r}"))
        res["timing"][f"null_pass_{r}"] = round(time.time() - t1, 1)
    if a.refcheck:
        z = np.load(R2 / "stats/obs.npz")
        vi = list(z["variants"]).index("K1000_fnone")
        res["refcheck"]["stats_K1000"] = dict(L=bool(np.array_equal(z["L"][vi], obs["L"][0])), G=bool(np.array_equal(z["G"][vi], obs["G"][0])),
                                              E=bool(np.array_equal(z["E"][vi], obs["E"][0])), W=bool(np.array_equal(z["W"][vi], obs["W"][0])))
        log("refcheck stats", res["refcheck"]["stats_K1000"])
        del z
    t1 = time.time()
    c = S2.build_candidate(CAND, SPEC, u, obs, nulls, left, d, ["K1000_fnone"])
    res["timing"]["rule_selection"] = round(time.time() - t1, 1)
    assert c is not None, "no rule passes"
    for k in ("sidx", "value", "cap", "b", "arg"):
        np.save(cd / f"{k}.npy", c[k])
    c["rules"].to_csv(cd / "rules.tsv", sep="\t", index=False)
    res["candidate"] = c["info"]
    log("candidate", json.dumps(c["info"], default=float))
    nobs = dict(L_ge2_observed=int(((obs["L"][0] >= 1)).sum()),
                L_ge2_null_mean=float(np.mean([(p["L"][0] >= 1).sum() for p in nulls])))
    res["pass_counts"] = nobs
    del obs, nulls
    if a.refcheck:
        rc = R2 / "cand" / CAND
        res["refcheck"]["sidx"] = bool(np.array_equal(np.load(rc / "sidx.npy"), c["sidx"]))
        res["refcheck"]["b"] = bool(np.array_equal(np.load(rc / "b.npy"), c["b"]))
    # ---- validated edges, Leiden families ---------------------------------------------------------------------------
    t1 = time.time()
    res["edges"] = S2.extract_edges(u, L, c["sidx"], left, d, c["cap"], a.threads, cd)
    res["timing"]["edges"] = round(time.time() - t1, 1)
    del L
    ea, eb, ew = np.load(cd / "edges_a.npy"), np.load(cd / "edges_b.npy"), np.load(cd / "edges_w.npy")
    if a.refcheck:
        rc = R2 / "cand" / CAND
        res["refcheck"]["edges"] = bool(np.array_equal(np.load(rc / "edges_a.npy"), ea) and np.array_equal(np.load(rc / "edges_b.npy"), eb)
                                        and np.array_equal(np.load(rc / "edges_w.npy"), ew))
        log("refcheck sidx/b/edges", res["refcheck"])
    U = int(np.load(SR.EMB / "uid_of_row.npy", mmap_mode="r").max()) + 1
    memb, t_build, t_leiden = leiden(ea, eb, ew, U)
    del ea, eb, ew
    np.save(cd / "fam_l2000.npy", memb)
    sz = np.bincount(memb)
    res["families"] = dict(communities=int(len(sz)), nontrivial=int((sz >= 2).sum()), largest=int(sz.max()))
    res["timing"].update(igraph_build=t_build, leiden=t_leiden)
    if a.refcheck:
        res["refcheck"]["leiden_l2000"] = bool(np.array_equal(np.load(R2 / "cand" / CAND / "fam_l2000.npy"), memb))
        log("refcheck leiden", res["refcheck"]["leiden_l2000"])
    log("families", res["families"], f"{time.time()-t0:.0f}s")
    # ---- modules, readings, scoring ----------------------------------------------------------------------------------
    t1 = time.time()
    ctx = RG.Ctx()
    ctx.u.lineage = lineage_codes(a.lineage_map, ctx.u.ng).astype(np.int64)       # engine pair statistics on the unit
    ctx.u.nl = int(ctx.u.lineage.max()) + 1
    land = RG.Land(CAND, ctx.left[c["sidx"]], ctx.d[c["sidx"]], c["value"], c["b"], c["arg"], info=c["info"])
    g = memb.astype(np.int64)[ctx.uq]
    _, tok = np.unique(g, return_inverse=True)
    mod = RG.modules(ctx, land, tok.astype(np.int64), res=0.3)
    res["modules"] = mod["info"]
    tabs = RG.readings(ctx, land, mod, ["R1", "DH"])
    ev, pers = RG.evaluate(ctx, tabs, land, "all", a.threads)
    res["readings"] = ev
    res["member_support"] = dict(even=RG.member_support(ctx, land.b, 0), odd=RG.member_support(ctx, land.b, 1))
    res["timing"]["modules_readings_scoring"] = round(time.time() - t1, 1)
    res["paired_vs_genus"] = {}
    import rules as BR
    for rn, (per, sc) in pers.items():
        per.to_parquet(LF / "scores" / f"sr_{a.tag}__{rn}.parquet", index=False)
        tabs[rn].to_parquet(out / f"loci_{rn}.parquet", index=False)
        ref = pd.read_parquet(R2 / "scores/all" / f"selected__{CAND}__l2000__0.3__{rn}.parquet")
        ref["excess"] = ref.f1 - ref.control_mean_f1
        ref = ref.set_index("instance_id")
        pi = per.set_index("instance_id")
        for sp in ("even", "odd"):
            res["paired_vs_genus"][f"{sp}:{rn}"] = BR.paired(pi, ref, ctx.gcode, sp)
        if a.refcheck:
            res["refcheck"][f"scores_{rn}"] = bool(np.array_equal(pi.loc[ref.index].f1.to_numpy(), ref.f1.to_numpy()))
        pr = res["paired_vs_genus"][f"even:{rn}"]
        log(a.tag, rn, "even d summed vs genus", [round(x, 4) for x in pr["summed"]],
            {k[:4]: round(pr[k]["d_f1"][0], 4) for k in BR.CALLERS})
    res["seconds"] = round(time.time() - t0, 1)
    jdump(out / "result.json", res)
    log("wrote", out / "result.json", res["seconds"], "refcheck", res["refcheck"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--lineage-map", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--refcheck", action="store_true")
    a = ap.parse_args()
    {"run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
