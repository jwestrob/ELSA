#!/usr/bin/env python
"""softrec2: validated families, token-free modules, the boundary workstream's reading rules, single-output scoring.

usage
  readings2.py families --cand NAME --res r1,r2,.. --threads T [--weights sim|one]
      Leiden (igraph, modularity, n_iterations -1, seed 17) on the candidate's validated-edge graph (unique sequences),
      one process per resolution (fork). Writes $RD/softrec/r2/cand/<NAME>/fam_l<res>.npy (+ fam_cc from cands stage).
  readings2.py tune --configs CAND:FAM[:RES],... --threads T [--readings R1,A,C,H,DH] [--tag TAG]
      Per configuration: gene tokens = validated family of the gene's unique sequence (FAM = cc | l<res> | rescue3)
      -> the engine's pair statistics on those tokens (colocalization + pair_statistics, w 3) -> significant token pairs
      (token pairs of significant instances, ta != tb) -> Louvain (networkx, resolution RES default 0.3, seed 17,
      weight J x log2 max(L, 2)) -> pair module per significant instance (-1 between modules; self pairs take the
      token's module) -> boundary label = module of the instance setting the boundary's support -> readings:
        R1 conserved@2; A R1 end-trimmed (f 0.5); C module-consistent segmentation of R1; H = C then end trimming f 0.5;
        DH = R1 loci plus H loci (nested output).
      Scores on ODD genomes only (shadow package $RD/boundary/pkg_odd), ribosomal check on odd runs, gold member genes
      touching support (odd). Writes $RD/softrec/r2/results/tune_<TAG>.json and per-instance scores.
  readings2.py refcheck --threads T
      (1) the boundary workstream's R1 and DH on rescue3 recomputed with its own code and scored (per-instance F1 must
      equal boundary/scores/rescue3/*.parquet); (2) this script's module code run on the token engine's landscape must
      give the same DH loci table.
  readings2.py test --configs ... --threads T [--tag TAG]
      Full package (even genomes for the verdict), ribosomal on all 1,686 runs, paired genome bootstrap (2,000, seed 17)
      against token R1 and token DH (boundary scores), member support (even); writes results/test_<TAG>.json.
Special candidates: 'lin_rescue3' = round-1 fam_rescue3_genus poisson arm (lineage-aware G and E on rescue3 families,
softrec machinery), rebuilt from runs/fam_rescue3_genus/observed_stats.npz and its selected rule; 'tok_rescue3' = the
token engine's own landscape (for the reproduction check and as the reference).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
sys.path.insert(0, str(RD / "boundary/code"))
import blib as B  # noqa: E402
import rules as BR  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import softrec as SR  # noqa: E402

R2 = RD / "softrec/r2"
EMB = SR.EMB
PKG = B.PKG
CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
W = 3
TOKENS_RESCUE3 = RD / "engine/dpann_tokens"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jdump(path, obj):
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    os.replace(tmp, path)


# ============================================================================== shared context
class Ctx:
    def __init__(self):
        self.u = B.R.Universe(PKG / "universe")
        assert np.array_equal(self.u.gid, np.arange(self.u.n))
        su = SR.Universe()
        nd, off, left, d = su.instances(W)
        self.left, self.d = left, d
        self.cs, self.ce = self.u.c_start, self.u.c_end
        self.genome_arr = self.u.genes.genome.to_numpy()
        self.contig_arr = self.u.genes.contig.to_numpy()
        self.gcode = B.odd_even_code()
        self.gc = self.u.gc
        self.uq = np.load(EMB / "uid_of_row.npy").astype(np.int64)
        assert (self.uq >= 0).all()
        runs = B.ribo_runs()
        self.runs_all = runs
        self.runs_odd = runs[runs.genome_code % 2 == 1].reset_index(drop=True)
        self.runs_even_mask = (runs.genome_code % 2 == 0).to_numpy()
        mem = pd.read_parquet(PKG / "gold/members.parquet", columns=["instance_id", "caller", "gid"])
        self.members = mem
        tp = B.token_path("dpann__mmseqs__seq__e3c50s75_rescue3")
        t = pd.read_parquet(tp)
        tok = np.full(self.u.n, -1, np.int64)
        tok[t.gid.to_numpy()] = t.token.to_numpy()
        assert (tok >= 0).all()
        self.tok_rescue3 = tok
        self.rescue3_path = tp

    def table(self, s, e, gene_module):
        s = np.asarray(s, np.int64)
        e = np.asarray(e, np.int64)
        keep = e > s
        s, e = s[keep], e[keep]
        return pd.DataFrame(dict(genome=self.genome_arr[s], contig=self.contig_arr[s], start=self.u.pos[s], end=self.u.pos[e],
                                 module_id=B.R.locus_modules(s, e, gene_module), gid_s=s, gid_e=e))


# ============================================================================== landscapes
class Land:
    """Significant instances (engine order (d, left)), support value, boundary support b and arg (position in sidx)."""

    def __init__(self, name, sig_left, sig_d, value, b=None, arg=None, n=None, info=None):
        self.name = name
        self.sig_left = np.asarray(sig_left, np.int64)
        self.sig_d = np.asarray(sig_d, np.int64)
        self.value = np.asarray(value, np.float32)
        if b is None:
            b, arg = SR.boundary_support_argmax(self.sig_left, self.sig_d, self.value, np.arange(len(self.sig_left), dtype=np.int64), n)
        self.b, self.arg = b, arg
        self.info = info or {}


def land_candidate(ctx, name):
    cd = R2 / "cand" / name
    sidx = np.load(cd / "sidx.npy")
    L = Land(name, ctx.left[sidx], ctx.d[sidx], np.load(cd / "value.npy"), np.load(cd / "b.npy"), np.load(cd / "arg.npy"),
             info=json.loads((cd / "cand.json").read_text()))
    return L


FAMRUNS = {"lin_rescue3": "fam_rescue3_genus", "gen_rescue3": "fam_rescue3_genome"}


def land_famrun(ctx, name):
    """Family-list softrec runs on rescue3 tokens with the Poisson rule chosen by instance-level FDR:
    lin_rescue3 = round-1 fam_rescue3_genus (G and E over genomes outside the genus: lineage-aware expectation);
    gen_rescue3 = fam_rescue3_genome (G and E over genomes other than the own genome: the engine's expectation, same
    machinery), the matched control."""
    run = FAMRUNS[name]
    rdir = RD / f"engine/results_dpann_softrec/dpann__softrec__{run}__fam__poisson__w3"
    summ = json.loads((rdir / "summary.json").read_text())
    rule = summ["rule"]
    z = np.load(RD / f"softrec/runs/{run}/observed_stats.npz")
    obs = dict(L=z["L"], G=z["G"], E=z["E"], S=z["S"], own=z["own"])
    ok, Lt, G, E, S = SR.derived(obs, 0)
    logP = SR.logp_poisson(G, E, ok & (Lt >= 2))
    e = int(rule.split("1e-")[1].split("&")[0])
    m = int(rule.split("L>=")[1])
    sig = (Lt >= m) & (logP <= -e * math.log(10))
    assert int(sig.sum()) == int(summ["significant_instances"]), (int(sig.sum()), summ["significant_instances"])
    sidx = np.flatnonzero(sig)
    sidx = sidx[np.lexsort((ctx.left[sidx], ctx.d[sidx]))]
    L = Land(name, ctx.left[sidx], ctx.d[sidx], Lt[sidx].astype(np.float32), n=ctx.u.n,
             info=dict(run=run, rule=rule, fdr=summ.get("fdr"), significant_instances=int(len(sidx)), source=str(rdir)))
    saved = np.load(rdir / "boundary_support.npy")
    L.info["boundary_support_equals_saved"] = bool(np.array_equal(saved, L.b))
    assert L.info["boundary_support_equals_saved"]
    return L


def land_token(ctx, arm="rescue3"):
    BL = B.Landscape(B.ARMS[arm])
    val = BL.sp.L.to_numpy().astype(np.float32)[BL.inst_pair]
    L = Land(f"tok_{arm}", BL.inst_left, BL.inst_dist, val, n=ctx.u.n, info=dict(arm=arm))
    assert np.array_equal(L.b, BL.b)
    return L, BL


def landscape(ctx, name):
    if name in FAMRUNS:
        return land_famrun(ctx, name)
    if name == "tok_rescue3":
        return land_token(ctx)[0]
    return land_candidate(ctx, name)


# ============================================================================== families and modules
def tokens_for(ctx, cand, fam):
    """Gene tokens: 'rescue3' = MMseqs2 rescue3 families; 'cc' / 'l<res>' = validated families of the candidate."""
    if fam == "rescue3":
        return ctx.tok_rescue3.copy()
    memb = np.load(R2 / "cand" / cand / f"fam_{fam}.npy").astype(np.int64)
    g = memb[ctx.uq]
    _, tok = np.unique(g, return_inverse=True)
    return tok.astype(np.int64)


def modules(ctx, land, tok, res=0.3, seed=17):
    import networkx as nx
    t0 = time.time()
    u = ctx.u
    key, left, dist, T = B.R.colocalization(u, tok, W)
    st = B.R.pair_statistics(u, tok, key, left, W, T)
    ta = tok[land.sig_left]
    tb = tok[land.sig_left + land.sig_d]
    lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
    ns = lo != hi
    skeys = np.unique(lo[ns] * T + hi[ns])
    idx = np.searchsorted(st["key"], skeys)
    assert (idx < len(st["key"])).all() and (st["key"][idx] == skeys).all(), "significant token pair without co-location"
    Jp, Lp = st["J"][idx], st["L"][idx]
    slo, shi = skeys // T, skeys % T
    H = nx.Graph()
    wt = Jp * np.log2(np.maximum(Lp, 2).astype(np.float64))
    H.add_weighted_edges_from(zip(slo.tolist(), shi.tolist(), wt.tolist()))
    tm = np.full(T, -1, np.int64)
    comms = nx.community.louvain_communities(H, weight="weight", resolution=res, seed=seed)
    for i, c in enumerate(comms):
        if len(c) >= 2:
            tm[np.fromiter(c, dtype=np.int64)] = i
    pm = tm[lo].copy()
    pm[tm[hi] != pm] = -1
    gene_module = tm[tok]
    sizes = np.bincount(tm[tm >= 0]) if (tm >= 0).any() else np.zeros(1, np.int64)
    sizes = sizes[sizes > 0]
    info = dict(tokens=int(T), significant_token_pairs=int(len(skeys)), self_pair_instances=int((~ns).sum()),
                self_pair_share=float((~ns).mean()), modules=int(len(sizes)), tokens_in_modules=int((tm >= 0).sum()),
                module_size_q50_q90_max=np.quantile(sizes, [0.5, 0.9, 1]).tolist() if len(sizes) else [],
                instances_with_module=float((pm >= 0).mean()), louvain_res=res, seconds=round(time.time() - t0, 1))
    return dict(tm=tm, pm=pm, gene_module=gene_module, info=info, sp=dict(lo=slo, hi=shi, J=Jp, L=Lp))


def readings(ctx, land, mod, which):
    b = land.b
    out = {}
    s2, e2 = B.R.runs_at_least(b, ctx.cs, ctx.ce, 2.0, 0)
    gm = mod["gene_module"] if mod is not None else np.full(ctx.u.n, -1, np.int64)
    if "R1" in which:
        out["R1"] = (s2, e2)
    if "A" in which:
        out["A"] = B.trim_ends(b, s2, e2, 0.5)
    if mod is not None and any(x in which for x in ("C", "H", "DH")):
        lab = np.where(land.arg >= 0, mod["pm"][np.maximum(land.arg, 0)], -2).astype(np.int64)
        cs_, ce_, _ = B.label_segments(b, lab, s2, e2)
        if "C" in which:
            out["C"] = (cs_, ce_)
        ok = ce_ > cs_
        hs, he = B.trim_ends(b, cs_[ok], ce_[ok], 0.5)
        if "H" in which:
            out["H"] = (hs, he)
        if "DH" in which:
            se = np.unique(np.c_[np.r_[s2, hs], np.r_[e2, he]], axis=0)
            out["DH"] = (se[:, 0], se[:, 1])
    return {k: ctx.table(s, e, gm) for k, (s, e) in out.items()}


def member_support(ctx, b, parity):
    touch = (b > 0) | (np.r_[0, b[:-1]] > 0)
    m = ctx.members[ctx.gc[ctx.members.gid.to_numpy()] % 2 == parity]
    return {cl: float(touch[g.gid.to_numpy()].mean()) for cl, g in m.groupby("caller")}


# ============================================================================== family comparison
def family_compare(ctx, cand, fam, rng):
    tok = tokens_for(ctx, cand, fam)
    ref = ctx.tok_rescue3
    deg = None
    ea = np.load(R2 / "cand" / cand / "edges_a.npy")
    eb = np.load(R2 / "cand" / cand / "edges_b.npy")
    U = int(ctx.uq.max()) + 1
    deg = np.bincount(ea, minlength=U) + np.bincount(eb, minlength=U)
    has = deg[ctx.uq] > 0

    def sizes_info(t):
        s = np.bincount(np.unique(t, return_inverse=True)[1])
        gs = s[np.unique(t, return_inverse=True)[1]]
        return dict(families=int(len(s)), families_ge2=int((s >= 2).sum()), genes_in_families_ge2=int((gs >= 2).sum()),
                    family_size_q50_q90_q99_max_ge2=np.quantile(s[s >= 2], [0.5, 0.9, 0.99, 1]).tolist(),
                    gene_weighted_family_size_q50_q90=np.quantile(gs, [0.5, 0.9]).tolist())

    def entropy(c):
        p = c / c.sum()
        return float(-(p * np.log(p)).sum())

    def mi(x, y):
        _, xi = np.unique(x, return_inverse=True)
        _, yi = np.unique(y, return_inverse=True)
        n = len(xi)
        k = xi.astype(np.int64) * (int(yi.max()) + 1) + yi
        kk, nij = np.unique(k, return_counts=True)
        ni = np.bincount(xi)
        nj = np.bincount(yi)
        a_ = kk // (int(yi.max()) + 1)
        b_ = kk % (int(yi.max()) + 1)
        val = (nij / n) * (np.log(nij) + np.log(n) - np.log(ni[a_]) - np.log(nj[b_]))
        # ARI
        c2 = lambda z: z * (z - 1) / 2.0
        sij, si, sj, sn = c2(nij.astype(float)).sum(), c2(ni.astype(float)).sum(), c2(nj.astype(float)).sum(), c2(float(n))
        expct = si * sj / sn
        ari = (sij - expct) / (0.5 * (si + sj) - expct)
        return float(val.sum()), entropy(ni.astype(float)), entropy(nj.astype(float)), float(ari)

    x, y = tok[has], ref[has]
    MI, Hx, Hy, ARI = mi(x, y)
    emis = [mi(rng.permutation(x), y)[0] for _ in range(3)]
    EMI = float(np.mean(emis))
    AMI = (MI - EMI) / (0.5 * (Hx + Hy) - EMI)
    NMI = MI / (0.5 * (Hx + Hy))
    # ribosomal Pfam purity
    mk = pd.read_parquet(B.RC.CACHE / "marked.parquet")
    g = mk.gid.to_numpy()

    def purity(t):
        df = pd.DataFrame(dict(fam=t[g], marker=mk.marker.to_numpy()))
        cnt = df.groupby(["fam", "marker"]).size().rename("n").reset_index()
        tot = cnt.groupby("fam").n.transform("sum")
        top = cnt.loc[cnt.groupby("fam").n.idxmax()].set_index("fam")
        df["plural"] = top.marker.reindex(df.fam).to_numpy()
        pur = float((df.plural == df.marker).mean())
        fams_per_marker = df.groupby("marker").fam.nunique()
        main_share = cnt.groupby("marker").n.max() / cnt.groupby("marker").n.sum()
        fam_size = np.bincount(t)[t[g]]
        return dict(marked_genes=int(len(g)), purity_marked_genes_in_plurality_marker_family=pur,
                    families_per_marker_median=float(fams_per_marker.median()),
                    largest_family_share_of_marker_median=float(main_share.median()),
                    marked_genes_in_singletons=float((fam_size == 1).mean()))
    return dict(validated=sizes_info(tok), validated_on_genes_with_edge=sizes_info(tok[has]), rescue3=sizes_info(ref),
                genes_with_validated_edge=int(has.sum()), agreement_on_genes_with_edge=dict(AMI=AMI, NMI=NMI, ARI=ARI, MI=MI,
                EMI_perm3=EMI, EMI_sd=float(np.std(emis)), H_validated=Hx, H_rescue3=Hy),
                ribosomal_purity=dict(validated=purity(tok), rescue3=purity(ref)))


# ============================================================================== scoring
def evaluate(ctx, tables, land, split, threads):
    pkg = B.make_split_pkg(1) if split == "odd" else PKG
    runs = ctx.runs_odd if split == "odd" else ctx.runs_all
    res, pers = {}, {}
    for name, df in tables.items():
        t0 = time.time()
        per, _ = B.score_table(pkg, df, threads)
        g = per.genome.map(ctx.gcode).to_numpy()
        summ = {sp: B.summarize_per(per[g % 2 == (1 if sp == "odd" else 0)].reset_index(drop=True))
                for sp in (("odd",) if split == "odd" else ("odd", "even"))}
        sc = B.ribo_score(runs, df.gid_s.to_numpy(), df.gid_e.to_numpy())
        rib = dict(scored=B.ribo_summary(runs, sc))
        if split != "odd":
            rib["even_runs"] = B.ribo_summary(runs[ctx.runs_even_mask].reset_index(drop=True), sc[ctx.runs_even_mask].reset_index(drop=True))
        res[name] = dict(summary=summ, ribo=rib, loci=B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), ctx.u.n),
                         seconds=round(time.time() - t0, 1))
        pers[name] = (per, sc)
        log(land.name, name, split, {sp: (round(v["summed_DF_TXSS_excess"], 4), {c[:4]: round(v[c]["f1"], 3) for c in CALLERS if c in v})
                                     for sp, v in summ.items()}, "ribo", rib["scored"]["recovered"], rib["scored"]["resolved"],
            rib["scored"]["parvarchaeaceae"], rib["scored"].get("parvarchaeaceae_resolved"), f"{time.time()-t0:.0f}s")
    return res, pers


def parse_configs(s):
    out = []
    for x in s.split(","):
        if not x:
            continue
        p = x.split(":")
        out.append(dict(cand=p[0], fam=(p[1] if len(p) > 1 else ""), res=(float(p[2]) if len(p) > 2 else 0.3)))
    return out


def run_configs(a, split):
    import numba
    numba.set_num_threads(a.threads)
    ctx = Ctx()
    rdir = R2 / "results"
    rdir.mkdir(parents=True, exist_ok=True)
    sdir = R2 / "scores" / split
    sdir.mkdir(parents=True, exist_ok=True)
    fn = rdir / f"{split}_{a.tag}.json"
    out = json.loads(fn.read_text()) if fn.exists() else {}
    which = a.readings.split(",")
    lands = {}
    refs = {}
    if True:
        bsc = RD / "boundary/scores/rescue3"
        for k, f in (("tok_R1", "R1_conserved2"), ("tok_DH", "DH_coreH_res0.3_f0.5")):
            r = pd.read_parquet(bsc / f"{f}.parquet")
            r["excess"] = r.f1 - r.control_mean_f1
            refs[k] = r.set_index("instance_id")
    for c in parse_configs(a.configs):
        key = f"{c['cand']}|{c['fam'] or '-'}|{c['res']:g}"
        if key in out and not a.redo:
            log("skip", key)
            continue
        t0 = time.time()
        if c["cand"] not in lands:
            lands[c["cand"]] = landscape(ctx, c["cand"])
        land = lands[c["cand"]]
        mod = None
        item = dict(cand=c["cand"], fam=c["fam"], res=c["res"], land=land.info)
        need_mod = any(x in which for x in ("C", "H", "DH"))
        if c["fam"] and need_mod:
            tok = tokens_for(ctx, c["cand"], c["fam"])
            mod = modules(ctx, land, tok, res=c["res"])
            item["modules"] = mod["info"]
            log(key, "modules", mod["info"])
        tabs = readings(ctx, land, mod, [x for x in which if (x in ("R1", "A") or mod is not None)])
        res, pers = evaluate(ctx, tabs, land, split, a.threads)
        item["readings"] = res
        parity = 1 if split == "odd" else 0
        item["member_support"] = {("odd" if parity else "even"): member_support(ctx, land.b, parity)}
        if split != "odd":
            item["member_support"]["odd"] = member_support(ctx, land.b, 1)
        item["supported_boundaries"] = int((land.b > 0).sum())
        if refs:
            item["paired"] = {}
            for rn, (per, sc) in pers.items():
                pi = per.set_index("instance_id")
                for refk, ref in refs.items():
                    for sp in (("odd",) if split == "odd" else ("even", "odd")):
                        item["paired"][f"{sp}:{rn}_vs_{refk}"] = BR.paired(pi, ref, ctx.gcode, sp)
        for rn, (per, sc) in pers.items():
            per.to_parquet(sdir / f"{a.tag}__{c['cand']}__{c['fam'] or '-'}__{c['res']:g}__{rn}.parquet", index=False)
        item["seconds"] = round(time.time() - t0, 1)
        out[key] = item
        jdump(fn, out)
        log("done", key, f"{item['seconds']}s")
    log("wrote", fn)


def cmd_tune(a):
    run_configs(a, "odd")


def cmd_test(a):
    run_configs(a, "all")


# ============================================================================== families
_GS = {}


def _leiden(task):
    import random
    import igraph as ig
    cand, res, seed = task
    G, wt = _GS[cand]
    t = time.time()
    random.seed(seed)
    ig.set_random_number_generator(random)
    part = G.community_leiden(objective_function="modularity", weights=wt, resolution=res, n_iterations=-1)
    return cand, res, np.asarray(part.membership, dtype=np.int32), round(time.time() - t, 1)


def cmd_families(a):
    """Leiden for one or more candidates' validated-edge graphs; all graphs are loaded before the fork so every
    (candidate, resolution) task shares them."""
    import multiprocessing as mp
    import igraph as ig
    t0 = time.time()
    U = int(np.load(EMB / "uid_of_row.npy", mmap_mode="r").max()) + 1
    cands = [x for x in a.cand.split(",") if x]
    for cand in cands:
        cd = R2 / "cand" / cand
        ea = np.load(cd / "edges_a.npy")
        eb = np.load(cd / "edges_b.npy")
        wt = np.maximum(np.load(cd / "edges_w.npy").astype(np.float64), 0.0) if a.weights == "sim" else None
        t1 = time.time()
        G = ig.Graph(n=U, edges=np.column_stack([ea, eb]), directed=False)
        del ea, eb
        _GS[cand] = (G, wt)
        log(f"graph {cand}: {U:,} nodes {G.ecount():,} edges, built {time.time() - t1:.0f}s")
    tasks = [(c, float(r), 17) for c in cands for r in a.res.split(",")]
    rows = []
    with mp.get_context("fork").Pool(min(len(tasks), a.procs)) as pool:
        for cand, res, memb, secs in pool.imap_unordered(_leiden, tasks):
            cd = R2 / "cand" / cand
            tag = f"l{res:g}" if a.weights == "sim" else f"lu{res:g}"
            np.save(cd / f"fam_{tag}.npy", memb)
            sz = np.bincount(memb)
            row = dict(cand=cand, fam=tag, resolution=res, weights=a.weights, communities=int(len(sz)),
                       nontrivial=int((sz >= 2).sum()), genes_nontrivial_unique=int(sz[sz >= 2].sum()), largest=int(sz.max()),
                       size_q50_q90_q99_nontrivial=np.quantile(sz[sz >= 2], [.5, .9, .99]).tolist(), leiden_seconds=secs)
            rows.append(row)
            log("leiden", json.dumps(row))
            fn = cd / "families.json"
            prev = json.loads(fn.read_text()) if fn.exists() else []
            jdump(fn, prev + [row])
    for cand in cands:
        cd = R2 / "cand" / cand
        cc = np.load(cd / "fam_cc.npy")
        sz = np.bincount(cc)
        row = dict(cand=cand, fam="cc", communities=int(len(sz)), nontrivial=int((sz >= 2).sum()), largest=int(sz.max()))
        fn = cd / "families.json"
        prev = json.loads(fn.read_text()) if fn.exists() else []
        if not any(r.get("fam") == "cc" for r in prev):
            jdump(fn, prev + [row])
    log("end", f"{time.time() - t0:.0f}s")


_CTX = None


def _famcmp_one(c):
    rng = np.random.default_rng(17)
    return f"{c['cand']}|{c['fam']}", family_compare(_CTX, c["cand"], c["fam"], rng)


def cmd_famcmp(a):
    """Validated families vs rescue3 for several configurations, one process per configuration (fork)."""
    import multiprocessing as mp
    global _CTX
    _CTX = Ctx()
    fn = R2 / "results" / "family_compare.json"
    (R2 / "results").mkdir(exist_ok=True, parents=True)
    out = json.loads(fn.read_text()) if fn.exists() else {}
    cfgs = parse_configs(a.configs)
    with mp.get_context("fork").Pool(min(len(cfgs), a.threads)) as pool:
        for k, r in pool.imap_unordered(_famcmp_one, cfgs):
            out[k] = r
            log(k, json.dumps(r["validated_on_genes_with_edge"]), json.dumps(r["agreement_on_genes_with_edge"]),
                json.dumps(r["ribosomal_purity"]))
            jdump(fn, out)


# ============================================================================== label diagnostics
@__import__("numba").njit(cache=True)
def _span_labels(lab, s, e, out_n, out_edge):
    """For spans [s, e] (gene ids, inclusive): number of distinct module labels (>= 0) on the internal boundaries
    s..e-1, and whether the label just outside each end (boundary s-1 or e) differs from the nearest internal label
    (1 = the module changes exactly at the span edge on both sides)."""
    for i in range(len(s)):
        seen = np.empty(max(e[i] - s[i], 1), dtype=np.int64)
        m = 0
        first = -9
        last = -9
        for k in range(s[i], e[i]):
            v = lab[k]
            if v >= 0:
                if first == -9:
                    first = v
                last = v
                new = True
                for j in range(m):
                    if seen[j] == v:
                        new = False
                        break
                if new:
                    seen[m] = v
                    m += 1
        out_n[i] = m
        lo = lab[s[i] - 1] if s[i] > 0 else -2
        hi = lab[e[i]] if e[i] < len(lab) else -2
        out_edge[i] = 1 if (first >= 0 and lo != first and last >= 0 and hi != last) else 0


_DIAG = {}


def _diag_one(c):
    ctx, mem = _DIAG["ctx"], _DIAG["mem"]
    land = landscape(ctx, c["cand"])
    if c["cand"] == "tok_rescue3":
        BL = B.Landscape(B.ARMS["rescue3"])
        tm, pm = BR.Builder(BL).modules(c["res"])
        lab = BL.boundary_labels(pm)
    else:
        tok = ctx.tok_rescue3.copy() if c["fam"] == "rescue3" else tokens_for(ctx, c["cand"], c["fam"])
        mod = modules(ctx, land, tok, res=c["res"])
        lab = np.where(land.arg >= 0, mod["pm"][np.maximum(land.arg, 0)], -2).astype(np.int64)
    item = {}
    r = ctx.runs_all
    n_ = np.zeros(len(r), np.int64)
    ed = np.zeros(len(r), np.int64)
    _span_labels(lab, r.gid_start.to_numpy(np.int64), r.gid_end.to_numpy(np.int64), n_, ed)
    parv = (r.family == "Parvarchaeaceae").to_numpy()
    item["ribosomal"] = dict(runs=int(len(r)), one_label=int((n_ == 1).sum()), two_plus=int((n_ >= 2).sum()),
                             no_label=int((n_ == 0).sum()), edge_change_both_sides=int(ed.sum()),
                             parv_one_label=int(((n_ == 1) & parv).sum()), labelled_boundaries_all=float((lab >= 0).mean()))
    for cl, g in mem.groupby("caller"):
        n2 = np.zeros(len(g), np.int64)
        e2 = np.zeros(len(g), np.int64)
        _span_labels(lab, g["min"].to_numpy(np.int64), g["max"].to_numpy(np.int64), n2, e2)
        item[cl] = dict(n=int(len(g)), one_label=float((n2 == 1).mean()), two_plus=float((n2 >= 2).mean()),
                        no_label=float((n2 == 0).mean()), edge_change_both_sides=float(e2.mean()))
    return f"{c['cand']}|{c['fam']}|{c['res']:g}", item


def cmd_diag(a):
    """Module labels on ribosomal runs (all 1,686) and on gold system spans (odd genomes only); one process per config."""
    import multiprocessing as mp
    ctx = Ctx()
    inst = pd.read_parquet(PKG / "gold/instances.parquet")
    mem = ctx.members.groupby("instance_id").gid.agg(["min", "max", "count"]).reset_index()
    mem = mem.merge(inst[["instance_id", "caller", "genome"]], on="instance_id")
    mem = mem[(mem["count"] >= 2) & (mem.genome.map(ctx.gcode) % 2 == 1)]
    _DIAG.update(ctx=ctx, mem=mem)
    fn = R2 / "results" / "label_diag.json"
    out = json.loads(fn.read_text()) if fn.exists() else {}
    cfgs = parse_configs(a.configs)
    with mp.get_context("fork").Pool(min(len(cfgs), a.threads)) as pool:
        for k, item in pool.imap_unordered(_diag_one, cfgs):
            out[k] = item
            log(k, json.dumps(item))
            jdump(fn, out)


# ============================================================================== reference reproduction
def cmd_refcheck(a):
    import numba
    numba.set_num_threads(a.threads)
    t0 = time.time()
    ctx = Ctx()
    out = {}
    # (1) boundary workstream code: R1 and DH on rescue3, all genomes, per-instance equality with the stored scores
    BL = B.Landscape(B.ARMS["rescue3"])
    bld = BR.Builder(BL)
    for label, fam, p, f in (("R1", "R1", {}, "R1_conserved2"), ("DH", "D", dict(core="H", res=0.3, f=0.5), "DH_coreH_res0.3_f0.5")):
        s, e = bld.build(fam, p)
        df = BL.table(s, e)
        stored = pd.read_parquet(RD / "boundary/loci/rescue3" / f"{f}.parquet")
        same_loci = bool(len(df) == len(stored) and (df[["gid_s", "gid_e", "module_id"]].to_numpy() == stored[["gid_s", "gid_e", "module_id"]].to_numpy()).all())
        per, _ = B.score_table(PKG, df, a.threads)
        ref = pd.read_parquet(RD / "boundary/scores/rescue3" / f"{f}.parquet").set_index("instance_id")
        pi = per.set_index("instance_id")
        same_f1 = bool(np.allclose(pi.loc[ref.index].f1.to_numpy(), ref.f1.to_numpy()) and
                       np.allclose(pi.loc[ref.index].control_mean_f1.to_numpy(), ref.control_mean_f1.to_numpy()))
        g = per.genome.map(ctx.gcode).to_numpy()
        summ = B.summarize_per(per[g % 2 == 0].reset_index(drop=True))
        sc = B.ribo_score(ctx.runs_all, df.gid_s.to_numpy(), df.gid_e.to_numpy())
        out[f"boundary_{label}"] = dict(loci_equal_stored=same_loci, per_instance_f1_and_control_equal_stored=same_f1, even=summ,
                                        ribo=B.ribo_summary(ctx.runs_all, sc), loci=B.locus_stats(df.gid_s.to_numpy(), df.gid_e.to_numpy(), ctx.u.n))
        log("boundary", label, "loci equal", same_loci, "scores equal", same_f1, "even summed", round(summ["summed_DF_TXSS_excess"], 4),
            {c[:4]: round(summ[c]["f1"], 3) for c in CALLERS}, out[f"boundary_{label}"]["ribo"])
    # (2) this script's module code on the token landscape
    land, BL2 = land_token(ctx)
    mod = modules(ctx, land, ctx.tok_rescue3, res=0.3)
    tm_ref, pm_ref = bld.modules(0.3)
    out["module_code"] = dict(token_modules_equal=bool(np.array_equal(mod["tm"], tm_ref)), info=mod["info"],
                              sig_pairs_equal=bool(len(mod["sp"]["lo"]) == len(BL.sp) and np.array_equal(mod["sp"]["lo"], BL.sp.lo.to_numpy())
                                                   and np.array_equal(mod["sp"]["hi"], BL.sp.hi.to_numpy())),
                              J_equal=bool(np.allclose(mod["sp"]["J"], BL.sp.J.to_numpy())), L_equal=bool(np.array_equal(mod["sp"]["L"], BL.sp.L.to_numpy())))
    tabs = readings(ctx, land, mod, ["R1", "DH"])
    for label, f in (("R1", "R1_conserved2"), ("DH", "DH_coreH_res0.3_f0.5")):
        stored = pd.read_parquet(RD / "boundary/loci/rescue3" / f"{f}.parquet")
        df = tabs[label]
        out["module_code"][f"{label}_loci_equal_stored"] = bool(len(df) == len(stored) and (df[["gid_s", "gid_e"]].to_numpy() == stored[["gid_s", "gid_e"]].to_numpy()).all())
    log("module code", out["module_code"])
    out["seconds"] = round(time.time() - t0, 1)
    (R2 / "results").mkdir(parents=True, exist_ok=True)
    jdump(R2 / "results" / "refcheck.json", out)
    log("wrote refcheck", out["seconds"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("families", "tune", "test", "refcheck", "famcmp", "diag"))
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--cand", default="")
    ap.add_argument("--res", default="")
    ap.add_argument("--weights", default="sim")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--configs", default="")
    ap.add_argument("--readings", default="R1,A,C,H,DH")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--paired-odd", action="store_true", help="tune: also paired odd comparisons vs token R1 / DH")
    a = ap.parse_args()
    {"families": cmd_families, "tune": cmd_tune, "test": cmd_test, "refcheck": cmd_refcheck, "famcmp": cmd_famcmp, "diag": cmd_diag}[a.cmd](a)


if __name__ == "__main__":
    main()
