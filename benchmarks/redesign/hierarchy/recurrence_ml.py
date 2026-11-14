#!/usr/bin/env python
"""Multi-resolution recurrence engine (workstream hierarchy): a copy of engine v1.5 with K token levels.

Each gene carries one token per level. Every pair of genes within `w` positions on a contig is an instance at each
level where both of its tokens are assigned and differ. Pair statistics (G, L, NP, J, E, P) are computed within each
level exactly as v1.5 `pair_statistics`. Each instance takes the level whose token pair has the smallest Poisson P
(ties to the level listed first). Multiplicity over levels: rules use P_adj = min(1, K * P); the null permutes gene
order within each genome jointly for all levels and applies the same per-instance choice, so the empirical FDR
includes the choice of level (as --adaptive does for the choice of window). Rules are evaluated on the (level,
token pair) units selected by at least one instance. The support of a boundary is the L of the selected pair of the
best significant instance spanning it, and loci are read from this combined support exactly as v1.5.

With one level the procedure is v1.5 at fixed w (checked exactly before any multi-level result is used).
Levels: --levels name=path,name=path,... where path is a (gid, token) parquet or a membership .npy in gid order.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import recurrence as v15          # the deployed v1.5 module (copied beside this file); kernels and rules reused as is

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
log = v15.log


def load_level(path, u):
    path = str(path)
    if path.endswith(".npy"):
        m = np.load(path).astype(np.int64)
        assert len(m) == u.n
        tok_by_gid = np.full(u.n, -1, dtype=np.int64)              # factorized as context_kernel cmd_arms writes arms
        ok = m >= 0
        _, tok_by_gid[ok] = np.unique(m[ok], return_inverse=True)
    else:
        tokens = pd.read_parquet(path)
        assert len(tokens) == u.n and tokens.gid.is_unique
        tok_by_gid = np.full(u.n, -1, dtype=np.int64)
        tok_by_gid[tokens.gid.to_numpy()] = tokens.token.to_numpy()
    return tok_by_gid[u.gid]


def instances(u, w):
    """All gene pairs within w on a contig, in v1.5 order (distance-major)."""
    lefts, dists = [], []
    for d in range(1, w + 1):
        left = np.flatnonzero(u.cc[:-d] == u.cc[d:])
        lefts.append(left)
        dists.append(np.full(len(left), d, dtype=np.int64))
    return np.concatenate(lefts), np.concatenate(dists)


NAMES = ("G", "L", "NP", "n_other", "J", "E", "P", "logP", "F_lo", "F_hi")


def ml_statistics(u, toks, left, dist, w):
    """Per-level statistics, per-instance level choice, and the global table of selected (level, pair) units.

    Returns st (global table sorted by global key), key/left/dist of the kept instances (global keys), and a
    per-instance level array."""
    K = len(toks)
    n_inst = len(left)
    best = np.full(n_inst, np.inf)
    lvl = np.full(n_inst, -1, dtype=np.int64)
    per = []
    for k, tok in enumerate(toks):
        T = int(tok.max()) + 1
        ta, tb = tok[left], tok[left + dist]
        ok = (ta >= 0) & (tb >= 0) & (ta != tb)
        key = np.full(n_inst, -1, dtype=np.int64)
        key[ok] = np.minimum(ta[ok], tb[ok]) * T + np.maximum(ta[ok], tb[ok])
        st = v15.pair_statistics(u, tok, key[ok], left[ok], w, T)
        lp = np.full(n_inst, np.inf)
        lp[ok] = st["logP"][np.searchsorted(st["key"], key[ok])]
        better = lp < best                                   # strict: ties keep the earlier level
        best[better] = lp[better]
        lvl[better] = k
        per.append((T, key, st))
    keep = lvl >= 0
    kidx = np.flatnonzero(keep)
    out = {n: [] for n in NAMES + ("key", "lo", "hi", "level")}
    inc_k, inc_g = [], []
    gkey_inst = np.empty(len(kidx), dtype=np.int64)
    koff, toff = 0, 0
    level_info = []
    for k, (T, key, st) in enumerate(per):
        sel = kidx[lvl[kidx] == k]
        used = np.zeros(len(st["key"]), dtype=bool)
        used[np.searchsorted(st["key"], key[sel])] = True
        for n in NAMES:
            out[n].append(st[n][used])
        out["key"].append(st["key"][used] + koff)
        out["lo"].append(st["lo"][used] + toff)
        out["hi"].append(st["hi"][used] + toff)
        out["level"].append(np.full(int(used.sum()), k, dtype=np.int64))
        k_of, g_of = st["incidence"]
        m = used[np.searchsorted(st["key"], k_of)]
        inc_k.append(k_of[m] + koff)
        inc_g.append(g_of[m])
        gkey_inst[lvl[kidx] == k] = key[sel] + koff
        level_info.append(dict(tokens=T, token_pairs=int(len(st["key"])), selected_pairs=int(used.sum()),
                               instances_valid=int((key >= 0).sum()), instances_selecting=int(len(sel))))
        assert koff + T * T > koff, "global key overflow"
        koff += T * T
        toff += T
    st = {n: np.concatenate(v) for n, v in out.items()}
    assert np.all(np.diff(st["key"]) > 0)
    if K > 1:
        st["P"] = np.minimum(1.0, K * st["P"])
        st["logP"] = np.minimum(0.0, st["logP"] + np.log(K))
    st["incidence"] = (np.concatenate(inc_k), np.concatenate(inc_g))
    return st, gkey_inst, left[kidx], dist[kidx], lvl[kidx], level_info, toff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--levels", required=True, help="name=path,... listed in priority order (finest first)")
    ap.add_argument("--universe", type=Path, default=RD / "ws_dpann/package/universe")
    ap.add_argument("--w", type=int, default=3)
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--rule")
    ap.add_argument("--module-res", type=float, default=10.0)
    ap.add_argument("--null", choices=["genome", "contig"], default="genome")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)

    tag = f"{a.arm}__w{a.w}"
    out = a.out / tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    u = v15.Universe(a.universe)
    specs = [x.split("=", 1) for x in a.levels.split(",")]
    toks = [load_level(p, u) for _, p in specs]
    K = len(toks)
    n_start, n_end = (u.g_start, u.g_end) if a.null == "genome" else (u.c_start, u.c_end)
    res = dict(engine="v1.5-ml", null=a.null, arm=a.arm, universe=str(a.universe), w=a.w, window_grid=None,
               oriented=False, perms=a.perms, genes=u.n, genomes=u.ng,
               assigned=int((toks[0] >= 0).sum()), tokens=int(len(np.unique(toks[0][toks[0] >= 0]))))
    res["levels"] = [dict(name=n, path=p, assigned=int((t >= 0).sum()), tokens=int(len(np.unique(t[t >= 0]))))
                     for (n, p), t in zip(specs, toks)]
    log("universe + tokens", res, f"{time.time()-t0:.0f}s")

    left0, dist0 = instances(u, a.w)
    st, key, left, dist, ilev, level_info, T = ml_statistics(u, toks, left0, dist0, a.w)
    res.update(instances=int(len(key)), token_pairs=int(len(st["key"])),
               token_pairs_L2=int((st["L"] >= 2).sum()), stats_seconds=round(time.time() - t0, 1))
    for d, li in zip(res["levels"], level_info):
        d.update(li)
    log("observed", {k: res[k] for k in ("instances", "token_pairs", "token_pairs_L2")}, level_info, f"{time.time()-t0:.0f}s")

    nulls = []
    for r in range(a.perms):
        rng = np.random.default_rng(1000 + r)
        order = np.arange(u.n)
        for s, e in zip(n_start, n_end):
            order[s:e] = s + rng.permutation(e - s)    # same swaps as v1.5 rng.permutation(tok[s:e])
        toks_n = [t[order] for t in toks]
        sn = ml_statistics(u, toks_n, left0, dist0, a.w)[0]
        nulls.append({k: sn[k] for k in ("key", "L", "J", "P")})      # the rules read L, J and P only
        del sn
        log("null", r, "token pairs", len(nulls[-1]["key"]), "L>=2", int((nulls[-1]["L"] >= 2).sum()),
            f"{time.time()-t0:.0f}s")

    rows = []
    for name, rule in v15.RULES.items():
        mo = rule(st)
        n_obs = int(mo.sum())
        n_null = float(np.mean([rule(s).sum() for s in nulls]))
        rows.append(dict(rule=name, observed=n_obs, null_mean=n_null,
                         fdr=(n_null / n_obs if n_obs else np.nan),
                         cross_phylum=int((mo & (st["NP"] >= 2)).sum())))
    rules = pd.DataFrame(rows)
    if a.rule:
        best = a.rule
    else:
        ok = rules[(rules.fdr <= a.fdr) & (rules.observed > 0)]
        best, best_cov = None, -1
        for name in ok.sort_values("observed", ascending=False).head(12).rule:
            _, genes = v15.covered_genes(u, st, v15.RULES[name](st), key, left, dist)
            cov = int(genes.sum())
            rules.loc[rules.rule == name, "genes_covered"] = cov
            rules.loc[rules.rule == name, "query_genes_covered"] = int((genes & u.is_query).sum())
            if cov > best_cov:
                best, best_cov = name, cov
    rules.to_csv(out / "rules.tsv", sep="\t", index=False)
    res["rule"] = best
    log("rule", best)
    if best is None:
        json.dump(res, open(out / "summary.json", "w"), indent=1)
        return

    mask = v15.RULES[best](st)
    inst, genes = v15.covered_genes(u, st, mask, key, left, dist)
    sig_idx = np.searchsorted(st["key"], key[inst])
    li, di = left[inst], dist[inst]
    q_inst = u.is_query[li]
    xph = st["n_other"][sig_idx] > 0
    qg_any = np.zeros(u.n, dtype=bool)
    qg_x = np.zeros(u.n, dtype=bool)
    qg_any[li[q_inst]] = True
    qg_any[li[q_inst] + di[q_inst]] = True
    qg_x[li[q_inst & xph]] = True
    qg_x[li[q_inst & xph] + di[q_inst & xph]] = True
    row = rules[rules.rule == best].iloc[0]
    res.update(significant_pairs=int(mask.sum()), fdr=float(row.fdr),
               significant_pairs_cross_phylum=int((mask & (st["NP"] >= 2)).sum()),
               significant_pairs_5_phyla=int((mask & (st["NP"] >= 5)).sum()),
               genes_covered=int(genes.sum()), genes_covered_frac=float(genes.mean()),
               query_genes=int(u.is_query.sum()), query_genes_covered=int((qg_any & u.is_query).sum()),
               query_genes_covered_other_phylum_support=int((qg_x & u.is_query).sum()),
               L_quantiles=np.quantile(st["L"][mask], [.1, .25, .5, .75, .9, .99]).tolist(),
               tokens_in_significant_pairs=int(len(np.unique(np.r_[st["lo"][mask], st["hi"][mask]]))))
    if K > 1:
        res["level_of_significant_pairs"] = {specs[k][0]: int((st["level"][mask] == k).sum()) for k in range(K)}
        res["level_of_significant_instances"] = {specs[k][0]: int((ilev[inst] == k).sum()) for k in range(K)}
        log("levels of significant pairs / instances", res["level_of_significant_pairs"], res["level_of_significant_instances"])

    # ---- modules (v1.5 louvain): tokens of all levels are distinct nodes -------------------------------
    import networkx as nx
    in_mod = np.zeros(T, dtype=bool)
    H = nx.Graph()
    wt = st["J"][mask] * np.log2(np.maximum(st["L"][mask], 2).astype(np.float64))
    H.add_weighted_edges_from(zip(st["lo"][mask].tolist(), st["hi"][mask].tolist(), wt.tolist()))
    token_module = np.full(T, -1, dtype=np.int64)
    for i, c in enumerate(nx.community.louvain_communities(H, weight="weight", resolution=a.module_res, seed=17)):
        if len(c) >= 2:
            idx = np.fromiter(c, dtype=np.int64)
            token_module[idx] = i
            in_mod[idx] = True
    sizes = np.bincount(token_module[in_mod]) if in_mod.any() else np.zeros(1, dtype=np.int64)
    sizes = sizes[sizes > 0]
    res["modules"] = dict(method="louvain", module_res=a.module_res, module_j=0.5, strong_pairs=int(mask.sum()),
                          modules=int(len(sizes)), tokens_in_modules=int(in_mod.sum()),
                          largest=int(sizes.max()) if len(sizes) else 0,
                          size_quantiles=np.quantile(sizes, [.5, .9, .99]).tolist() if len(sizes) else [])
    log("modules", res["modules"])
    # a gene's module: that of its token at the first listed level that has one
    gene_module = np.full(u.n, -1, dtype=np.int64)
    off = 0
    for tok in toks:
        m = np.where(tok >= 0, token_module[np.maximum(tok, 0) + off], -1)
        gene_module = np.where(gene_module < 0, m, gene_module)
        off += int(tok.max()) + 1
    pair_module = token_module[st["lo"][mask]].copy()
    pair_module[token_module[st["hi"][mask]] != pair_module] = -1
    sig = pd.DataFrame({k: st[k][mask] for k in ("key", "lo", "hi", "G", "L", "NP", "n_other", "J", "E", "P", "F_lo", "F_hi")})
    sig["module"] = pair_module
    if K > 1:
        sig["level"] = st["level"][mask]
    sig.to_parquet(out / "significant_pairs.parquet", index=False)
    pd.DataFrame({"token": np.flatnonzero(in_mod), "module": token_module[in_mod]}).to_parquet(
        out / "token_modules.parquet", index=False)

    # ---- support landscape and loci (v1.5) ----------------------------------------------------------------
    value = st["L"][sig_idx].astype(np.float32)
    sig_rank = np.cumsum(mask) - 1
    b, arg = v15.boundary_support_argmax(li, di, value, sig_rank[sig_idx], u.n)
    np.save(out / "boundary_support.npy", b)
    k_of, g_of = st["incidence"]
    inc_idx = np.searchsorted(st["key"], k_of)
    keep = mask[inc_idx]
    bits = v15.fill_bitsets(sig_rank[inc_idx[keep]], g_of[keep], int(mask.sum()), (u.ng + 63) // 64)
    genome_arr, contig_arr = u.genes.genome.to_numpy(), u.genes.contig.to_numpy()

    def loci_for(cs, ce, levels, gaps, taus):
        parts = []

        def frame(s_idx, e_idx, kind, level):
            return pd.DataFrame(dict(genome=genome_arr[s_idx], contig=contig_arr[s_idx],
                                     start=u.pos[s_idx], end=u.pos[e_idx], kind=kind, level=float(level),
                                     module_id=v15.locus_modules(s_idx, e_idx, gene_module)))
        for gap in gaps:
            for level in levels:
                s_idx, e_idx = v15.runs_at_least(b, cs, ce, float(level), gap)
                parts.append(frame(s_idx, e_idx, "conserved" if gap == 0 else f"conserved_g{gap}", level))
        for tau in taus:
            s_idx, e_idx = v15.profile_units(arg, bits, cs, ce, tau)
            parts.append(frame(s_idx, e_idx, "unit", tau))
        return pd.concat(parts, ignore_index=True)

    loci_all = loci_for(u.c_start, u.c_end, (2, 5, 12, 30, 80), (0, 2), (0.5,))
    loci_all = loci_all[loci_all.end > loci_all.start].drop_duplicates(
        ["genome", "contig", "start", "end", "kind"]).reset_index(drop=True)
    loci_all.to_parquet(out / "loci_all_genomes.parquet", index=False)
    res["loci_all_genomes"] = {k: int(v) for k, v in loci_all.groupby("kind").size().items()}
    log("all-genome loci", res["loci_all_genomes"])
    res["seconds"] = round(time.time() - t0, 1)
    with open(out / "summary.json", "w") as fh:
        json.dump(res, fh, indent=1)
    log("done", tag, res["seconds"])


if __name__ == "__main__":
    main()
