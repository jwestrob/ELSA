#!/usr/bin/env python
"""Recurrence-first neighbourhood discovery, prototype engine (v1).

Input : a token file (gid -> family token) and a gene universe directory
        (genes.parquet, genomes.parquet).
Step 1: every pair of genes within `w` positions on a contig is an *instance* of an
        unordered token pair.
Step 2: per token pair, count the genomes (G), lineages (L, genus level) and phyla (NP)
        in which it is co-located; J = G / (genomes carrying the rarer token);
        E = expected genomes under random gene order given per-genome copy numbers;
        P = Poisson tail of G given E.
Step 3: null = the same computation after shuffling tokens within each genome
        (default; contig sizes kept) or within each contig.
        A rule's false-discovery rate is (null pairs passing) / (observed passing).
Step 4: modules = connected components of tokens joined by significant pairs with
        J >= `module_j`.
Step 5: support landscape. Each boundary between adjacent genes carries the largest
        support of any significant instance spanning it. Loci are read from it:
          conserved   maximal runs of boundaries with support >= m, for a ladder of m
          conserved_g the same, bridging up to g unsupported boundaries
          unit        runs whose best pairs share a genome profile (Jaccard >= tau)
          island      genes between supported flanks with no support inside
Step 6: optional benchmark (source207 on the Spacedust collection) and an all-genome
        locus table for external scorers.

No pairwise chaining and no per-gene neighbour budget: cost is linear in genes.
v1 over v0: universe directory as an argument, modules, gap-tolerant loci, all-genome
locus table. The statistics are unchanged, so v0 and v1 summaries are comparable.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import poisson

ROOT = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/spacedust")
RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
SCALE = ROOT / "model_scaling_20260913"
LEVELS = (2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500, 800)
SPAN = ["query_genome", "query_contig", "query_start", "query_end"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- kernels
@njit(parallel=True, cache=True)
def expected_genomes(lo, hi, indptr, gen, cnt, n_g, w):
    out = np.zeros(len(lo), dtype=np.float64)
    for p in prange(len(lo)):
        i, ie = indptr[lo[p]], indptr[lo[p] + 1]
        j, je = indptr[hi[p]], indptr[hi[p] + 1]
        s = 0.0
        while i < ie and j < je:
            if gen[i] == gen[j]:
                x = 2.0 * w * cnt[i] * cnt[j] / n_g[gen[i]]
                s += x if x < 1.0 else 1.0
                i += 1
                j += 1
            elif gen[i] < gen[j]:
                i += 1
            else:
                j += 1
        out[p] = s
    return out


@njit(cache=True)
def boundary_support_argmax(left, dist, value, sig_pos, n):
    b = np.zeros(n, dtype=np.float32)
    arg = np.full(n, -1, dtype=np.int64)
    for t in range(len(left)):
        v = value[t]
        for k in range(left[t], left[t] + dist[t]):
            if v > b[k]:
                b[k] = v
                arg[k] = sig_pos[t]
    return b, arg


@njit(cache=True)
def runs_at_least(b, starts, ends, level, gap):
    """Maximal runs of boundaries with b >= level inside each contig [start, end),
    bridging up to `gap` consecutive boundaries below the level. Inclusive gene ranges."""
    out_s, out_e = [], []
    for c in range(len(starts)):
        s, e = starts[c], ends[c]
        k = s
        while k < e - 1:
            if b[k] >= level:
                k0 = k
                last = k
                k += 1
                while k < e - 1:
                    if b[k] >= level:
                        last = k
                        k += 1
                    elif k - last <= gap:
                        k += 1
                    else:
                        break
                out_s.append(k0)
                out_e.append(last + 1)
                k = last + 1
            else:
                k += 1
    return np.array(out_s, dtype=np.int64), np.array(out_e, dtype=np.int64)


@njit(cache=True)
def islands(b, starts, ends, flank_level, max_len):
    out_s, out_e = [], []
    for c in range(len(starts)):
        s, e = starts[c], ends[c]
        k = s
        while k < e - 1:
            if b[k] == 0.0:
                k0 = k
                while k < e - 1 and b[k] == 0.0:
                    k += 1
                k1 = k - 1
                left_ok = k0 - 1 >= s and b[k0 - 1] >= flank_level
                right_ok = k1 + 1 < e - 1 and b[k1 + 1] >= flank_level
                n_genes = k1 - k0
                if left_ok and right_ok and 1 <= n_genes <= max_len:
                    out_s.append(k0 + 1)
                    out_e.append(k1)
            else:
                k += 1
    return np.array(out_s, dtype=np.int64), np.array(out_e, dtype=np.int64)


@njit(cache=True)
def fill_bitsets(rows, genomes, n_rows, n_words):
    bits = np.zeros((n_rows, n_words), dtype=np.uint64)
    one = np.uint64(1)
    for i in range(len(rows)):
        bits[rows[i], genomes[i] // 64] |= one << np.uint64(genomes[i] % 64)
    return bits


@njit(cache=True)
def _popcount(x):
    c = 0
    while x:
        x &= x - np.uint64(1)
        c += 1
    return c


@njit(cache=True)
def _jaccard(a, b):
    inter, union = 0, 0
    for w in range(len(a)):
        inter += _popcount(a[w] & b[w])
        union += _popcount(a[w] | b[w])
    return inter / union if union else 0.0


@njit(cache=True)
def profile_units(arg, bits, starts, ends, tau):
    out_s, out_e = [], []
    for c in range(len(starts)):
        s, e = starts[c], ends[c]
        k = s
        while k < e - 1:
            if arg[k] < 0:
                k += 1
                continue
            k0 = k
            while (k + 1 < e - 1 and arg[k + 1] >= 0
                   and _jaccard(bits[arg[k]], bits[arg[k + 1]]) >= tau):
                k += 1
            out_s.append(k0)
            out_e.append(k + 1)
            k += 1
    return np.array(out_s, dtype=np.int64), np.array(out_e, dtype=np.int64)


@njit(cache=True)
def union_find(n, a, b):
    parent = np.arange(n)
    for i in range(len(a)):
        x = a[i]
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        y = b[i]
        while parent[y] != y:
            parent[y] = parent[parent[y]]
            y = parent[y]
        if x != y:
            if x < y:
                parent[y] = x
            else:
                parent[x] = y
    for i in range(n):
        x = i
        while parent[x] != x:
            x = parent[x]
        parent[i] = x
    return parent


@njit(cache=True)
def locus_modules(s_idx, e_idx, arg, pair_module):
    """Plurality module among the best pairs of a locus's boundaries (-1 if none)."""
    out = np.full(len(s_idx), -1, dtype=np.int64)
    for i in range(len(s_idx)):
        n = e_idx[i] - s_idx[i]
        if n <= 0:
            continue
        buf = np.empty(n, dtype=np.int64)
        m = 0
        for k in range(s_idx[i], e_idx[i]):
            if arg[k] >= 0 and pair_module[arg[k]] >= 0:
                buf[m] = pair_module[arg[k]]
                m += 1
        if m == 0:
            continue
        buf = np.sort(buf[:m])
        best, best_n, run = buf[0], 1, 1
        for j in range(1, m):
            if buf[j] == buf[j - 1]:
                run += 1
            else:
                run = 1
            if run > best_n:
                best, best_n = buf[j], run
        out[i] = best
    return out


# --------------------------------------------------------------------------- core
class Universe:
    def __init__(self, directory):
        directory = Path(directory)
        g = pd.read_parquet(directory / "genes.parquet")
        if "side" not in g:
            g["side"] = "ref"
        g = g[["gid", "side", "genome", "genome_code", "contig", "contig_code", "position_index", "strand"]]
        g = g.sort_values(["contig_code", "position_index"], kind="stable").reset_index(drop=True)
        self.genes = g
        self.n = len(g)
        self.gid = g.gid.to_numpy()
        self.cc = g.contig_code.to_numpy(dtype=np.int64)
        self.gc = g.genome_code.to_numpy(dtype=np.int64)
        self.pos = g.position_index.to_numpy(dtype=np.int64)
        self.is_query = (g.side == "query").to_numpy()
        change = np.flatnonzero(np.diff(self.cc)) + 1
        self.c_start = np.r_[0, change].astype(np.int64)
        self.c_end = np.r_[change, self.n].astype(np.int64)
        # genome blocks: contigs of one genome are adjacent in contig_code order
        gchange = np.flatnonzero(np.diff(self.gc)) + 1
        self.g_start = np.r_[0, gchange].astype(np.int64)
        self.g_end = np.r_[gchange, self.n].astype(np.int64)
        assert len(self.g_start) == len(np.unique(self.gc)), "genes of a genome are not contiguous"
        gm = pd.read_parquet(directory / "genomes.parquet").sort_values("genome_code").reset_index(drop=True)
        assert np.array_equal(gm.genome_code.to_numpy(), np.arange(len(gm))), "genome codes must be 0..n-1"
        self.genomes = gm
        self.ng = len(gm)
        self.n_g = gm.n_genes.to_numpy(dtype=np.float64)
        genus = gm.genus.where(gm.genus.notna() & (gm.genus != ""), "genome:" + gm.genome)
        self.lineage, _ = pd.factorize(genus)
        self.nl = int(self.lineage.max()) + 1
        phylum = gm.phylum.where(gm.phylum.notna() & (gm.phylum != ""), "unk:" + gm.genome)
        self.phylum, _ = pd.factorize(phylum)
        self.np_ = int(self.phylum.max()) + 1
        self.other_phylum = ((gm.stratum == "3_other_phylum").to_numpy() if "stratum" in gm
                             else np.zeros(self.ng, dtype=bool))
        self.query_contig_mask = self.is_query[self.c_start]


def colocalization(u, tok, w, oriented=False, strand=None):
    T = int(tok.max()) + 1
    keys, lefts, dists = [], [], []
    for d in range(1, w + 1):
        left = np.flatnonzero(u.cc[:-d] == u.cc[d:])
        ta, tb = tok[left], tok[left + d]
        ok = (ta >= 0) & (tb >= 0) & (ta != tb)
        left, ta, tb = left[ok], ta[ok], tb[ok]
        key = np.minimum(ta, tb) * T + np.maximum(ta, tb)
        if oriented:
            key = key * 2 + (strand[left] == strand[left + d]).astype(np.int64)
        keys.append(key)
        lefts.append(left)
        dists.append(np.full(len(left), d, dtype=np.int64))
    return np.concatenate(keys), np.concatenate(lefts), np.concatenate(dists), T


def pair_statistics(u, tok, key, left, w, T, oriented=False):
    g = u.gc[left]
    kg = np.unique(key * u.ng + g)
    k_of, g_of = kg // u.ng, kg % u.ng
    ukey, first, G = np.unique(k_of, return_index=True, return_counts=True)
    kl = np.unique(k_of * u.nl + u.lineage[g_of])
    L = np.unique(kl // u.nl, return_counts=True)[1]
    kp = np.unique(k_of * u.np_ + u.phylum[g_of])
    NP = np.unique(kp // u.np_, return_counts=True)[1]
    n_other = np.add.reduceat(u.other_phylum[g_of].astype(np.int64), first)
    base = ukey // 2 if oriented else ukey
    lo, hi = base // T, base % T
    valid = tok >= 0
    tg, cnt = np.unique(tok[valid] * u.ng + u.gc[valid], return_counts=True)
    t_of, gen = tg // u.ng, tg % u.ng
    indptr = np.searchsorted(t_of, np.arange(T + 1)).astype(np.int64)
    F = np.diff(indptr)
    E = expected_genomes(lo, hi, indptr, gen.astype(np.int64), cnt.astype(np.float64), u.n_g, float(w))
    if oriented:
        E = E / 2.0
    J = G / np.minimum(F[lo], F[hi])
    P = poisson.sf(G - 1, np.maximum(E, 1e-12))
    return dict(key=ukey, lo=lo, hi=hi, G=G, L=L, NP=NP, n_other=n_other, J=J, E=E, P=P,
                F_lo=F[lo], F_hi=F[hi], incidence=(k_of, g_of))


def shuffle_within(starts, ends, tok, rng):
    out = tok.copy()
    for s, e in zip(starts, ends):
        out[s:e] = rng.permutation(tok[s:e])
    return out


RULES = {}
for m in (2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50):
    RULES[f"L>={m}"] = (lambda st, m=m: st["L"] >= m)
    for j in (0.2, 0.4, 0.6, 0.8):
        RULES[f"L>={m}&J>={j}"] = (lambda st, m=m, j=j: (st["L"] >= m) & (st["J"] >= j))
for e in (2, 3, 4, 6, 8, 10, 15, 20, 30):
    RULES[f"P<=1e-{e}&L>=2"] = (lambda st, e=e: (st["P"] <= 10.0 ** -e) & (st["L"] >= 2))
    RULES[f"P<=1e-{e}&L>=3"] = (lambda st, e=e: (st["P"] <= 10.0 ** -e) & (st["L"] >= 3))


def covered_genes(u, st, mask, key, left, dist):
    sig = np.zeros(len(st["key"]), dtype=bool)
    sig[mask] = True
    inst = sig[np.searchsorted(st["key"], key)]
    genes = np.zeros(u.n, dtype=bool)
    genes[left[inst]] = True
    genes[left[inst] + dist[inst]] = True
    return inst, genes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tokens", type=Path, help="token parquet (default: $RD/tokens/<arm>.parquet)")
    ap.add_argument("--universe", type=Path, default=RD / "universe")
    ap.add_argument("--w", type=int, default=3)
    ap.add_argument("--perms", type=int, default=3)
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--rule", help="use this rule instead of choosing by coverage at the FDR target")
    ap.add_argument("--module-j", type=float, default=0.5)
    ap.add_argument("--oriented", action="store_true")
    ap.add_argument("--null", choices=["genome", "contig"], default="genome",
                    help="shuffle gene order within each genome (contig sizes kept) or within each contig")
    ap.add_argument("--benchmark", choices=["source207", "none"], default="source207")
    ap.add_argument("--all-genome-loci", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)

    tag = f"{a.arm}__w{a.w}{'__oriented' if a.oriented else ''}"
    out = a.out / tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    u = Universe(a.universe)
    tokens = pd.read_parquet(a.tokens or RD / "tokens" / f"{a.arm}.parquet")
    assert len(tokens) == u.n and tokens.gid.is_unique
    tok_by_gid = np.full(u.n, -1, dtype=np.int64)
    tok_by_gid[tokens.gid.to_numpy()] = tokens.token.to_numpy()
    tok = tok_by_gid[u.gid]
    strand = u.genes.strand.to_numpy(dtype=np.int64)
    n_start, n_end = (u.g_start, u.g_end) if a.null == "genome" else (u.c_start, u.c_end)
    res = dict(engine="v1.1", null=a.null, arm=a.arm, universe=str(a.universe), w=a.w, oriented=a.oriented,
               perms=a.perms, genes=u.n, genomes=u.ng, assigned=int((tok >= 0).sum()),
               tokens=int(len(np.unique(tok[tok >= 0]))))
    log("universe + tokens", res, f"{time.time()-t0:.0f}s")

    key, left, dist, T = colocalization(u, tok, a.w, a.oriented, strand)
    st = pair_statistics(u, tok, key, left, a.w, T, a.oriented)
    res.update(instances=int(len(key)), token_pairs=int(len(st["key"])),
               token_pairs_L2=int((st["L"] >= 2).sum()), stats_seconds=round(time.time() - t0, 1))
    log("observed", {k: res[k] for k in ("instances", "token_pairs", "token_pairs_L2")}, f"{time.time()-t0:.0f}s")

    nulls = []
    for r in range(a.perms):
        rng = np.random.default_rng(1000 + r)
        if a.oriented:
            order = np.arange(u.n)
            for s, e in zip(n_start, n_end):
                order[s:e] = rng.permutation(order[s:e])
            tok_n, strand_n = tok[order], strand[order]
        else:
            tok_n, strand_n = shuffle_within(n_start, n_end, tok, rng), strand
        k_n, l_n, d_n, _ = colocalization(u, tok_n, a.w, a.oriented, strand_n)
        nulls.append(pair_statistics(u, tok_n, k_n, l_n, a.w, T, a.oriented))
        log("null", r, "token pairs", len(nulls[-1]["key"]), "L>=2", int((nulls[-1]["L"] >= 2).sum()),
            f"{time.time()-t0:.0f}s")

    rows = []
    for name, rule in RULES.items():
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
            _, genes = covered_genes(u, st, RULES[name](st), key, left, dist)
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

    mask = RULES[best](st)
    inst, genes = covered_genes(u, st, mask, key, left, dist)
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

    # ---- modules: tokens joined by significant, consistent pairs --------------------
    strong = mask & (st["J"] >= a.module_j)
    parent = union_find(T, st["lo"][strong].astype(np.int64), st["hi"][strong].astype(np.int64))
    in_mod = np.zeros(T, dtype=bool)
    in_mod[st["lo"][strong]] = True
    in_mod[st["hi"][strong]] = True
    token_module = np.where(in_mod, parent, -1)
    sizes = np.bincount(parent[in_mod]) if in_mod.any() else np.zeros(1, dtype=np.int64)
    sizes = sizes[sizes > 0]
    res["modules"] = dict(module_j=a.module_j, strong_pairs=int(strong.sum()), modules=int(len(sizes)),
                          tokens_in_modules=int(in_mod.sum()), largest=int(sizes.max()) if len(sizes) else 0,
                          size_quantiles=np.quantile(sizes, [.5, .9, .99]).tolist() if len(sizes) else [])
    pair_module = token_module[st["lo"][mask]]
    sig = pd.DataFrame({k: st[k][mask] for k in ("key", "lo", "hi", "G", "L", "NP", "n_other", "J", "E", "P", "F_lo", "F_hi")})
    sig["module"] = pair_module
    sig.to_parquet(out / "significant_pairs.parquet", index=False)
    pd.DataFrame({"token": np.flatnonzero(in_mod), "module": parent[in_mod]}).to_parquet(
        out / "token_modules.parquet", index=False)

    # ---- support landscape and loci ----------------------------------------------------
    value = st["L"][sig_idx].astype(np.float32)
    sig_rank = np.cumsum(mask) - 1
    b, arg = boundary_support_argmax(li, di, value, sig_rank[sig_idx], u.n)
    np.save(out / "boundary_support.npy", b)
    k_of, g_of = st["incidence"]
    inc_idx = np.searchsorted(st["key"], k_of)
    keep = mask[inc_idx]
    bits = fill_bitsets(sig_rank[inc_idx[keep]], g_of[keep], int(mask.sum()), (u.ng + 63) // 64)
    genome_arr, contig_arr = u.genes.genome.to_numpy(), u.genes.contig.to_numpy()

    def loci_for(cs, ce, levels, gaps, taus, flanks):
        parts = []

        def frame(s_idx, e_idx, kind, level):
            return pd.DataFrame(dict(genome=genome_arr[s_idx], contig=contig_arr[s_idx],
                                     start=u.pos[s_idx], end=u.pos[e_idx], kind=kind, level=float(level),
                                     module_id=locus_modules(s_idx, e_idx, arg, pair_module)))
        for gap in gaps:
            for level in levels:
                s_idx, e_idx = runs_at_least(b, cs, ce, float(level), gap)
                parts.append(frame(s_idx, e_idx, "conserved" if gap == 0 else f"conserved_g{gap}", level))
        for flank in flanks:
            s_idx, e_idx = islands(b, cs, ce, float(flank), 300)
            parts.append(frame(s_idx, e_idx, "island", flank))
        for tau in taus:
            s_idx, e_idx = profile_units(arg, bits, cs, ce, tau)
            parts.append(frame(s_idx, e_idx, "unit", tau))
        return pd.concat(parts, ignore_index=True)

    if a.all_genome_loci:
        loci_all = loci_for(u.c_start, u.c_end, (2, 5, 12, 30, 80), (0, 2), (0.5,), ())
        loci_all = loci_all[loci_all.end > loci_all.start].drop_duplicates(
            ["genome", "contig", "start", "end", "kind"]).reset_index(drop=True)
        loci_all.to_parquet(out / "loci_all_genomes.parquet", index=False)
        res["loci_all_genomes"] = {k: int(v) for k, v in loci_all.groupby("kind").size().items()}
        log("all-genome loci", res["loci_all_genomes"])

    if a.benchmark == "source207":
        sys.path.insert(0, str(SCALE / "code"))
        from evaluate_spacedust_queries import evaluate
        import run_spacedust_scaling as rss
        qs, qe = u.c_start[u.query_contig_mask], u.c_end[u.query_contig_mask]
        loci = loci_for(qs, qe, LEVELS, (0, 1, 2), (0.3, 0.5, 0.7, 0.9), (2, 5, 20))
        loci = loci.rename(columns=dict(genome="query_genome", contig="query_contig",
                                        start="query_start", end="query_end"))
        loci.to_parquet(out / "query_loci.parquet", index=False)
        res["query_loci"] = {f"{k}@{lv:g}": int(n) for (k, lv), n in loci.groupby(["kind", "level"]).size().items()}
        regions = pd.read_parquet(ROOT / "evaluation/deepbgc_s8_207/regions.parquet")
        qmeta = rss.complete_metadata(pd.read_parquet(ROOT / "bgc_queries/metadata.parquet"), rss.QUERY_ROWS)
        rng = np.random.default_rng(17)

        def score(df, label):
            spans = df[SPAN].drop_duplicates()
            if spans.empty:
                return dict(selection=label, n_intervals=0)
            observed, ctrl, report = evaluate(regions, qmeta, spans, threads=a.threads, controls=200, seed=17)
            per = (observed.set_index("region_id").f1 - ctrl.groupby("region_id").f1.mean()).to_numpy()
            boot = rng.choice(per, size=(10000, len(per)), replace=True).mean(axis=1)
            r = dict(selection=label, n_intervals=len(spans),
                     f1_observed=report["observed"]["mean_best_single_block_f1"],
                     f1_random=report["controls"]["mean_best_single_block_f1"], f1_excess=float(per.mean()),
                     ci_lo=float(np.quantile(boot, .025)), ci_hi=float(np.quantile(boot, .975)),
                     localized_observed=report["observed"]["localized_fraction"],
                     localized_random=report["controls"]["localized_fraction"],
                     per_region_f1=observed.set_index("region_id").f1.round(6).to_dict(),
                     per_region_random_f1=ctrl.groupby("region_id").f1.mean().round(6).to_dict())
            log(label, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                        if k not in ("selection", "per_region_f1", "per_region_random_f1")})
            return r

        k = loci.kind
        res["source207"] = [
            score(loci[k == "conserved"], "conserved loci, all levels"),
            score(loci[k == "conserved_g1"], "conserved loci, gap 1"),
            score(loci[k == "conserved_g2"], "conserved loci, gap 2"),
            score(loci[k.str.startswith("conserved")], "conserved loci, gaps 0-2"),
            score(loci[k == "island"], "islands"),
            score(loci[k == "unit"], "profile units, all tau"),
            score(loci[k.str.startswith("conserved") | (k == "unit")], "conserved (gaps 0-2) + profile units"),
            score(loci, "all locus kinds"),
        ]
    res["seconds"] = round(time.time() - t0, 1)
    with open(out / "summary.json", "w") as fh:
        json.dump(res, fh, indent=1)
    log("done", tag, res["seconds"])


if __name__ == "__main__":
    main()
