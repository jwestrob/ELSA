#!/usr/bin/env python
"""Shared code for the boundary workstream (locus-reading rules on the recurrence engine's own outputs).

Landscape: the engine's boundary support recomputed from the arm's tokens and significant_pairs.parquet with
the engine's own functions (read-only import of $RD/engine/code/recurrence.py) and checked equal to the saved
boundary_support.npy. Per boundary k (between gid k and k+1):
  b      max L (genera) over significant instances spanning k (= the engine's boundary_support)
  arg    index (row of significant_pairs.parquet) of the pair that sets b (engine tie rule: first max)
  bJ     max J over significant instances spanning k
  nspan  number of significant instances spanning k
Gene module = token module of the gene's token (token_modules.parquet; -1 if none).
Pair module = significant_pairs.module (-1 for a pair between two modules).

Rules return inclusive gid intervals (s, e) with e > s inside one contig; `table()` turns them into the
scorer's locus table (genome, contig, start, end, module_id) with the engine's plurality-module rule.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
BD = RD / "boundary"
sys.path.insert(0, str(RD / "engine/code"))
import recurrence as R  # noqa: E402  (read-only import)
sys.path.insert(0, str(PKG / "scorer"))
import score_dpann_loci as S  # noqa: E402
sys.path.insert(0, str(RD / "ribosomal/code"))
import ribosomal_check as RC  # noqa: E402

ARMS = {
    "rescue3": RD / "engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3",
    "base": RD / "engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3",
    "hybrid": RD / "engine/results_dpann_emb1/dpann__hybrid__esmpp_small__genomecentred__e3c50cpmr0p05__w3",
}
LADDER = (2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500, 800)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def token_path(arm):
    x = arm[len("dpann__"):] if arm.startswith("dpann__") else arm
    for p in (RD / "engine/dpann_tokens" / f"{x}__dpann.parquet", RD / "engine/dpann_tokens" / f"{x}.parquet",
              PKG / "tokens" / f"{x}__dpann.parquet", RD / "tokens" / f"{x}.parquet"):
        if p.exists():
            return p
    raise FileNotFoundError(arm)


# --------------------------------------------------------------------------- landscape
class Landscape:
    def __init__(self, rdir, universe=None, tokens=None, verify=True):
        rdir = Path(rdir)
        self.rdir = rdir
        self.summary = json.load(open(rdir / "summary.json"))
        self.w = int(self.summary["w"])
        u = R.Universe(Path(universe) if universe else PKG / "universe")
        assert np.array_equal(u.gid, np.arange(u.n)), "universe order != gid order"
        self.u = u
        tp = Path(tokens) if tokens else token_path(self.summary["arm"])
        self.tokens_path = tp
        t = pd.read_parquet(tp)
        tok = np.full(u.n, -1, dtype=np.int64)
        tok[t.gid.to_numpy()] = t.token.to_numpy()
        self.tok = tok
        key, left, dist, T = R.colocalization(u, tok, self.w)
        sp = pd.read_parquet(rdir / "significant_pairs.parquet")
        assert (sp.key.to_numpy() == sp.lo.to_numpy() * T + sp.hi.to_numpy()).all(), "pair key / T mismatch"
        assert np.all(np.diff(sp.key.to_numpy()) > 0)
        self.sp = sp
        skey = sp.key.to_numpy()
        pidx = np.minimum(np.searchsorted(skey, key), len(skey) - 1)
        found = skey[pidx] == key
        li, di, pi = left[found], dist[found], pidx[found].astype(np.int64)
        self.inst_left, self.inst_dist, self.inst_pair = li, di, pi
        L = sp.L.to_numpy().astype(np.float32)
        J = sp.J.to_numpy().astype(np.float32)
        b, arg = R.boundary_support_argmax(li, di, L[pi], pi, u.n)
        self.b, self.arg = b, arg
        if verify:
            saved = np.load(rdir / "boundary_support.npy")
            assert np.array_equal(b, saved), "boundary support not reproduced"
        self.bJ, _ = R.boundary_support_argmax(li, di, J[pi], pi, u.n)
        self.nspan = span_count(li, di, u.n)
        tm = pd.read_parquet(rdir / "token_modules.parquet")
        token_module = np.full(T, -1, dtype=np.int64)
        token_module[tm.token.to_numpy()] = tm.module.to_numpy()
        self.T = T
        self.token_module = token_module
        self.gene_module = np.where(tok >= 0, token_module[np.maximum(tok, 0)], -1).astype(np.int64)
        self.pair_module = sp.module.to_numpy().astype(np.int64)
        self.cs, self.ce = u.c_start, u.c_end
        self.genome_arr = u.genes.genome.to_numpy()
        self.contig_arr = u.genes.contig.to_numpy()
        self.genome_code = u.gc
        self.s2, self.e2 = R.runs_at_least(b, self.cs, self.ce, 2.0, 0)

    def boundary_labels(self, pair_module=None):
        pm = self.pair_module if pair_module is None else pair_module
        return np.where(self.arg >= 0, pm[np.maximum(self.arg, 0)], -2).astype(np.int64)

    def table(self, s, e, gene_module=None, extra=None):
        s = np.asarray(s, dtype=np.int64)
        e = np.asarray(e, dtype=np.int64)
        keep = e > s
        s, e = s[keep], e[keep]
        gm = self.gene_module if gene_module is None else gene_module
        df = pd.DataFrame(dict(genome=self.genome_arr[s], contig=self.contig_arr[s], start=self.u.pos[s],
                               end=self.u.pos[e], module_id=R.locus_modules(s, e, gm), gid_s=s, gid_e=e))
        if extra:
            for k, v in extra.items():
                df[k] = np.asarray(v)[keep]
        return df


@njit(cache=True)
def span_count(left, dist, n):
    c = np.zeros(n, dtype=np.int32)
    for t in range(len(left)):
        for k in range(left[t], left[t] + dist[t]):
            c[k] += 1
    return c


# --------------------------------------------------------------------------- rule kernels
@njit(cache=True)
def trim_ends(b, s2, e2, f):
    """(a) Relative trimming: drop end genes while the end boundary's support < f * median internal support
    of the level-2 locus (median fixed from the untrimmed locus). Keeps >= 1 boundary when f <= 1."""
    os_, oe = s2.copy(), e2.copy()
    for i in range(len(s2)):
        s, e = s2[i], e2[i]
        thr = f * np.median(b[s:e])
        while e - s > 1 and b[s] < thr:
            s += 1
        while e - s > 1 and b[e - 1] < thr:
            e -= 1
        os_[i] = s
        oe[i] = e
    return os_, oe


@njit(cache=True)
def trim_ends_log(b, s2, e2, f):
    """(a') Relative trimming in log space: drop end genes while log(b_end) < f * median log(b) internal."""
    os_, oe = s2.copy(), e2.copy()
    for i in range(len(s2)):
        s, e = s2[i], e2[i]
        thr = f * np.median(np.log(b[s:e]))
        while e - s > 1 and np.log(b[s]) < thr:
            s += 1
        while e - s > 1 and np.log(b[e - 1]) < thr:
            e -= 1
        os_[i] = s
        oe[i] = e
    return os_, oe


@njit(cache=True)
def runs_within(b, s, e, level, out_s, out_e, n):
    """Append maximal runs of boundaries with b >= level inside the gene interval [s, e] (boundaries s..e-1)."""
    k = s
    while k < e:
        if b[k] >= level:
            k0 = k
            while k < e and b[k] >= level:
                k += 1
            out_s[n] = k0
            out_e[n] = k
            n += 1
        else:
            k += 1
    return n


@njit(cache=True)
def edge_contrast(b, s, e, level, h):
    """Mean over the edges of the runs at `level` inside [s, e] of
    log(mean b over the h boundaries just inside) - log(max(mean b over the h boundaries just outside, 1)),
    where 'outside' stops at the level-2 locus ends (support 0 beyond them). Returns (contrast, n_runs)."""
    tot, ne, nr = 0.0, 0, 0
    k = s
    while k < e:
        if b[k] >= level:
            k0 = k
            while k < e and b[k] >= level:
                k += 1
            k1 = k - 1
            nr += 1
            # left edge
            ins, ni = 0.0, 0
            for j in range(k0, min(k0 + h, k1 + 1)):
                ins += b[j]
                ni += 1
            out, no = 0.0, 0
            for j in range(k0 - 1, k0 - 1 - h, -1):
                if j < s:
                    break
                out += b[j]
                no += 1
            o = out / no if no else 0.0
            tot += np.log(ins / ni) - np.log(max(o, 1.0))
            ne += 1
            # right edge
            ins, ni = 0.0, 0
            for j in range(k1, max(k1 - h, k0 - 1), -1):
                ins += b[j]
                ni += 1
            out, no = 0.0, 0
            for j in range(k1 + 1, k1 + 1 + h):
                if j >= e:
                    break
                out += b[j]
                no += 1
            o = out / no if no else 0.0
            tot += np.log(ins / ni) - np.log(max(o, 1.0))
            ne += 1
        else:
            k += 1
    return (tot / ne if ne else -1e9), nr


@njit(cache=True)
def per_locus_level(b, s2, e2, ladder, h, min_gain, keep_low):
    """(b) Per-locus level. For each level-2 locus, every ladder level <= its max support is a candidate; the
    reading at a level is its runs of boundaries >= level. The chosen level maximizes the mean edge contrast
    (edge_contrast with window h); a level above 2 is taken only if its contrast exceeds level 2's by
    `min_gain` (natural-log units). keep_low=1 also emits the stretches between the chosen runs (>= 2 genes)
    so that no gene of the level-2 locus leaves every locus.
    Returns (s, e, locus index, chosen level, is_core)."""
    cap = 0
    for i in range(len(s2)):
        cap += (e2[i] - s2[i]) * 2 + 2
    out_s = np.empty(cap, dtype=np.int64)
    out_e = np.empty(cap, dtype=np.int64)
    out_i = np.empty(cap, dtype=np.int64)
    out_l = np.empty(cap, dtype=np.float64)
    out_c = np.empty(cap, dtype=np.int8)
    n = 0
    for i in range(len(s2)):
        s, e = s2[i], e2[i]
        tmp_s = np.empty(e - s + 1, dtype=np.int64)
        tmp_e = np.empty(e - s + 1, dtype=np.int64)
        mx = b[s:e].max()
        c2, _ = edge_contrast(b, s, e, 2.0, h)
        best_l, best_c = 2.0, c2
        for lv in ladder:
            if lv <= 2.0 or lv > mx:
                continue
            c, nr = edge_contrast(b, s, e, lv, h)
            if c > best_c:
                best_c, best_l = c, lv
        if best_l > 2.0 and best_c - c2 < min_gain:
            best_l = 2.0
        m = runs_within(b, s, e, best_l, tmp_s, tmp_e, 0)
        prev_end = s  # first gene not yet covered by a core run
        for r in range(m):
            rs, re = tmp_s[r], tmp_e[r]
            if keep_low == 1 and rs - 1 > prev_end:
                # low stretch between prev_end and rs: genes prev_end .. rs (the junction gene rs stays with the core)
                out_s[n] = prev_end
                out_e[n] = rs - 1
                out_i[n] = i
                out_l[n] = best_l
                out_c[n] = 0
                n += 1
            out_s[n] = rs
            out_e[n] = re
            out_i[n] = i
            out_l[n] = best_l
            out_c[n] = 1
            n += 1
            prev_end = re + 1
        if keep_low == 1 and e > prev_end:
            out_s[n] = prev_end
            out_e[n] = e
            out_i[n] = i
            out_l[n] = best_l
            out_c[n] = 0
            n += 1
    return out_s[:n], out_e[:n], out_i[:n], out_l[:n], out_c[:n]


@njit(cache=True)
def label_segments(b, lab, s2, e2):
    """(c) Module-consistent reading. Boundary labels inside each level-2 locus (-1 = pair between modules,
    -2 = no pair) are filled forward then backward from the nearest labelled boundary; the locus is cut where
    the filled label changes. Adjacent segments share one gene, which stays with the side whose junction
    boundary has the larger support. Returns (s, e, locus index)."""
    cap = 0
    for i in range(len(s2)):
        cap += e2[i] - s2[i] + 1
    out_s = np.empty(cap, dtype=np.int64)
    out_e = np.empty(cap, dtype=np.int64)
    out_i = np.empty(cap, dtype=np.int64)
    n = 0
    for i in range(len(s2)):
        s, e = s2[i], e2[i]
        nb = e - s
        f = np.empty(nb, dtype=np.int64)
        last = -9
        for j in range(nb):
            v = lab[s + j]
            if v >= 0:
                last = v
            f[j] = last
        nxt = -9
        for j in range(nb - 1, -1, -1):
            if f[j] == -9:
                f[j] = nxt
            else:
                nxt = f[j]
        # segments of equal label over boundaries; gene ranges with junction-gene assignment
        seg_start = s  # first gene of the current segment
        for j in range(nb - 1):
            if f[j + 1] != f[j]:
                k1 = s + j  # last boundary of the left segment; junction gene k1 + 1
                if b[k1] >= b[k1 + 1]:
                    left_end, right_start = k1 + 1, k1 + 2
                else:
                    left_end, right_start = k1, k1 + 1
                out_s[n] = seg_start
                out_e[n] = left_end
                out_i[n] = i
                n += 1
                seg_start = right_start
        out_s[n] = seg_start
        out_e[n] = e
        out_i[n] = i
        n += 1
    return out_s[:n], out_e[:n], out_i[:n]


# --------------------------------------------------------------------------- scoring
def odd_even_code():
    g = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome", "genome_code"])
    return dict(zip(g.genome, g.genome_code))


def make_split_pkg(parity):
    """Shadow package whose gold holds only genomes with genome_code % 2 == parity (universe symlinked)."""
    d = BD / ("pkg_odd" if parity == 1 else "pkg_even")
    if (d / "gold/instances.parquet").exists():
        return d
    (d / "gold").mkdir(parents=True, exist_ok=True)
    if not (d / "universe").exists():
        (d / "universe").symlink_to(PKG / "universe")
    inst = pd.read_parquet(PKG / "gold/instances.parquet")
    code = odd_even_code()
    keep = inst.genome.map(code).to_numpy() % 2 == parity
    tmp = d / "gold/instances.tmp.parquet"
    inst[keep].to_parquet(tmp, index=False)
    tmp.rename(d / "gold/instances.parquet")
    return d


def score_table(pkg, df, threads, controls=200):
    per, report, _ = S.score(pkg, df[["genome", "contig", "start", "end", "module_id"]], controls=controls, seed=17,
                             threads=threads, min_members=2)
    per["excess"] = per.f1 - per.control_mean_f1
    return per, report


def summarize_per(per):
    out = {}
    for c, g in per.groupby("caller"):
        hit = g.coverage >= 0.5
        out[c] = dict(n=int(len(g)), f1=float(g.f1.mean()), control=float(g.control_mean_f1.mean()),
                      excess=float(g.excess.mean()),
                      median_bprec_covered=float(g.boundary_precision[hit].median()) if hit.any() else None,
                      covered50=float(g.covered50.mean()), localized=float(g.localized.mean()),
                      median_best_span=float(g.best_span[g.best_span > 0].median()) if (g.best_span > 0).any() else None)
    out["summed_DF_TXSS_excess"] = out.get("DefenseFinder", {}).get("excess", 0) + out.get("TXSScan", {}).get("excess", 0)
    return out


def ribo_runs():
    r = pd.read_parquet(RC.CACHE / "runs.parquet")
    return r[(r.run_set == "g2") & (r.n_marked >= 16)].reset_index(drop=True)


def ribo_locset(s, e, mod=None):
    s = np.asarray(s, dtype=np.int64)
    e = np.asarray(e, dtype=np.int64)
    mod = np.full(len(s), -1, dtype=np.int64) if mod is None else np.asarray(mod, dtype=np.int64)
    o = np.argsort(s, kind="stable")
    return dict(a=s[o], b=e[o], mod=mod[o], maxlen=int((e - s + 1).max()) if len(s) else 1, n=int(len(s)))


def ribo_score(runs, s, e, mod=None):
    sc = RC.score(ribo_locset(s, e, mod), runs.gid_start.to_numpy(), runs.gid_end.to_numpy())
    return sc


def ribo_summary(runs, sc):
    d = RC.summarize(runs, sc)
    keys = ("runs", "recovered", "resolved", "median_precision_recovered", "median_ext_left_recovered",
            "median_ext_right_recovered", "median_fragments", "parvarchaeaceae", "parvarchaeaceae_resolved",
            "median_coverage")
    return {k: d[k] for k in keys}


def locus_stats(s, e, n_genes):
    s = np.asarray(s)
    e = np.asarray(e)
    keep = e > s
    s, e = s[keep], e[keep]
    cov = np.zeros(n_genes + 1, dtype=np.int32)
    np.add.at(cov, s, 1)
    np.add.at(cov, e + 1, -1)
    covered = int((np.cumsum(cov)[:n_genes] > 0).sum())
    span = e - s + 1
    return dict(loci=int(len(s)), genes_covered=covered, median_span=float(np.median(span)) if len(s) else None,
                mean_span=float(span.mean()) if len(s) else None)
