#!/usr/bin/env python
"""Ribosomal-locus standing check for recurrence-engine result directories (DPANN package).

Gold runs (built once, cached under $RD/ribosomal/gold/):
  per contig, maximal runs of genes carrying a Pfam domain whose name matches
  ^ribosom(al)?_ (case-insensitive; an observed domain used only as a positional marker),
  allowing up to `gap` unmarked genes inside a run (g0 strict, g2 tolerant); runs with
  >= 8 marked genes are kept and the >= 16 subset is the primary set.
Controls (cached once, identical for every arm):
  random     20 length-matched windows per run on the same contig, avoiding every gold run
             (other contigs of the same genome when the contig has no room)
  transplant genome labels shuffled across runs (5 permutations): each run's length and
             contig-relative offset placed on a contig of another genome
Per arm, for each run and each locus set (kind@level):
  coverage        fraction of the run's genes inside any locus of the set
  fragments       number of loci of the set overlapping the run
  best            locus with the largest overlap (ties: the shorter one); best_frac = overlap / run
  precision       overlap / best-locus length; ext_left / ext_right = genes past each run end
  recovered       best_frac >= 0.5 (July rule); resolved: best_frac >= 0.8 and precision >= 0.8
  module          module_id of the best locus (engine plurality module); n_modules over all
                  overlapping loci; top-module purity over recovered runs
Locus sets: the three complete sets of loci_all_genomes.parquet (conserved@2, conserved_g2@2,
unit@0.5) and, when boundary_support.npy exists, the conserved ladder (levels 2,5,12,30,80 at
gaps 0 and 2) recomputed with the engine's run rule; module ids for ladder loci are joined
from the table (a locus's module depends only on its genes).

usage:
  ribosomal_check.py gold [--threads 2]
  ribosomal_check.py check RESULT_DIR [--out DIR] [--threads N] [--per-run]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
DB = RD / "ws_dpann/input/sharur.duckdb"
CACHE = RD / "ribosomal/gold"
MARKER_SQL = "regexp_matches(lower(name), '^ribosom(al)?_')"
GAPS = (0, 2)
MIN_MARKED = (8, 16)
LADDER = (2, 5, 12, 30, 80)
N_RANDOM = 20
N_TRANSPLANT = 5
SEED = 17
COMPLETENESS_BINS = [0, 70, 80, 90, 101]
COMPLETENESS_LABELS = ["<70", "70-80", "80-90", ">=90"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- gold
def load_universe(pkg):
    genes = pd.read_parquet(pkg / "universe/genes.parquet",
                            columns=["gid", "protein_id", "genome_code", "contig_code", "position_index"])
    genes = genes.sort_values("gid").reset_index(drop=True)
    assert np.array_equal(genes.gid.to_numpy(), np.arange(len(genes))), "gid must be 0..n-1"
    cc = genes.contig_code.to_numpy()
    assert (np.diff(cc) >= 0).all(), "gid order must follow contig_code (engine Universe order)"
    contigs = pd.read_parquet(pkg / "universe/contigs.parquet").sort_values("first_gid").reset_index(drop=True)
    genomes = pd.read_parquet(pkg / "universe/genomes.parquet").sort_values("genome_code").reset_index(drop=True)
    return genes, contigs, genomes


def build_runs(marked_gid, contig_of_gid, gap):
    m = np.sort(marked_gid)
    new = np.r_[True, (np.diff(m) > gap + 1) | (np.diff(contig_of_gid[m]) != 0)]
    rid = np.cumsum(new) - 1
    last = np.r_[new[1:], True]
    return pd.DataFrame(dict(gid_start=m[new], gid_end=m[last], n_marked=np.bincount(rid)))


def window_candidates(cs, ce, n, forb_s, forb_e):
    """Start gids x in [cs, ce - n] with [x, x + n - 1] disjoint from every forbidden interval."""
    if ce - cs < n:
        return np.empty(0, dtype=np.int64)
    x = np.arange(cs, ce - n + 1)
    ok = np.ones(len(x), dtype=bool)
    for s, e in zip(forb_s, forb_e):
        ok &= (x + n - 1 < s) | (x > e)
    return x[ok]


def build_gold(a):
    import duckdb
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    genes, contigs, genomes = load_universe(PKG)
    con = duckdb.connect(str(DB), read_only=True)
    con.execute(f"SET threads={a.threads}")
    hits = con.execute(f"SELECT protein_id, accession, name, score FROM annotations "
                       f"WHERE source = 'pfam' AND {MARKER_SQL}").df()
    names_all = con.execute(f"SELECT name, count(*) n, count(DISTINCT protein_id) np FROM annotations "
                            f"WHERE source = 'pfam' AND {MARKER_SQL} GROUP BY 1 ORDER BY 1").df()
    con.close()
    log("marker hits", len(hits), "proteins", hits.protein_id.nunique(), "names", len(names_all))
    pid2gid = pd.Series(genes.gid.to_numpy(), index=genes.protein_id.to_numpy())
    assert pid2gid.index.is_unique
    hits["gid"] = pid2gid.reindex(hits.protein_id.to_numpy()).to_numpy()
    unmapped = int(hits.gid.isna().sum())
    hits = hits.dropna(subset=["gid"])
    hits["gid"] = hits.gid.astype(np.int64)
    top = hits.sort_values("score", ascending=False).drop_duplicates("gid").set_index("gid").name
    allnames = hits.groupby("gid").name.apply(lambda s: ";".join(sorted(set(s))))
    marked = pd.DataFrame({"gid": top.index.to_numpy(), "marker": top.to_numpy()})
    marked["markers"] = allnames.reindex(marked.gid).to_numpy()
    marked = marked.sort_values("gid").reset_index(drop=True)
    marked.to_parquet(CACHE / "marked.parquet", index=False)

    cc_of_gid = genes.contig_code.to_numpy()
    pos_of_gid = genes.position_index.to_numpy()
    cinfo = contigs.set_index("contig_code")
    gm = genomes.set_index("genome_code")
    marker_of = pd.Series(marked.marker.to_numpy(), index=marked.gid.to_numpy())
    frames = []
    for gap in GAPS:
        r = build_runs(marked.gid.to_numpy(), cc_of_gid, gap)
        r = r[r.n_marked >= min(MIN_MARKED)].reset_index(drop=True)
        r.insert(0, "run_set", f"g{gap}")
        frames.append(r)
    runs = pd.concat(frames, ignore_index=True)
    runs["n_genes"] = runs.gid_end - runs.gid_start + 1
    runs["contig_code"] = cc_of_gid[runs.gid_start]
    runs["genome_code"] = genes.genome_code.to_numpy()[runs.gid_start]
    runs["pos_start"] = pos_of_gid[runs.gid_start]
    runs["pos_end"] = pos_of_gid[runs.gid_end]
    ci = cinfo.loc[runs.contig_code]
    runs["contig"] = ci.contig.to_numpy()
    runs["genome"] = ci.genome.to_numpy()
    runs["contig_first_gid"] = ci.first_gid.to_numpy()
    runs["contig_n_genes"] = ci.n_genes.to_numpy()
    runs["at_contig_edge"] = (runs.pos_start == 0) | (runs.pos_end == runs.contig_n_genes - 1)
    g = gm.loc[runs.genome_code]
    for col in ("phylum", "class", "ord", "family", "genus", "completeness"):
        runs[col] = g[col].to_numpy()
    runs["completeness_bin"] = pd.cut(runs.completeness, COMPLETENESS_BINS, right=False,
                                      labels=COMPLETENESS_LABELS).astype(str)
    runs["order"] = [" ".join(marker_of.get(x, "-") for x in range(s, e + 1))
                     for s, e in zip(runs.gid_start, runs.gid_end)]
    runs.insert(1, "run_id", np.arange(len(runs)))
    runs.to_parquet(CACHE / "runs.parquet", index=False)

    # ---- controls (seeded once; identical for every arm) --------------------------------
    rng = np.random.default_rng(SEED)
    forb = runs[runs.run_set == "g2"].sort_values("gid_start")
    forb_by_contig = {c: (d.gid_start.to_numpy(), d.gid_end.to_numpy()) for c, d in forb.groupby("contig_code")}
    contig_by_genome = {gc: d for gc, d in contigs.merge(genomes[["genome", "genome_code"]], on="genome")
                        .groupby("genome_code")}
    empty = (np.empty(0, np.int64), np.empty(0, np.int64))
    rows = []
    src_counts = {"same_contig": 0, "same_genome": 0, "none": 0}
    for r in runs.itertuples():
        n = r.n_genes
        cs, ce = r.contig_first_gid, r.contig_first_gid + r.contig_n_genes
        x = window_candidates(cs, ce, n, *forb_by_contig.get(r.contig_code, empty))
        src = "same_contig"
        if len(x) == 0:
            src = "same_genome"
            cand = []
            for c in contig_by_genome[r.genome_code].itertuples():
                if c.contig_code != r.contig_code and c.n_genes >= n:
                    cand.append(window_candidates(c.first_gid, c.first_gid + c.n_genes, n,
                                                  *forb_by_contig.get(c.contig_code, empty)))
            x = np.concatenate(cand) if cand else np.empty(0, np.int64)
        if len(x) == 0:
            src_counts["none"] += 1
            continue
        src_counts[src] += 1
        pick = rng.choice(x, size=N_RANDOM, replace=True)
        for k, s in enumerate(pick):
            rows.append((r.run_id, k, src, int(s), int(s) + n - 1))
    ctrl = pd.DataFrame(rows, columns=["run_id", "draw", "source", "gid_start", "gid_end"])
    ctrl.to_parquet(CACHE / "controls_random.parquet", index=False)

    rows = []
    tsrc = {"placed": 0, "none": 0}
    for rs, d in runs.groupby("run_set"):
        d = d.reset_index(drop=True)
        for p in range(N_TRANSPLANT):
            perm = rng.permutation(len(d))
            for i, r in enumerate(d.itertuples()):
                tgt = d.genome_code.iat[perm[i]]
                j = 0
                while tgt == r.genome_code and j < 20:
                    tgt = d.genome_code.iat[rng.integers(len(d))]
                    j += 1
                n = r.n_genes
                cands = contig_by_genome[tgt]
                cands = cands[cands.n_genes >= n]
                if tgt == r.genome_code or cands.empty:
                    tsrc["none"] += 1
                    continue
                c = cands.iloc[rng.choice(len(cands), p=(cands.n_genes / cands.n_genes.sum()).to_numpy())]
                x = window_candidates(c.first_gid, c.first_gid + c.n_genes, n,
                                      *forb_by_contig.get(c.contig_code, empty))
                if len(x) == 0:
                    tsrc["none"] += 1
                    continue
                f = (r.pos_start / max(r.contig_n_genes - n, 1))
                want = c.first_gid + int(round(f * (c.n_genes - n)))
                s = x[np.argmin(np.abs(x - want))]
                tsrc["placed"] += 1
                rows.append((r.run_id, p, int(tgt), int(s), int(s) + n - 1))
    tr = pd.DataFrame(rows, columns=["run_id", "draw", "target_genome_code", "gid_start", "gid_end"])
    tr.to_parquet(CACHE / "controls_transplant.parquet", index=False)

    # ---- description ------------------------------------------------------------------
    desc = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), database=str(DB), marker_rule=MARKER_SQL,
                marker_hits=int(len(hits) + unmapped), marker_hits_unmapped=unmapped,
                marked_genes=int(len(marked)), marker_names=names_all.to_dict("records"),
                genomes_total=int(len(genomes)),
                phylum_genomes=genomes.phylum.value_counts().to_dict(),
                controls=dict(random_per_run=N_RANDOM, random_source=src_counts, transplant_permutations=N_TRANSPLANT,
                              transplant=tsrc, seed=SEED), sets={})
    gtot = genomes.assign(completeness_bin=pd.cut(genomes.completeness, COMPLETENESS_BINS, right=False,
                                                  labels=COMPLETENESS_LABELS).astype(str))
    for rs in runs.run_set.unique():
        for mm in MIN_MARKED:
            d = runs[(runs.run_set == rs) & (runs.n_marked >= mm)]
            key = f"{rs}_m{mm}"
            gw = d.groupby("genome_code").size()
            desc["sets"][key] = dict(
                runs=int(len(d)), genomes_with_run=int(d.genome_code.nunique()),
                genomes_with_2plus_runs=int((gw >= 2).sum()),
                n_genes_quantiles=np.quantile(d.n_genes, [0, .1, .25, .5, .75, .9, 1]).tolist() if len(d) else [],
                n_marked_quantiles=np.quantile(d.n_marked, [0, .1, .25, .5, .75, .9, 1]).tolist() if len(d) else [],
                at_contig_edge=int(d.at_contig_edge.sum()),
                by_phylum={k: dict(genomes_with_run=int(d[d.phylum == k].genome_code.nunique()), genomes=int(v))
                           for k, v in genomes.phylum.value_counts().items()},
                by_completeness={k: dict(genomes_with_run=int(d[d.completeness_bin == k].genome_code.nunique()),
                                         genomes=int((gtot.completeness_bin == k).sum()))
                                 for k in COMPLETENESS_LABELS},
                parvarchaeaceae_genomes_with_run=int(d[d.family == "Parvarchaeaceae"].genome_code.nunique()),
                parvarchaeaceae_genomes=int((genomes.family == "Parvarchaeaceae").sum()))
    desc["seconds"] = round(time.time() - t0, 1)
    with open(CACHE / "gold.json", "w") as fh:
        json.dump(desc, fh, indent=1)
    log("gold done", {k: v["runs"] for k, v in desc["sets"].items()}, f"{desc['seconds']}s")


# --------------------------------------------------------------------------- scoring
@njit(cache=True)
def runs_at_least(b, starts, ends, level, gap):
    """Copy of the engine rule (recurrence.py): maximal runs of boundaries with b >= level,
    bridging up to `gap` boundaries below it; inclusive gene ranges."""
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


@njit(parallel=True, cache=True)
def score_intervals(a, b, mod, maxlen, qs, qe):
    """a, b sorted by a (inclusive gid intervals). For each query [qs, qe]: coverage genes,
    fragments, best overlap, best length, best index, ext_left, ext_right, n_modules."""
    n = len(qs)
    out = np.zeros((n, 8), dtype=np.int64)
    for r in prange(n):
        s, e = qs[r], qe[r]
        j = np.searchsorted(a, s - maxlen + 1)
        cov, nfrag, best_ov, best_len, best_j = 0, 0, 0, 1 << 60, -1
        last = s - 1
        buf = np.empty(64, dtype=np.int64)
        nb = 0
        while j < len(a) and a[j] <= e:
            if b[j] >= s:
                lo = max(a[j], s)
                hi = min(b[j], e)
                ov = hi - lo + 1
                lo2 = max(lo, last + 1)
                if hi >= lo2:
                    cov += hi - lo2 + 1
                if hi > last:
                    last = hi
                nfrag += 1
                ln = b[j] - a[j] + 1
                if ov > best_ov or (ov == best_ov and ln < best_len):
                    best_ov, best_len, best_j = ov, ln, j
                if mod[j] >= 0 and nb < 64:
                    buf[nb] = mod[j]
                    nb += 1
            j += 1
        nm = 0
        if nb > 0:
            u = np.sort(buf[:nb])
            nm = 1
            for t in range(1, nb):
                if u[t] != u[t - 1]:
                    nm += 1
        out[r, 0] = cov
        out[r, 1] = nfrag
        out[r, 2] = best_ov
        out[r, 3] = best_len if best_j >= 0 else 0
        out[r, 4] = best_j
        out[r, 5] = max(0, s - a[best_j]) if best_j >= 0 else 0
        out[r, 6] = max(0, b[best_j] - e) if best_j >= 0 else 0
        out[r, 7] = nm
    return out


def score(locset, qs, qe):
    a, b, mod = locset["a"], locset["b"], locset["mod"]
    if len(a) == 0:
        a, b, mod = np.array([-10], np.int64), np.array([-10], np.int64), np.array([-1], np.int64)
    o = score_intervals(a, b, mod, locset["maxlen"], qs.astype(np.int64), qe.astype(np.int64))
    n = (qe - qs + 1).astype(np.float64)
    best_j = o[:, 4]
    with np.errstate(invalid="ignore", divide="ignore"):
        df = pd.DataFrame(dict(
            coverage=o[:, 0] / n, fragments=o[:, 1], best_frac=o[:, 2] / n,
            precision=np.where(o[:, 3] > 0, o[:, 2] / np.maximum(o[:, 3], 1), np.nan),
            best_len=o[:, 3], ext_left=np.where(best_j >= 0, o[:, 5], -1), ext_right=np.where(best_j >= 0, o[:, 6], -1),
            n_modules=o[:, 7], module=np.where(best_j >= 0, mod[np.maximum(best_j, 0)], -2)))
    df["recovered"] = df.best_frac >= 0.5
    df["resolved"] = (df.best_frac >= 0.8) & (df.precision >= 0.8)
    return df


def module_purity(mods):
    """Top-module purity over all given (recovered) runs: share whose best locus carries the most
    common module id; runs whose best locus has no module (-1) count in the denominator."""
    allm = np.asarray(mods)
    m = allm[allm >= 0]
    if len(m) == 0:
        return dict(top_module=None, purity=None, purity_among_assigned=None, n_with_module=0, n_distinct=0)
    v, c = np.unique(m, return_counts=True)
    return dict(top_module=int(v[c.argmax()]), purity=float(c.max() / len(allm)),
                purity_among_assigned=float(c.max() / len(m)), n_with_module=int(len(m)), n_distinct=int(len(v)),
                top3=[[int(v[i]), int(c[i])] for i in np.argsort(-c)[:3]])


def summarize(runs, sc, with_tax=True):
    n = len(sc)
    rec = sc.recovered.to_numpy()
    res = sc.resolved.to_numpy()
    out = dict(runs=n, recovered=int(rec.sum()), resolved=int(res.sum()),
               recovered_frac=float(rec.mean()) if n else None, resolved_frac=float(res.mean()) if n else None,
               median_coverage=float(sc.coverage.median()), mean_coverage=float(sc.coverage.mean()),
               median_fragments=float(sc.fragments.median()), mean_fragments=float(sc.fragments.mean()),
               median_best_frac=float(sc.best_frac.median()),
               median_precision_recovered=float(sc.precision[rec].median()) if rec.any() else None,
               median_precision_any=float(sc.precision.median()) if sc.precision.notna().any() else None,
               median_ext_left_recovered=float(sc.ext_left[rec].median()) if rec.any() else None,
               median_ext_right_recovered=float(sc.ext_right[rec].median()) if rec.any() else None,
               median_best_len_recovered=float(sc.best_len[rec].median()) if rec.any() else None,
               no_locus=int((sc.fragments == 0).sum()),
               median_n_modules=float(sc.n_modules.median()))
    mp = module_purity(sc.module[rec])
    out.update({f"module_{k}": v for k, v in mp.items()})
    out["recovered_best_module_missing"] = int((sc.module[rec] < 0).sum())
    if with_tax:
        r = runs[rec]
        out.update(genomes=int(r.genome_code.nunique()), genera=int(r.genus.nunique()),
                   phyla=int(r.phylum.nunique()), genomes_with_run=int(runs.genome_code.nunique()),
                   genera_with_run=int(runs.genus.nunique()), phyla_with_run=int(runs.phylum.nunique()),
                   parvarchaeaceae=f"{r[r.family == 'Parvarchaeaceae'].genome_code.nunique()}/"
                                   f"{runs[runs.family == 'Parvarchaeaceae'].genome_code.nunique()}",
                   parvarchaeaceae_resolved=f"{runs[res & (runs.family == 'Parvarchaeaceae').to_numpy()].genome_code.nunique()}/"
                                            f"{runs[runs.family == 'Parvarchaeaceae'].genome_code.nunique()}")
    return out


def load_locus_sets(rdir, contigs, n_genes_total):
    t0 = time.time()
    l = pd.read_parquet(rdir / "loci_all_genomes.parquet")
    if contigs.contig.is_unique:
        cat = pd.Categorical(l.contig.to_numpy(), categories=contigs.contig.to_numpy())
        code = cat.codes
        assert (code >= 0).all(), "locus contig not in universe"
        first = contigs.first_gid.to_numpy()[code]
        gchk = contigs.genome.to_numpy()[code]
        assert (gchk == l.genome.to_numpy()).all(), "contig/genome mismatch"
    else:
        first = l.merge(contigs[["genome", "contig", "first_gid"]], on=["genome", "contig"], how="left").first_gid.to_numpy()
    l["a"] = first + l.start.to_numpy()
    l["b"] = first + l.end.to_numpy()
    log("loci table", len(l), f"{time.time()-t0:.1f}s")
    sets = {}

    def mk(a, b, mod, name):
        o = np.argsort(a, kind="stable")
        a, b, mod = a[o], b[o], mod[o]
        sets[name] = dict(a=a, b=b, mod=mod, maxlen=int((b - a + 1).max()) if len(a) else 1, n=int(len(a)))

    for kind, lev in (("conserved", 2.0), ("conserved_g2", 2.0), ("unit", 0.5)):
        d = l[(l.kind == kind) & (l.level == lev)]
        mk(d.a.to_numpy(), d.b.to_numpy(), d.module_id.to_numpy(), f"{kind}@{lev:g}")
    consistency = {}
    bpath = rdir / "boundary_support.npy"
    if bpath.exists():
        bsup = np.load(bpath)
        assert len(bsup) == n_genes_total, "boundary_support length != universe genes"
        cs = contigs.first_gid.to_numpy().astype(np.int64)
        ce = cs + contigs.n_genes.to_numpy().astype(np.int64)
        # module lookup: (a, b, kind) is unique in the table
        for gap in (0, 2):
            kind = "conserved" if gap == 0 else "conserved_g2"
            t = l[l.kind == kind]
            key_t = t.a.to_numpy() * n_genes_total + t.b.to_numpy()
            ser = pd.Series(t.module_id.to_numpy(), index=key_t)
            ser = ser[~ser.index.duplicated()]
            for lev in LADDER:
                s_idx, e_idx = runs_at_least(bsup, cs, ce, float(lev), gap)
                keep = e_idx > s_idx
                s_idx, e_idx = s_idx[keep], e_idx[keep]
                mod = ser.reindex(s_idx * n_genes_total + e_idx).to_numpy()
                missing = int(np.isnan(mod).sum()) if mod.dtype.kind == "f" else 0
                mod = np.nan_to_num(mod, nan=-3).astype(np.int64)
                mk(s_idx, e_idx, mod, f"ladder_{kind}@{lev}")
                consistency[f"ladder_{kind}@{lev}"] = dict(n=int(len(s_idx)), module_join_missing=missing)
        a2 = sets["ladder_conserved@2"]
        t2 = sets["conserved@2"]
        consistency["ladder_level2_equals_table"] = bool(np.array_equal(a2["a"], t2["a"]) and np.array_equal(a2["b"], t2["b"]))
    log("locus sets", {k: v["n"] for k, v in sets.items()}, f"{time.time()-t0:.1f}s")
    return sets, consistency


def check(a):
    t0 = time.time()
    import numba
    numba.set_num_threads(a.threads)
    rdir = Path(a.result_dir)
    out = Path(a.out) if a.out else rdir
    out.mkdir(parents=True, exist_ok=True)
    if not (CACHE / "gold.json").exists():
        build_gold(a)
    runs = pd.read_parquet(CACHE / "runs.parquet")
    rnd = pd.read_parquet(CACHE / "controls_random.parquet")
    trn = pd.read_parquet(CACHE / "controls_transplant.parquet")
    contigs = pd.read_parquet(PKG / "universe/contigs.parquet").sort_values("first_gid").reset_index(drop=True)
    n_genes_total = int((contigs.first_gid + contigs.n_genes).max())
    summ = json.load(open(rdir / "summary.json")) if (rdir / "summary.json").exists() else {}
    sets, consistency = load_locus_sets(rdir, contigs, n_genes_total)

    res = dict(result_dir=str(rdir), arm=summ.get("arm"), engine=summ.get("engine"), w=summ.get("w"),
               rule=summ.get("rule"), tokens=summ.get("tokens"), significant_pairs=summ.get("significant_pairs"),
               genes_covered_frac=summ.get("genes_covered_frac"), modules=summ.get("modules", {}).get("modules"),
               gold_cache=str(CACHE), locus_set_sizes={k: v["n"] for k, v in sets.items()},
               consistency=consistency, results={}, controls={}, by_phylum={}, by_completeness={})
    per_run = []
    for rs in ("g2", "g0"):
        for mm in (16, 8):
            key = f"{rs}_m{mm}"
            rr = runs[(runs.run_set == rs) & (runs.n_marked >= mm)].reset_index(drop=True)
            rres, cres = {}, {}
            rc = rnd[rnd.run_id.isin(rr.run_id)]
            tc = trn[trn.run_id.isin(rr.run_id)]
            any_res = np.zeros(len(rr), dtype=bool)
            for name, ls in sets.items():
                sc = score(ls, rr.gid_start.to_numpy(), rr.gid_end.to_numpy())
                rres[name] = summarize(rr, sc)
                any_res |= sc.resolved.to_numpy() if name.startswith("ladder") else False
                if key == "g2_m16" and a.per_run:
                    per_run.append(pd.concat([rr[["run_id"]], sc.add_prefix(name + ":")], axis=1))
                if name in ("conserved@2", "conserved_g2@2", "unit@0.5") or name.startswith("ladder_conserved@"):
                    c1 = score(ls, rc.gid_start.to_numpy(), rc.gid_end.to_numpy())
                    c2 = score(ls, tc.gid_start.to_numpy(), tc.gid_end.to_numpy())
                    cres[name] = dict(random=summarize(None, c1, with_tax=False),
                                      transplant=summarize(None, c2, with_tax=False))
                    # per-run excess of recovered over the run's own random draws
                    rec_ctrl = c1.recovered.groupby(rc.run_id.to_numpy()).mean()
                    cres[name]["random_recovered_mean_per_run"] = float(rec_ctrl.mean()) if len(rec_ctrl) else None
                if key == "g2_m16" and name == "conserved@2":
                    for col, store in (("phylum", res["by_phylum"]), ("completeness_bin", res["by_completeness"])):
                        for v, idx in rr.groupby(col).groups.items():
                            s = sc.loc[idx]
                            store[str(v)] = dict(runs=int(len(idx)), recovered=int(s.recovered.sum()),
                                                 resolved=int(s.resolved.sum()),
                                                 genomes_with_run=int(rr.loc[idx].genome_code.nunique()),
                                                 median_coverage=float(s.coverage.median()),
                                                 median_fragments=float(s.fragments.median()))
                    edge = rr.at_contig_edge.to_numpy()
                    res["contig_edge"] = dict(edge_runs=int(edge.sum()), edge_recovered=int(sc.recovered[edge].sum()),
                                              edge_resolved=int(sc.resolved[edge].sum()),
                                              internal_runs=int((~edge).sum()),
                                              internal_recovered=int(sc.recovered[~edge].sum()),
                                              internal_resolved=int(sc.resolved[~edge].sum()))
            if any(k.startswith("ladder") for k in sets):
                rres["any_ladder_level_oracle"] = dict(runs=int(len(rr)), resolved=int(any_res.sum()))
            res["results"][key] = rres
            res["controls"][key] = cres
    p = res["results"]["g2_m16"]["conserved@2"]
    pc = res["controls"]["g2_m16"]["conserved@2"]["random"]
    res["primary"] = dict(run_set="g2_m16", locus_set="conserved@2", **p,
                          random_recovered_frac=pc["recovered_frac"], random_resolved_frac=pc["resolved_frac"])
    res["seconds"] = round(time.time() - t0, 1)
    with open(out / "ribosomal_check.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    if per_run:
        pr = per_run[0]
        for d in per_run[1:]:
            pr = pr.merge(d, on="run_id")
        pr.to_parquet(out / "ribosomal_runs.parquet", index=False)
    alt = res["results"]["g2_m16"]
    a80 = alt.get("ladder_conserved@80", {}).get("resolved", "na")
    au = alt["unit@0.5"]["resolved"]
    line = (f"RIBOSOMAL {res['arm']} w={res['w']}: runs={p['runs']} recovered={p['recovered']} "
            f"resolved={p['resolved']} (random windows {pc['recovered_frac']:.3f}/{pc['resolved_frac']:.3f}) "
            f"median_coverage={p['median_coverage']:.3f} median_fragments={p['median_fragments']:g} "
            f"median_precision={p['median_precision_recovered'] or 0:.3f} "
            f"top_module_purity={p['module_purity'] if p['module_purity'] is None else round(p['module_purity'], 3)} "
            f"Parvarchaeaceae={p['parvarchaeaceae']} | resolved conserved@80={a80} unit@0.5={au} "
            f"[{res['seconds']}s]")
    with open(out / "ribosomal_check.txt", "w") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gold")
    g.add_argument("--threads", type=int, default=2)
    c = sub.add_parser("check")
    c.add_argument("result_dir")
    c.add_argument("--out")
    c.add_argument("--threads", type=int, default=2)
    c.add_argument("--per-run", action="store_true", help="also write ribosomal_runs.parquet (g2_m16, every locus set)")
    a = ap.parse_args()
    build_gold(a) if a.cmd == "gold" else check(a)


if __name__ == "__main__":
    main()
