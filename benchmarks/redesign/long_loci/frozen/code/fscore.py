#!/usr/bin/env python
"""Frozen-run scorer for the lut test set (protocol: frozen/FROZEN.json 'scoring'; read-only scorers imported).

  prep   shadow packages (universe symlinked; gold/instances.parquet from gold/lut_instances.parquet, caller main / strict),
         complete-cluster table, lut-homolog table (slice_lut part-1 RB component genes mapped to package gids)
  score  every available output: lut loci (score_dpann_loci.score, 200 controls, seed 17, min_members 2), ribosomal runs
         (ribosomal_check.score, standing + any fitting locus), complete clusters, false-positive side, coverage;
         bootstrap intervals over truth loci / runs (10,000 draws, seed 17), paired differences
usage: fscore.py prep | fscore.py score --threads T
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
FZ = RD / "long_loci/frozen"
PKG = RD / "slice_lut/testset/package_k8"
SL = RD / "slice_lut"
SIN = FZ / "score_in"
sys.path.insert(0, str(RD / "ws_dpann/package/scorer"))
sys.path.insert(0, str(RD / "ribosomal/code"))
import score_dpann_loci as S  # noqa: E402  (read-only)
import ribosomal_check as RC  # noqa: E402  (read-only)

COMPS = [f"META1_{x}" for x in range(1778, 1788)]
LAB = dict(zip(COMPS, list("ABCDEFGH") + ["M", "I"]))
FULL = set("ABCDEFGHI")
MODULE = set("ABCEFG")
NBOOT = 10_000
OUTPUTS = {   # label -> loci parquet (s, e inclusive gids); FROZEN.json configurations and comparators
    "PRIMARY_WA_L4b_t0.2": FZ / "loci/lut_WA/L4b_t0.2.parquet",
    "WA_L1": FZ / "loci/lut_WA/L1.parquet",
    "SECONDARY_Z_L4bp_t0.02": FZ / "loci/lut_Z/L4bp_t0.02.parquet",
    "Z_L1": FZ / "loci/lut_Z/L1.parquet",
    "engine_DH": FZ / "loci/lut_engine/DH.parquet",
    "synton_faithful": FZ / "synton/runs/faithful/post/loci.parquet",
    "synton_matched": FZ / "synton/runs/matched/post/loci.parquet",
}
PAIRS = [("PRIMARY_WA_L4b_t0.2", b) for b in ("engine_DH", "WA_L1", "SECONDARY_Z_L4bp_t0.02", "synton_faithful", "synton_matched")] + \
        [("SECONDARY_Z_L4bp_t0.02", b) for b in ("engine_DH", "Z_L1", "synton_faithful", "synton_matched")]


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def jdump(obj, fn):
    Path(fn).write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    log("wrote", fn)


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)


# =========================================================================== prep
def cmd_prep(a):
    SIN.mkdir(parents=True, exist_ok=True)
    genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome", "genome_code", "contig_code", "position_index"])
    contigs = pd.read_parquet(PKG / "universe/contigs.parquet")
    gm = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome", "genome_code", "layer", "split", "source_scope"])
    I = pd.read_parquet(PKG / "gold/lut_instances.parquet")
    info = {}
    for caller, d in (("lut_homology_main", "pkg_lut_main"), ("lut_homology_strict", "pkg_lut_strict")):
        x = I[I.caller == caller].copy()
        mg = x.member_gids.map(lambda v: np.asarray(v, np.int64))
        mp = x.member_positions.map(lambda v: np.asarray(v, np.int64))
        first = contigs.set_index("contig_code").first_gid.loc[x.contig_code].to_numpy()
        ok_pos = all(np.array_equal(f + p, g) for f, p, g in zip(first, mp, mg))
        same_contig = all((genes.contig_code.to_numpy()[g] == c).all() for g, c in zip(mg, x.contig_code))
        x["member_positions"] = mp.map(lambda v: v.tolist())
        x["n_members_resolved"] = mp.map(lambda v: len(np.unique(v)))
        x["resolved"] = True
        x["order_trusted"] = True
        x["caller"] = caller
        cols = ["instance_id", "caller", "system_type", "system_subtype", "genome", "genome_code", "contig", "contig_code",
                "start_idx", "end_idx", "span_genes", "member_positions", "n_members_resolved", "resolved", "order_trusted"]
        pd_dir = FZ / d
        (pd_dir / "gold").mkdir(parents=True, exist_ok=True)
        if not (pd_dir / "universe").exists():
            (pd_dir / "universe").symlink_to(PKG / "universe")
        x[cols].to_parquet(pd_dir / "gold/instances.parquet", index=False)
        info[caller] = dict(instances=int(len(x)), positions_equal_gids=bool(ok_pos), single_contig=bool(same_contig),
                            members_ge2=int((x.n_members_resolved >= 2).sum()),
                            by_type=x.system_type.value_counts().to_dict())
    # complete clusters (main)
    x = I[I.caller == "lut_homology_main"]
    comp = x.member_components.map(lambda v: set(v))
    cpl = x[comp.map(lambda s: FULL <= s)]
    rows = []
    for _, r in cpl.iterrows():
        g = np.asarray(r.member_gids, np.int64)
        c = np.asarray(r.member_components, dtype=object)
        rows.append(dict(instance_id=int(r.instance_id), genome=r.genome, contig_code=int(r.contig_code),
                         members=g.tolist(), module=g[np.isin(c, list(MODULE))].tolist(), architecture=r.architecture))
    pd.DataFrame(rows).to_parquet(SIN / "complete.parquet", index=False)
    lay = cpl.genome.map(dict(zip(gm.genome, gm.layer)))
    info["complete"] = dict(n=int(len(cpl)), layers=lay.value_counts().to_dict())
    # lut homologs (RB tier) mapped to package gids
    parts = []
    pk_first = contigs.set_index(["genome", "contig"]).first_gid
    for sc, udir in (("order", SL / "package/rhizobiales/universe"), ("alpha_wide", SL / "work/alpha_wide/pkg/universe"),
                     ("beyond", SL / "work/beyond/pkg/universe")):
        cg = pd.read_parquet(SL / f"results/{sc}/component_genes.parquet")
        cg = cg[cg.genome.isin(gm.genome[gm.source_scope == sc])]
        cg = cg[cg.is_rb.fillna(False) & cg.comp.isin(COMPS) & (cg.evalue <= 1e-5)]
        sc_contigs = pd.read_parquet(udir / "contigs.parquet", columns=["genome", "contig", "contig_code"]).set_index("contig_code")
        cg["contig"] = sc_contigs.contig.loc[cg.contig_code].to_numpy()
        key = pd.MultiIndex.from_arrays([cg.genome, cg.contig])
        fg = pk_first.reindex(key).to_numpy()
        assert not np.isnan(fg.astype(float)).any(), f"{sc}: contigs missing in the package"
        cg["gid"] = fg.astype(np.int64) + cg.position_index.to_numpy(np.int64)
        cg["component"] = cg.comp.map(LAB)
        cg["scope"] = sc
        parts.append(cg[["gid", "genome", "component", "evalue", "bits", "scope"]])
    H = pd.concat(parts, ignore_index=True).drop_duplicates("gid")
    gg = genes.set_index("gid").loc[H.gid]
    assert (gg.genome.to_numpy() == H.genome.to_numpy()).all()
    H["contig_code"] = gg.contig_code.to_numpy()
    H["position_index"] = gg.position_index.to_numpy()
    H["layer"] = H.genome.map(dict(zip(gm.genome, gm.layer)))
    H = H.sort_values("gid").reset_index(drop=True)
    gid = H.gid.to_numpy()
    cc = H.contig_code.to_numpy()
    iso = np.ones(len(H), bool)
    for d in (1, 2, 3):
        same = (cc[d:] == cc[:-d]) & (gid[d:] - gid[:-d] <= 3)
        iso[d:] &= ~same
        iso[:-d] &= ~same
    H["isolated"] = iso
    H.to_parquet(SIN / "homologs.parquet", index=False)
    info["homologs"] = dict(rb_genes=int(len(H)), by_layer=H.layer.value_counts().to_dict(),
                            b1b2=int(H.layer.str[:2].isin(["b1", "b2"]).sum()),
                            b1b2_isolated=int((H.layer.str[:2].isin(["b1", "b2"]) & H.isolated).sum()),
                            b1b2_by_component=H[H.layer.str[:2].isin(["b1", "b2"])].component.value_counts().to_dict())
    R = pd.read_parquet(PKG / "gold/ribosomal_runs.parquet")
    R = R[(R.run_set == "g2") & (R.n_marked >= 16)]
    info["ribosomal_runs"] = dict(runs=int(len(R)), genomes=int(R.genome_code.nunique()), even=int((R.genome_code % 2 == 0).sum()))
    info["genomes_by_layer"] = gm.layer.value_counts().to_dict()
    jdump(info, SIN / "prep.json")


# =========================================================================== helpers
def load_loci(label, path, first_gid):
    df = pd.read_parquet(path)
    if label.startswith("synton"):
        s = first_gid[df.contig_code.to_numpy(np.int64)] + df.start.to_numpy(np.int64)
        e = first_gid[df.contig_code.to_numpy(np.int64)] + df.end.to_numpy(np.int64)
    else:
        s, e = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
    k = e > s
    se = pd.DataFrame(dict(s=s[k], e=e[k])).drop_duplicates().sort_values(["s", "e"]).reset_index(drop=True)
    return se.s.to_numpy(np.int64), se.e.to_numpy(np.int64)


def boot_mean(v, rng_seed=17, n=NBOOT):
    v = np.asarray(v, np.float64)
    if len(v) == 0:
        return [None, None]
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    m = v[idx].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def summ(per):
    if len(per) == 0:
        return dict(n=0)
    return dict(n=int(len(per)), f1=float(per.f1.mean()), f1_ci=boot_mean(per.f1), control=float(per.control_mean_f1.mean()),
                excess=float(per.excess.mean()), excess_ci=boot_mean(per.excess), coverage=float(per.coverage.mean()),
                precision=float(per.boundary_precision.mean()),
                median_precision_covered50=float(per.boundary_precision[per.coverage >= .5].median()) if (per.coverage >= .5).any() else None,
                covered50=float(per.covered50.mean()), localized=float(per.localized.mean()),
                no_overlap=int((per.coverage == 0).sum()), median_best_span=float(per.best_span[per.best_span > 0].median()) if (per.best_span > 0).any() else None)


def paired(a, b, n=NBOOT):
    """a, b: per-instance frames indexed by instance_id (same set). Paired bootstrap over loci."""
    j = a[["f1", "excess"]].join(b[["f1", "excess"]], lsuffix="_a", rsuffix="_b", how="inner")
    rng = np.random.default_rng(17)
    idx = rng.integers(0, len(j), size=(n, len(j)))
    out = {}
    for c in ("f1", "excess"):
        d = (j[f"{c}_a"] - j[f"{c}_b"]).to_numpy()
        m = d[idx].mean(1)
        out[c] = [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]
    d, lo, hi = out["excess"]
    out["verdict"] = "BEATS" if lo > 0 else ("MATCHES" if d >= -0.02 else "BELOW")
    out["n"] = int(len(j))
    out["better_worse_f1"] = [int((j.f1_a > j.f1_b).sum()), int((j.f1_a < j.f1_b).sum())]
    return out


@njit(cache=True)
def any_fit(a, b, maxlen, qs, qe):
    """Per run: 1 if some locus [a, b] has overlap / run >= .8 and overlap / locus length >= .8."""
    out = np.zeros(len(qs), np.int8)
    for r in range(len(qs)):
        s, e = qs[r], qe[r]
        j = np.searchsorted(a, s - maxlen + 1)
        n = e - s + 1
        while j < len(a) and a[j] <= e:
            if b[j] >= s:
                ov = min(b[j], e) - max(a[j], s) + 1
                if ov >= 0.8 * n and ov >= 0.8 * (b[j] - a[j] + 1):
                    out[r] = 1
                    break
            j += 1
    return out


def ribo(runs, s, e):
    ls = dict(a=s, b=e, mod=np.full(len(s), -1, np.int64), maxlen=int((e - s + 1).max()) if len(s) else 1, n=int(len(s)))
    sc = RC.score(ls, runs.gid_start.to_numpy(), runs.gid_end.to_numpy())
    sc["any_fit"] = any_fit(s, e, ls["maxlen"], runs.gid_start.to_numpy(np.int64), runs.gid_end.to_numpy(np.int64)).astype(bool)
    return sc


def ribo_summ(sc):
    rec = sc.recovered.to_numpy()
    return dict(runs=int(len(sc)), recovered=int(rec.sum()), resolved=int(sc.resolved.sum()), resolved_any_fit=int(sc.any_fit.sum()),
                median_precision_recovered=float(sc.precision[rec].median()) if rec.any() else None,
                median_best_frac=float(sc.best_frac.median()), median_fragments=float(sc.fragments.median()),
                median_ext_left_right_recovered=[float(sc.ext_left[rec].median()), float(sc.ext_right[rec].median())] if rec.any() else None,
                no_locus=int((sc.fragments == 0).sum()))


def ribo_paired(a, b, n=NBOOT):
    d = a.resolved.to_numpy().astype(int) - b.resolved.to_numpy().astype(int)
    rng = np.random.default_rng(17)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].sum(1)
    return [int(d.sum()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def covered_mask(s, e, n):
    cov = np.zeros(n + 1, np.int32)
    np.add.at(cov, s, 1)
    np.add.at(cov, e + 1, -1)
    return np.cumsum(cov)[:n] > 0


def smallest_containing(s, e, g, maxlen):
    """span of the smallest locus containing each gene g (0 if none)."""
    out = np.zeros(len(g), np.int64)
    for i, x in enumerate(g):
        lo = np.searchsorted(s, x - maxlen + 1)
        hi = np.searchsorted(s, x, side="right")
        ss, ee = s[lo:hi], e[lo:hi]
        m = ee >= x
        if m.any():
            out[i] = int((ee[m] - ss[m] + 1).min())
    return out


def complete_side(C, s, e, maxlen):
    held, prec, matched, nested, rows = 0, [], 0, 0, []
    for _, r in C.iterrows():
        mem = np.asarray(r.members, np.int64)
        mod = np.asarray(r.module, np.int64)
        a0, b0 = mem.min(), mem.max()
        lo = np.searchsorted(s, a0 - maxlen + 1)
        hi = np.searchsorted(s, b0, side="right")
        ss, ee = s[lo:hi], e[lo:hi]
        k = ee >= a0
        ss, ee = ss[k], ee[k]
        if len(ss) == 0:
            rows.append(dict(instance_id=int(r.instance_id), held=False, cov=0.0, prec=None, matched=False, nested=False))
            continue
        inside = (mem[None, :] >= ss[:, None]) & (mem[None, :] <= ee[:, None])
        cov = inside.sum(1) / len(mem)
        span = ee - ss + 1
        o = np.lexsort((span, -cov))[0]
        h = cov[o] >= 0.8
        p = inside[o].sum() / span[o]
        mi = (mod[None, :] >= ss[:, None]) & (mod[None, :] <= ee[:, None])
        mcov = mi.sum(1) / max(len(mod), 1)
        mprec = mi.sum(1) / span
        mm = (mcov >= 0.8) & (mprec >= 0.8) & (len(mod) > 0)
        nst = bool(h and (mm & (ss >= ss[o]) & (ee <= ee[o])).any())
        held += int(h)
        matched += int(mm.any())
        nested += int(nst)
        if h:
            prec.append(p)
        rows.append(dict(instance_id=int(r.instance_id), held=bool(h), cov=float(cov[o]), prec=float(p), span=int(span[o]),
                         matched=bool(mm.any()), nested=nst))
    prec = np.asarray(prec)
    return dict(n=int(len(C)), held=held, held_precision_median=float(np.median(prec)) if len(prec) else None,
                held_precision_q10_q90=np.quantile(prec, [.1, .9]).tolist() if len(prec) else None,
                held_precision_ge_0_8=int((prec >= 0.8).sum()), module_matched=matched, module_matched_nested=nested), rows


# =========================================================================== score
def cmd_score(a):
    t0 = time.time()
    genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome", "genome_code", "contig", "position_index"])
    n = len(genes)
    gen_arr, con_arr, pos = genes.genome.to_numpy(), genes.contig.to_numpy(), genes.position_index.to_numpy()
    gcode = genes.genome_code.to_numpy()
    contigs = pd.read_parquet(PKG / "universe/contigs.parquet").sort_values("contig_code")
    first_gid = np.zeros(int(contigs.contig_code.max()) + 1, np.int64)
    first_gid[contigs.contig_code.to_numpy()] = contigs.first_gid.to_numpy()
    gm = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome", "genome_code", "layer"]).sort_values("genome_code")
    layer_of_genome = dict(zip(gm.genome, gm.layer))
    layer_of_gene = gm.layer.to_numpy()[gcode]
    runs = pd.read_parquet(PKG / "gold/ribosomal_runs.parquet")
    runs = runs[(runs.run_set == "g2") & (runs.n_marked >= 16)].sort_values("gid_start").reset_index(drop=True)
    even_run = (runs.genome_code % 2 == 0).to_numpy()
    C = pd.read_parquet(SIN / "complete.parquet")
    H = pd.read_parquet(SIN / "homologs.parquet")
    neg = H.layer.str[:2].isin(["b1", "b2"]).to_numpy()
    Hn = H[neg]
    Hiso = Hn[Hn.isolated]
    neg_gene = np.isin(layer_of_gene, ["b1_same_genus_negative", "b2_same_family_negative"])
    (FZ / "scores").mkdir(parents=True, exist_ok=True)
    res = dict(outputs={}, pairs={}, written=None, absent=[])
    pers, ribos = {}, {}
    for label, path in OUTPUTS.items():
        if not path.exists():
            res["absent"].append(label)
            log("absent", label, path)
            continue
        tl = time.time()
        s, e = load_loci(label, path, first_gid)
        maxlen = int((e - s + 1).max()) if len(s) else 1
        df = pd.DataFrame(dict(genome=gen_arr[s], contig=con_arr[s], start=pos[s], end=pos[e], module_id=np.full(len(s), -1)))
        item = {}
        for caller, d in (("main", "pkg_lut_main"), ("strict", "pkg_lut_strict")):
            per, _, _ = S.score(FZ / d, df, controls=200, seed=17, threads=a.threads, min_members=2)
            per["excess"] = per.f1 - per.control_mean_f1
            per["layer"] = per.genome.map(layer_of_genome)
            per["even"] = per.genome.map(dict(zip(gm.genome, gm.genome_code % 2 == 0)))
            per.to_parquet(FZ / "scores" / f"{label}__{caller}.parquet", index=False)
            item[f"lut_{caller}"] = dict(all=summ(per), even=summ(per[per.even]),
                                         by_class={k: summ(g) for k, g in per.groupby("system_type")},
                                         by_layer={k: summ(g) for k, g in per.groupby("layer")})
            if caller == "main":
                pers[label] = per.set_index("instance_id")
        sc = ribo(runs, s, e)
        ribos[label] = sc
        item["ribosomal"] = dict(all=ribo_summ(sc), even=ribo_summ(sc[even_run].reset_index(drop=True)))
        cs, rows = complete_side(C, s, e, maxlen)
        item["complete"] = cs
        pd.DataFrame(rows).to_parquet(FZ / "scores" / f"{label}__complete.parquet", index=False)
        cov = covered_mask(s, e, n)
        ge3 = (e - s + 1) >= 3
        cov3 = covered_mask(s[ge3], e[ge3], n)
        sm = smallest_containing(s, e, Hiso.gid.to_numpy(), maxlen)
        # loci in b1/b2 holding RB homologs of >= 2 / >= 3 distinct components
        hn_g = Hn.gid.to_numpy()
        hn_c = Hn.component.to_numpy()
        o = np.argsort(hn_g)
        hn_g, hn_c = hn_g[o], hn_c[o]
        lneg = np.isin(layer_of_gene[s], ["b1_same_genus_negative", "b2_same_family_negative"])
        k2 = k3 = 0
        g2 = set()
        for si, ei in zip(s[lneg], e[lneg]):
            lo, hi = np.searchsorted(hn_g, si), np.searchsorted(hn_g, ei, side="right")
            if hi - lo >= 2:
                nd = len(set(hn_c[lo:hi]))
                k2 += int(nd >= 2)
                k3 += int(nd >= 3)
                if nd >= 3:
                    g2.add(gen_arr[si])
        item["false_positive_side"] = dict(
            isolated_homologs=int(len(Hiso)), isolated_in_any_locus=float(cov[Hiso.gid.to_numpy()].mean()) if len(Hiso) else None,
            isolated_in_locus_ge3=float(cov3[Hiso.gid.to_numpy()].mean()) if len(Hiso) else None,
            isolated_smallest_locus_span_median=float(np.median(sm[sm > 0])) if (sm > 0).any() else None,
            isolated_by_component_in_locus={c: float(cov[g.gid.to_numpy()].mean()) for c, g in Hiso.groupby("component")},
            background_b1b2_in_any_locus=float(cov[neg_gene].mean()), background_b1b2_in_locus_ge3=float(cov3[neg_gene].mean()),
            all_rb_homologs_b1b2=int(len(Hn)), loci_b1b2_ge2_components=k2, loci_b1b2_ge3_components=k3,
            genomes_b1b2_with_ge3_component_locus=int(len(g2)))
        item["coverage"] = dict(loci=int(len(s)), genes_in_loci_frac=float(cov.mean()),
                                span_q50_q90_q99_max=np.quantile(e - s + 1, [.5, .9, .99, 1]).tolist() if len(s) else None,
                                by_layer={k: float(cov[layer_of_gene == k].mean()) for k in np.unique(layer_of_gene)})
        item["seconds"] = round(time.time() - tl, 1)
        res["outputs"][label] = item
        L = item["lut_main"]["all"]
        log(label, "lut F1 %.3f ctl %.3f exc %.3f" % (L["f1"], L["control"], L["excess"]), "ribo", item["ribosomal"]["all"],
            "complete", cs, "cov", round(item["coverage"]["genes_in_loci_frac"], 3), f"{item['seconds']}s")
    for A, Bn in PAIRS:
        if A in pers and Bn in pers:
            res["pairs"][f"{A}__vs__{Bn}"] = dict(
                lut_main_all=paired(pers[A], pers[Bn]),
                lut_main_even=paired(pers[A][pers[A].even], pers[Bn][pers[Bn].even]),
                ribo_resolved_all=ribo_paired(ribos[A], ribos[Bn]),
                ribo_resolved_even=ribo_paired(ribos[A][even_run].reset_index(drop=True), ribos[Bn][even_run].reset_index(drop=True)))
            log("pair", A, Bn, res["pairs"][f"{A}__vs__{Bn}"]["lut_main_all"], res["pairs"][f"{A}__vs__{Bn}"]["ribo_resolved_all"])
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["seconds"] = round(time.time() - t0, 1)
    res["peak_rss_gb"] = rss_gb()
    jdump(res, FZ / "results" / f"score_lut{a.suffix}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "score"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    {"prep": cmd_prep, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
