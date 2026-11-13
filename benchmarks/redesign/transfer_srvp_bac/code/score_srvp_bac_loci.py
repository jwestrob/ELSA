#!/usr/bin/env python
"""Score predicted loci against curated-caller gold loci on the SR-VP bacteria package.
(Copy of ws_dpann/package/scorer/score_dpann_loci.py, sha256 51400ca5...; only this docstring differs.)

Input loci: a parquet or TSV table with columns
    genome, contig, start, end, module_id
in gene-index coordinates (position_index of the package universe; start and end inclusive).
module_id may be any hashable label; use -1 or empty for "no module".

For every resolved gold instance (curated caller call, all members on one contig) the scorer finds
the single predicted locus with the best gene-level F1, using the same rule as `evaluate` in
$SCALE/code/evaluate_spacedust_queries.py:
    overlap   = number of member genes inside the locus
    coverage  = overlap / n_members
    precision = overlap / locus span            ("boundary precision", member-based)
    F1        = 2 * overlap / (n_members + span)
    best      = max F1, then max coverage, then min span, then min start, then min end
It also reports span_precision = |locus ∩ [first member, last member]| / locus span, and
covered50 = some single locus covers >= 50 % of the members; localized = some single locus has
coverage >= 0.5 and precision >= 0.5.

Control: each instance's exact member pattern is rigidly translated to a uniform legal origin on
its own contig, `controls` times (default 200), RNG seeded by sha256(f"{seed}:{instance_id}")
exactly as `evaluate` does, and scored the same way.

Module consistency: for each (caller, system_type), the share of instances whose best locus
carries the plurality module (instances without any overlapping locus count in the denominator
of `module_plurality_share_all` and are excluded from `module_plurality_share_hit`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

NO_MODULE = -1


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_loci(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, sep="\t")
    need = {"genome", "contig", "start", "end", "module_id"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"loci table lacks columns {sorted(missing)}")
    return df[["genome", "contig", "start", "end", "module_id"]].copy()


def prepare(pkg: Path, loci: pd.DataFrame):
    contigs = pd.read_parquet(pkg / "universe" / "contigs.parquet")
    key = contigs.set_index(["genome", "contig"])
    j = loci.merge(contigs[["genome", "contig", "contig_code", "n_genes"]], on=["genome", "contig"], how="left")
    if j.contig_code.isna().any():
        bad = j[j.contig_code.isna()].head(5)
        raise ValueError(f"{int(j.contig_code.isna().sum())} loci on contigs absent from the package, e.g. {bad.to_dict('records')}")
    if ((j.start < 0) | (j.end >= j.n_genes) | (j.end < j.start)).any():
        raise ValueError("loci outside their contig or with end < start")
    # integer module codes; NO_MODULE for missing
    mod = j.module_id.astype("string").fillna("")
    mod = mod.where(~mod.isin(["", "-1", "<NA>", "nan"]), "")
    codes, labels = pd.factorize(mod.where(mod != "", None), use_na_sentinel=True)
    j["module_code"] = np.where(codes < 0, NO_MODULE, codes)
    j = j.drop_duplicates(["contig_code", "start", "end", "module_code"])
    return j, list(labels), key


def score_contig(task):
    """task = (contig_code, n_genes, loci arrays, list of instances); returns per (instance, draw) rows."""
    cc, n, s, e, m, insts, controls, seed = task
    order = np.lexsort((e, s))
    s, e, m = s[order], e[order], m[order]
    span = e - s + 1
    out = []
    for inst_id, members in insts:
        members = np.unique(np.asarray(members, dtype=np.int64))
        k = len(members)
        offs = members - members[0]
        width = int(offs[-1])
        origins = [int(members[0])]
        if controls:
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{inst_id}".encode()).digest()[:8], "big"))
            origins += rng.integers(0, n - width, size=controls).tolist()
        o = np.asarray(origins, dtype=np.int64)[:, None]
        R = o.shape[0]
        if len(s) == 0:
            z = np.zeros(R)
            out.append((inst_id, z, z, z, z, z.astype(bool), z.astype(bool), np.full(R, NO_MODULE), np.zeros(R, np.int64)))
            continue
        lo = np.searchsorted(offs, s[None, :] - o, side="left")
        hi = np.searchsorted(offs, e[None, :] - o, side="right")
        ov = (hi - lo).astype(np.float64)
        cov = ov / k
        prec = ov / span[None, :]
        f1 = 2 * ov / (k + span[None, :])
        inter = np.minimum(e[None, :], o + width) - np.maximum(s[None, :], o) + 1
        sprec = np.clip(inter, 0, None) / span[None, :]
        covered50 = (cov >= 0.5).any(axis=1)
        localized = ((cov >= 0.5) & (prec >= 0.5)).any(axis=1)
        L = len(s)
        rows = np.repeat(np.arange(R), L)
        keys = np.lexsort((np.tile(e, R), np.tile(s, R), np.tile(span, R), -cov.ravel(), -f1.ravel(), rows))
        first = keys[np.searchsorted(rows[keys], np.arange(R))] - np.arange(R) * L
        idx = np.arange(R)
        hit = ov[idx, first] > 0
        bf1 = np.where(hit, f1[idx, first], 0.0)
        bcov = np.where(hit, cov[idx, first], 0.0)
        bprec = np.where(hit, prec[idx, first], 0.0)
        bsprec = np.where(hit, sprec[idx, first], 0.0)
        bmod = np.where(hit, m[first], NO_MODULE)
        bspan = np.where(hit, span[first], 0)
        out.append((inst_id, bf1, bcov, bprec, bsprec, covered50, localized, bmod, bspan))
    return out


def plurality(mods: np.ndarray):
    """(share among instances with a module, share among all, n distinct, plurality module)."""
    hitm = mods[mods != NO_MODULE]
    if len(hitm) == 0:
        return 0.0, 0.0, 0, NO_MODULE
    vals, cnt = np.unique(hitm, return_counts=True)
    top = cnt.argmax()
    return cnt[top] / len(hitm), cnt[top] / len(mods), len(vals), int(vals[top])


def score(pkg: Path, loci: pd.DataFrame, *, controls=200, seed=17, threads=1, include_untrusted=False,
          callers=None, min_members=1, max_span=None):
    pkg = Path(pkg)
    inst = pd.read_parquet(pkg / "gold" / "instances.parquet")
    inst = inst[inst.resolved.fillna(False)]
    if not include_untrusted:
        inst = inst[inst.order_trusted.fillna(False)]
    if callers:
        inst = inst[inst.caller.isin(callers)]
    inst = inst[inst.n_members_resolved >= min_members].copy()
    if max_span is not None:
        inst = inst[inst.span_genes <= max_span].copy()
    contigs = pd.read_parquet(pkg / "universe" / "contigs.parquet").set_index("contig_code")
    lj, labels, _ = prepare(pkg, loci)
    by_contig = {cc: g for cc, g in lj.groupby("contig_code")}
    tasks = []
    for cc, g in inst.groupby("contig_code"):
        n = int(contigs.loc[cc, "n_genes"])
        lg = by_contig.get(cc)
        if lg is None:
            s = e = m = np.empty(0, np.int64)
        else:
            s, e, m = (lg.start.to_numpy(np.int64), lg.end.to_numpy(np.int64), lg.module_code.to_numpy(np.int64))
        tasks.append((int(cc), n, s, e, m, list(zip(g.instance_id.tolist(), g.member_positions.tolist())), controls, seed))
    if threads > 1:
        with ProcessPoolExecutor(threads) as ex:
            res = [r for chunk in ex.map(score_contig, tasks, chunksize=64) for r in chunk]
    else:
        res = [r for t in tasks for r in score_contig(t)]
    R = controls + 1
    ids = np.array([r[0] for r in res])
    stack = lambda i: np.vstack([r[i] for r in res]) if res else np.empty((0, R))
    F1, COV, PREC, SPREC, C50, LOC, MOD, SPAN = (stack(i) for i in range(1, 9))
    per = pd.DataFrame(dict(instance_id=ids, f1=F1[:, 0], coverage=COV[:, 0], boundary_precision=PREC[:, 0],
                            span_precision=SPREC[:, 0], covered50=C50[:, 0].astype(bool), localized=LOC[:, 0].astype(bool),
                            best_module=[labels[x] if x != NO_MODULE else None for x in MOD[:, 0]],
                            best_span=SPAN[:, 0], control_mean_f1=F1[:, 1:].mean(1) if controls else np.nan,
                            control_covered50=C50[:, 1:].mean(1) if controls else np.nan))
    per = inst[["instance_id", "caller", "system_type", "genome", "contig", "start_idx", "end_idx",
                "n_members_resolved"]].merge(per, on="instance_id")
    pos = {iid: i for i, iid in enumerate(ids)}

    def summarize(sub: pd.DataFrame):
        rows = np.array([pos[i] for i in sub.instance_id])
        d = dict(instances=int(len(rows)))
        if not len(rows):
            return d
        d.update(covered50=float(C50[rows, 0].mean()), localized=float(LOC[rows, 0].mean()),
                 mean_best_f1=float(F1[rows, 0].mean()),
                 median_boundary_precision_at_best=float(np.median(PREC[rows, 0])),
                 mean_boundary_precision_at_best=float(PREC[rows, 0].mean()),
                 median_span_precision_at_best=float(np.median(SPREC[rows, 0])),
                 median_coverage_at_best=float(np.median(COV[rows, 0])))
        if controls:
            draw_f1 = F1[rows, 1:].mean(0)
            draw_c50 = C50[rows, 1:].mean(0)
            draw_prec = np.median(PREC[rows, 1:], axis=0)
            d.update(control_mean_best_f1=float(draw_f1.mean()),
                     control_mean_best_f1_ci95=[float(np.quantile(draw_f1, .025)), float(np.quantile(draw_f1, .975))],
                     control_covered50=float(draw_c50.mean()),
                     control_covered50_ci95=[float(np.quantile(draw_c50, .025)), float(np.quantile(draw_c50, .975))],
                     control_median_boundary_precision=float(draw_prec.mean()),
                     f1_minus_control=float(F1[rows, 0].mean() - draw_f1.mean()),
                     # one-sided empirical p over draws; ties within 1e-12 count against the observation
                     control_draws_ge_observed=int((draw_f1 >= F1[rows, 0].mean() - 1e-12).sum()),
                     p_f1_vs_control=float((1 + (draw_f1 >= F1[rows, 0].mean() - 1e-12).sum()) / (1 + controls)))
        hit_share, all_share, nmod, top = plurality(MOD[rows, 0])
        d.update(module_plurality_share_hit=float(hit_share), module_plurality_share_all=float(all_share),
                 distinct_best_modules=int(nmod), plurality_module=labels[top] if top != NO_MODULE else None)
        if controls:
            cs = np.array([plurality(MOD[rows, j])[0] for j in range(1, R)])
            d.update(control_module_plurality_share_hit=float(cs.mean()))
        return d

    report = dict(scope=dict(controls=controls, seed=seed, include_untrusted=include_untrusted,
                             callers=callers, min_members=min_members, max_span=max_span, loci=int(len(lj)),
                             loci_modules=len(labels)),
                  by_caller={}, by_type={}, by_size={})
    for c, g in per.groupby("caller"):
        report["by_caller"][c] = summarize(g)
        size = pd.cut(g.n_members_resolved, [0, 1, 3, 6, 10**6], labels=["1", "2-3", "4-6", "7+"])
        report["by_size"][c] = {str(k): summarize(gg) for k, gg in g.groupby(size, observed=True)}
        report["by_type"][c] = {}
        for t, gt in g.groupby("system_type", dropna=False):
            report["by_type"][c][str(t)] = summarize(gt)
    # instance-weighted module consistency over types with >= 2 instances
    for c in report["by_type"]:
        types = [v for v in report["by_type"][c].values() if v["instances"] >= 2]
        w = np.array([v["instances"] for v in types], dtype=float)
        if len(types):
            report["by_caller"][c]["module_consistency_weighted"] = float(
                np.dot(w, [v["module_plurality_share_hit"] for v in types]) / w.sum())
            if controls:
                report["by_caller"][c]["control_module_consistency_weighted"] = float(
                    np.dot(w, [v["control_module_plurality_share_hit"] for v in types]) / w.sum())
    return per, report, dict(F1=F1, COV=COV, PREC=PREC, SPREC=SPREC, C50=C50, MOD=MOD, ids=ids, labels=labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", type=Path, required=True)
    ap.add_argument("--loci", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--controls", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--include-untrusted", action="store_true")
    ap.add_argument("--callers", nargs="*")
    ap.add_argument("--min-members", type=int, default=1)
    ap.add_argument("--max-span", type=int, default=None,
                    help="drop instances whose first-to-last member span exceeds this many genes")
    a = ap.parse_args()
    per, report, _ = score(a.pkg, load_loci(a.loci), controls=a.controls, seed=a.seed, threads=a.threads,
                           include_untrusted=a.include_untrusted, callers=a.callers, min_members=a.min_members,
                           max_span=a.max_span)
    a.out.mkdir(parents=True, exist_ok=True)
    per.to_parquet(a.out / "instance_scores.parquet", index=False)
    report["inputs"] = dict(loci=str(a.loci), loci_sha256=_sha(a.loci),
                            instances_sha256=_sha(a.pkg / "gold" / "instances.parquet"), scorer_sha256=_sha(__file__))
    (a.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["by_caller"], indent=2))


if __name__ == "__main__":
    main()
