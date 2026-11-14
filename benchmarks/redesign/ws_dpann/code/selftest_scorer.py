#!/usr/bin/env python
"""Self-test of score_dpann_loci.py on inputs whose answers are known.

A. Whole-contig loci (one locus [0, n-1] per contig, module = contig): every instance must have
   coverage 1, covered50 = 1, boundary precision k/n, F1 2k/(k+n), span precision S/n; every
   control draw must give the same values, so F1 minus control is exactly 0. Module plurality share
   per system type must equal the share of its instances on its most common contig.
B. Gold instances as loci (module = system type): covered50 = 1 for every instance (its own span
   covers all members). For instances with contiguous members (span = k) best F1 = 1 and
   precision = 1. Every best F1 is >= the own-span value 2k/(k+S). Module plurality share = 1 for
   every type on the subset where no instance of another type has a locus overlapping it.
C. No loci: every measure is 0.
D. Random loci: the vectorized scorer equals an independent brute-force implementation, for the
   observed placement and for the first 5 control draws, on a random sample of instances.
Writes selftest.json and exits non-zero on any failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from score_dpann_loci import score  # noqa: E402

TOL = 1e-12


def brute(members, loci):
    """Reference: loop over loci; same tie-break; returns (f1, cov, prec, sprec, c50, localized, module)."""
    members = sorted(set(int(x) for x in members))
    k, first, last = len(members), members[0], members[-1]
    best = None
    c50 = loc = False
    for s, e, m in loci:
        ov = sum(1 for x in members if s <= x <= e)
        span = e - s + 1
        cov, prec, f1 = ov / k, ov / span, 2 * ov / (k + span)
        c50 |= cov >= 0.5
        loc |= cov >= 0.5 and prec >= 0.5
        if ov == 0:
            continue
        sp = max(0, min(e, last) - max(s, first) + 1) / span
        key = (-f1, -cov, span, s, e)
        if best is None or key < best[0]:
            best = (key, f1, cov, prec, sp, m)
    if best is None:
        return 0.0, 0.0, 0.0, 0.0, c50, loc, None
    return best[1], best[2], best[3], best[4], c50, loc, best[5]


def draws(inst_id, n, members, controls, seed):
    members = np.unique(np.asarray(members, dtype=np.int64))
    offs = members - members[0]
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{seed}:{inst_id}".encode()).digest()[:8], "big"))
    return [members] + [offs + o for o in rng.integers(0, n - int(offs[-1]), size=controls)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pkg = a.pkg
    inst = pd.read_parquet(pkg / "gold" / "instances.parquet")
    inst = inst[inst.resolved.fillna(False) & inst.order_trusted.fillna(False)].copy()
    contigs = pd.read_parquet(pkg / "universe" / "contigs.parquet")
    nmap = contigs.set_index("contig_code").n_genes
    checks = {}
    ok = True

    def check(name, cond, detail=None):
        nonlocal ok
        checks[name] = dict(passed=bool(cond), detail=detail)
        ok &= bool(cond)

    # ---------------- A: whole-contig loci
    used = contigs[contigs.contig_code.isin(inst.contig_code.unique())]
    loci = pd.DataFrame(dict(genome=used.genome, contig=used.contig, start=0, end=used.n_genes - 1,
                             module_id=used.contig_code.astype(str)))
    per, rep, raw = score(pkg, loci, controls=200, seed=17, threads=a.threads)
    pm = per.merge(inst[["instance_id", "contig_code"]], on="instance_id")
    k = pm.n_members_resolved.to_numpy(float)
    n = pm.contig_code.map(nmap).to_numpy(float)
    S = (pm.end_idx - pm.start_idx + 1).to_numpy(float)
    check("A_coverage_all_1", np.all(pm.coverage == 1.0))
    check("A_covered50_all", bool(pm.covered50.all()))
    check("A_precision_eq_k_over_n", np.max(np.abs(pm.boundary_precision - k / n)) < TOL,
          float(np.max(np.abs(pm.boundary_precision - k / n))))
    check("A_f1_eq_2k_over_k_plus_n", np.max(np.abs(pm.f1 - 2 * k / (k + n))) < TOL)
    check("A_span_precision_eq_S_over_n", np.max(np.abs(pm.span_precision - S / n)) < TOL)
    check("A_controls_identical_to_observed", np.max(np.abs(raw["F1"] - raw["F1"][:, :1])) < TOL)
    deltas = {c: v["f1_minus_control"] for c, v in rep["by_caller"].items()}
    check("A_f1_minus_control_zero", all(abs(x) < 1e-12 for x in deltas.values()), deltas)
    pvals = {c: v["p_f1_vs_control"] for c, v in rep["by_caller"].items()}
    check("A_p_value_one", all(x == 1.0 for x in pvals.values()), pvals)
    exp_share = {}
    bad = []
    for (c, t), g in pm.groupby(["caller", "system_type"]):
        e = g.contig_code.value_counts().iloc[0] / len(g)
        got = rep["by_type"][c][str(t)]["module_plurality_share_hit"]
        exp_share[f"{c}|{t}"] = e
        if abs(got - e) > TOL:
            bad.append((c, t, e, got))
    check("A_module_share_eq_contig_plurality", not bad, bad[:5])
    summ_A = {c: dict(mean_f1=v["mean_best_f1"], mean_precision=v["mean_boundary_precision_at_best"],
                      expected_mean_precision=float(np.mean((k / n)[pm.caller == c])),
                      mean_f1_expected=float(np.mean((2 * k / (k + n))[pm.caller == c])))
              for c, v in rep["by_caller"].items()}
    for c, v in summ_A.items():
        check(f"A_mean_precision_{c}", abs(v["mean_precision"] - v["expected_mean_precision"]) < 1e-12)

    # ---------------- B: gold instances as loci, one caller at a time
    summ_B = {}
    for c, gi in inst.groupby("caller"):
        loci = pd.DataFrame(dict(genome=gi.genome, contig=gi.contig, start=gi.start_idx, end=gi.end_idx,
                                 module_id=gi.system_type.astype(str)))
        per, rep, raw = score(pkg, loci, controls=0, threads=a.threads, callers=[c])
        pm = per.merge(gi[["instance_id", "contig_code"]], on="instance_id")
        k = pm.n_members_resolved.to_numpy(float)
        S = (pm.end_idx - pm.start_idx + 1).to_numpy(float)
        contiguous = S == k
        check(f"B_{c}_covered50_all", bool(pm.covered50.all()))
        check(f"B_{c}_contiguous_f1_1", bool(np.all(pm.f1[contiguous] == 1.0)), int(contiguous.sum()))
        check(f"B_{c}_contiguous_precision_1", bool(np.all(pm.boundary_precision[contiguous] == 1.0)))
        check(f"B_{c}_f1_ge_own_span", bool(np.all(pm.f1 >= 2 * k / (k + S) - TOL)))
        # isolated subset: no locus of another type overlaps the instance span
        iso = []
        for cc, g in gi.groupby("contig_code"):
            st, en, ty = g.start_idx.to_numpy(), g.end_idx.to_numpy(), g.system_type.astype(str).to_numpy()
            for i in range(len(g)):
                other = (ty != ty[i]) & (st <= en[i]) & (en >= st[i])
                if not other.any():
                    iso.append(int(g.instance_id.iloc[i]))
        sub = pm[pm.instance_id.isin(iso)]
        wrong = int((sub.best_module != sub.system_type.astype(str)).sum())
        check(f"B_{c}_isolated_module_is_own_type", wrong == 0, dict(isolated=len(sub), wrong=wrong))
        shares = {}
        for t, g in sub.groupby("system_type"):
            shares[str(t)] = float((g.best_module == str(t)).mean())
        check(f"B_{c}_isolated_module_share_1", all(v == 1.0 for v in shares.values()))
        summ_B[c] = dict(instances=len(pm), contiguous=int(contiguous.sum()), isolated=len(sub),
                         mean_f1=float(pm.f1.mean()), covered50=float(pm.covered50.mean()),
                         module_consistency_all=rep["by_caller"][c].get("module_consistency_weighted"))

    # ---------------- C: no loci
    empty = pd.DataFrame(dict(genome=[], contig=[], start=[], end=[], module_id=[]))
    per, rep, raw = score(pkg, empty, controls=3, threads=a.threads)
    check("C_empty_all_zero", float(per.f1.abs().sum() + per.coverage.sum() + per.covered50.sum()) == 0.0)

    # ---------------- D: random loci vs brute force
    rng = np.random.default_rng(17)
    sample = inst.sample(n=min(1500, len(inst)), random_state=17)
    cc = sample.contig_code.unique()
    rows = []
    for c_ in cc:
        nn = int(nmap[c_])
        for _ in range(int(rng.integers(0, 8))):
            s = int(rng.integers(0, nn))
            e = int(min(nn - 1, s + rng.integers(0, 12)))
            rows.append((c_, s, e, f"m{int(rng.integers(0, 5))}"))
    # one module per exact span, so the tie-break between identical spans cannot differ by row order
    rl = pd.DataFrame(rows, columns=["contig_code", "start", "end", "module_id"]).drop_duplicates(
        ["contig_code", "start", "end"]).merge(contigs[["contig_code", "genome", "contig"]], on="contig_code")
    per, rep, raw = score(pkg, rl[["genome", "contig", "start", "end", "module_id"]], controls=5, seed=17,
                          threads=a.threads, callers=None)
    pos = {i: j for j, i in enumerate(raw["ids"])}
    labels = raw["labels"]
    mism = 0
    tested = 0
    by_c = {c_: [(int(r.start), int(r.end), str(r.module_id)) for r in g.drop_duplicates(["start", "end", "module_id"]).itertuples()]
            for c_, g in rl.groupby("contig_code")}
    for r in sample.itertuples():
        L = by_c.get(r.contig_code, [])
        for d, mem in enumerate(draws(r.instance_id, int(nmap[r.contig_code]), r.member_positions, 5, 17)):
            f1, cov, prec, sp, c50, loc, mod = brute(mem, L)
            j = pos[r.instance_id]
            gm = raw["MOD"][j, d]
            got_mod = labels[gm] if gm >= 0 else None
            same = (abs(raw["F1"][j, d] - f1) < TOL and abs(raw["COV"][j, d] - cov) < TOL
                    and abs(raw["PREC"][j, d] - prec) < TOL and abs(raw["SPREC"][j, d] - sp) < TOL
                    and bool(raw["C50"][j, d]) == c50 and got_mod == mod)
            mism += not same
            tested += 1
    check("D_vectorized_equals_brute_force", mism == 0, dict(tested=tested, mismatches=mism, loci=len(rl)))

    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    result = dict(passed=ok, checks=checks, whole_contig=summ_A, gold_as_loci=summ_B,
                  instances_tested=int(len(inst)),
                  scorer_sha256=sha(Path(__file__).parent / "score_dpann_loci.py"), selftest_sha256=sha(__file__),
                  instances_sha256=sha(pkg / "gold" / "instances.parquet"))
    a.out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
