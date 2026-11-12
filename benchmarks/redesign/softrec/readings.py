#!/usr/bin/env python
"""Single-reading comparisons for softrec result directories (coordinator request, 2026-11-12).

The DPANN scorer takes, per gold instance, the best locus of the whole loci_all_genomes table (11 readings), so the
full-table F1 is a best-of-11 figure. Here every result directory is also scored on single readings:
  R1  default reading: conserved@2 (maximal runs of boundaries with support >= 2 genera)
  A   rule A of the boundary workstream (relative end trimming, f 0.5): end genes of each R1 locus are dropped
      while the end boundary's support < 0.5 x the locus median support
  UA  token-free analogue of rule H (H = module-consistent segmentation at Louvain res 0.3, then A f 0.5; softrec
      has no tokens and no modules): the engine's profile units (unit@0.5, cut where the genome sets of the best
      instances at adjacent boundaries have Jaccard < 0.5), then A f 0.5
Token references: the boundary workstream's per-instance scores of R1 and H for base and rescue3, and this script's
A and UA applied to the token engines' own landscapes (to show what the analogue loses against true H).
Paired genome-bootstrap comparisons (2,000 draws, seed 17) on even and odd genomes; ribosomal check on each reading.

usage: readings.py --pairs SOFTDIR:TOKENREF ... --threads T --out results/readings.json
       TOKENREF in base | rescue3; every SOFTDIR reading X is compared with the token reading X
       (R1 vs R1, A vs A, UA vs UA) and with token H.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
OUT = RD / "softrec/readings"
TOKEN = {"base": RD / "engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3",
         "rescue3": RD / "engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3"}
BSC = RD / "boundary/scores"
CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")
sys.path.insert(0, str(PKG / "scorer"))
import score_dpann_loci as S  # noqa: E402
sys.path.insert(0, str(RD / "ribosomal/code"))
import ribosomal_check as RC  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


@njit(cache=True)
def runs_at_least(b, starts, ends, level, gap):
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
def trim_ends(b, s2, e2, f):
    """Copy of boundary/blib.py trim_ends (rule A)."""
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


class Ctx:
    def __init__(self):
        c = pd.read_parquet(PKG / "universe/contigs.parquet").sort_values("first_gid").reset_index(drop=True)
        self.contigs = c
        self.cs = c.first_gid.to_numpy(np.int64)
        self.ce = self.cs + c.n_genes.to_numpy(np.int64)
        self.n = int(self.ce.max())
        g = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome", "contig", "position_index"])
        self.genome = g.genome.to_numpy()
        self.contig = g.contig.to_numpy()
        self.pos = g.position_index.to_numpy(np.int64)
        gm = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome", "genome_code"])
        self.code = dict(zip(gm.genome, gm.genome_code))
        r = pd.read_parquet(RC.CACHE / "runs.parquet")
        self.runs = r[(r.run_set == "g2") & (r.n_marked >= 16)].reset_index(drop=True)

    def table(self, s, e):
        keep = e > s
        s, e = s[keep], e[keep]
        return pd.DataFrame(dict(genome=self.genome[s], contig=self.contig[s], start=self.pos[s], end=self.pos[e],
                                 module_id=-1)), s, e

    def unit_gids(self, rdir):
        l = pd.read_parquet(rdir / "loci_all_genomes.parquet")
        l = l[l.kind == "unit"].merge(self.contigs[["genome", "contig", "first_gid"]], on=["genome", "contig"], how="left")
        s = (l.first_gid + l.start).to_numpy(np.int64)
        e = (l.first_gid + l.end).to_numpy(np.int64)
        return s, e


def readings(ctx, rdir):
    b = np.load(rdir / "boundary_support.npy")
    s2, e2 = runs_at_least(b, ctx.cs, ctx.ce, 2.0, 0)
    sa, ea = trim_ends(b, s2, e2, 0.5)
    us, ue = ctx.unit_gids(rdir)
    ok = ue > us
    sua, eua = trim_ends(b, us[ok], ue[ok], 0.5)
    return {"R1": (s2, e2), "A": (sa, ea), "UA": (sua, eua)}


def score(ctx, s, e, threads):
    df, s, e = ctx.table(s, e)
    per, _, _ = S.score(PKG, df, controls=200, seed=17, threads=threads, min_members=2)
    per["excess"] = per.f1 - per.control_mean_f1
    sc = RC.score(dict(a=np.sort(s), b=e[np.argsort(s, kind="stable")], mod=np.full(len(s), -1, np.int64),
                       maxlen=int((e - s + 1).max()), n=int(len(s))),
                  ctx.runs.gid_start.to_numpy(), ctx.runs.gid_end.to_numpy())
    rs = RC.summarize(ctx.runs, sc)
    span = e - s + 1
    cov = np.zeros(ctx.n + 1, np.int32)
    np.add.at(cov, s, 1)
    np.add.at(cov, e + 1, -1)
    loci = dict(loci=int(len(s)), genes_in_loci=int((np.cumsum(cov)[:ctx.n] > 0).sum()), median_span=float(np.median(span)))
    ribo = {k: rs[k] for k in ("runs", "recovered", "resolved", "median_precision_recovered", "parvarchaeaceae")}
    return per, ribo, loci


def boot(values, groups, rng, draws=2000):
    codes, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(codes))
    n = np.bincount(inv, minlength=len(codes)).astype(float)
    pick = rng.integers(0, len(codes), size=(draws, len(codes)))
    m = s[pick].sum(1) / n[pick].sum(1)
    return [float(values.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def paired(A, B, code, split):
    rng = np.random.default_rng(17)
    A = A.set_index("instance_id")
    B = B.set_index("instance_id")
    ids = A.index.intersection(B.index)
    x, y = A.loc[ids], B.loc[ids]
    g = x.genome.map(code).to_numpy()
    m = (g % 2 == 0) if split == "even" else (g % 2 == 1)
    out, tot = {}, {}
    for c in CALLERS:
        k = m & (x.caller.to_numpy() == c)
        out[c] = dict(n=int(k.sum()), f1_A=float(x.f1.to_numpy()[k].mean()), f1_B=float(y.f1.to_numpy()[k].mean()),
                      excess_A=float(x.excess.to_numpy()[k].mean()), excess_B=float(y.excess.to_numpy()[k].mean()),
                      d_f1=boot((x.f1.to_numpy() - y.f1.to_numpy())[k], g[k], rng),
                      d_excess=boot((x.excess.to_numpy() - y.excess.to_numpy())[k], g[k], rng))
        tot[c] = ((x.excess.to_numpy() - y.excess.to_numpy())[k], g[k])
    d1, g1 = tot["DefenseFinder"]
    d2, g2 = tot["TXSScan"]
    allg = np.unique(np.r_[g1, g2])
    idx = {c: i for i, c in enumerate(allg)}
    i1 = np.array([idx[c] for c in g1], np.int64)
    i2 = np.array([idx[c] for c in g2], np.int64)
    s1, n1 = np.bincount(i1, weights=d1, minlength=len(allg)), np.bincount(i1, minlength=len(allg))
    s2, n2 = np.bincount(i2, weights=d2, minlength=len(allg)), np.bincount(i2, minlength=len(allg))
    pick = rng.integers(0, len(allg), size=(2000, len(allg)))
    bs = s1[pick].sum(1) / np.maximum(n1[pick].sum(1), 1) + s2[pick].sum(1) / np.maximum(n2[pick].sum(1), 1)
    out["summed_A"] = out["DefenseFinder"]["excess_A"] + out["TXSScan"]["excess_A"]
    out["summed_B"] = out["DefenseFinder"]["excess_B"] + out["TXSScan"]["excess_B"]
    out["summed_d"] = [float(d1.mean() + d2.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
    return out


def ladder(ctx, rdir, threads, splits):
    """Each of the 11 readings of loci_all_genomes scored alone: mean F1 and excess per caller and split, and the share
    of instances whose full-table best F1 is reached by that reading."""
    l = pd.read_parquet(rdir / "loci_all_genomes.parquet")
    full = pd.read_parquet(rdir / "score_min2/instance_scores.parquet").set_index("instance_id")
    out = {}
    for (kind, level), d in l.groupby(["kind", "level"]):
        df = d[["genome", "contig", "start", "end"]].assign(module_id=-1)
        per, _, _ = S.score(PKG, df, controls=200, seed=17, threads=threads, min_members=2)
        per["excess"] = per.f1 - per.control_mean_f1
        per = per.set_index("instance_id")
        g = per.genome.map(ctx.code).to_numpy()
        row = dict(loci=int(len(d)))
        for sp in splits:
            m = (g % 2 == 0) if sp == "even" else (g % 2 == 1)
            for c in CALLERS:
                k = m & (per.caller.to_numpy() == c)
                f = full.loc[per.index[k], "f1"].to_numpy()
                row[f"{sp}:{c}"] = dict(f1=float(per.f1.to_numpy()[k].mean()), excess=float(per.excess.to_numpy()[k].mean()),
                                        reaches_full_best=float((per.f1.to_numpy()[k] >= f - 1e-9).mean()))
        out[f"{kind}@{level:g}"] = row
        log(rdir.name, kind, level, {k: round(v["f1"], 3) for k, v in row.items() if isinstance(v, dict)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--splits", default="odd,even")
    ap.add_argument("--ladder", action="store_true", help="also score each of the 11 readings alone")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = Ctx()
    cache = {}
    ladders = {}

    def get(rdir):
        rdir = Path(rdir)
        if str(rdir) in cache:
            return cache[str(rdir)]
        res = {}
        for name, (s, e) in readings(ctx, rdir).items():
            f = OUT / f"{rdir.name}__{name}.parquet"
            per, ribo, loci = score(ctx, s, e, a.threads)
            per.to_parquet(f, index=False)
            res[name] = dict(per=per, ribo=ribo, loci=loci)
            log(rdir.name, name, ribo, loci)
        cache[str(rdir)] = res
        return res

    report = {}
    for pr in a.pairs:
        sdir, tref = pr.split(":")
        soft = get(sdir)
        tok = get(TOKEN[tref])
        bH = pd.read_parquet(BSC / tref / "H_res0.3_f0.5.parquet")
        bR1 = pd.read_parquet(BSC / tref / "R1_conserved2.parquet")
        bR1["excess"] = bR1.f1 - bR1.control_mean_f1
        bH["excess"] = bH.f1 - bH.control_mean_f1
        item = dict(soft={k: dict(ribo=v["ribo"], loci=v["loci"]) for k, v in soft.items()},
                    token={k: dict(ribo=v["ribo"], loci=v["loci"]) for k, v in tok.items()}, compare={})
        # this script's token R1 must reproduce the boundary workstream's R1 scores
        chk = tok["R1"]["per"].set_index("instance_id").f1.sort_index()
        ref = bR1.set_index("instance_id").f1.sort_index()
        item["token_R1_matches_boundary_R1"] = bool(np.allclose(chk.loc[ref.index].to_numpy(), ref.to_numpy()))
        for sp in a.splits.split(","):
            for x in ("R1", "A", "UA"):
                item["compare"][f"{sp}:{x}_vs_token_{x}"] = paired(soft[x]["per"], tok[x]["per"], ctx.code, sp)
                item["compare"][f"{sp}:{x}_vs_token_H"] = paired(soft[x]["per"], bH, ctx.code, sp)
            item["compare"][f"{sp}:token_UA_vs_token_H"] = paired(tok["UA"]["per"], bH, ctx.code, sp)
            item["compare"][f"{sp}:token_A_vs_token_R1"] = paired(tok["A"]["per"], tok["R1"]["per"], ctx.code, sp)
        if a.ladder:
            for d_ in (Path(sdir), TOKEN[tref]):
                if str(d_) not in ladders:
                    ladders[str(d_)] = ladder(ctx, d_, a.threads, a.splits.split(","))
            item["ladder_soft"] = ladders[str(Path(sdir))]
            item["ladder_token"] = ladders[str(TOKEN[tref])]
        report[pr] = item
        for k, v in item["compare"].items():
            log(pr, k, "summed A %.3f B %.3f d %+.3f [%+.3f, %+.3f]" % (v["summed_A"], v["summed_B"], *v["summed_d"]),
                "F1 A", {c[:4]: round(v[c]["f1_A"], 3) for c in CALLERS}, "F1 B", {c[:4]: round(v[c]["f1_B"], 3) for c in CALLERS})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    log("wrote", a.out)


if __name__ == "__main__":
    main()
