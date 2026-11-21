#!/usr/bin/env python
"""Frozen run 2 scorer (SR-VP bins; protocol FROZEN2.json 'scoring'; read-only scorers imported).

curated   package gold (DefenseFinder, TXSScan) through ws_dpann/package/scorer/score_dpann_loci.score (the package has the
          DPANN gold columns), 200 controls, seed 17, min_members 2; TXSScan as called and span-limited (span_genes <= 30);
          every number also without genome G03c08f50ba11988e; means with instance bootstrap (10,000), paired differences
          by genome bootstrap (10,000), seed 17
ribosomal ribosomal_check.score on the g2 runs with >= 16 marked genes; any fitting locus; accurate reading (bio.py rules,
          restricted annotation of annot2.py) strict / loose when frozen2/annot/genes_annot.parquet exists
subsets   all / even (test half) / Bacteria / Archaea / multi-contig / single-contig
usage: fscore2.py score --threads T [--suffix S]
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
FZ = RD / "long_loci/frozen2"
PKG = RD / "ws_srvp_bins/package"
sys.path.insert(0, str(RD / "ws_dpann/package/scorer"))
sys.path.insert(0, str(RD / "ribosomal/code"))
sys.path.insert(0, str(RD / "ribo_audit/code"))
import score_dpann_loci as S  # noqa: E402
import ribosomal_check as RC  # noqa: E402

BAD = "G03c08f50ba11988e"
NB = 10_000
CALLERS = ("DefenseFinder", "TXSScan")
OUTPUTS = {"PRIMARY_Z_L4bp_t0.02": FZ / "loci/srvp_Z/L4bp_t0.02.parquet", "Z_L1": FZ / "loci/srvp_Z/L1.parquet",
           "SECONDARY_WA_L4b_t0.2": FZ / "loci/srvp_WA/L4b_t0.2.parquet", "WA_L1": FZ / "loci/srvp_WA/L1.parquet",
           "engine_DH": FZ / "loci/srvp_engine/DH.parquet",
           "synton_faithful": FZ / "synton/runs/faithful/post/loci.parquet", "synton_matched": FZ / "synton/runs/matched/post/loci.parquet"}
REF = "engine_DH"
PAIRS = [("PRIMARY_Z_L4bp_t0.02", REF), ("SECONDARY_WA_L4b_t0.2", REF), ("PRIMARY_Z_L4bp_t0.02", "SECONDARY_WA_L4b_t0.2"),
         ("PRIMARY_Z_L4bp_t0.02", "Z_L1"), ("SECONDARY_WA_L4b_t0.2", "WA_L1"), ("Z_L1", REF), ("WA_L1", REF),
         ("PRIMARY_Z_L4bp_t0.02", "synton_faithful"), ("PRIMARY_Z_L4bp_t0.02", "synton_matched")]


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def jdump(obj, fn):
    Path(fn).write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    log("wrote", fn)


def boot_ci(v, n=NB):
    v = np.asarray(v, float)
    if len(v) == 0:
        return [None, None]
    rng = np.random.default_rng(17)
    m = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def summ(p):
    if len(p) == 0:
        return dict(n=0)
    return dict(n=int(len(p)), f1=float(p.f1.mean()), f1_ci=boot_ci(p.f1), control=float(p.control_mean_f1.mean()),
                excess=float(p.excess.mean()), excess_ci=boot_ci(p.excess), coverage=float(p.coverage.mean()),
                precision=float(p.boundary_precision.mean()), covered50=float(p.covered50.mean()),
                median_bprec_covered50=float(p.boundary_precision[p.coverage >= .5].median()) if (p.coverage >= .5).any() else None,
                no_overlap=int((p.coverage == 0).sum()))


def views(per):
    """caller views: DF, TXSS as called, TXSS span <= 30; each also without the pathological genome."""
    v = {"DefenseFinder": per[per.caller == "DefenseFinder"], "TXSScan": per[per.caller == "TXSScan"],
         "TXSScan_span30": per[(per.caller == "TXSScan") & (per.span_genes <= 30)]}
    v.update({k + "_noG03": x[x.genome != BAD] for k, x in list(v.items())})
    return v


def subsets(per, gm):
    dom = per.genome.map(dict(zip(gm.genome, gm.domain)))
    nc = per.genome.map(dict(zip(gm.genome, gm.n_contigs)))
    ev = per.genome.map(dict(zip(gm.genome, gm.genome_code % 2 == 0)))
    return {"all": per, "even": per[ev.to_numpy()], "Bacteria": per[(dom == "Bacteria").to_numpy()],
            "Archaea": per[(dom == "Archaea").to_numpy()], "multi_contig": per[(nc >= 2).to_numpy()],
            "single_contig": per[(nc == 1).to_numpy()]}


def summed(vv, a="DefenseFinder", b="TXSScan"):
    return float(vv[a].excess.mean() + vv[b].excess.mean())


def paired_genome(A, Bp, n=NB):
    """genome bootstrap of per-caller mean differences (F1, excess) and summed excess (DF + TXSS as called), plus the
    span-limited and no-G03 variants. A, Bp: per-instance frames indexed by instance_id."""
    j = A[["caller", "genome", "span_genes", "f1", "excess"]].join(Bp[["f1", "excess"]], rsuffix="_b", how="inner")
    genomes = pd.Index(sorted(j.genome.unique()))
    G = len(genomes)
    rng = np.random.default_rng(17)
    Wt = rng.multinomial(G, np.full(G, 1.0 / G), size=n).astype(np.float64)
    out = {}
    sets = {"DefenseFinder": j[j.caller == "DefenseFinder"], "TXSScan": j[j.caller == "TXSScan"],
            "TXSScan_span30": j[(j.caller == "TXSScan") & (j.span_genes <= 30)]}
    sets.update({k + "_noG03": x[x.genome != BAD] for k, x in list(sets.items())})
    draws = {}
    for k, x in sets.items():
        gi = genomes.get_indexer(x.genome)
        cnt = np.bincount(gi, minlength=G).astype(float)
        item = {}
        for c in ("f1", "excess"):
            d = (x[c] - x[c + "_b"]).to_numpy()
            s = np.bincount(gi, weights=d, minlength=G)
            m = (Wt @ s) / np.maximum(Wt @ cnt, 1)
            item[c] = [float(d.mean()) if len(d) else None, float(np.quantile(m, .025)), float(np.quantile(m, .975))]
            draws[(k, c)] = m
        item["n"] = int(len(x))
        item["better_worse_f1"] = [int((x.f1 > x.f1_b).sum()), int((x.f1 < x.f1_b).sum())]
        out[k] = item
    for tag, b in (("summed", "TXSScan"), ("summed_span30", "TXSScan_span30"), ("summed_noG03", "TXSScan_noG03")):
        a = "DefenseFinder" if tag != "summed_noG03" else "DefenseFinder_noG03"
        pt = out[a]["excess"][0] + out[b]["excess"][0]
        m = draws[(a, "excess")] + draws[(b, "excess")]
        lo, hi = float(np.quantile(m, .025)), float(np.quantile(m, .975))
        out[tag] = dict(d=[pt, lo, hi], verdict="BEATS" if lo > 0 else ("MATCHES" if pt >= -0.02 else "BELOW"))
    return out


@njit(cache=True)
def _any_fit(a, b, maxlen, qs, qe):
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


class Accurate:
    """ribo_audit bio.py accurate targets on this package (Bio.__init__ / run_units logic with the package arrays)."""
    def __init__(self, genes, runs, d_op, d_mem):
        import bio
        self.bio = bio
        an = pd.read_parquet(FZ / "annot/genes_annot.parquet").sort_values("gid")
        n = len(genes)
        cc = genes.contig_code.to_numpy()
        strand = genes.strand.to_numpy()
        start, end = genes.start.to_numpy(np.int64), genes.end.to_numpy(np.int64)
        lo, hi = np.minimum(start, end), np.maximum(start, end)
        member = np.isin(an.cls.to_numpy(), ["R", "N"])
        isR = an.cls.to_numpy() == "R"
        same = cc[:-1] == cc[1:]
        dist = np.full(n - 1, 10 ** 9, np.int64)
        dist[same] = (lo[1:] - hi[:-1] - 1)[same]
        sstr = strand[:-1] == strand[1:]
        mm = member[:-1] & member[1:]
        link = same & sstr & ((dist <= d_op) | (mm & (dist <= d_mem)))
        cs, ce, nm, nr = bio.build_components(link, member, isR, bio.X_CUT)
        comp_of = np.full(n, -1, np.int64)
        for i, (s, e) in enumerate(zip(cs, ce)):
            comp_of[s:e + 1] = i
        marked = np.zeros(n, bool)
        marked[pd.read_parquet(PKG / "ribosomal/marked.parquet").gid.to_numpy()] = True
        self.targets = []
        for s, e in zip(runs.gid_start.to_numpy(), runs.gid_end.to_numpy()):
            mk = [k for k in range(s, e + 1) if marked[k]]
            comps = sorted(set(int(comp_of[k]) for k in mk if comp_of[k] >= 0))
            single = [k for k in mk if comp_of[k] < 0]
            if len(comps) + len(single) == 1 and comps:
                t = [(int(cs[comps[0]]), int(ce[comps[0]]))]
            else:
                t = [(int(cs[c]), int(ce[c])) for c in comps if nm[c] >= bio.MIN_TARGET_MEMBERS]
            self.targets.append(t)
        self.info = dict(components=int(len(cs)), members=int(member.sum()), R=int(isR.sum()),
                         runs_with_target=int(sum(1 for t in self.targets if t)),
                         multi_target_runs=int(sum(1 for t in self.targets if len(t) > 1)))

    def ok(self, s, e):
        import audit as AU
        L = AU.Loci.__new__(AU.Loci)
        o = np.argsort(s, kind="stable")
        L.a, L.b = s[o], e[o]
        L.maxlen = int((L.b - L.a + 1).max())
        L.n = len(s)
        return np.array([bool(t) and all(self.bio.score_target(L, ts, te)["ok"] for ts, te in t) for t in self.targets])


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


def cmd_score(a):
    t0 = time.time()
    genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome", "genome_code", "contig", "contig_code",
                                                                     "position_index", "strand", "start", "end"]).sort_values("gid")
    n = len(genes)
    gen_arr, con_arr, pos = genes.genome.to_numpy(), genes.contig.to_numpy(), genes.position_index.to_numpy()
    contigs = pd.read_parquet(PKG / "universe/contigs.parquet")
    first_gid = np.zeros(int(contigs.contig_code.max()) + 1, np.int64)
    first_gid[contigs.contig_code.to_numpy()] = contigs.first_gid.to_numpy()
    gm = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome", "genome_code", "domain", "n_contigs"]).sort_values("genome_code")
    gdom = gm.domain.to_numpy()[genes.genome_code.to_numpy()]
    gmulti = gm.n_contigs.to_numpy()[genes.genome_code.to_numpy()] >= 2
    inst = pd.read_parquet(PKG / "gold/instances.parquet", columns=["instance_id", "span_genes"])
    runs = pd.read_parquet(PKG / "ribosomal/runs.parquet")
    runs = runs[(runs.run_set == "g2") & (runs.n_marked >= 16)].sort_values("gid_start").reset_index(drop=True)
    rsub = {"all": np.ones(len(runs), bool), "even": (runs.genome_code % 2 == 0).to_numpy(),
            "Bacteria": (runs.domain == "Bacteria").to_numpy(), "Archaea": (runs.domain == "Archaea").to_numpy(),
            "multi_contig": gm.n_contigs.to_numpy()[runs.genome_code.to_numpy()] >= 2}
    rsub["single_contig"] = ~rsub["multi_contig"]
    acc = {}
    if (FZ / "annot/genes_annot.parquet").exists():
        for rd, dd in (("strict", (150, 500)), ("loose", (500, 500))):
            acc[rd] = Accurate(genes, runs, *dd)
            log("accurate", rd, acc[rd].info)
    (FZ / "scores").mkdir(parents=True, exist_ok=True)
    res = dict(outputs={}, pairs={}, absent=[], accurate_info={k: v.info for k, v in acc.items()},
               runs={k: int(v.sum()) for k, v in rsub.items()})
    pers, ribs = {}, {}
    for label, path in OUTPUTS.items():
        if not path.exists():
            res["absent"].append(label)
            log("absent", label)
            continue
        tl = time.time()
        s, e = load_loci(label, path, first_gid)
        df = pd.DataFrame(dict(genome=gen_arr[s], contig=con_arr[s], start=pos[s], end=pos[e], module_id=np.full(len(s), -1)))
        per, _, _ = S.score(PKG, df, controls=200, seed=17, threads=a.threads, min_members=2)
        per["excess"] = per.f1 - per.control_mean_f1
        per = per.merge(inst, on="instance_id", how="left")
        per.to_parquet(FZ / "scores" / f"{label}.parquet", index=False)
        pers[label] = per.set_index("instance_id")
        item = {"curated": {}}
        for sk, sub in subsets(per, gm).items():
            vv = views(sub)
            item["curated"][sk] = {k: summ(x) for k, x in vv.items()}
            item["curated"][sk]["summed"] = summed(vv)
            item["curated"][sk]["summed_span30"] = summed(vv, b="TXSScan_span30")
            item["curated"][sk]["summed_noG03"] = summed(vv, a="DefenseFinder_noG03", b="TXSScan_noG03")
        ls = dict(a=s, b=e, mod=np.full(len(s), -1, np.int64), maxlen=int((e - s + 1).max()) if len(s) else 1, n=int(len(s)))
        sc = RC.score(ls, runs.gid_start.to_numpy(), runs.gid_end.to_numpy())
        sc["any_fit"] = _any_fit(s, e, ls["maxlen"], runs.gid_start.to_numpy(np.int64), runs.gid_end.to_numpy(np.int64)).astype(bool)
        for rd, A_ in acc.items():
            sc[f"acc_{rd}"] = A_.ok(s, e)
        ribs[label] = sc
        item["ribosomal"] = {k: dict(runs=int(m.sum()), recovered=int(sc.recovered[m].sum()), resolved=int(sc.resolved[m].sum()),
                                     any_fit=int(sc.any_fit[m].sum()),
                                     **{f"accurate_{rd}": int(sc[f"acc_{rd}"][m].sum()) for rd in acc},
                                     median_precision_recovered=float(sc.precision[m & sc.recovered.to_numpy()].median()) if (m & sc.recovered.to_numpy()).any() else None)
                             for k, m in rsub.items()}
        cov = np.zeros(n + 1, np.int32)
        np.add.at(cov, s, 1)
        np.add.at(cov, e + 1, -1)
        inl = np.cumsum(cov)[:n] > 0
        item["coverage"] = dict(loci=int(len(s)), genes_in_loci_frac=float(inl.mean()),
                                Bacteria=float(inl[gdom == "Bacteria"].mean()), Archaea=float(inl[gdom == "Archaea"].mean()),
                                multi_contig=float(inl[gmulti].mean()), single_contig=float(inl[~gmulti].mean()),
                                span_q50_q90_q99_max=np.quantile(e - s + 1, [.5, .9, .99, 1]).tolist() if len(s) else None)
        item["seconds"] = round(time.time() - tl, 1)
        res["outputs"][label] = item
        c = item["curated"]["all"]
        log(label, "DF %.3f/%.3f/%+.3f TXSS %.3f/%+.3f summed %+.3f" % (c["DefenseFinder"]["f1"], c["DefenseFinder"]["control"],
            c["DefenseFinder"]["excess"], c["TXSScan"]["f1"], c["TXSScan"]["excess"], c["summed"]), "ribo", item["ribosomal"]["all"],
            "genes", round(item["coverage"]["genes_in_loci_frac"], 3), f"{item['seconds']}s")
    for A_, B_ in PAIRS:
        if A_ in pers and B_ in pers:
            pr = paired_genome(pers[A_], pers[B_])
            d = ribs[A_].resolved.to_numpy().astype(int) - ribs[B_].resolved.to_numpy().astype(int)
            rng = np.random.default_rng(17)
            m = d[rng.integers(0, len(d), size=(NB, len(d)))].sum(1)
            pr["ribo_resolved"] = [int(d.sum()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]
            for rd in acc:
                d2 = ribs[A_][f"acc_{rd}"].to_numpy().astype(int) - ribs[B_][f"acc_{rd}"].to_numpy().astype(int)
                pr[f"accurate_{rd}"] = int(d2.sum())
            res["pairs"][f"{A_}__vs__{B_}"] = pr
            log("pair", A_, B_, pr["summed"], pr["ribo_resolved"])
    res["written"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["seconds"] = round(time.time() - t0, 1)
    jdump(res, FZ / "results" / f"score_srvp{a.suffix}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score"])
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    cmd_score(a)


if __name__ == "__main__":
    main()
