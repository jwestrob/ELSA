#!/usr/bin/env python
"""ribo_audit part 6 scorer: the frozen ribosomal benchmark (predeclared6.json) on any package.

usage (one thread; login node or Slurm):
  score6.py --package PKG --annot ANNOTDIR --primary LOCI --secondary LOCI --engine LOCI --out OUTDIR
            [--groups GROUPS.parquet] [--verdicts] [--name LABEL]
  PKG: package root with universe/{genes,contigs,genomes}.parquet and ribosomal tables (ribosomal/{marked,runs}.parquet or
       gold/ribosomal_{marked,runs}.parquet); optional universe/gene_flags.parquet.
  ANNOTDIR: annot6.py output (genes_annot.parquet, extra_ko.parquet, window_pfam.parquet, window_gids.npy).
  LOCI: parquet with (s, e) or (gid_s, gid_e) inclusive gids.
Writes OUTDIR/score6.json, score6_runs.parquet, table6.txt (and cases6_<output>.tsv with --verdicts).
Definition (a) = bio5.py variant 'a'; standing, part-4 and the part-2 accurate reading are reported without bars.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bio as Bm  # noqa: E402
import bio4 as B4  # noqa: E402
import bio5 as B5  # noqa: E402
import audit as A  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUTS = ("Z", "WA", "DH")
LABEL = {"Z": "primary (embedding-list bond tree)", "WA": "secondary (family-bond adaptive tree)", "DH": "engine reference"}


class Data6(B4.Data4):
    """Data4 with optional groups / flags and package-layout detection (CFG entry 'pkg')."""
    def __init__(self, ds):
        c = B4.CFG[ds]
        self.ds = ds
        g = pd.read_parquet(c["genes"], columns=["gid", "genome_code", "contig_code", "position_index", "strand", "length",
                                                 "start", "end"]).sort_values("gid").reset_index(drop=True)
        assert np.array_equal(g.gid.to_numpy(), np.arange(len(g)))
        self.n = len(g)
        self.cc = g.contig_code.to_numpy(np.int64)
        self.gc = g.genome_code.to_numpy(np.int64)
        self.pos = g.position_index.to_numpy(np.int64)
        self.strand = g.strand.to_numpy(np.int8)
        self.length = g.length.to_numpy(np.int64)
        self.start = g.start.to_numpy(np.int64)
        self.end = g.end.to_numpy(np.int64)
        ct = pd.read_parquet(c["contigs"]).set_index("contig_code")
        self.cfirst = ct.first_gid.reindex(self.cc).to_numpy(np.int64)
        self.clast = self.cfirst + ct.n_genes.reindex(self.cc).to_numpy(np.int64) - 1
        gm = pd.read_parquet(c["genomes"]).sort_values("genome_code").reset_index(drop=True)
        assert np.array_equal(gm.genome_code.to_numpy(), np.arange(len(gm)))
        dom = gm.domain.to_numpy() if "domain" in gm.columns else gm.tax.str.split(";").str[0].str.replace("d__", "").to_numpy()
        self.genomes = gm
        self.domain_of_genome = dom
        if c.get("groups") and Path(c["groups"]).exists():
            m = pd.read_parquet(c["groups"], columns=["genome_code", "group"]).sort_values("genome_code")
            self.grp = pd.factorize(m.group.to_numpy())[0].astype(np.int64)
        else:
            self.grp = np.arange(len(gm), dtype=np.int64)
        self.G = int(self.grp.max()) + 1
        mk = pd.read_parquet(c["marked"])
        self.marked = np.zeros(self.n, bool)
        self.marked[mk.gid.to_numpy()] = True
        self.marker = pd.Series(mk.marker.to_numpy(), index=mk.gid.to_numpy())
        r = pd.read_parquet(c["runs"])
        r = r[(r.run_set == "g2") & (r.m16 if "m16" in r.columns else (r.n_marked >= 16))].reset_index(drop=True)
        gcode = r.genome_code.to_numpy()
        r["domain"] = dom[gcode]
        closed = gm.n_contigs.to_numpy()[gcode] == 1
        r["gtype"] = np.where(r.domain == "Archaea", "Archaea", np.where(closed, "Bact_closed", "Bact_multi"))
        r["closed"] = closed
        r["phylum"] = gm.phylum.to_numpy()[gcode] if "phylum" in gm.columns else ""
        r["stratum"] = r.gtype
        r["group"] = self.grp[gcode]
        self.runs = r
        if c.get("flags") and Path(c["flags"]).exists():
            fl = pd.read_parquet(c["flags"]).sort_values("gid")
            pa = fl.partial.astype(str).to_numpy()
            self.partial = (pa != "00") & fl.partial.notna().to_numpy()
            self.truncated = fl.truncated.fillna(False).astype(bool).to_numpy() if "truncated" in fl.columns else np.zeros(self.n, bool)
        else:
            self.partial = np.zeros(self.n, bool)
            self.truncated = np.zeros(self.n, bool)


def first(*paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(paths[0])


def register(a):
    P = Path(a.package)
    an = Path(a.annot)
    B4.CFG["pkg"] = dict(
        genes=P / "universe/genes.parquet", contigs=P / "universe/contigs.parquet", genomes=P / "universe/genomes.parquet",
        marked=first(P / "ribosomal/marked.parquet", P / "gold/ribosomal_marked.parquet"),
        runs=first(P / "ribosomal/runs.parquet", P / "gold/ribosomal_runs.parquet"),
        groups=a.groups, flags=P / "universe/gene_flags.parquet", annot=an, base=an / "genes_annot.parquet",
        out={"Z": Path(a.primary), "WA": Path(a.secondary), "DH": Path(a.engine)}, target={n: (None, None) for n in OUTS})
    B4.Data4 = Data6


def boot_sum(d, n=2000, seed=17):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    s = d[idx].sum(1)
    return float(d.sum()), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def main():
    p = argparse.ArgumentParser()
    for k in ("package", "annot", "primary", "secondary", "engine", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--groups")
    p.add_argument("--name", default="package")
    p.add_argument("--verdicts", action="store_true")
    a = p.parse_args()
    t0 = time.time()
    register(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = B4.Bio4("pkg", "union")
    D = B.D
    runs = D.runs
    Ls = {n: A.Loci(B4.CFG["pkg"]["out"][n]) for n in OUTS}
    mk = np.flatnonzero(B.member)
    base_brk = np.r_[True, (np.diff(mk) > 3) | (D.cc[mk[1:]] != D.cc[mk[:-1]])]
    TG = {"standing": [[(int(s), int(e), -1)] for s, e in zip(runs.gid_start, runs.gid_end)]}
    for v in ("part4", "a"):
        cs, ce, cid = B5.chains(B, mk, base_brk, None, v)
        pcs = B5.run_targets(B, mk, cs, ce, cid, runs)
        TG[v] = ([[q for q in ps if q[0] <= s <= q[1]] for ps, s in zip(pcs, runs.gid_start)] if v == "part4"
                 else [[q for q in ps if q[2] >= B5.MINMARK] for ps in pcs])
        if v == "a":
            TG["a_pieces"] = pcs
    acc = [B.span_units(int(s), int(e))[4] for s, e in zip(runs.gid_start, runs.gid_end)]
    rows, chance = [], {}
    for d in ("standing", "part4", "a"):
        rng = np.random.default_rng(17)
        ch = {n: {} for n in OUTS}
        for ri in range(len(runs)):
            tg = TG[d][ri]
            row = dict(run_id=int(runs.run_id.iat[ri]), definition=d, dom=runs.domain.iat[ri], gtype=runs.gtype.iat[ri],
                       closed=bool(runs.closed.iat[ri]), phylum=runs.phylum.iat[ri], n_targets=len(tg), kept=len(tg) > 0,
                       targets=";".join(f"{s}..{e}" for s, e, _ in tg))
            if d == "a":
                row["pieces_marked"] = ",".join(str(m) for _, _, m in TG["a_pieces"][ri])
            for n in OUTS:
                st = [A.standard(Ls[n], s, e) for s, e, _ in tg]
                row[f"{n}_recovered"] = bool(tg) and all(x["recovered"] for x in st)
                row[f"{n}_resolved"] = bool(tg) and all(x["resolved"] for x in st)
                row[f"{n}_anyfit"] = bool(tg) and all(Bm.any_fit(Ls[n], s, e) for s, e, _ in tg)
                if d == "standing":
                    row[f"{n}_accurate"] = bool(acc[ri]) and all(Bm.score_target(Ls[n], s, e)["ok"] for s, e in acc[ri])
            rows.append(row)
            for s, e, _ in sorted(tg):
                cf, cl = int(D.cfirst[s]), int(D.clast[s])
                ln = e - s + 1
                cand = [x for x in range(cf, cl - ln + 2) if x + ln - 1 < s or x > e]
                if cand:
                    for stt in rng.choice(cand, size=5, replace=True):
                        for n in OUTS:
                            c_ = ch[n].setdefault(runs.domain.iat[ri], [0, 0])
                            c_[0] += int(Bm.score_target(Ls[n], int(stt), int(stt) + ln - 1)["ok"])
                            c_[1] += 1
        chance[d] = {n: {dm: dict(resolved=v[0], windows=v[1], chance=v[0] / max(v[1], 1)) for dm, v in ch[n].items()} for n in OUTS}
    df = pd.DataFrame(rows)
    df.to_parquet(out / "score6_runs.parquet", index=False)
    res = dict(name=a.name, package=str(a.package), annot=str(a.annot), loci={n: str(B4.CFG["pkg"]["out"][n]) for n in OUTS},
               runs=int(len(runs)), by_domain=runs.domain.value_counts().to_dict(), definitions={}, bars={})
    for d in ("standing", "part4", "a"):
        x = df[df.definition == d]
        dd = {}
        for dm in ["all"] + sorted(x.dom.unique()):
            y = x if dm == "all" else x[x.dom == dm]
            e = dict(runs=int(len(y)), below_min=int((~y.kept).sum()), targets=int(y.n_targets.sum()))
            for n in OUTS:
                e[n] = dict(recovered=int(y[f"{n}_recovered"].sum()), resolved=int(y[f"{n}_resolved"].sum()),
                            anyfit=int(y[f"{n}_anyfit"].sum()),
                            chance=(chance[d][n].get(dm, {}).get("chance") if dm != "all" else None))
                if d == "standing":
                    e[n]["accurate_reading_resolved"] = int(y[f"{n}_accurate"].sum())
            dd[dm] = e
        res["definitions"][d] = dd
    # bars (definition a)
    x = df[df.definition == "a"]
    N = len(x)
    for n, role in (("Z", "primary"), ("WA", "secondary")):
        r1 = x[f"{n}_recovered"].mean()
        bac = x[x.dom == "Bacteria"]
        arc = x[x.dom == "Archaea"]
        r2 = bac[f"{n}_resolved"].mean() if len(bac) else None
        r3 = arc[f"{n}_resolved"].mean() if len(arc) else None
        dlt = x[f"{n}_resolved"].astype(int).to_numpy() - x["DH_resolved"].astype(int).to_numpy()
        tot, lo, hi = boot_sum(dlt)
        ch5 = {dm: chance["a"][n].get(dm, {}).get("chance") for dm in ("Bacteria", "Archaea") if dm in chance["a"][n]}
        bars = dict(R1=dict(value=float(r1), n=int(N), bar=0.97, pass_=bool(r1 >= 0.97)),
                    R2=dict(value=None if r2 is None else float(r2), n=int(len(bac)), bar=0.90,
                            pass_=None if r2 is None else bool(r2 >= 0.90)),
                    R3=dict(value=None if r3 is None else float(r3), n=int(len(arc)), bar=0.85,
                            pass_=None if r3 is None else bool(r3 >= 0.85)),
                    R4=dict(difference=tot, ci95=[lo, hi], resamples=2000, pass_=bool(lo > 0)),
                    R5=dict(chance=ch5, bar=0.05, pass_=bool(all(v <= 0.05 for v in ch5.values()))))
        bars["PASS"] = bool(all(b["pass_"] for k, b in bars.items() if k.startswith("R") and b["pass_"] is not None))
        res["bars"][role] = bars
    # descriptive: closed vs multi, phylum (definition a)
    res["closed_vs_multi"] = {k: {n: dict(runs=int(len(y)), resolved=int(y[f"{n}_resolved"].sum())) for n in OUTS}
                              for k, y in (("closed", x[(x.dom == "Bacteria") & x.closed]), ("multi", x[(x.dom == "Bacteria") & ~x.closed]))}
    ph = x.groupby(["dom", "phylum"]).agg(runs=("run_id", "size"), Z=("Z_resolved", "sum"), WA=("WA_resolved", "sum"),
                                         DH=("DH_resolved", "sum"), below=("kept", lambda s: int((~s).sum())))
    res["phylum"] = [dict(dom=k[0], phylum=k[1], **{c: int(v) for c, v in r.items()}) for k, r in ph.iterrows()]
    if a.verdicts:
        res["verdicts"] = verdicts(B, Ls, runs, x, out)
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out / "score6.json", "w"), indent=1, default=str)
    (out / "table6.txt").write_text(render(res) + "\n")
    print(render(res))


def verdicts(B, Ls, runs, x, out):
    D = B.D
    summ = {}
    for n in OUTS:
        cases = []
        for ri, r in x.reset_index(drop=True).iterrows():
            if r[f"{n}_resolved"]:
                continue
            if not r.kept:
                cases.append(dict(run_id=int(r.run_id), dom=r.dom, verdict="below_minimum", pieces_marked=r.pieces_marked))
                continue
            for tg in r.targets.split(";"):
                s, e = map(int, tg.split(".."))
                st = A.standard(Ls[n], s, e)
                if st["resolved"]:
                    continue
                mk_, comps, single, n_units, targets = B.span_units(s, e)
                L = Ls[n]
                decisive, kind = [], None
                if not targets:
                    v = 7
                else:
                    if n_units > 1:
                        uid = [int(B.comp_of[k]) if B.comp_of[k] >= 0 else -10 - k for k in mk_]
                        for x_, y_, ux, uy in zip(mk_[:-1], mk_[1:], uid[:-1], uid[1:]):
                            if ux != uy:
                                decisive.append(("glue", next((q for q in range(x_, y_) if not B.link[q]), x_)))
                        kind = "glue"
                    else:
                        ts, te = targets[0]
                        sc = Bm.score_target(L, ts, te)
                        if sc["ok"]:
                            kind = "ok"
                        elif Bm.any_fit(L, ts, te):
                            kind = "fit"
                        elif sc["bf"] < 0.8:
                            kind = "split"
                            if sc["a"] > ts:
                                decisive.append(("cut", sc["a"] - 1))
                            if 0 <= sc["b"] < te:
                                decisive.append(("cut", sc["b"]))
                            if sc["a"] < 0:
                                decisive.append(("cut", ts))
                        else:
                            kind = "ext"
                    dat = [q for (_, p) in decisive for q in (p - 1, p, p + 1, p + 2) if D.cfirst[s] <= q <= D.clast[s] and B.anom[q]]
                    v = 6 if (dat and kind in ("split", "glue")) else {"glue": 2, "ok": 3, "fit": 5, "split": 1, "ext": 4}[kind]
                cases.append(dict(run_id=int(r.run_id), dom=r.dom, span=f"{s}..{e}", verdict=v,
                                  junctions=" ; ".join(f"{kd}@{p - s} {Bm.gene_label(B, p)}|{Bm.gene_label(B, p + 1)} "
                                                       f"{'same' if D.strand[p] == D.strand[p + 1] else 'OPP'}" for kd, p in decisive)))
        cd = pd.DataFrame(cases)
        cd.to_csv(out / f"cases6_{n}.tsv", sep="\t", index=False)
        summ[n] = cd.groupby(["dom", "verdict"]).run_id.nunique().unstack(fill_value=0).to_dict("index") if len(cd) else {}
    return summ


def render(res):
    L = [f"FROZEN RIBOSOMAL BENCHMARK (predeclared6.json) on {res['name']}: {res['runs']} runs {res['by_domain']}"]
    for role in ("primary", "secondary"):
        b = res["bars"][role]
        L.append(f"  {role}: R1 recovered {b['R1']['value']:.1%} (>= 97%) {'PASS' if b['R1']['pass_'] else 'FAIL'}; "
                 f"R2 bacteria resolved {('%.1f%%' % (100 * b['R2']['value'])) if b['R2']['value'] is not None else 'n/a'} (>= 90%) "
                 f"{'PASS' if b['R2']['pass_'] else ('n/a' if b['R2']['pass_'] is None else 'FAIL')}; "
                 f"R3 archaea resolved {('%.1f%%' % (100 * b['R3']['value'])) if b['R3']['value'] is not None else 'n/a'} (>= 85%) "
                 f"{'PASS' if b['R3']['pass_'] else ('n/a' if b['R3']['pass_'] is None else 'FAIL')}; "
                 f"R4 minus engine {b['R4']['difference']:+.0f} [{b['R4']['ci95'][0]:+.0f}, {b['R4']['ci95'][1]:+.0f}] "
                 f"{'PASS' if b['R4']['pass_'] else 'FAIL'}; R5 chance " +
                 ", ".join(f"{k} {v:.2%}" for k, v in b['R5']['chance'].items()) + f" {'PASS' if b['R5']['pass_'] else 'FAIL'}"
                 f"  => {'PASS' if b['PASS'] else 'FAIL'}")
    L.append("  definition / domain: runs, below minimum; resolved Z / WA / DH; recovered Z; any-fit Z / WA / DH; chance Z / WA / DH")
    for d, dd in res["definitions"].items():
        for dm, e in dd.items():
            ch = " / ".join(("%.2f%%" % (100 * e[n]["chance"])) if e[n]["chance"] is not None else "-" for n in OUTS)
            acc = (" ; accurate reading Z / WA / DH " + " / ".join(str(e[n]["accurate_reading_resolved"]) for n in OUTS)) if d == "standing" else ""
            L.append(f"  {d:9s} {dm:9s} {e['runs']:5d} {e['below_min']:4d}; {e['Z']['resolved']} / {e['WA']['resolved']} / "
                     f"{e['DH']['resolved']}; {e['Z']['recovered']}; {e['Z']['anyfit']} / {e['WA']['anyfit']} / {e['DH']['anyfit']}; {ch}{acc}")
    L.append("  closed vs multi-contig bacterial genomes (definition a): " + json.dumps(res["closed_vs_multi"]))
    L.append("  phylum table (definition a): dom, phylum, runs, resolved Z / WA / DH, below minimum")
    for r in sorted(res["phylum"], key=lambda r: -r["runs"]):
        L.append(f"    {r['dom']:9s} {str(r['phylum']):28s} {r['runs']:5d} {r['Z']:5d} {r['WA']:5d} {r['DH']:5d} {r['below']:3d}")
    if "verdicts" in res:
        L.append("  verdicts for unresolved runs (definition a): " + json.dumps(res["verdicts"], default=str))
    return "\n".join(L)


if __name__ == "__main__":
    main()
