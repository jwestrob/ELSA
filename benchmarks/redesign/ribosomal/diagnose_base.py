#!/usr/bin/env python
"""Diagnosis of the ribosomal-locus check on one engine arm (default: the base arm).

Recomputes the arm's pair statistics with the engine's own functions (read-only import of
$RD/engine/code/recurrence.py), verifies them against the saved significant-pair count and
boundary support, then classifies every boundary inside the g2/m16 gold runs:
  break      boundary support b < 2 (no significant instance spans it; conserved@2 splits here)
  features   flank genes marked / unmarked, flank token genome frequency, flank genes that are an
             endpoint of no significant instance, the best spanning pair of any significance
             (max lineages L; smallest P among pairs with L >= 2)
Also: per ribosomal Pfam family (top marker), token splitting and break involvement; token
granularity of the ribosomal families in every DPANN token arm; three typical failures and three
successes printed gene by gene.
Outputs under $RD/ribosomal/diag/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
GOLD = RD / "ribosomal/gold"
OUT = RD / "ribosomal/diag"
BASE = RD / "engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def token_path(arm):
    """Engine arm name (summary.json 'arm') -> token parquet."""
    x = arm[len("dpann__"):] if arm.startswith("dpann__") else arm
    for p in (RD / "engine/dpann_tokens" / f"{x}__dpann.parquet", RD / "engine/dpann_tokens" / f"{x}.parquet",
              PKG / "tokens" / f"{x}__dpann.parquet", RD / "tokens" / f"{x}.parquet"):
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", type=Path, default=BASE)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--arms-file", type=Path, default=RD / "ribosomal/arms.txt")
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    sys.path.insert(0, str(RD / "engine/code"))
    import recurrence as R
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summ = json.load(open(a.result_dir / "summary.json"))
    w = int(summ["w"])
    u = R.Universe(PKG / "universe")
    assert np.array_equal(u.gid, np.arange(u.n)), "universe order != gid order"
    tp = token_path(summ["arm"])
    tokens = pd.read_parquet(tp)
    tok = np.full(u.n, -1, dtype=np.int64)
    tok[tokens.gid.to_numpy()] = tokens.token.to_numpy()
    key, left, dist, T = R.colocalization(u, tok, w)
    st = R.pair_statistics(u, tok, key, left, w, T)
    mask = R.RULES[summ["rule"]](st)
    rep = dict(arm=summ["arm"], tokens_file=str(tp), w=w, rule=summ["rule"],
               significant_pairs_recomputed=int(mask.sum()), significant_pairs_summary=summ["significant_pairs"])
    assert rep["significant_pairs_recomputed"] == rep["significant_pairs_summary"], rep
    idx = np.searchsorted(st["key"], key)
    sig = mask[idx]
    Lr = st["L"][idx].astype(np.float32)
    b_saved = np.load(a.result_dir / "boundary_support.npy")
    b_sig, _ = R.boundary_support_argmax(left[sig], dist[sig], Lr[sig], np.arange(int(sig.sum())), u.n)
    rep["boundary_support_reproduced"] = bool(np.array_equal(b_sig, b_saved))
    b = b_saved
    b_L_any, _ = R.boundary_support_argmax(left, dist, Lr, np.arange(len(left)), u.n)
    l2 = Lr >= 2
    mlogp = (-st["logP"][idx]).astype(np.float32)
    b_mlogp_L2, _ = R.boundary_support_argmax(left[l2], dist[l2], mlogp[l2], np.arange(int(l2.sum())), u.n)
    sig_deg = np.bincount(left[sig], minlength=u.n) + np.bincount(left[sig] + dist[sig], minlength=u.n)
    tg = np.unique(tok[tok >= 0] * u.ng + u.gc[tok >= 0])
    F = np.bincount(tg // u.ng, minlength=T)
    gene_F = np.where(tok >= 0, F[np.maximum(tok, 0)], 0)
    tm = pd.read_parquet(a.result_dir / "token_modules.parquet")
    token_module = np.full(T, -1, dtype=np.int64)
    token_module[tm.token.to_numpy()] = tm.module.to_numpy()
    gene_mod = np.where(tok >= 0, token_module[np.maximum(tok, 0)], -1)
    log("statistics", rep, f"{time.time()-t0:.0f}s")
    rep["genome_wide"] = dict(genes=int(u.n), genes_sig_endpoint_frac=float((sig_deg > 0).mean()),
                              boundaries_supported_frac=float((b >= 2).mean()))

    marked = pd.read_parquet(GOLD / "marked.parquet")
    mk = np.zeros(u.n, dtype=bool)
    mk[marked.gid.to_numpy()] = True
    marker = np.full(u.n, "-", dtype=object)
    marker[marked.gid.to_numpy()] = marked.marker.to_numpy()
    runs = pd.read_parquet(GOLD / "runs.parquet")
    runs = runs[(runs.run_set == "g2") & (runs.n_marked >= 16)].reset_index(drop=True)

    # ---- boundaries inside runs ----------------------------------------------------------
    rows = []
    for r in runs.itertuples():
        for k in range(r.gid_start, r.gid_end):
            rows.append((r.run_id, k))
    bd = pd.DataFrame(rows, columns=["run_id", "k"])
    k = bd.k.to_numpy()
    bd["b"] = b[k]
    bd["brk"] = b[k] < 2
    bd["left_marked"], bd["right_marked"] = mk[k], mk[k + 1]
    bd["left_marker"], bd["right_marker"] = marker[k], marker[k + 1]
    bd["flank_unmarked"] = ~(mk[k] & mk[k + 1])
    bd["flank_min_F"] = np.minimum(gene_F[k], gene_F[k + 1])
    bd["flank_singleton"] = bd.flank_min_F <= 1
    bd["flank_no_sig"] = (sig_deg[k] == 0) | (sig_deg[k + 1] == 0)
    bd["best_L_any"] = b_L_any[k]
    bd["best_mlogP_L2"] = b_mlogp_L2[k]
    bd["same_token"] = (tok[k] == tok[k + 1]) & (tok[k] >= 0)

    def cause(x):
        if x.flank_unmarked:
            return "A_unmarked_gene_at_break"
        if x.flank_singleton:
            return "B_flank_token_in_1_genome"
        if x.flank_no_sig:
            return "C_flank_gene_no_significant_pair"
        if x.best_L_any >= 2:
            return "D_spanning_pair_L>=2_but_P_fails"
        return "E_spanning_pairs_all_L<2"
    bd["cause"] = [cause(x) for x in bd.itertuples()]
    br, nb = bd[bd.brk], bd[~bd.brk]
    feats = ["flank_unmarked", "flank_singleton", "flank_no_sig", "same_token"]
    rep["boundaries"] = dict(
        total=int(len(bd)), breaks=int(len(br)), runs_with_break=int(br.run_id.nunique()), runs=int(len(runs)),
        feature_rate_break={f: float(br[f].mean()) for f in feats},
        feature_rate_nonbreak={f: float(nb[f].mean()) for f in feats},
        best_L_any_at_break=br.best_L_any.value_counts().sort_index().head(12).to_dict(),
        cause_at_break=br.cause.value_counts().to_dict(),
        cause_at_nonbreak=nb.cause.value_counts().to_dict(),
        flank_min_F_median_break=float(br.flank_min_F.median()), flank_min_F_median_nonbreak=float(nb.flank_min_F.median()))
    # breaks among marked-marked boundaries only
    mm = bd[~bd.flank_unmarked]
    rep["boundaries"]["marked_marked"] = dict(total=int(len(mm)), breaks=int(mm.brk.sum()),
                                              break_rate=float(mm.brk.mean()))
    um = bd[bd.flank_unmarked]
    rep["boundaries"]["with_unmarked_flank"] = dict(total=int(len(um)), breaks=int(um.brk.sum()),
                                                    break_rate=float(um.brk.mean()) if len(um) else None)
    # families bordering breaks (marked-marked breaks), per family incidence
    fam = []
    for side in ("left_marker", "right_marker"):
        fam.append(bd[[side, "brk"]].rename(columns={side: "family"}))
    fam = pd.concat(fam)
    fam = fam[fam.family != "-"]
    ft = fam.groupby("family").brk.agg(["size", "sum", "mean"]).rename(
        columns={"size": "flank_boundaries", "sum": "breaks", "mean": "break_rate"}).sort_values("breaks", ascending=False)
    # per-family token splitting among run genes and genome-wide marked genes
    run_genes = np.concatenate([np.arange(r.gid_start, r.gid_end + 1) for r in runs.itertuples()])
    rg = pd.DataFrame(dict(gid=run_genes, family=marker[run_genes], token=tok[run_genes], F=gene_F[run_genes],
                           sig_deg=sig_deg[run_genes], module=gene_mod[run_genes], genome=u.gc[run_genes]))
    rgm = rg[rg.family != "-"]
    tsplit = rgm.groupby("family").agg(genes=("gid", "size"), genomes=("genome", "nunique"), tokens=("token", "nunique"),
                                       top_token_share=("token", lambda s: s.value_counts().iloc[0] / len(s)),
                                       median_token_genomes=("F", "median"),
                                       no_sig_frac=("sig_deg", lambda s: float((s == 0).mean())),
                                       modules=("module", lambda s: s[s >= 0].nunique()))
    ft = ft.join(tsplit, how="outer").sort_values("breaks", ascending=False)
    ft.to_csv(OUT / "family_table.tsv", sep="\t")
    rep["families_top_breaks"] = ft.head(15).reset_index().to_dict("records")
    rep["run_genes"] = dict(genes=int(len(rg)), marked=int(len(rgm)),
                            sig_endpoint_frac=float((rg.sig_deg > 0).mean()),
                            sig_endpoint_frac_marked=float((rgm.sig_deg > 0).mean()),
                            sig_endpoint_frac_unmarked=float((rg[rg.family == "-"].sig_deg > 0).mean()),
                            gene_modules_per_run_median=float(rg.assign(run=np.repeat(runs.run_id.to_numpy(), runs.n_genes.to_numpy()))
                                                              .query("module >= 0").groupby("run").module.nunique().median()))
    vc = rgm[rgm.module >= 0].module.value_counts()
    rep["run_genes"]["marked_gene_top_modules"] = [[int(m), int(c)] for m, c in vc.head(8).items()]
    rep["run_genes"]["marked_gene_module_missing_frac"] = float((rgm.module < 0).mean())
    rep["run_genes"]["marked_gene_top_module_share"] = float(vc.iloc[0] / len(rgm)) if len(vc) else None
    fam_mod = rgm[rgm.module >= 0].groupby("family").module.agg(lambda s: int(s.value_counts().index[0]))
    ft["top_module"] = fam_mod.reindex(ft.index)
    ft.to_csv(OUT / "family_table.tsv", sep="\t")
    rep["module_of_family"] = {str(m): sorted(fam_mod[fam_mod == m].index.tolist()) for m in fam_mod.unique()}

    # ---- observed Pfam domains of unmarked genes inside runs and of the conserved@2 extensions ----
    import duckdb
    genes_pid = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "protein_id"]).sort_values("gid")
    pid = genes_pid.protein_id.to_numpy()
    pr_path0 = RD / "ribosomal/arms" / f"{a.result_dir.parent.name}__{a.result_dir.name}" / "ribosomal_runs.parquet"
    ext_rows = []
    if pr_path0.exists():
        pr0 = runs.merge(pd.read_parquet(pr_path0), on="run_id")
        for rid, gs, ge, el, er in zip(pr0.run_id, pr0.gid_start, pr0.gid_end,
                                       pr0["conserved@2:ext_left"].clip(lower=0), pr0["conserved@2:ext_right"].clip(lower=0)):
            for j, g in enumerate(range(gs - el, gs)):
                ext_rows.append((rid, g, "left", el - j))
            for j, g in enumerate(range(ge + 1, ge + er + 1)):
                ext_rows.append((rid, g, "right", j + 1))
    ext = pd.DataFrame(ext_rows, columns=["run_id", "gid", "side", "dist"])
    gap_g = rg[rg.family == "-"].gid.to_numpy()
    q = pd.DataFrame({"protein_id": pid[np.unique(np.r_[gap_g, ext.gid.to_numpy()]).astype(np.int64)]})
    con = duckdb.connect(str(RD / "ws_dpann/input/sharur.duckdb"), read_only=True)
    con.execute(f"SET threads={min(a.threads, 8)}")
    con.register("q", q)
    ann = con.execute("SELECT a.protein_id, a.name FROM annotations a JOIN q USING (protein_id) "
                      "WHERE a.source = 'pfam'").df()
    con.close()
    p2g = pd.Series(np.arange(len(pid)), index=pid)
    ann["gid"] = p2g.reindex(ann.protein_id.to_numpy()).to_numpy()
    names = ann.groupby("gid").name.apply(lambda s: ";".join(sorted(set(s))))

    def tally(gids, label):
        gids = np.asarray(gids)
        nm = names.reindex(gids)
        no_pfam = int(nm.isna().sum())
        vc2 = pd.Series([x for v in nm.dropna() for x in v.split(";")]).value_counts()
        return dict(genes=int(len(gids)), no_pfam=no_pfam, top_pfam=[[k2, int(v2)] for k2, v2 in vc2.head(15).items()])
    rep["observed_domains"] = dict(
        note="Pfam names as observed on the genes; positional description only, no system call",
        gap_genes_inside_runs=tally(gap_g, "gap"),
        extension_left_adjacent=tally(ext[(ext.side == "left") & (ext.dist == 1)].gid, "l1"),
        extension_right_adjacent=tally(ext[(ext.side == "right") & (ext.dist == 1)].gid, "r1"),
        extension_all=tally(ext.gid, "all"),
        extension_genes_per_run=dict(runs_with_extension=int(ext.run_id.nunique()), genes=int(len(ext))))
    log("boundaries", rep["boundaries"]["breaks"], "breaks of", rep["boundaries"]["total"])

    # ---- token granularity of the ribosomal families across token arms ----------------------
    gran = []
    arms = [l.strip() for l in open(a.arms_file) if l.strip() and not l.startswith("#")] if a.arms_file.exists() else []
    seen = set()
    for d in arms:
        try:
            arm = json.load(open(Path(d) / "summary.json"))["arm"]
        except Exception:
            continue
        p = token_path(arm)
        if p is None or str(p) in seen:
            continue
        seen.add(str(p))
        t = pd.read_parquet(p)
        tk = np.full(u.n, -1, dtype=np.int64)
        tk[t.gid.to_numpy()] = t.token.to_numpy()
        x = pd.DataFrame(dict(family=marker[run_genes], token=tk[run_genes], genome=u.gc[run_genes]))
        x = x[x.family != "-"]
        g = x.groupby("family").agg(tokens=("token", "nunique"),
                                    top=("token", lambda s: s.value_counts().iloc[0] / len(s)))
        tg2 = np.unique(tk[tk >= 0] * u.ng + u.gc[tk >= 0])
        F2 = np.bincount(tg2 // u.ng, minlength=int(tk.max()) + 1)
        gran.append(dict(token_arm=p.stem, tokens_total=int(len(np.unique(tk[tk >= 0]))),
                         ribo_families=int(len(g)), median_tokens_per_family=float(g.tokens.median()),
                         mean_tokens_per_family=float(g.tokens.mean()),
                         median_top_token_share=float(g.top.median()),
                         marked_gene_token_genomes_median=float(np.median(
                             F2[np.maximum(tk[run_genes[x.index.to_numpy()]], 0)]))))
    if gran:
        pd.DataFrame(gran).to_csv(OUT / "granularity.tsv", sep="\t", index=False)
        rep["granularity"] = gran
    log("granularity", len(gran))

    # ---- examples ------------------------------------------------------------------------
    pr_path = RD / "ribosomal/arms" / f"{a.result_dir.parent.name}__{a.result_dir.name}" / "ribosomal_runs.parquet"
    ex_lines = []
    if pr_path.exists():
        pr = pd.read_parquet(pr_path)
        pr = runs.merge(pr, on="run_id")
        c = "conserved@2:"
        fail = pr[~pr[c + "resolved"]]
        ok = pr[pr[c + "resolved"]]
        rng = np.random.default_rng(17)

        def pick(df, nfrag=None):
            if nfrag is not None:
                df2 = df[df[c + "fragments"] == nfrag]
                df = df2 if len(df2) >= 3 else df
            out = []
            for ph in df.phylum.value_counts().index:
                d = df[df.phylum == ph]
                out.append(d.iloc[rng.integers(len(d))])
                if len(out) == 3:
                    break
            return out
        med = int(fail[c + "fragments"].median()) if len(fail) else None
        loci = pd.read_parquet(a.result_dir / "loci_all_genomes.parquet")
        for label, sel in (("FAILURE (typical: fragments = median of unresolved runs)", pick(fail, med)),
                           ("SUCCESS", pick(ok))):
            for r in sel:
                cs = int(r.contig_first_gid)
                lo, hi = max(cs, r.gid_start - 3), min(cs + r.contig_n_genes - 1, r.gid_end + 3)
                ll = loci[(loci.genome == r.genome) & (loci.contig == r.contig) & (loci.end >= lo - cs) & (loci.start <= hi - cs)]
                ex_lines.append(f"== {label}: run {r.run_id} {r.genome} {r.phylum}/{r.family}/{r.genus} contig {r.contig} "
                                f"pos {r.pos_start}-{r.pos_end} ({r.n_genes} genes, {r.n_marked} marked), contig {r.contig_n_genes} genes; "
                                f"conserved@2 fragments={r[c+'fragments']} best_frac={r[c+'best_frac']:.2f} "
                                f"precision={r[c+'precision']:.2f} ext={r[c+'ext_left']}/{r[c+'ext_right']} module={r[c+'module']}; "
                                f"unit@0.5 best_frac={r['unit@0.5:best_frac']:.2f} prec={r['unit@0.5:precision']:.2f}")
                for kind, lev in (("conserved", 2.0), ("conserved_g2", 2.0), ("unit", 0.5)):
                    q = ll[(ll.kind == kind) & (ll.level == lev)]
                    ex_lines.append(f"   {kind}@{lev:g} loci (pos, module): " +
                                    ", ".join(f"{s}-{e}(m{m})" for s, e, m in zip(q.start, q.end, q.module_id)))
                ex_lines.append("   pos  in_run  marker               token    tokGenomes module sigDeg  b(next)  bestL_any(next)")
                for g in range(lo, hi + 1):
                    inrun = "*" if r.gid_start <= g <= r.gid_end else " "
                    nb_ = f"{b[g]:7.0f}" if g < hi else "       "
                    nl_ = f"{b_L_any[g]:6.0f}" if g < hi else ""
                    ex_lines.append(f"   {g - cs:4d}  {inrun:^6}  {marker[g]:<20} {tok[g]:<8d} {gene_F[g]:10d} {gene_mod[g]:6d} {sig_deg[g]:6d} {nb_}  {nl_}")
        with open(OUT / "examples.txt", "w") as fh:
            fh.write("\n".join(ex_lines) + "\n")
    rep["seconds"] = round(time.time() - t0, 1)
    with open(OUT / "diagnosis.json", "w") as fh:
        json.dump(rep, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    bd.to_parquet(OUT / "boundaries.parquet", index=False)
    log("done", rep["seconds"])


if __name__ == "__main__":
    main()
