#!/usr/bin/env python
"""Signal 2, tests 2-4 of ws_signals: per-gene foreignness, coherence of adjacent residual directions, boundary signal.

Inputs from s2_classify.py: s2_gene.parquet (f = -log2 p(host fam_jac group | r_i), cross-fitted by family half) and
s2_v_famjac_f16.npy (centred log-probabilities; rows = eligible genes in gid order).
Host-relative direction u_i = v_i - mean of v over the genome's other eligible genes; coherence c_ij = cos(u_i, u_j).
Writes $RD/signals/work/s2_tests.json and small tables s2_*.tsv. Even genomes are the primary reporting set.
"""
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RD, PKG, W, ENG, DB, SEED, BLK, log, load_genes, load_tokens, load_groups, auc_w, boot_weights, ci  # noqa: E402

G = 3564
TAG = os.environ.get("SIG_TAG", "")
TNP = r"(?i)(transpos|DDE_Tnp|^rve|Phage_integrase|Recombinase|Resolvase|HTH_Tnp|Y1_Tnp|OrfB_|IS605)"


def gsum(gc, x, minlength=G):
    return np.bincount(gc, weights=x, minlength=minlength)


def boot_mean_diff(gc_a, xa, gc_b, xb, Wb):
    """difference of means (a - b) with genome bootstrap."""
    sa, na = gsum(gc_a, xa), gsum(gc_a, np.ones(len(xa)))
    sb, nb = gsum(gc_b, xb), gsum(gc_b, np.ones(len(xb)))
    d = sa.sum() / na.sum() - sb.sum() / nb.sum()
    db = (Wb @ sa) / (Wb @ na) - (Wb @ sb) / (Wb @ nb)
    return float(d), ci(db)


def stratified_diff(gc_all, strat, f, in_set, Wb, S):
    """sum_s n_set,s (mean_set,s - mean_all,s) / n_set with genome bootstrap; strata S."""
    idx = gc_all * S + strat
    L = G * S
    bs = np.bincount(idx, weights=f, minlength=L).reshape(G, S)
    bn = np.bincount(idx, minlength=L).reshape(G, S).astype(np.float64)
    ss = np.bincount(idx[in_set], weights=f[in_set], minlength=L).reshape(G, S)
    sn = np.bincount(idx[in_set], minlength=L).reshape(G, S).astype(np.float64)

    def stat(bs_, bn_, ss_, sn_):
        mb = bs_ / np.maximum(bn_, 1e-12)
        ms = ss_ / np.maximum(sn_, 1e-12)
        ok = (sn_ > 0) & (bn_ > 0)
        return (sn_ * np.where(ok, ms - mb, 0)).sum(-1) / np.maximum((sn_ * ok).sum(-1), 1e-12)
    point = float(stat(bs.sum(0), bn.sum(0), ss.sum(0), sn.sum(0)))
    boot = stat(Wb @ bs, Wb @ bn, Wb @ ss, Wb @ sn)
    return point, ci(boot)


def main():
    t0 = time.time()
    threads = int(os.environ.get("SIG_THREADS", "1"))
    g = load_genes()
    tok = load_tokens()
    gc = g.genome_code.to_numpy(np.int64)
    cc = g.contig_code.to_numpy(np.int64)
    st = g.strand.to_numpy(np.int64)
    n = len(g)
    gs = pd.read_parquet(W / "gene_stats.parquet")
    elig = gs.eligible.to_numpy(bool)
    ge = np.flatnonzero(elig)
    row = np.full(n, -1, np.int64)
    row[ge] = np.arange(len(ge))
    s2 = pd.read_parquet(W / f"s2_gene{TAG}.parquet")
    assert (s2.gid.to_numpy() == ge).all()
    f = np.full(n, np.nan)
    f[ge] = s2.f.to_numpy()
    V = np.load(W / f"s2_v_famjac_f16{TAG}.npy").astype(np.float32)
    log("loaded", V.shape, f"{time.time() - t0:.0f}s")
    out = {}
    Wb_all = boot_weights(G)

    # ---------------------------------------------------------------- gene sets
    inst = pd.read_parquet(PKG / "gold/instances.parquet", columns=["instance_id", "caller", "system_type", "n_members_resolved"])
    mem = pd.read_parquet(PKG / "gold/members.parquet")
    sets = {}
    for c in ("DefenseFinder", "CRISPRCasTyper", "TXSScan"):
        sets[c] = np.unique(mem[mem.caller == c].gid.to_numpy(np.int64))
    con = duckdb.connect(str(DB), read_only=True, config={"threads": threads})
    tp = con.execute(f"SELECT DISTINCT protein_id FROM annotations WHERE source = 'pfam' AND regexp_matches(name, '{TNP}')").df()
    con.close()
    pid = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "protein_id"])
    p2g = pd.Series(pid.gid.to_numpy(), index=pid.protein_id.to_numpy())
    sets["transposase_domain"] = np.unique(p2g.reindex(tp.protein_id).dropna().astype(np.int64).to_numpy())
    rib = pd.read_parquet(RD / "ribosomal/gold/marked.parquet", columns=["gid"])
    sets["ribosomal_marker"] = np.unique(rib.gid.to_numpy(np.int64))
    out["set_sizes"] = {k: dict(genes=int(len(v)), eligible=int(elig[v].sum())) for k, v in sets.items()}
    log("sets", out["set_sizes"])

    # ---------------------------------------------------------------- test 2: foreignness
    m = gs.fam_m.to_numpy()
    famjac = load_groups("fam_jac_average_t0.2")
    tg = pd.DataFrame(dict(t=tok, grp=famjac[gc])).drop_duplicates()
    breadth = tg.groupby("t").size().reindex(np.arange(int(tok.max()) + 1)).fillna(0).to_numpy()[tok]
    mb = np.digitize(m, [6, 11, 31, 101])
    bb = np.digitize(breadth, [2, 4, 11])
    rn = gs.rnorm.to_numpy()
    dec = np.full(n, 0)
    dec[ge] = np.minimum((pd.Series(rn[ge]).rank(pct=True).to_numpy() * 10).astype(int), 9)
    strat = (mb * 4 + bb) * 10 + dec
    gmf = np.bincount(gc[ge], weights=f[ge], minlength=G) / np.maximum(np.bincount(gc[ge], minlength=G), 1)
    fc = f - gmf[gc]
    S = 200
    t2 = {}
    for scope in ("even", "all"):
        sc = elig & ((gc % 2 == 0) if scope == "even" else True)
        idx = np.flatnonzero(sc)
        res = {}
        for k, v in sets.items():
            ins = np.zeros(n, bool)
            ins[v] = True
            ins_s = ins[idx]
            raw, raw_ci = boot_mean_diff(gc[idx][ins_s], f[idx][ins_s], gc[idx], f[idx], Wb_all)
            sd, sd_ci = stratified_diff(gc[idx], strat[idx], f[idx], ins_s, Wb_all, S)
            # post hoc addition: genome-centred foreignness fc = f - mean f of the genome's eligible genes (removes
            # the host group's prior and any genome-level offset); same comparisons
            rawc, rawc_ci = boot_mean_diff(gc[idx][ins_s], fc[idx][ins_s], gc[idx], fc[idx], Wb_all)
            sdc, sdc_ci = stratified_diff(gc[idx], strat[idx], fc[idx], ins_s, Wb_all, S)
            res[k] = dict(n=int(ins_s.sum()), mean_f=float(f[idx][ins_s].mean()), raw_diff=raw, raw_ci=raw_ci,
                          strat_diff=sd, strat_ci=sd_ci, auroc_vs_all=auc_w(f[idx], ins_s),
                          gcentred_diff=rawc, gcentred_ci=rawc_ci, gcentred_strat_diff=sdc, gcentred_strat_ci=sdc_ci,
                          auroc_gcentred=auc_w(fc[idx], ins_s))
        res["_background_mean_f"] = float(f[idx].mean())
        t2[scope] = res
        log("test2", scope, json.dumps(res))
    out["test2_foreignness"] = t2
    # per system type (DefenseFinder, even genomes): raw and stratified
    ev = elig & (gc % 2 == 0)
    idx = np.flatnonzero(ev)
    dfm = mem[mem.caller == "DefenseFinder"].merge(inst[["instance_id", "system_type"]], on="instance_id")
    rows = []
    for stype, d in dfm.groupby("system_type"):
        ins = np.zeros(n, bool)
        ins[d.gid.to_numpy()] = True
        k = ins[idx]
        if k.sum() < 30:
            continue
        sd, sdci = stratified_diff(gc[idx], strat[idx], f[idx], k, Wb_all, S)
        rows.append(dict(system_type=stype, n=int(k.sum()), mean_f=float(f[idx][k].mean()), strat_diff=sd, lo=sdci[0], hi=sdci[1]))
    pd.DataFrame(rows).sort_values("strat_diff").to_csv(W / f"s2_foreign_by_dftype{TAG}.tsv", sep="\t", index=False)

    # ---------------------------------------------------------------- test 3: coherence
    import scipy.sparse as sps
    Sg = np.asarray(sps.csr_matrix((np.ones(len(ge)), (gc[ge], np.arange(len(ge)))), shape=(G, len(ge))) @ V, np.float64)
    ng = np.bincount(gc[ge], minlength=G).astype(np.float64)

    def u_of(gids):
        r = row[gids]
        v = V[r].astype(np.float64)
        return v - (Sg[gc[gids]] - v) / np.maximum(ng[gc[gids]] - 1, 1)[:, None]

    def coh(a, b):
        outc = np.empty(len(a))
        for s in range(0, len(a), BLK):
            ua, ub = u_of(a[s:s + BLK]), u_of(b[s:s + BLK])
            outc[s:s + BLK] = (ua * ub).sum(1) / np.maximum(np.linalg.norm(ua, axis=1) * np.linalg.norm(ub, axis=1), 1e-12)
        return outc
    rng = np.random.default_rng(SEED)
    decay = {}
    pairs_off1 = None
    for off in (1, 2, 3, 5, 10, 20, 50):
        i = np.arange(n - off)
        j = i + off
        ok = (cc[i] == cc[j]) & elig[i] & elig[j]
        i, j = i[ok], j[ok]
        if off == 1:
            pairs_off1 = (i.copy(), j.copy())
        if len(i) > 1_500_000:
            s = np.sort(rng.choice(len(i), 1_500_000, replace=False))
            i, j = i[s], j[s]
        c = coh(i, j)
        decay[off] = (i, j, c)
    # different contigs, same genome
    gi = rng.choice(ge, 3_000_000)
    order = np.argsort(gc[ge], kind="mergesort")
    ge_by_g = ge[order]
    gstart = np.searchsorted(gc[ge_by_g], np.arange(G))
    gcount = np.bincount(gc[ge], minlength=G)
    gj = ge_by_g[gstart[gc[gi]] + (rng.random(len(gi)) * gcount[gc[gi]]).astype(np.int64)]
    ok = cc[gi] != cc[gj]
    gi, gj = gi[ok][:1_000_000], gj[ok][:1_000_000]
    cdiff = coh(gi, gj)
    t3 = {}
    for scope in ("even", "all"):
        keep = (lambda a: gc[a] % 2 == 0) if scope == "even" else (lambda a: np.ones(len(a), bool))
        rr = {}
        kd = keep(gi)
        rr["diff_contig_mean"] = float(cdiff[kd].mean())
        for off, (i, j, c) in decay.items():
            k = keep(i)
            d, dci = boot_mean_diff(gc[i][k], c[k], gc[gi][kd], cdiff[kd], Wb_all)
            sm = st[i] == st[j]
            rr[f"off{off}"] = dict(n=int(k.sum()), mean=float(c[k].mean()), minus_diff_contig=d, ci=dci,
                                   same_strand=float(c[k & sm].mean()), opp_strand=float(c[k & ~sm].mean()),
                                   frac_same=float(sm[k].mean()))      # strand split: post hoc
        i, j, c = decay[1]
        k = keep(i)
        same = st[i] == st[j]
        rr["off1_same_strand"] = float(c[k & same].mean())
        rr["off1_opp_strand"] = float(c[k & ~same].mean())
        t3[scope] = rr
    log("test3 decay", json.dumps(t3))

    # loci: adjacent member pairs of one instance vs all offset-1 pairs with the same strand configuration
    i1, j1 = pairs_off1
    c1 = coh(i1, j1)
    same1 = st[i1] == st[j1]
    pair_inst = mem.merge(mem, on=["instance_id", "caller"])
    pair_inst = pair_inst[pair_inst.gid_y == pair_inst.gid_x + 1]
    key1 = i1
    loci = {}
    for c in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
        lp = np.unique(pair_inst[pair_inst.caller == c].gid_x.to_numpy(np.int64))
        isl = np.isin(key1, lp)
        for scope in ("even", "all"):
            ks = (gc[i1] % 2 == 0) if scope == "even" else np.ones(len(i1), bool)
            # strand-matched background: weight background same/opp classes by the locus pairs' strand mix
            rr = {}
            a = isl & ks
            if a.sum() == 0:
                continue
            ws = np.where(same1, (a & same1).sum() / max((ks & same1).sum(), 1), (a & ~same1).sum() / max((ks & ~same1).sum(), 1))
            sa, na = gsum(gc[i1][a], c1[a]), gsum(gc[i1][a], np.ones(a.sum()))
            sb, nb = gsum(gc[i1][ks], c1[ks] * ws[ks]), gsum(gc[i1][ks], ws[ks])
            d = sa.sum() / na.sum() - sb.sum() / nb.sum()
            db = (Wb_all @ sa) / (Wb_all @ na) - (Wb_all @ sb) / (Wb_all @ nb)
            rr = dict(n_pairs=int(a.sum()), mean_c=float(c1[a].mean()), bg_strand_matched=float(sb.sum() / nb.sum()),
                      diff=float(d), ci=ci(db), frac_same_strand=float(same1[a].mean()))
            loci[f"{c}_{scope}"] = rr
    out["test3_coherence"] = dict(decay=t3, loci=loci)
    log("test3 loci", json.dumps(loci))

    # ---------------------------------------------------------------- test 4: boundaries
    b_sup = np.load(ENG / "boundary_support.npy")
    member_any = np.zeros(n, bool)
    member_any[mem.gid.to_numpy()] = True
    mi = mem.merge(inst[["instance_id", "caller", "n_members_resolved"]], on=["instance_id", "caller"])
    t4 = {}
    rows_b = []
    for c in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
        d = mi[(mi.caller == c) & (mi.n_members_resolved >= 2)]
        edges, inter = [], []
        for iid, dd in d.groupby("instance_id"):
            gg = np.sort(dd.gid.to_numpy())
            if cc[gg[0]] != cc[gg[-1]] or gg[-1] - gg[0] + 1 != len(gg):
                continue                                   # contiguous single-contig instances only
            if c == "TXSScan" and len(gg) > 30:
                continue
            lft, rgt = gg[0] - 1, gg[-1] + 1
            if lft >= 0 and cc[lft] == cc[gg[0]] and not member_any[lft]:
                edges.append((lft, gg[0]))
            if rgt < n and cc[rgt] == cc[gg[-1]] and not member_any[rgt]:
                edges.append((gg[-1], rgt))
            for x in gg[:-1]:
                inter.append((x, x + 1))
        E_ = np.array(edges, dtype=np.int64).reshape(-1, 2)
        I_ = np.array(inter, dtype=np.int64).reshape(-1, 2)
        rnd_ok = ~member_any[i1] & ~member_any[j1]
        R_ = np.column_stack([i1[rnd_ok], j1[rnd_ok]])
        R_ = R_[rng.choice(len(R_), min(300_000, len(R_)), replace=False)]
        frames = []
        for kind, P in (("edge", E_), ("interior", I_), ("random", R_)):
            P = P[elig[P[:, 0]] & elig[P[:, 1]]]
            frames.append(pd.DataFrame(dict(kind=kind, a=P[:, 0], b=P[:, 1])))
        D = pd.concat(frames, ignore_index=True)
        D["c"] = coh(D.a.to_numpy(), D.b.to_numpy())
        D["support"] = b_sup[D.a.to_numpy()]
        D["gc"] = gc[D.a.to_numpy()]
        D["same"] = st[D.a.to_numpy()] == st[D.b.to_numpy()]
        D["caller"] = c
        rows_b.append(D)
        rr = {}
        for scope in ("even", "odd"):
            Ds = D[(D.gc % 2 == 0) if scope == "even" else (D.gc % 2 == 1)]
            for neg in ("interior", "random"):
                s_ = Ds[Ds.kind.isin(["edge", neg])]
                y = (s_.kind == "edge").to_numpy()
                rr[f"{scope}_edge_vs_{neg}"] = dict(n_edge=int(y.sum()), n_neg=int((~y).sum()), auroc_neg_c=auc_w(-s_.c.to_numpy(), y),
                                                    auroc_neg_support=auc_w(-s_.support.to_numpy(), y))
        # post hoc: strand composition of each boundary kind and the AUROC among same-strand boundaries only
        for scope in ("even",):
            Ds = D[D.gc % 2 == 0]
            rr["frac_same_strand"] = {k_: float(Ds[Ds.kind == k_].same.mean()) for k_ in ("edge", "interior", "random")}
            s_ = Ds[Ds.kind.isin(["edge", "interior"]) & Ds.same]
            y = (s_.kind == "edge").to_numpy()
            rr["even_edge_vs_interior_same_strand"] = dict(n_edge=int(y.sum()), n_neg=int((~y).sum()),
                                                           auroc_neg_c=auc_w(-s_.c.to_numpy(), y),
                                                           auroc_neg_support=auc_w(-s_.support.to_numpy(), y))
        # genome bootstrap on even genomes for AUROC(-c) edge vs interior and the support combination
        tr = D[(D.gc % 2 == 1) & D.kind.isin(["edge", "interior"])]
        te = D[(D.gc % 2 == 0) & D.kind.isin(["edge", "interior"])].reset_index(drop=True)
        if len(tr) > 20 and len(te) > 20 and tr.kind.nunique() == 2 and te.kind.nunique() == 2:
            Xtr = np.column_stack([np.log1p(tr.support), tr.c])
            Xte = np.column_stack([np.log1p(te.support), te.c])
            ytr, yte = (tr.kind == "edge").to_numpy(), (te.kind == "edge").to_numpy()
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            both = LogisticRegression().fit((Xtr - mu) / sd, ytr)
            sup = LogisticRegression().fit(((Xtr - mu) / sd)[:, :1], ytr)
            s_both = both.decision_function((Xte - mu) / sd)
            s_sup = sup.decision_function(((Xte - mu) / sd)[:, :1])
            s_c = -te.c.to_numpy()
            gte = te.gc.to_numpy()
            ug, ginv = np.unique(gte, return_inverse=True)
            Wb = boot_weights(len(ug))
            ab, as_, ac = [], [], []
            for w in Wb[:2000]:
                ww = w[ginv]
                ab.append(auc_w(s_both, yte, ww))
                as_.append(auc_w(s_sup, yte, ww))
                ac.append(auc_w(s_c, yte, ww))
            ab, as_, ac = map(np.array, (ab, as_, ac))
            rr["even_model"] = dict(coef_both=both.coef_.tolist(), coef_support=sup.coef_.tolist(),
                                    auroc_both=auc_w(s_both, yte), auroc_support=auc_w(s_sup, yte), auroc_c=auc_w(s_c, yte),
                                    auroc_c_ci=ci(ac), diff_both_minus_support=float(auc_w(s_both, yte) - auc_w(s_sup, yte)),
                                    diff_ci=ci(ab - as_))
        t4[c] = rr
        log("test4", c, json.dumps(rr))
    out["test4_boundary"] = t4
    pd.concat(rows_b).to_parquet(W / f"s2_boundaries{TAG}.parquet", index=False)

    # ---------------------------------------------------------------- exploratory: contig-level foreignness vs contamination
    gm = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome_code", "completeness", "contamination", "phylum"]).sort_values("genome_code")
    cdf = pd.DataFrame(dict(cc=cc[ge], gc=gc[ge], f=f[ge]))
    cstat = cdf.groupby("cc").agg(gc=("gc", "first"), n=("f", "size"), mf=("f", "mean"))
    gmean = cdf.groupby("gc").f.mean()
    cstat["rel"] = cstat.mf - gmean.reindex(cstat.gc).to_numpy()
    big = cstat[cstat.n >= 10].copy()
    thr = np.quantile(big.rel, 0.99)
    big["foreign"] = big.rel >= thr
    per_g = big.groupby("gc").agg(n_contigs=("n", "size"), n_foreign=("foreign", "sum"))
    per_g = per_g.join(gm.set_index("genome_code"))
    from scipy.stats import spearmanr
    rho = spearmanr(per_g.contamination, per_g.n_foreign / per_g.n_contigs)
    hi = per_g.contamination >= 5
    out["exploratory_contigs"] = dict(contigs_ge10=int(len(big)), threshold_rel_f_q99=float(thr),
                                      spearman_contamination_vs_foreign_frac=[float(rho.statistic), float(rho.pvalue)],
                                      foreign_frac_contam_ge5=float(per_g[hi].n_foreign.sum() / per_g[hi].n_contigs.sum()),
                                      foreign_frac_contam_lt5=float(per_g[~hi].n_foreign.sum() / per_g[~hi].n_contigs.sum()),
                                      genomes_contam_ge5=int(hi.sum()))
    big.reset_index().to_parquet(W / f"s2_contigs{TAG}.parquet", index=False)
    log("exploratory contigs", json.dumps(out["exploratory_contigs"]))
    out["seconds"] = time.time() - t0
    json.dump(out, open(W / f"s2_tests{TAG}.json", "w"), indent=1, default=float)
    log("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
