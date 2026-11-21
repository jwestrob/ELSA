#!/usr/bin/env python
"""Extract one real bond-tree region for the explainer page (bond_tree.html).

Reads the third frozen run (testset3: 842 complete GTDB genomes) read-only on biotite and prints one JSON document to
stdout. Nothing is written on the cluster. Run from the Mac:

  ssh jwestrob@biotite.berkeley.edu 'OMP_NUM_THREADS=1 /home/jwestrob/miniconda3/envs/elsa/bin/python - 565' \
      < extract.py > bond_tree_data.json

Region: the longest ribosomal run (g2, >= 16 marked genes) of the chosen genome, the PRIMARY bonded run that contains it,
and the SECONDARY bonded run(s) that overlap that run; the window is their union plus 3 genes on each side.

Inputs (RD = .../ELSA/redesign_20261111; F3 = RD/long_loci/frozen3; RA = RD/ribo_audit/work6/testset3):
  RD/testset3/package/universe/genes.parquet, genomes.parquet     gene order, strand, coordinates, taxonomy
  RA/ribosomal/runs.parquet                                        ribosomal runs
  RA/annot/genes_annot.parquet, extra_ko.parquet                   KOfam calls (Aksha) of ribosomal / cluster / selected KOs
  RA/annot/window_pfam.parquet                                     observed Pfam domains near the runs
  PRIMARY  (embedding-list bond tree, Z L4b+ theta .02)
           F3/nodes/gtdb3_Z/bonds_obs.npz (k, d, Lambda), obs4b_knn_t0.02.parquet (tree nodes)
           F3/loci/gtdb3_Z/L1.parquet, L4bp_t0.02.parquet; F3/lists/kappa_var_knn50.npy
  SECONDARY (family-bond tree, WA L4b theta .2)
           F3/nodes/gtdb3_WA/bonds_obs.npz; F3/loci/gtdb3_WA/L1, L4bp_t0.2, L4b_t0.2; F3/emb/kappa_var_tokens.npy
  one bond F3/gtable/{counts.npy, genome.bin, local.bin, z.bin}, F3/emb/meta.npz, F3/groups/emb_hom_average_t0.75.parquet
           (z lists rebuilt for two genes; n, E, Lambda recomputed with vl.pass_count's group-counting formula)
"""
import json
import math
import sys
import time

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
F3 = RD + "/long_loci/frozen3"
PKG = RD + "/testset3/package"
RA = RD + "/ribo_audit/work6/testset3"
GCODE = int(sys.argv[1]) if len(sys.argv) > 1 else 565
MARGIN = 3
ZCUT, WGAP = 10.6, 3
TAU_Z, TAU_WA, TH_Z, TH_WA = 8.0, 8.0, 0.02, 0.2         # frozen3 values (REPORT_frozen3.txt, FROZEN3.json)
T0 = time.time()

# KEGG gene symbols of the non-ribosomal KOs met in this region (ribosomal proteins take the KO definition's name)
KO_SYMBOL = {"K03043": "rpoB", "K03046": "rpoC", "K03073": "secE", "K02601": "nusG", "K02518": "infA", "K01265": "map",
             "K00939": "adk", "K16785": "ecfT", "K16786": "ecfA1", "K16787": "ecfA2", "K06173": "truA", "K01883": "cysS",
             "K03218": "rlmB", "K02355": "fusA", "K02358": "tuf", "K03040": "rpoA", "K03076": "secY"}


def tlog(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, file=sys.stderr, flush=True)


def boundary_score(k, d, lam, n):
    """s[b] = largest Lambda of a bond spanning boundary b (ll.boundary_score), plus the index of that bond."""
    s = np.zeros(n)
    arg = np.full(n, -1, np.int64)
    for i in np.argsort(lam, kind="mergesort"):          # ascending: the largest Lambda is written last
        a, b = max(int(k[i]), 0), min(int(k[i] + d[i]), n)
        s[a:b] = lam[i]
        arg[a:b] = i
    return s, arg


def tree_port(w, lo, hi):
    """Top-down reading of ll.build_tree: cut the run at its weakest boundary weight (equal minima cut together)."""
    out = []

    def rec(a, b, parent):
        if b <= a:
            return
        ww = w[a:b]                                      # boundaries a..b-1 (boundary j lies between genes j, j+1)
        m = ww.min()
        nid = len(out)
        out.append(dict(lo=int(a), hi=int(b), m=float(m), parent=parent))
        st = a
        for c in [a + i for i in np.flatnonzero(ww == m)]:
            rec(st, c, nid)
            st = c + 1
        rec(st, b, nid)

    rec(lo, hi, -1)
    return out


def runs_in(df, A, B):
    return df[(df.e >= A) & (df.s <= B)]


def main():
    g = pd.read_parquet(PKG + "/universe/genes.parquet",
                        columns=["gid", "genome_code", "contig_code", "position_index", "strand", "start", "end", "length"])
    cc = g.contig_code.to_numpy(np.int64)
    gcode = g.genome_code.to_numpy(np.int64)
    gm = pd.read_parquet(PKG + "/universe/genomes.parquet").sort_values("genome_code").reset_index(drop=True)
    tlog("genes")
    runs = pd.read_parquet(RA + "/ribosomal/runs.parquet")
    rr = runs[(runs.genome_code == GCODE) & (runs.run_set == "g2") & runs.m16].sort_values("n_marked").iloc[-1]
    ra, rb = int(rr.gid_start), int(rr.gid_end)

    ZL1 = pd.read_parquet(F3 + "/loci/gtdb3_Z/L1.parquet")
    WL1 = pd.read_parquet(F3 + "/loci/gtdb3_WA/L1.parquet")
    l1 = ZL1[(ZL1.s <= rb) & (ZL1.e >= ra)].iloc[0]
    lo, hi = int(l1.s), int(l1.e)
    wov = runs_in(WL1, lo, hi)
    A = min(lo, int(wov.s.min())) - MARGIN
    B = max(hi, int(wov.e.max())) + MARGIN
    NB = B - A                                            # boundaries A..B-1
    zr = runs_in(ZL1, A, B)
    wr_ = runs_in(WL1, A, B)
    zr_in = zr[(zr.s >= A) & (zr.e <= B)]
    wr_in = wr_[(wr_.s >= A) & (wr_.e <= B)]
    checks = dict(primary_runs_cut_by_window=int(len(zr) - len(zr_in)), secondary_runs_cut_by_window=int(len(wr_) - len(wr_in)))

    # ---------------- PRIMARY: embedding-list bonds, kNN cohesion, theta .02, every tree node reported (L4b+)
    zb = np.load(F3 + "/nodes/gtdb3_Z/bonds_obs.npz")
    zk, zd, zl = zb["k"], zb["d"], zb["lam"]
    sel = (zk < B) & (zk + zd > A)
    sk, sd, sl = zk[sel], zd[sel], zl[sel]
    s, arg = boundary_score(sk - A, sd, sl, NB)
    kap = np.load(F3 + "/lists/kappa_var_knn50.npy", mmap_mode="r")[A:B].astype(np.float64)
    w = np.where((kap < TH_Z) & (s > 0), 1e-6 + np.maximum(np.nan_to_num(kap, nan=0.0), 0.0), s)
    T = pd.read_parquet(F3 + "/nodes/gtdb3_Z/obs4b_knn_t0.02.parquet")
    T["id"] = np.arange(len(T))
    tn = T[(T.lo >= A) & (T.hi <= B)].copy()
    ids = set(tn.id)
    P = pd.read_parquet(F3 + "/loci/gtdb3_Z/L4bp_t0.02.parquet")
    pz = P[(P.s >= A) & (P.e <= B)]
    tlog("primary loaded")
    port = []
    for r in zr_in.itertuples():
        port += tree_port(w, int(r.s) - A, int(r.e) - A)
    stored = set(zip(tn.lo - A, tn.hi - A))
    checks["primary_tree_port_equals_stored_nodes"] = {(x["lo"], x["hi"]) for x in port} == stored
    checks["primary_reported_loci_equal_tree_nodes"] = set(zip(pz.s - A, pz.e - A)) == stored
    checks["primary_stored_m_equals_min_w"] = bool(all(abs(r.m - w[r.lo - A:r.hi - A].min()) < 1e-9 for r in tn.itertuples()))
    checks["primary_s_positive_inside_runs"] = bool(all((s[int(r.s) - A:int(r.e) - A] > 0).all() for r in zr_in.itertuples()))
    nodes = []
    for r in tn.sort_values(["lo", "hi"]).itertuples():
        nodes.append(dict(id=int(r.id), lo=int(r.lo) - A, hi=int(r.hi) - A, m=float(r.m),
                          parent=int(r.par) if r.par in ids else -1))

    # ---------------- SECONDARY: family bonds (adaptive distance <= 8), token cohesion, theta .2, family node test
    wb = np.load(F3 + "/nodes/gtdb3_WA/bonds_obs.npz")
    wk, wd, wl = wb["k"], wb["d"], wb["lam"]
    sel2 = (wk < B) & (wk + wd > A)
    fk, fd, fl = wk[sel2], wd[sel2], wl[sel2]
    s2, arg2 = boundary_score(fk - A, fd, fl, NB)
    kt = np.load(F3 + "/emb/kappa_var_tokens.npy", mmap_mode="r")[A:B].astype(np.float64)
    w2 = np.where((kt < TH_WA) & (s2 > 0), 1e-6 + np.maximum(np.nan_to_num(kt, nan=0.0), 0.0), s2)
    WP = pd.read_parquet(F3 + "/loci/gtdb3_WA/L4bp_t0.2.parquet")
    WR = pd.read_parquet(F3 + "/loci/gtdb3_WA/L4b_t0.2.parquet")
    wp = WP[(WP.s >= A) & (WP.e <= B)]
    wrep = WR[(WR.s >= A) & (WR.e <= B)]
    port2 = []
    for r in wr_in.itertuples():
        part = tree_port(w2, int(r.s) - A, int(r.e) - A)
        for x in part:                                   # parent indices become positions in the combined list
            x["parent"] = x["parent"] + len(port2) if x["parent"] >= 0 else -1
        port2 += part
    checks["secondary_tree_port_equals_stored_nodes"] = {(x["lo"], x["hi"]) for x in port2} == set(zip(wp.s - A, wp.e - A))
    rep2 = set(zip(wrep.s - A, wrep.e - A))
    checks["secondary_reported_subset_of_nodes"] = rep2 <= {(x["lo"], x["hi"]) for x in port2}
    tlog("secondary loaded")

    # ---------------- genes and annotation
    ga = pd.read_parquet(RA + "/annot/genes_annot.parquet").set_index("gid").loc[A:B]
    xk = pd.read_parquet(RA + "/annot/extra_ko.parquet")
    xk = xk[(xk.gid >= A) & (xk.gid <= B)].sort_values("score", ascending=False).drop_duplicates("gid").set_index("gid")
    wpf = pd.read_parquet(RA + "/annot/window_pfam.parquet")
    wpf = wpf[(wpf.gid >= A) & (wpf.gid <= B)].drop_duplicates("gid").set_index("gid")
    gg = g.set_index("gid").loc[A:B]
    genes = []
    for gid, r in gg.iterrows():
        a_ = ga.loc[gid]
        ko = a_.ko if a_.ko else (xk.loc[gid, "ko"] if gid in xk.index else None)
        if a_.cls == "R" and a_.ko_name:
            label, src = a_.ko_name.split(" protein ")[-1], "KO"
        elif ko:
            label, src = KO_SYMBOL.get(ko, ko), "KO"
        elif gid in wpf.index:
            label, src = wpf.loc[gid, "top"], "Pfam"
        else:
            label, src = None, None
        genes.append(dict(gid=int(gid), pos=int(r.position_index), strand=int(r.strand), start=int(r.start),
                          end=int(r.end), aa=int(r.length), label=label, src=src, ko=ko,
                          pfam=(wpf.loc[gid, "arch"] if gid in wpf.index else None), ribo=bool(ra <= gid <= rb)))
    bounds = []
    for i in range(NB):
        j, j2 = arg[i], arg2[i]
        bounds.append(dict(s=round(float(s[i]), 3), kap=None if np.isnan(kap[i]) else round(float(kap[i]), 5),
                           bk=int(sk[j]) - A if j >= 0 else None, bd=int(sd[j]) if j >= 0 else None,
                           s2=round(float(s2[i]), 3), kap2=None if np.isnan(kt[i]) else round(float(kt[i]), 5),
                           bk2=int(fk[j2]) - A if j2 >= 0 else None, bd2=int(fd[j2]) if j2 >= 0 else None))
    tlog("annotation")

    # ---------------- example bonds, recomputed from the z lists (vl.pass_count, group counting)
    meta = np.load(F3 + "/emb/meta.npz")
    uq = meta["uq"].astype(np.int64)
    gstart = meta["g_start"].astype(np.int64)
    gt = pd.read_parquet(F3 + "/groups/emb_hom_average_t0.75.parquet").sort_values("genome_code")
    grp = pd.factorize(gt.group.to_numpy())[0].astype(np.int64)
    mg = gm.n_genes.to_numpy(np.float64)
    cnt = np.load(F3 + "/gtable/counts.npy").astype(np.int64)
    off = np.r_[0, np.cumsum(cnt)]
    zG = np.memmap(F3 + "/gtable/genome.bin", np.int16, "r")
    zLc = np.memmap(F3 + "/gtable/local.bin", np.int16, "r")
    zz = np.memmap(F3 + "/gtable/z.bin", np.float16, "r")

    def zlist(gid):
        u = uq[gid]
        a_, b_ = off[u], off[u + 1]
        G = np.asarray(zG[a_:b_]).astype(np.int64)
        Lc = np.asarray(zLc[a_:b_]).astype(np.int64)
        Z = np.asarray(zz[a_:b_]).astype(np.float32)
        k_ = (Z >= ZCUT) & (grp[G] != grp[gcode[gid]])
        return {int(Gi): int(gstart[Gi] + Li) for Gi, Li in zip(G[k_], Lc[k_])}

    def score(x, y):
        lx, ly = zlist(x), zlist(y)
        both = sorted(set(lx) & set(ly))
        per = []
        for G in sorted(set(lx) | set(ly)):
            px, py = lx.get(G), ly.get(G)
            if px is not None and py is not None:
                same = bool(cc[px] == cc[py])
                dist = int(abs(py - px)) if same else None
                cat = "support" if (same and 1 <= dist <= WGAP) else "apart"
            else:
                same, dist, cat = None, None, "xonly" if py is None else "yonly"
            per.append(dict(G=G, grp=int(grp[G]), dist=dist, cat=cat))
        sup = [p for p in per if p["cat"] == "support"]
        grp_lp = {}
        for G in both:
            p = min(1.0, 2.0 * WGAP / mg[G])
            grp_lp[grp[G]] = grp_lp.get(grp[G], 0.0) + (math.log1p(-p) if p < 1 else -1e300)
        own = min(1.0, 2.0 * WGAP / mg[gcode[x]])
        Eg = sum(1 - math.exp(v) for v in grp_lp.values()) + own
        ng = len({p["grp"] for p in sup})
        nn = ng + 1.0
        lam = nn * math.log(nn / Eg) - nn + Eg if nn > Eg else 0.0
        dist_counts = {}
        for p in sup:
            dist_counts[str(p["dist"])] = dist_counts.get(str(p["dist"]), 0) + 1
        agg = dict(x=x - A, y=y - A, genomes_x=len(lx), genomes_y=len(ly), groups_x=len({grp[G] for G in lx}),
                   groups_y=len({grp[G] for G in ly}), genomes_both=len(both), groups_both=len(grp_lp),
                   genomes_support=len(sup), groups_support=ng, groups_apart_only=len(grp_lp) - ng,
                   genomes_apart=sum(p["cat"] == "apart" for p in per), n=nn, E=Eg, E_own=own, lam=lam,
                   support_by_distance=dist_counts)
        # deterministic sample for the schematic: supporting genomes from distinct groups and phyla, one second genome
        # of an already-shown group, then genomes where the two matches lie apart, then single-match genomes
        ph = gm.phylum.to_numpy()
        pick, seen_g, seen_p = [], set(), set()
        for p in sorted(sup, key=lambda p: (ph[p["G"]], p["G"])):
            if ph[p["G"]] not in seen_p and p["grp"] not in seen_g and len(pick) < 5:
                pick.append(p)
                seen_g.add(p["grp"])
                seen_p.add(ph[p["G"]])
        twin = [p for p in sup if p["grp"] in seen_g and p not in pick]
        if twin:
            pick.append(twin[0])
        pick += [p for p in per if p["cat"] == "apart"][:3]
        pick += [p for p in per if p["cat"] == "xonly"][:1] + [p for p in per if p["cat"] == "yonly"][:1]
        for p in pick:
            r_ = gm.loc[p["G"]]
            p.update(genome=r_.genome, species=r_.species, phylum=r_.phylum)
        agg["sample"] = pick
        return agg

    names = {gd["gid"]: gd["label"] for gd in genes}
    zbonds = [(int(a), int(b), float(c)) for a, b, c in zip(sk, sd, sl)]
    examples = []
    for want in (("L2", "S19"), ("truA", "L13"), ("FolB", "HPPK")):
        hit = [b for b in zbonds if b[1] == 1 and names.get(b[0]) == want[0] and names.get(b[0] + 1) == want[1]]
        if not hit:
            continue
        k_, d_, lam_ = hit[0]
        e = score(k_, k_ + d_)
        e["stored_lam"] = lam_
        e["recomputed_equals_stored"] = abs(e["lam"] - lam_) < 1e-6 * max(1.0, lam_)
        examples.append(e)
    tlog("examples")
    gi = gm.loc[GCODE]
    out = dict(
        source="ELSA redesign, long_loci third frozen run (testset3), read-only extraction by extract.py",
        genome=dict(accession=gi.genome, species=gi.species, phylum=gi.phylum, cls=gi["class"], order=gi.ord,
                    family=gi.family, n_genes=int(gi.n_genes), n_contigs=int(gi.n_contigs), group=int(grp[GCODE]),
                    group_size=int((grp == grp[GCODE]).sum()), first_gid=int(A), first_pos=int(gg.position_index.iloc[0])),
        collection=dict(name="testset3", genomes=int(len(gm)), groups=int(grp.max() + 1)),
        window=dict(n_genes=int(B - A + 1), main_run=[lo - A, hi - A], ribo_run=[ra - A, rb - A]),
        params=dict(tau_primary=TAU_Z, theta_primary=TH_Z, tau_secondary=TAU_WA, theta_secondary=TH_WA, zcut=ZCUT,
                    wgap=WGAP),
        genes=genes, bounds=bounds,
        primary=dict(runs=[[int(r.s) - A, int(r.e) - A] for r in zr_in.itertuples()], nodes=nodes),
        secondary=dict(runs=[[int(r.s) - A, int(r.e) - A] for r in wr_in.itertuples()],
                       nodes=[dict(lo=x["lo"], hi=x["hi"], m=x["m"], parent=x["parent"],
                                   reported=(x["lo"], x["hi"]) in rep2) for x in port2]),
        examples=examples, checks=checks)
    print(json.dumps(out, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    tlog("done")


if __name__ == "__main__":
    main()
