#!/usr/bin/env python
"""lut_loci n01 (biotite login node, single process): neighborhood table of the 733 main lut loci.

For every main lut instance (gold/lut_instances.parquet, homology + co-location calls, not a curated caller) take the
instance span +-10 genes on its contig. Per window gene: oriented relative position, gold component, nearest component and
offset, whole-proteome Pfam (annot_gpu, GA), rescue3 token, slot call (pass_port lut pass + tiers), Z bonds (frozen run,
embedding-list arm) to instance members, and whether it lies in the method's best-matching Z L4b+ locus.
Z lists (z >= 10.6, best gene per genome, own genome group excluded; identical to slot_rescue/code/score.py z_chunk) are
written for every window gene.

usage: n01_neighborhood.py OUTDIR
"""
import sys
import time

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PK = RD + "/slice_lut/testset/package_k8"
FZ = RD + "/long_loci/frozen"
PP = RD + "/dpann_leads/pass_port/out/lut"
ZCUT, FLANK = 10.6, 10
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def main(out):
    gn = pd.read_parquet(PK + "/universe/genes.parquet",
                         columns=["gid", "protein_id", "genome_code", "contig_code", "position_index", "strand", "length"])
    n = len(gn)
    gc = gn.genome_code.to_numpy(np.int64)
    cid = gn.contig_code.to_numpy(np.int64)
    pos = gn.position_index.to_numpy(np.int64)
    strand = gn.strand.to_numpy(np.int64)
    cstart = np.r_[0, np.flatnonzero(np.diff(cid)) + 1]
    cfirst = np.zeros(cid.max() + 1, np.int64)
    cfirst[cid[cstart]] = cstart
    clen = np.bincount(cid)
    g_start = np.r_[0, np.flatnonzero(np.diff(gc)) + 1].astype(np.int64)
    gm = pd.read_parquet(PK + "/universe/genomes.parquet")
    groups = pd.read_parquet(FZ + "/groups/emb_hom_average_t0.75.parquet").sort_values("genome_code")
    lin_g = pd.factorize(groups.group.to_numpy())[0].astype(np.int64)
    lin = lin_g[gc]
    log("genes", n)

    I = pd.read_parquet(PK + "/gold/lut_instances.parquet")
    I = I[I.definition == "main"].reset_index(drop=True)
    log("main instances", len(I))

    # window genes
    rows = []
    for r in I.itertuples():
        mg = np.array(r.member_gids, np.int64)
        mpos = np.array(r.member_positions, np.int64)
        comps = list(r.member_components)
        orient = 1 if (strand[mg] > 0).sum() >= (strand[mg] < 0).sum() else -1
        c0 = cfirst[r.contig_code]
        lo, hi = max(0, r.start_idx - FLANK), min(clen[r.contig_code] - 1, r.end_idx + FLANK)
        for p in range(lo, hi + 1):
            g = c0 + p
            assert pos[g] == p and cid[g] == r.contig_code
            rel = (p - r.start_idx) if orient > 0 else (r.end_idx - p)
            d = (p - mpos) * orient
            j = int(np.argmin(np.abs(d) * 100 + (d < 0)))
            comp = comps[int(np.flatnonzero(mg == g)[0])] if g in set(mg.tolist()) else ""
            rows.append((r.instance_id, g, p, rel, r.span_genes, int(strand[g] * orient), comp, comps[j], int(d[j])))
    W = pd.DataFrame(rows, columns=["instance_id", "gid", "pos", "rel", "span", "ostrand", "component", "nearest_comp",
                                    "nearest_off"])
    W["region"] = np.where(W.rel < 0, "up", np.where(W.rel >= W.span, "down", "inside"))
    W["length"] = gn.length.to_numpy()[W.gid]
    W["protein_id"] = gn.protein_id.to_numpy()[W.gid]
    W["genome_code"] = gc[W.gid]
    tax = gm.set_index("genome_code")
    for c in ["genome", "phylum", "class", "ord", "family", "genus", "species", "layer", "lut_status", "completeness"]:
        W[c] = tax.loc[W.genome_code, c].to_numpy()
    W = W.merge(I[["instance_id", "system_type", "architecture", "at_contig_edge_le2", "mdh_ko_within_25_genes"]],
                on="instance_id")
    log("window rows", len(W), "unique genes", W.gid.nunique())

    # Pfam
    pf = pd.read_parquet(RD + "/annot_gpu/lut/PFAM.parquet", columns=["gid", "accession", "name", "ali_from", "ali_to"])
    wg = np.unique(W.gid.to_numpy())
    pfw = pf[pf.gid.isin(wg)].sort_values(["gid", "ali_from"])
    agg = pfw.groupby("gid").agg(pfam=("accession", lambda x: ";".join(x)), pfam_names=("name", lambda x: ";".join(x)))
    W = W.merge(agg, left_on="gid", right_index=True, how="left")
    W["has_pfam"] = W.pfam.notna()
    W["duf_only"] = W.pfam_names.fillna("").map(
        lambda s: bool(s) and all(x.startswith("DUF") or x.startswith("UPF") for x in s.split(";")))
    pfam_any = np.zeros(n, bool)
    pfam_any[pf.gid.unique()] = True
    np.save(out + "/pfam_any.npy", pfam_any)
    del pf
    log("pfam joined")

    # tokens
    tok = pd.read_parquet(FZ + "/tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet")
    assert (tok.gid.to_numpy() == np.arange(n)).all()
    W["token"] = tok.token.to_numpy()[W.gid]

    # slot call
    ps = pd.read_parquet(PP + "/lut_slot_rescue_pass.parquet")
    ti = pd.read_parquet(PP + "/tiers.parquet")[["protein_id", "tier", "tier_family"]]
    ps = ps.merge(ti, on="protein_id", how="left")
    keep = ["gid", "proposed_label_positional_homology", "proposed_label_name", "votes_for", "supporting_genomes",
            "supporting_groups", "support", "runner_up_name", "runner_up_votes", "label_amplification",
            "label_annotated_count", "n_carriers", "unannotated_supporters", "locus_size", "n_bonds", "max_bond_lambda",
            "tier", "tier_family"]
    W = W.merge(ps[keep], on="gid", how="left")

    # bonds
    z = np.load(FZ + "/nodes/lut_Z/bonds_obs.npz")
    k, d, lam = z["k"].astype(np.int64), z["d"].astype(np.int64), z["lam"]
    inw = np.isin(k, wg) | np.isin(k + d, wg)
    bk, bd, bl = k[inw], d[inw], lam[inw]
    bonds = pd.DataFrame({"a": bk, "b": bk + bd, "d": bd, "lam": bl})
    bonds.to_parquet(out + "/window_bonds.parquet")
    nb = pd.concat([bonds[["a"]].rename(columns={"a": "gid"}), bonds[["b"]].rename(columns={"b": "gid"})]).gid.value_counts()
    W["zbonds"] = W.gid.map(nb).fillna(0).astype(int)
    mem = I[["instance_id", "member_gids"]].explode("member_gids").rename(columns={"member_gids": "m"})
    mem["m"] = mem.m.astype(np.int64)
    memset = mem.groupby("instance_id").m.apply(set).to_dict()
    bl_a = bonds.groupby("a")
    bdict = {}
    for a, b, l in zip(bonds.a.to_numpy(), bonds.b.to_numpy(), bonds.lam.to_numpy()):
        bdict.setdefault(a, []).append((b, l))
        bdict.setdefault(b, []).append((a, l))
    bc, bcl = [], []
    for iid, g in zip(W.instance_id.to_numpy(), W.gid.to_numpy()):
        ms = memset[iid]
        ls = [l for (o, l) in bdict.get(g, []) if o in ms and o != g]
        bc.append(len(ls))
        bcl.append(max(ls) if ls else 0.0)
    W["bonds_to_members"], W["max_lam_to_members"] = bc, bcl
    log("bonds", len(bonds))

    # best-matching Z L4b+ locus per instance (gene-level F1 between instance span and locus) and smallest covering locus
    L = pd.read_parquet(FZ + "/loci/lut_Z/L4bp_t0.02.parquet").sort_values(["s", "e"])
    s, e = L.s.to_numpy(np.int64), L.e.to_numpy(np.int64)
    best = []
    for r in I.itertuples():
        c0 = cfirst[r.contig_code]
        a0, a1 = c0 + r.start_idx, c0 + r.end_idx
        cand = np.flatnonzero((s <= a1) & (e >= a0) & (s >= c0))
        if len(cand) == 0:
            best.append((r.instance_id, -1, -1, 0.0, -1, -1))
            continue
        ov = np.minimum(e[cand], a1) - np.maximum(s[cand], a0) + 1
        f1 = 2 * ov / ((e[cand] - s[cand] + 1) + (a1 - a0 + 1))
        b = cand[int(np.argmax(f1))]
        cov = cand[(s[cand] <= a0) & (e[cand] >= a1)]
        sm = cov[int(np.argmin(e[cov] - s[cov]))] if len(cov) else -1
        best.append((r.instance_id, s[b], e[b], float(f1.max()), s[sm] if sm >= 0 else -1, e[sm] if sm >= 0 else -1))
    B = pd.DataFrame(best, columns=["instance_id", "best_s", "best_e", "best_f1", "cover_s", "cover_e"])
    W = W.merge(B, on="instance_id")
    W["in_best_locus"] = (W.gid >= W.best_s) & (W.gid <= W.best_e)
    W["in_cover_locus"] = (W.gid >= W.cover_s) & (W.gid <= W.cover_e)
    B.to_csv(out + "/instance_loci.tsv", sep="\t", index=False)
    log("loci matched; median best F1", B.best_f1.median())

    # Z lists for window genes
    uq = np.load(FZ + "/emb/meta.npz")["uq"].astype(np.int64)
    T_ = FZ + "/gtable"
    zoff = np.r_[0, np.cumsum(np.load(T_ + "/counts.npy").astype(np.int64))]
    zG = np.memmap(T_ + "/genome.bin", np.int16, "r")
    zL = np.memmap(T_ + "/local.bin", np.int16, "r")
    zz = np.memmap(T_ + "/z.bin", np.float16, "r")
    lr = []
    for g in wg:
        u = uq[g]
        if u < 0:
            lr.append((g, np.zeros(0, np.int64), np.zeros(0, np.float32)))
            continue
        a, b = zoff[u], zoff[u + 1]
        zv = np.asarray(zz[a:b]).astype(np.float32)
        gg = np.asarray(zG[a:b]).astype(np.int64)
        kk = np.flatnonzero((zv >= ZCUT) & (lin_g[gg] != lin[g]))
        o = kk[np.argsort(-zv[kk], kind="mergesort")]
        p = g_start[gg[o]] + np.asarray(zL[a:b]).astype(np.int64)[o]
        lr.append((g, p, zv[o]))
    Z = pd.DataFrame({"gid": [x[0] for x in lr], "partners": [x[1] for x in lr], "z": [x[2] for x in lr]})
    Z["n_partners"] = Z.partners.map(len)
    Z.to_parquet(out + "/window_zlists.parquet")
    W = W.merge(Z[["gid", "n_partners"]], on="gid", how="left")
    W.to_parquet(out + "/window_genes.parquet")
    log("done", W.shape)


if __name__ == "__main__":
    main(sys.argv[1])
