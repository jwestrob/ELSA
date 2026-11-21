#!/usr/bin/env python
"""ws_calib Q1 follow-up (POST HOC, descriptive; written after q1_classes.json was read).

(a) physical kind conditional on no homology evidence, band vs length-matched baseline, query bootstrap
(b) homology evidence among pairs with an unannotated side, by which side is unannotated
(c) alignment geometry of alignable pairs (coverage of the shorter protein and of each protein; identity)
(d) three-way headline per band: remote homology / same kind with contradicting annotation / unexplained
Single-threaded, login-node sized. Writes results/q1_extra.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
C = RD / "calib"
W = C / "work/q1"
P = C / "work/prep"
KN = ["rank1_10", "rank11_100", "rank101_1000", "cos20_32", "cos32_45", "cos45_60", "cos60_100"]


def boot_ratio(j, num, den, nq, draws=2000, seed=17):
    a = np.bincount(j, weights=num.astype(float), minlength=nq)
    b = np.bincount(j, weights=den.astype(float), minlength=nq)
    rng = np.random.default_rng(seed)
    Wt = rng.multinomial(nq, np.full(nq, 1.0 / nq), size=draws).astype(float)
    r = (Wt @ a) / np.maximum(Wt @ b, 1)
    return [round(a.sum() / max(b.sum(), 1), 4)] + np.quantile(r, [0.025, 0.975]).round(4).tolist()


def main():
    pr = pd.read_parquet(W / "pairs_classified.parquet")
    Q = dict(np.load(P / "queries.npz"))
    A = dict(np.load(P / "ann.npz"))
    uf = dict(np.load(P / "uidfeat.npz"))
    R = np.flatnonzero(Q["stratum"] == 0)
    remap = np.full(len(Q["uid"]), -1)
    remap[R] = np.arange(len(R))
    pr = pr[remap[pr.j.to_numpy()] >= 0].reset_index(drop=True)
    jj = remap[pr.j.to_numpy()]
    nq = len(R)
    kind = pr.kind.to_numpy()
    grp = np.where(kind == 8, 8 + pr.rank_or_genome.to_numpy(), kind)
    q, t = pr.q.to_numpy(), pr.t.to_numpy()
    annot = (np.diff(A["pfam_ptr"]) > 0) | (np.diff(A["cdd_ptr"]) > 0)
    aq, at = annot[q], annot[t]
    HOM = pr.HOM.to_numpy()
    out = {"a_kind_given_no_homology": {}, "b_unannotated": {}, "c_alignment": {}, "d_headline": {}}
    for g, name in enumerate(KN):
        for tag, gg in ((name, g), ("lenmatch_" + name, 8 + g)):
            m = grp == gg
            nh = m & ~HOM
            d = dict(n_no_hom=int(nh.sum()))
            for f in ("K_kind", "same_TM_class", "len_ratio_ge_067", "comp_le_p25", "both_TM"):
                d[f] = boot_ratio(jj[nh], pr[f].to_numpy()[nh], np.ones(nh.sum()), nq)
            out["a_kind_given_no_homology"][tag] = d
        m = grp == g
        e = {}
        for lab, mm in (("both_annotated", aq & at), ("query_only_annotated", aq & ~at), ("target_only_annotated", ~aq & at),
                        ("neither_annotated", ~aq & ~at)):
            sel = m & mm
            e[lab] = dict(share_of_pairs=round(float(sel.sum() / max(m.sum(), 1)), 4),
                          HOM=boot_ratio(jj[sel], HOM[sel], np.ones(sel.sum()), nq),
                          A_align=boot_ratio(jj[sel], pr.A_align.to_numpy()[sel], np.ones(sel.sum()), nq))
        out["b_unannotated"][name] = e
        # headline
        contra = pr.contradicted_any.to_numpy()
        same_kind_other_origin = (pr.KIND_contradicted.to_numpy() | pr.ROLE.to_numpy())
        unexplained = ~HOM & ~same_kind_other_origin
        out["d_headline"][name] = dict(
            remote_homology=boot_ratio(jj[m], HOM[m], np.ones(m.sum()), nq),
            same_kind_contradicted=boot_ratio(jj[m], same_kind_other_origin[m], np.ones(m.sum()), nq),
            unexplained=boot_ratio(jj[m], unexplained[m], np.ones(m.sum()), nq),
            unexplained_with_unannotated_side=boot_ratio(jj[m], (unexplained & ~(aq & at))[m], np.ones(m.sum()), nq),
            different_domains_not_same_kind=boot_ratio(jj[m], (unexplained & contra)[m], np.ones(m.sum()), nq))
    # alignment geometry (all strata; pairs with E <= 1e-3; the stored alignment for the pair)
    aln = pd.read_parquet(W / "aln.parquet").sort_values("evalue").drop_duplicates(["query", "target"])
    a2 = pr[pr.A_align & (pr.q != pr.t)][["q", "t", "kind", "cos"]].merge(aln, left_on=["q", "t"], right_on=["query", "target"])
    a2["cov_short"] = (np.minimum(a2.qend - a2.qstart, a2.tend - a2.tstart) + 1) / np.minimum(a2.qlen, a2.tlen)
    a2["cov_q"] = (a2.qend - a2.qstart + 1) / a2.qlen
    a2["cov_t"] = (a2.tend - a2.tstart + 1) / a2.tlen
    a2["both50"] = (a2.cov_q >= 0.5) & (a2.cov_t >= 0.5)
    for g, name in enumerate(KN):
        x = a2[a2.kind == g]
        if len(x):
            out["c_alignment"][name] = dict(n=int(len(x)), fident_q50=round(float(x.fident.median()), 3),
                                            cov_short_q50=round(float(x.cov_short.median()), 3),
                                            share_both_cov_ge_05=round(float(x.both50.mean()), 4),
                                            share_cov_short_ge_08=round(float((x.cov_short >= 0.8).mean()), 4),
                                            evalue_q50=float(x.evalue.median()),
                                            len_ratio_q50=round(float((np.minimum(x.qlen, x.tlen) / np.maximum(x.qlen, x.tlen)).median()), 3))
    (C / "results/q1_extra.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["d_headline"], indent=0))
    print(json.dumps(out["c_alignment"], indent=0))


if __name__ == "__main__":
    main()
