#!/usr/bin/env python
"""pass_port: the slot-rescue pass for any collection (code/collections.json), frozen call rule unchanged.

Call (identical to slot_rescue/code/pass_run.py; its functions are imported): a gene without a Pfam row, inside a locus of
the embedding-list tree (Z L4b+ theta .02) and with >= 1 stored Z bond; supporting partners = per genome the best-ranked
z >= 10.6 list partner (other genome groups) lying 1-3 genes from a list partner of a bonded neighbor, pooled over the
gene's bonds; vote = plurality key (Pfam accession; KO -> proxy Pfams, else the KO) with >= 2 votes.
Bond gate (every bond processed): recomputed support reproduces the stored bond. Where the bond file holds n / ng (DPANN)
both n and ng and Lambda are compared; where it holds Lambda only (the frozen collections) Lambda is recomputed from the
lists with vl.pass_count's group-counting formula and compared (relative tolerance 1e-9).
Added columns: support, label_annotated_count, label_amplification, unannotated_supporters, partner-derived function
(plurality KO among supporters that carried the winning label: partner_ko, _n, _share, _second, _counts, n_carriers,
n_carriers_ko), list_type (both / disagree / embedding-only / no alignment lists). Supporters per called gene go to
OUT/supporters/part_*.parquet (gid, supporters in rank order, carries_label).
DPANN regression (--collection dpann): the original 31 columns must equal slot_rescue's pass table row for row and the
partner function must equal gapscreen/partner_ko.parquet; otherwise exit 1.

usage: cpass.py --collection NAME --threads T [--smoke N] [--annot-dir DIR] [--config code/collections.json]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
sys.path.insert(0, RD + "/slot_rescue/code")
import score as S  # noqa: E402  (z_chunk, a_chunk, support, vote, right_mask, expand)
import pass_run as PR  # noqa: E402  (BASIS, vbin, sbin, runner_up)

G = S.G
WGAP = 3
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def P_(p):
    return p.replace("RD/", RD + "/") if isinstance(p, str) else p


# ------------------------------------------------------------------ worker side
def getl(arm, g, cache):
    v = cache.get((arm, g))
    if v is None:
        fn = S.z_chunk if arm == "Z" else S.a_chunk
        v = fn(np.array([g], np.int64))[1].astype(np.int64)
        if len(cache) > 30000:
            cache.clear()
        cache[(arm, g)] = v
    return v


def lam_pair(P, Q, j, ngs):
    """vl.pass_count group counting: E_grp = sum over groups of 1 - prod_G (1 - min(1, 2 W ca cb / m_G)) + own term."""
    gc, mg = G["gc"], G["mg"]
    up, ca = np.unique(gc[P], return_counts=True)
    uq_, cb = np.unique(gc[Q], return_counts=True)
    com, ia, ib = np.intersect1d(up, uq_, assume_unique=True, return_indices=True)
    Eg = 0.0
    if len(com):
        p = np.minimum(1.0, 2.0 * WGAP * ca[ia] * cb[ib] / mg[com])
        with np.errstate(divide="ignore"):
            lp = np.log1p(-p)
        _, inv = np.unique(G["grp"][com], return_inverse=True)
        s = np.bincount(inv, weights=lp)
        Eg = float(np.sum(1.0 - np.exp(s)))
    own = min(1.0, 2.0 * WGAP / mg[gc[j]])
    n, E = ngs + 1.0, Eg + own
    return n * np.log(n / E) - n + E if n > E else 0.0


def slot(arm, g, cache):
    P = getl(arm, g, cache)
    bp, bv = G["BPTR"][arm], G["BVAL"][arm]
    B = G["BONDS"][arm]
    best, nb, bad, lam = {}, 0, 0, 0.0
    for b in bv[bp[g]:bp[g + 1]]:
        j, d = int(B["k"][b]), int(B["d"][b])
        other = j + d if j == g else j
        ns, ngs, ug, pos = S.support(P, getl(arm, other, cache))
        if "n" in B and (ns != int(B["n"][b]) - 1 or ngs != int(B["ng"][b]) - 1):
            bad += 1
        lr = lam_pair(P, getl(arm, other, cache), j, ngs)
        if abs(lr - float(B["lam"][b])) > 1e-9 * max(1.0, abs(float(B["lam"][b]))):
            bad += 1
        nb += 1
        lam = max(lam, float(B["lam"][b]))
        for G_, p_ in zip(ug.tolist(), pos.tolist()):
            if G_ not in best or p_ < best[G_]:
                best[G_] = p_
    Pp = np.zeros(0, np.int64)
    if nb and best:
        pos = np.array(sorted(best.values()), np.int64)
        Pp = P[pos]
        c, lab, cR, co, na = S.vote(Pp, np.zeros(len(Pp), bool))
    elif nb:
        c, lab, co, na = "none_nosupport", -1, 0, 0
    else:
        c, lab, co, na = "none_nobond", -1, 0, 0
    cls = c if G["INLOC"][arm][g] else ("none_notinlocus" if nb else c)
    ngrp = len(np.unique(G["grp"][np.array(list(best.keys()), np.int64)])) if best else 0
    return dict(cls=cls, lab=int(lab), co=co, na=na, n_support=len(best), n_groups=ngrp, nb=nb, bad=bad, lam=lam, Pp=Pp,
                listlen=len(P))


def partner_function(Pp, kw):
    keyp, keyv, labp, labv, isko = G["KEYP"], G["KEYV"], G["LABP"], G["LABV"], G["ISKO"]
    carriers = [p for p in Pp.tolist() if kw in set(keyv[keyp[p]:keyp[p + 1]].tolist())]
    cnt, nko = {}, 0
    for p in carriers:
        kos = {x for x in labv[labp[p]:labp[p + 1]].tolist() if isko[x]}
        nko += bool(kos)
        for x in kos:
            cnt[x] = cnt.get(x, 0) + 1
    top = sorted(cnt.items(), key=lambda t: (-t[1], t[0]))
    return carriers, nko, top


def task(job):
    kind, cid_, items = job
    cache, out, sup = {}, [], []
    voc = G["VOCAB"]
    for g in items:
        if kind == "Z":
            r = slot("Z", g, cache)
            if r["cls"] == "other":
                ru, rc, tied = PR.runner_up(r["Pp"], r["lab"])
                carriers, nko, top = partner_function(r["Pp"], r["lab"])
                cset = set(carriers)
                sup.append((g, r["Pp"].astype(np.int32), np.array([p in cset for p in r["Pp"].tolist()], bool)))
                out.append((g, r["cls"], r["lab"], r["co"], r["na"], r["n_support"], r["n_groups"], r["nb"], r["bad"], r["lam"],
                            ru, rc, ";".join(voc[t] for t in tied), r["listlen"], len(carriers), nko,
                            voc[top[0][0]] if top else "", top[0][1] if top else 0,
                            voc[top[1][0]] if len(top) > 1 else "", top[1][1] if len(top) > 1 else 0,
                            ";".join(f"{voc[k]}:{c}" for k, c in top[:6])))
            else:
                out.append((g, r["cls"], -1, 0, r["na"], r["n_support"], r["n_groups"], r["nb"], r["bad"], r["lam"], -1, 0, "",
                            r["listlen"], 0, 0, "", 0, "", 0, ""))
        elif kind == "cal":
            r = slot("Z", g, cache)
            own = set(G["LABV"][G["LABP"][g]:G["LABP"][g + 1]].tolist()) | set(G["KEYV"][G["KEYP"][g]:G["KEYP"][g + 1]].tolist())
            ownpf = {x for x in own if G["ISPF"][x]} & set(G["LABV"][G["LABP"][g]:G["LABP"][g + 1]].tolist())
            out.append((g, r["cls"], r["lab"], r["co"], r["na"], r["nb"], r["bad"], int(r["lab"] in own), int(r["lab"] in ownpf)))
        elif kind == "A":
            r = slot("A", g, cache)
            out.append((g, r["cls"], r["lab"], r["listlen"], r["nb"], r["bad"]))
    if sup:
        pd.DataFrame(dict(gid=[s[0] for s in sup], supporters=[s[1] for s in sup], carries_label=[s[2] for s in sup])).to_parquet(
            f"{G['OUT']}/supporters/part_{cid_:05d}.parquet", index=False)
    return out


# ------------------------------------------------------------------ parent
def setup(cfg):
    gn = pd.read_parquet(P_(cfg["genes"]), columns=["gid", "protein_id", "genome", "genome_code", "contig", "contig_code",
                                                    "position_index", "strand", "length"])
    assert (gn.gid.to_numpy() == np.arange(len(gn))).all()
    n = len(gn)
    G["n"] = n
    G["gc"] = gn.genome_code.to_numpy(np.int64)
    G["cid"] = gn.contig_code.to_numpy(np.int64)
    assert (np.diff(G["gc"]) >= 0).all() and (np.diff(G["cid"]) >= 0).all()
    G["cstart"] = np.r_[0, np.flatnonzero(np.diff(G["cid"])) + 1].astype(np.int64)
    G["g_start"] = np.r_[0, np.flatnonzero(np.diff(G["gc"])) + 1].astype(np.int64)
    ng = len(G["g_start"])
    G["mg"] = np.bincount(G["gc"], minlength=ng).astype(np.float64)
    lm = pd.read_parquet(P_(cfg["groups"]), columns=["genome_code", "group"]).sort_values("genome_code")
    assert np.array_equal(lm.genome_code.to_numpy(), np.arange(ng))
    G["lin_of_genome"] = pd.factorize(lm.group.to_numpy())[0].astype(np.int64)
    G["grp"] = G["lin_of_genome"]
    G["lin"] = G["lin_of_genome"][G["gc"]]
    u = P_(cfg["uq"])
    G["uq"] = (np.load(u.split(":")[0])[u.split(":")[1]] if ":" in u else np.load(u)).astype(np.int64)
    assert len(G["uq"]) == n
    # annotation (label tables): protein_id -> gid by the collection's id scheme
    A = pd.read_parquet(P_(cfg["annot"]))
    if cfg["annot_id"] == "protein_id":
        A["gid"] = A.protein_id.map(pd.Series(np.arange(n), index=gn.protein_id.to_numpy()))
    else:
        A["gid"] = A.protein_id.astype(np.int64)
    assert A.gid.notna().all()
    A["gid"] = A.gid.astype(np.int64)
    kp = pd.read_csv(P_(cfg["ko_proxy"]), sep="\t")
    proxy = kp.groupby("ko").pf.apply(sorted).to_dict()
    vocab = sorted({x for v in A.pfam for x in v} | {x for v in A.ko for x in v} | set(kp.pf))
    lid = {x: i for i, x in enumerate(vocab)}
    G["VOCAB"], G["nlab"] = vocab, len(vocab)
    G["ISPF"] = np.array([v.startswith("PF") for v in vocab])
    G["ISKO"] = np.array([v.startswith("K") for v in vocab])
    ep = A[["gid", "pfam"]].explode("pfam").dropna().rename(columns={"pfam": "lab"})
    ek = A[["gid", "ko"]].explode("ko").dropna().rename(columns={"ko": "lab"})
    lab = pd.concat([ep, ek], ignore_index=True)
    lab["i"] = lab.lab.map(lid).astype(np.int64)
    lab = lab.drop_duplicates(["gid", "i"]).sort_values(["gid", "i"], kind="stable")
    ek["key"] = ek.lab.map(lambda k: proxy.get(k, [k]))
    kk = ek[["gid", "key"]].explode("key").rename(columns={"key": "lab"})
    key = pd.concat([ep, kk], ignore_index=True)
    key["i"] = key.lab.map(lid).astype(np.int64)
    key = key.drop_duplicates(["gid", "i"]).sort_values(["gid", "i"], kind="stable")
    G["LABP"] = np.r_[0, np.cumsum(np.bincount(lab.gid.to_numpy(), minlength=n))].astype(np.int64)
    G["KEYP"] = np.r_[0, np.cumsum(np.bincount(key.gid.to_numpy(), minlength=n))].astype(np.int64)
    G["LABV"] = lab.i.to_numpy(np.int64)
    G["KEYV"] = key.i.to_numpy(np.int64)
    G["LABCOUNT"] = np.bincount(G["LABV"], minlength=len(vocab))
    # lists
    T_ = P_(cfg["ztable"])
    G["zoff"] = np.r_[0, np.cumsum(np.load(T_ + "/counts.npy").astype(np.int64))]
    G["zG"] = np.memmap(T_ + "/genome.bin", np.int16, "r")
    G["zL"] = np.memmap(T_ + "/local.bin", np.int16, "r")
    G["zz"] = np.memmap(T_ + "/z.bin", np.float16, "r")
    arms = ["Z"]
    if cfg.get("aln_I"):
        G["I"] = np.load(P_(cfg["aln_I"]), mmap_mode="r")
        okq = G["uq"] >= 0
        G["members"] = np.flatnonzero(okq)[np.argsort(G["uq"][okq], kind="stable")].astype(np.int64)
        G["mptr"] = np.r_[0, np.cumsum(np.bincount(G["uq"][okq], minlength=G["I"].shape[0]))].astype(np.int64)
        arms.append("A")
    G["BONDS"], G["BPTR"], G["BVAL"], G["INLOC"], G["LOCI"] = {}, {}, {}, {}, {}
    for arm in arms:
        z = np.load(P_(cfg[f"bonds_{arm}"]))
        B = {k: z[k] for k in z.files}
        B["k"], B["d"] = B["k"].astype(np.int64), B["d"].astype(np.int64)
        G["BONDS"][arm] = B
        e = np.r_[B["k"], B["k"] + B["d"]]
        bi = np.r_[np.arange(len(B["k"])), np.arange(len(B["k"]))]
        o = np.argsort(e, kind="stable")
        G["BPTR"][arm] = np.r_[0, np.cumsum(np.bincount(e, minlength=n))].astype(np.int64)
        G["BVAL"][arm] = bi[o].astype(np.int64)
        L = pd.read_parquet(P_(cfg[f"loci_{arm}"])).sort_values(["s", "e"])
        s, e2 = L.s.to_numpy(np.int64), L.e.to_numpy(np.int64)
        G["LOCI"][arm] = (s, e2)
        cov = np.zeros(n + 1, np.int64)
        np.add.at(cov, s, 1)
        np.add.at(cov, e2 + 1, -1)
        G["INLOC"][arm] = np.cumsum(cov)[:n] > 0
    return gn, A, proxy, vocab, lid, arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "collections.json"))
    ap.add_argument("--annot-dir", help="override: annot.parquet, ko_proxy.tsv, label_names.tsv from this directory")
    a = ap.parse_args()
    cfg = json.load(open(a.config))[a.collection]
    if a.annot_dir:
        cfg.update(annot=a.annot_dir + "/annot.parquet", ko_proxy=a.annot_dir + "/ko_proxy.tsv", label_names=a.annot_dir + "/label_names.tsv")
    OUT = P_(cfg["out"]) + ("/smoke" if a.smoke else "")
    os.makedirs(OUT + "/supporters", exist_ok=True)
    for f in os.listdir(OUT + "/supporters"):
        os.remove(OUT + "/supporters/" + f)
    G["OUT"] = OUT
    th = a.threads
    run = dict(collection=a.collection, threads=th, smoke=a.smoke, timing={}, config=cfg)
    gn, A, proxy, vocab, lid, arms = setup(cfg)
    n = G["n"]
    log("setup", a.collection, "genes", n, "genomes", len(G["g_start"]), "groups", int(G["grp"].max()) + 1, "labels", len(vocab), "arms", arms)
    pool = Pool(th)

    def pmap(kind, items, nchunk):
        parts = [x for x in np.array_split(np.arange(len(items)), nchunk) if len(x)]
        jobs = [(kind, k, [items[i] for i in p]) for k, p in enumerate(parts)]
        return [r for part in pool.map(task, jobs) for r in part]

    # candidates: genes without a Pfam row
    haspf = np.zeros(n, bool)
    haspf[A.gid[A.pfam.str.len() > 0].to_numpy()] = True
    if cfg.get("pfam_nameonly"):
        no = pd.read_csv(P_(cfg["pfam_nameonly"]), sep="\t").protein_id
        haspf[no.map(pd.Series(np.arange(n), index=gn.protein_id.to_numpy())).to_numpy(np.int64)] = True
    hasko = np.zeros(n, bool)
    hasko[A.gid[A.ko.str.len() > 0].to_numpy()] = True
    cand = np.flatnonzero(~haspf)
    bonded = np.diff(G["BPTR"]["Z"]) > 0
    elig = cand[bonded[cand] & G["INLOC"]["Z"][cand]]
    if a.smoke:
        elig = elig[:a.smoke]
    counts = dict(genes=int(n), candidates=int(len(cand)), candidates_with_own_ko=int(hasko[cand].sum()), z_bonded=int(bonded[cand].sum()),
                  in_z_locus=int(G["INLOC"]["Z"][cand].sum()), eligible=int(len(cand[bonded[cand] & G["INLOC"]["Z"][cand]])))
    t = time.time()
    cols = ["gid", "cls", "lab", "votes_for", "voters_annotated", "supporting_genomes", "supporting_groups", "n_bonds", "bad",
            "max_bond_lambda", "ru", "ru_votes", "tied", "z_list_len", "n_carriers", "n_carriers_ko", "partner_ko", "partner_ko_n",
            "partner_ko_second", "partner_ko_second_n", "partner_ko_counts"]
    Zr = pd.DataFrame(pmap("Z", elig.tolist(), th * 32), columns=cols)
    run["timing"]["z_pass"] = round(time.time() - t, 1)
    counts.update({f"eligible_{k}": int(v) for k, v in Zr.cls.value_counts().items()})
    log("Z pass", len(Zr), Zr.cls.value_counts().to_dict())
    # calibration on bonded genes with a Pfam (pass_run rule)
    t = time.time()
    calpool = np.flatnonzero(haspf & (np.diff(G["LABP"]) > 0) & np.isin(np.arange(n), A.gid[A.pfam.str.len() > 0].to_numpy()) & bonded)
    rng = np.random.default_rng(20261120)
    cal = np.sort(rng.choice(calpool, size=min(50000, len(calpool)), replace=False)) if len(calpool) else np.zeros(0, np.int64)
    if a.smoke:
        cal = cal[:a.smoke]
    Cr = pd.DataFrame(pmap("cal", cal.tolist(), th * 8), columns=["gid", "cls", "lab", "votes_for", "voters_annotated", "nb", "bad", "agree",
                                                                  "agree_pfam"])
    run["timing"]["calibration"] = round(time.time() - t, 1)
    bad = int(Zr.bad.sum() + Cr.bad.sum())
    run["bond_gate"] = dict(bonds_checked=int(Zr.n_bonds.sum() + Cr.nb.sum()), mismatches=bad,
                            mode="n, ng and Lambda" if "n" in G["BONDS"]["Z"] else "Lambda recomputed")
    if bad:
        json.dump(run, open(f"{OUT}/run.json", "w"), indent=1, default=str)
        raise SystemExit(f"STOP: bond gate mismatches {bad}")
    Cc = Cr[Cr.cls == "other"].copy()
    Cc["share"] = Cc.votes_for / Cc.voters_annotated
    Cc["vbin"] = Cc.votes_for.map(PR.vbin)
    Cc["sbin"] = Cc.share.map(PR.sbin)
    cells = Cc.groupby(["vbin", "sbin"]).agg(n=("agree", "size"), agree=("agree", "mean"), agree_pfam=("agree_pfam", "mean")).reset_index()
    cells.to_csv(f"{OUT}/calibration_cells.tsv", sep="\t", index=False)
    run["calibration"] = dict(pool=int(len(calpool)), sample=int(len(Cr)), called=int(len(Cc)),
                              agree=float(Cc.agree.mean()) if len(Cc) else None)
    # A support column
    Out = Zr[Zr.cls == "other"].copy()
    if "A" in arms:
        t = time.time()
        Ar = pd.DataFrame(pmap("A", Out.gid.tolist(), th * 32), columns=["gid", "a_cls", "a_lab", "a_list_len", "a_nb", "a_bad"])
        run["timing"]["a_support"] = round(time.time() - t, 1)
        if int(Ar.a_bad.sum()):
            raise SystemExit("STOP: A bond gate")
        Out = Out.merge(Ar, on="gid")
    # table
    nm = {}
    if cfg.get("label_names"):
        ln = pd.read_csv(P_(cfg["label_names"]), sep="\t", keep_default_na=False)
        nm = dict(zip(ln.label, ln.name))
    g_ = Out.gid.to_numpy()
    df = pd.DataFrame(dict(protein_id=gn.protein_id.to_numpy()[g_], gid=g_, genome=gn.genome.to_numpy()[g_], genome_code=G["gc"][g_],
                           contig=gn.contig.to_numpy()[g_], position_index=gn.position_index.to_numpy()[g_],
                           strand=gn.strand.to_numpy()[g_], length=gn.length.to_numpy()[g_]))
    kos = A.set_index("gid").ko
    df["own_ko"] = [";".join(kos.get(x, [])) if hasko[x] else "" for x in g_]
    df["proposed_label_positional_homology"] = [vocab[v] for v in Out.lab]
    df["proposed_label_name"] = [nm.get(vocab[v], "") for v in Out.lab]
    df["proposed_label_kind"] = ["pfam" if vocab[v].startswith("PF") else "ko" for v in Out.lab]
    df["votes_for"] = Out.votes_for.to_numpy()
    df["voters_annotated"] = Out.voters_annotated.to_numpy()
    df["vote_share"] = np.round(df.votes_for / df.voters_annotated, 4)
    df["supporting_genomes"] = Out.supporting_genomes.to_numpy()
    df["supporting_groups"] = Out.supporting_groups.to_numpy()
    df["runner_up_label"] = [vocab[v] if v >= 0 else "" for v in Out.ru]
    df["runner_up_name"] = [nm.get(vocab[v], "") if v >= 0 else "" for v in Out.ru]
    df["runner_up_votes"] = Out.ru_votes.to_numpy()
    df["tied_keys"] = Out.tied.to_numpy()
    sZ, eZ = G["LOCI"]["Z"]
    loc = []
    for x in g_:
        lo = np.searchsorted(sZ, G["cstart"][G["cid"][x]], "left")
        hi = np.searchsorted(sZ, x, "right")
        ss, ee = sZ[lo:hi], eZ[lo:hi]
        m = ee >= x
        j = int(np.argmin(ee[m] - ss[m]))
        loc.append((int(ss[m][j]), int(ee[m][j])))
    df["locus_id"] = [f"Z_L4bp_t0.02:{s_}-{e_}" for s_, e_ in loc]
    df["locus_size"] = [e_ - s_ + 1 for s_, e_ in loc]
    df["n_bonds"] = Out.n_bonds.to_numpy()
    df["max_bond_lambda"] = np.round(Out.max_bond_lambda.to_numpy(), 2)
    if "A" in arms:
        al = [vocab[v] if v >= 0 else "" for v in Out.a_lab]
        df["alignment_label"] = [x if c == "other" else "" for x, c in zip(al, Out.a_cls)]
        df["alignment_lists_same_label"] = np.where(Out.a_list_len.to_numpy() == 0, "no list",
                                                    np.where(Out.a_cls.to_numpy() != "other", "no call",
                                                             np.where(df.alignment_label == df.proposed_label_positional_homology, "yes", "no")))
    else:
        df["alignment_label"] = ""
        df["alignment_lists_same_label"] = "no alignment lists"
    cell_rate = {(r.vbin, r.sbin): (r.agree, r.n) for r in cells.itertuples()}
    vb, sb = df.votes_for.map(PR.vbin), df.vote_share.map(PR.sbin)
    df["confidence_bin"] = [f"{v}|{s}" + ("|sparse" if cell_rate.get((v, s), (0, 0))[1] < 30 else "") for v, s in zip(vb, sb)]
    df["calibration_agreement"] = [round(float(cell_rate.get((v, s), (np.nan, 0))[0]), 4) for v, s in zip(vb, sb)]

    def ko_agree(kostr, lab_):
        if not kostr:
            return "no_ko"
        st = set()
        for k in kostr.split(";"):
            if lab_ == k:
                st.add("agree")
            elif k in proxy:
                st.add("agree" if lab_ in proxy[k] else "disagree")
            elif not lab_.startswith("PF"):
                st.add("disagree")
            else:
                st.add("unmapped")
        return "agree" if "agree" in st else ("disagree" if "disagree" in st else "unmapped")
    df["own_ko_agrees"] = [ko_agree(k, l_) for k, l_ in zip(df.own_ko, df.proposed_label_positional_homology)]
    df["basis"] = PR.BASIS
    # added columns
    df["support"] = np.round(df.votes_for / df.supporting_genomes, 4)
    lc = pd.Series(G["LABCOUNT"], index=vocab)
    df["label_annotated_count"] = df.proposed_label_positional_homology.map(lc).astype(int).to_numpy()
    nlab = df.proposed_label_positional_homology.map(df.proposed_label_positional_homology.value_counts())
    df["label_amplification"] = np.round(nlab / df.label_annotated_count.where(df.label_annotated_count > 0), 3)
    df["unannotated_supporters"] = df.supporting_genomes - df.voters_annotated
    df["n_carriers"] = Out.n_carriers.to_numpy()
    df["n_carriers_ko"] = Out.n_carriers_ko.to_numpy()
    df["partner_ko"] = Out.partner_ko.to_numpy()
    kn = pd.read_csv(RD + "/dpann_leads/pass_port/data/ko_names.tsv", sep="\t", header=None, names=["ko", "name"]).set_index("ko").name
    df["partner_ko_name"] = df.partner_ko.map(kn).fillna("")
    df["partner_ko_n"] = Out.partner_ko_n.to_numpy()
    df["partner_ko_share"] = np.round(df.partner_ko_n / df.n_carriers_ko.where(df.n_carriers_ko > 0), 4)
    df["partner_ko_second"] = Out.partner_ko_second.to_numpy()
    df["partner_ko_second_n"] = Out.partner_ko_second_n.to_numpy()
    df["partner_ko_counts"] = Out.partner_ko_counts.to_numpy()
    df["partner_function_strong"] = (df.partner_ko_n >= 3) & (df.partner_ko_share >= 0.70)
    df["list_type"] = df.alignment_lists_same_label.map({"yes": "both", "no": "disagree", "no call": "embedding-only",
                                                         "no list": "embedding-only", "no alignment lists": "no alignment lists"})
    df = df.sort_values("gid").reset_index(drop=True)
    df.to_parquet(f"{OUT}/{a.collection}_slot_rescue_pass.parquet", index=False)
    counts["called"] = int(len(df))
    counts["partner_function_strong"] = int(df.partner_function_strong.sum())
    counts["list_type"] = df.list_type.value_counts().to_dict()
    run["counts"] = counts
    # DPANN regression
    if cfg.get("regression") and not a.smoke:
        ref = pd.read_parquet(P_(cfg["regression"]["pass"]))
        m = ref.merge(df, on="protein_id", how="outer", suffixes=("_ref", "_new"), indicator=True)
        reg = dict(rows_ref=len(ref), rows_new=len(df), both=int((m._merge == "both").sum()))
        mism = {}
        for c in ref.columns:
            if c == "protein_id":
                continue
            x, y = m[c + "_ref"], m[c + "_new"]
            if x.dtype.kind == "f":
                bad_c = ~(np.isclose(x.astype(float), y.astype(float), equal_nan=True, rtol=0, atol=1e-6))
            else:
                bad_c = x.astype(str) != y.astype(str)
            mism[c] = int(bad_c.sum())
        reg["column_mismatches"] = mism
        pk = pd.read_parquet(P_(cfg["regression"]["partner_ko"]))
        mm = pk.merge(df, on="protein_id", how="inner")
        reg["partner_function_rows"] = len(mm)
        reg["partner_function_mismatches"] = dict(
            ko=int((mm.ko_top.fillna("") != mm.partner_ko.fillna("")).sum()), ko_n=int((mm.ko_top_n != mm.partner_ko_n).sum()),
            carriers=int((mm.n_carriers_x != mm.n_carriers_y).sum()) if "n_carriers_x" in mm else int((mm.n_carriers != mm.n_carriers).sum()),
            carriers_ko=int((mm.n_carriers_ko_x != mm.n_carriers_ko_y).sum()) if "n_carriers_ko_x" in mm else 0)
        reg["passed"] = bool(reg["rows_ref"] == reg["rows_new"] == reg["both"] and not any(mism.values())
                             and not any(reg["partner_function_mismatches"].values()) and reg["partner_function_rows"] == len(ref))
        run["regression"] = reg
        log("REGRESSION", json.dumps(reg))
    run["timing"]["total"] = round(time.time() - T0, 1)
    json.dump(run, open(f"{OUT}/run.json", "w"), indent=1, default=str)
    pool.close()
    log("done", json.dumps(counts))
    if run.get("regression") and not run["regression"]["passed"]:
        raise SystemExit("REGRESSION FAILED")


if __name__ == "__main__":
    main()
