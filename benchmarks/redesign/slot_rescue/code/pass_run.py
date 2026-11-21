#!/usr/bin/env python
"""slot_rescue whole-DPANN pass (predeclared_pass.json): the tested Z slot call on every gene without a Pfam row.

Code path of the test: score.py z_chunk / a_chunk (lists), support(), vote(), right_mask(), same label vocabulary.
Order: (a) regression on the 19,146 test rows (a1 identity with calls.parquet, a2 empty right-family set) -> STOP on
failure; candidates; Z pass; (b) calibration on 50,000 bonded Pfam genes; bond gate -> STOP on any mismatch; A support
column for called genes; (c) KO cross-check; tables.
Writes $RD/slot_rescue/pass/: dpann_slot_rescue_pass.parquet, counts.tsv, labels_top50.tsv, breadth.tsv,
calibration_cells.tsv, calibration_length.tsv, ko_agreement.tsv, regression.json, run.json (STATUS by the wrapper).

usage: pass_run.py --threads T [--smoke N]
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import score as S

G = S.G
PASS = S.SRD + "/pass"
ZOUT, AOUT = "L4bp_t0.02", "L4b_t0.02"
BASIS = ("hypothesis: plurality Pfam/KO key among the best-ranked embedding-list partners (one per genome, other genome "
         "groups only) that lie 1-3 genes from a list partner of a bonded neighbor (ELSA slot call, slot_rescue test); "
         "not a curated caller call")
VB = [(2, 2, "v2"), (3, 4, "v3-4"), (5, 9, "v5-9"), (10, 10 ** 9, "v10+")]
SB = [(0.0, 0.6, "s<0.60", False), (0.6, 0.8, "s0.60-0.80", True), (0.8, 1.01, "s>0.80", False)]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def vbin(v):
    return next(nm for lo, hi, nm in VB if lo <= v <= hi)


def sbin(s):
    if s < 0.6:
        return "s<0.60"
    if s <= 0.8:
        return "s0.60-0.80"
    return "s>0.80"


def lbin(L):
    return "<80" if L < 80 else ("80-150" if L <= 150 else ">150")


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


def slot(arm, g, Rids, cache):
    P = getl(arm, g, cache)
    bp, bv = G["BPTR"][arm], G["BVAL"][arm]
    BK, BD, BN, BNG, BL = G["BONDS"][arm]
    best, nb, bad, lam = {}, 0, 0, 0.0
    for b in bv[bp[g]:bp[g + 1]]:
        j, d = int(BK[b]), int(BD[b])
        other = j + d if j == g else j
        ns, ngs, ug, pos = S.support(P, getl(arm, other, cache))
        if ns != int(BN[b]) - 1 or ngs != int(BNG[b]) - 1:
            bad += 1
        nb += 1
        lam = max(lam, float(BL[b]))
        for G_, p_ in zip(ug.tolist(), pos.tolist()):
            if G_ not in best or p_ < best[G_]:
                best[G_] = p_
    Pp = np.zeros(0, np.int64)
    if nb and best:
        pos = np.array(sorted(best.values()), np.int64)
        Pp = P[pos]
        right = S.right_mask(P, Rids)[pos] if Rids is not None else np.zeros(len(Pp), bool)
        c, lab, cR, co, na = S.vote(Pp, right)
    elif nb:
        c, lab, cR, co, na = "none_nosupport", -1, 0, 0, 0
    else:
        c, lab, cR, co, na = "none_nobond", -1, 0, 0, 0
    inl = bool(G["INLOC"][arm][g])
    cls = c if inl else ("none_notinlocus" if nb else c)
    ngrp = len(np.unique(G["grp"][np.array(list(best.keys()), np.int64)])) if best else 0
    return dict(cls=cls, lab=int(lab), cR=cR, co=co, na=na, n_support=len(best), n_groups=ngrp, nb=nb, bad=bad, lam=lam,
                Pp=Pp, listlen=len(P))


def runner_up(Pp, kw):
    keys, own = S.expand(Pp, G["KEYP"], G["KEYV"])
    if len(keys) == 0:
        return -1, 0, []
    nl = G["nlab"]
    pair = np.unique(own.astype(np.int64) * nl + keys)
    k, c = np.unique(pair % nl, return_counts=True)
    cw = int(c[k == kw][0]) if (k == kw).any() else 0
    tied = [int(x) for x in k[(c == cw) & (k != kw)]] if cw else []
    carriers = np.unique(pair[pair % nl == kw] // nl)
    rest = pair[~np.isin(pair // nl, carriers)]
    if len(rest) == 0:
        return -1, 0, tied
    k2, c2 = np.unique(rest % nl, return_counts=True)
    m = int(c2.max())
    return int(k2[c2 == m].min()), m, tied


def task(job):
    kind, items = job
    cache, out = {}, []
    if kind == "reg":
        for row_id, g, Rids in items:
            r = slot("Z", g, Rids, cache)
            f = slot("Z", g, np.zeros(0, np.int64), cache)
            out.append((row_id, r["cls"], r["lab"], r["n_support"], r["nb"], r["bad"], f["cls"], f["lab"]))
    elif kind == "Z":
        for g in items:
            r = slot("Z", g, None, cache)
            if r["cls"] == "other":
                ru, rc, tied = runner_up(r["Pp"], r["lab"])
                out.append((g, r["cls"], r["lab"], r["co"], r["na"], r["n_support"], r["n_groups"], r["nb"], r["bad"], r["lam"],
                            ru, rc, ";".join(map(str, tied)), r["listlen"]))
            else:
                out.append((g, r["cls"], -1, 0, r["na"], r["n_support"], r["n_groups"], r["nb"], r["bad"], r["lam"], -1, 0, "",
                            r["listlen"]))
    elif kind == "cal":
        for g in items:
            r = slot("Z", g, None, cache)
            own = set(G["LABV"][G["LABP"][g]:G["LABP"][g + 1]].tolist()) | set(G["KEYV"][G["KEYP"][g]:G["KEYP"][g + 1]].tolist())
            ownpf = {x for x in own if G["ISPF"][x]} & set(G["LABV"][G["LABP"][g]:G["LABP"][g + 1]].tolist())
            out.append((g, r["cls"], r["lab"], r["co"], r["na"], r["nb"], r["bad"], int(r["lab"] in own), int(r["lab"] in ownpf)))
    elif kind == "A":
        for g in items:
            r = slot("A", g, None, cache)
            out.append((g, r["cls"], r["lab"], r["listlen"], r["nb"], r["bad"]))
    return out


# ------------------------------------------------------------------ parent
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    th = a.threads
    os.makedirs(PASS, exist_ok=True)
    run = dict(threads=th, smoke=a.smoke, timing={})
    gn = pd.read_parquet(S.GENES, columns=["gid", "protein_id", "genome", "genome_code", "contig", "contig_code", "position_index",
                                             "strand", "length"])
    assert (gn.gid.to_numpy() == np.arange(len(gn))).all()
    n = len(gn)
    G["n"] = n
    G["gc"] = gn.genome_code.to_numpy(np.int64)
    G["cid"] = gn.contig_code.to_numpy(np.int64)
    G["cstart"] = np.r_[0, np.flatnonzero(np.diff(G["cid"])) + 1].astype(np.int64)
    G["g_start"] = np.r_[0, np.flatnonzero(np.diff(G["gc"])) + 1].astype(np.int64)
    lm = pd.read_parquet(S.GROUPS, columns=["genome_code", "group"]).sort_values("genome_code")
    assert np.array_equal(lm.genome_code.to_numpy(), np.arange(len(G["g_start"])))
    G["lin_of_genome"] = pd.factorize(lm.group.to_numpy())[0].astype(np.int64)
    G["grp"] = G["lin_of_genome"]
    G["lin"] = G["lin_of_genome"][G["gc"]]
    uq = np.load(S.EMB + "/uid_of_row.npy").astype(np.int64)
    G["uq"] = uq
    pid2gid = pd.Series(np.arange(n), index=gn.protein_id.to_numpy())

    # labels: identical vocabulary to score.py (annotations + KO proxies + right-family labels of the test rows)
    T = pd.read_csv(S.SRD + "/data/truth.tsv", sep="\t", keep_default_na=False)
    C = pd.read_csv(S.SRD + "/data/controls.tsv", sep="\t", keep_default_na=False)
    Qr = pd.concat([T[["protein_id", "R_pfam", "R_ko"]], C[["protein_id", "R_pfam", "R_ko"]]], ignore_index=True)
    Qr["row_id"] = np.arange(len(Qr))
    Qr["gid"] = Qr.protein_id.map(pid2gid).astype(np.int64)
    A = pd.read_parquet(S.SRD + "/data/annot.parquet")
    kp = pd.read_csv(S.SRD + "/data/ko_proxy.tsv", sep="\t")
    proxy = kp.groupby("ko").pf.apply(sorted).to_dict()
    vocab = sorted({x for v in A.pfam for x in v} | {x for v in A.ko for x in v} | set(kp.pf))
    for s_ in list(Qr.R_pfam) + list(Qr.R_ko):
        vocab += [x for x in str(s_).split(";") if x]
    vocab = sorted(set(vocab))
    lid = {x: i for i, x in enumerate(vocab)}
    G["nlab"] = len(vocab)
    G["ISPF"] = np.array([v.startswith("PF") for v in vocab])
    A["gid"] = A.protein_id.map(pid2gid).astype(np.int64)
    ep = A[["gid", "pfam"]].explode("pfam").dropna().rename(columns={"pfam": "lab"})
    ek = A[["gid", "ko"]].explode("ko").dropna().rename(columns={"ko": "lab"})
    lab = pd.concat([ep, ek], ignore_index=True)
    lab["i"] = lab.lab.map(lid).astype(np.int64)
    lab = lab.drop_duplicates(["gid", "i"]).sort_values(["gid", "i"], kind="stable")
    ek2 = ek.copy()
    ek2["key"] = ek2.lab.map(lambda k: proxy.get(k, [k]))
    kk = ek2[["gid", "key"]].explode("key").rename(columns={"key": "lab"})
    key = pd.concat([ep, kk], ignore_index=True)
    key["i"] = key.lab.map(lid).astype(np.int64)
    key = key.drop_duplicates(["gid", "i"]).sort_values(["gid", "i"], kind="stable")
    G["LABP"] = np.r_[0, np.cumsum(np.bincount(lab.gid.to_numpy(), minlength=n))].astype(np.int64)
    G["KEYP"] = np.r_[0, np.cumsum(np.bincount(key.gid.to_numpy(), minlength=n))].astype(np.int64)
    G["LABV"] = lab.i.to_numpy(np.int64)
    G["KEYV"] = key.i.to_numpy(np.int64)
    Qr["Rids"] = [np.array(sorted({lid[x] for x in (str(p).split(";") + str(k).split(";")) if x}), np.int64)
                  for p, k in zip(Qr.R_pfam, Qr.R_ko)]
    log("labels", len(vocab))

    # lists, bonds, loci
    T_ = S.RD + "/calib/table/"
    G["zoff"] = np.r_[0, np.cumsum(np.load(T_ + "counts.npy").astype(np.int64))]
    G["zG"] = np.memmap(T_ + "genome.bin", np.int16, "r")
    G["zL"] = np.memmap(T_ + "local.bin", np.int16, "r")
    G["zz"] = np.memmap(T_ + "z.bin", np.float16, "r")
    G["I"] = np.load(S.RD + "/vocab_lists/mm/aln__k1000__I.npy", mmap_mode="r")
    okq = uq >= 0
    G["members"] = np.flatnonzero(okq)[np.argsort(uq[okq], kind="stable")].astype(np.int64)
    G["mptr"] = np.r_[0, np.cumsum(np.bincount(uq[okq], minlength=G["I"].shape[0]))].astype(np.int64)
    G["BONDS"], G["BPTR"], G["BVAL"], G["INLOC"], G["LOCI"] = {}, {}, {}, {}, {}
    for arm, out in (("Z", ZOUT), ("A", AOUT)):
        z = np.load(f"{S.RD}/long_loci/nodes/dpann_{arm}/bonds_obs.npz")
        k, d = z["k"].astype(np.int64), z["d"].astype(np.int64)
        G["BONDS"][arm] = (k, d, z["n"], z["ng"], z["lam"])
        e = np.r_[k, k + d]
        bi = np.r_[np.arange(len(k)), np.arange(len(k))]
        o = np.argsort(e, kind="stable")
        G["BPTR"][arm] = np.r_[0, np.cumsum(np.bincount(e, minlength=n))].astype(np.int64)
        G["BVAL"][arm] = bi[o].astype(np.int64)
        L = pd.read_parquet(f"{S.RD}/long_loci/loci/dpann_{arm}/{out}.parquet").sort_values(["s", "e"])
        s, e2 = L.s.to_numpy(np.int64), L.e.to_numpy(np.int64)
        G["LOCI"][arm] = (s, e2)
        cov = np.zeros(n + 1, np.int64)
        np.add.at(cov, s, 1)
        np.add.at(cov, e2 + 1, -1)
        G["INLOC"][arm] = np.cumsum(cov)[:n] > 0
        run[f"bonds_{arm}"] = int(len(k))
        run[f"loci_{arm}_{out}"] = int(len(L))
    log("bonds and loci")

    pool = Pool(th)

    def pmap(kind, items, nchunk):
        parts = [x for x in np.array_split(np.arange(len(items)), nchunk) if len(x)]
        jobs = [(kind, [items[i] for i in p]) for p in parts]
        return [r for part in pool.map(task, jobs) for r in part]

    # ---- (a) regression
    t = time.time()
    reg_items = list(zip(Qr.row_id.tolist(), Qr.gid.tolist(), Qr.Rids.tolist()))
    if a.smoke:
        reg_items = reg_items[:a.smoke]
    R = pd.DataFrame(pmap("reg", reg_items, th * 8), columns=["row_id", "cls", "lab", "n_support", "nb", "bad", "free_cls", "free_lab"])
    R["lab"] = [vocab[v] if v >= 0 else "" for v in R.lab]
    R["free_lab"] = [vocab[v] if v >= 0 else "" for v in R.free_lab]
    cp = pd.read_parquet(S.SRD + "/results/calls.parquet", columns=["row_id", "arm", "slot_pri_class", "slotname_label", "slot_n_support", "n_bonds"])
    cp = cp[cp.arm == "Z"].merge(R, on="row_id")
    a1 = dict(rows=len(cp), class_mismatch=int((cp.slot_pri_class != cp.cls).sum()),
              label_mismatch=int((cp.slotname_label != cp.lab).sum()),
              support_mismatch=int((cp.slot_n_support != cp.n_support).sum()), bonds_mismatch=int((cp.n_bonds != cp.nb).sum()),
              bond_gate_bad=int(R.bad.sum()))
    rs = json.load(open(S.SRD + "/data/regression_sets.json"))
    core = set(rs["core"])
    Rset = {r: set(x for x in (p.split(";") + k.split(";")) if x) for r, p, k in zip(Qr.row_id, Qr.R_pfam, Qr.R_ko)}
    Ri = R.set_index("row_id")
    vr = [r for r in rs["verified_row_ids"] if r in Ri.index]
    cr = [r for r in rs["control_row_ids"] if r in Ri.index]
    a1["verified_right"] = int(sum(Ri.at[r, "cls"] == "right" for r in vr))
    a1["controls_false"] = int(sum(Ri.at[r, "cls"] == "right" or (Ri.at[r, "cls"] == "other" and Ri.at[r, "lab"] in core) for r in cr))
    a2 = dict(verified_right=int(sum(Ri.at[r, "free_cls"] == "other" and Ri.at[r, "free_lab"] in Rset[r] for r in vr)),
              controls_false=int(sum(Ri.at[r, "free_cls"] == "other" and Ri.at[r, "free_lab"] in core for r in cr)),
              verified_n=len(vr), controls_n=len(cr),
              verified_changed_vs_tested=int(sum((Ri.at[r, "cls"] == "right") != (Ri.at[r, "free_cls"] == "other" and Ri.at[r, "free_lab"] in Rset[r]) for r in vr)))
    a1["passed"] = bool(a1["class_mismatch"] == a1["label_mismatch"] == a1["support_mismatch"] == a1["bonds_mismatch"] == a1["bond_gate_bad"] == 0
                    and (a.smoke or (a1["verified_right"] == rs["expected"]["verified_right"] and a1["controls_false"] == rs["expected"]["controls_false"])))
    a2["passed"] = bool(a.smoke or (2330 <= a2["verified_right"] <= 2373 and a2["controls_false"] <= 20))
    run["timing"]["regression"] = round(time.time() - t, 1)
    json.dump(dict(a1=a1, a2=a2), open(f"{PASS}/regression.json", "w"), indent=1)
    log("regression", a1, a2)
    if not (a1["passed"] and a2["passed"]):
        open(f"{PASS}/stop_reason.txt", "w").write(json.dumps(dict(a1=a1, a2=a2)) + "\n")
        raise SystemExit("STOP: regression check failed")

    # ---- candidates and the Z pass
    t = time.time()
    nameonly = set(pd.read_csv(S.SRD + "/data/pfam_nameonly.tsv", sep="\t").protein_id)
    haspf = np.zeros(n, bool)
    haspf[A.gid[A.pfam.str.len() > 0].to_numpy()] = True
    haspf[pid2gid[list(nameonly)].to_numpy()] = True
    hasko = np.zeros(n, bool)
    hasko[A.gid[A.ko.str.len() > 0].to_numpy()] = True
    cand = np.flatnonzero(~haspf)
    bonded = np.diff(G["BPTR"]["Z"]) > 0
    elig = cand[bonded[cand] & G["INLOC"]["Z"][cand]]
    if a.smoke:
        elig = elig[:a.smoke]
    counts = dict(candidates=int(len(cand)), candidates_with_own_ko=int(hasko[cand].sum()),
                  candidates_with_embedding=int((uq[cand] >= 0).sum()), z_bonded=int(bonded[cand].sum()),
                  in_z_locus=int(G["INLOC"]["Z"][cand].sum()), eligible_bonded_and_in_locus=int(len(cand[bonded[cand] & G["INLOC"]["Z"][cand]])))
    Zr = pd.DataFrame(pmap("Z", elig.tolist(), th * 32), columns=["gid", "cls", "lab", "votes_for", "voters_annotated", "supporting_genomes",
                                                                  "supporting_groups", "n_bonds", "bad", "max_bond_lambda", "ru", "ru_votes",
                                                                  "tied", "z_list_len"])
    run["timing"]["z_pass"] = round(time.time() - t, 1)
    counts.update({f"eligible_{k}": int(v) for k, v in Zr.cls.value_counts().items()})
    log("Z pass", len(Zr), Zr.cls.value_counts().to_dict())

    # ---- (b) calibration
    t = time.time()
    calpool = np.flatnonzero((G["LABP"][1:] - G["LABP"][:-1] > 0) & np.isin(np.arange(n), A.gid[A.pfam.str.len() > 0].to_numpy()) & bonded)
    rng = np.random.default_rng(20261120)
    cal = np.sort(rng.choice(calpool, size=min(50000, len(calpool)), replace=False))
    if a.smoke:
        cal = cal[:a.smoke]
    Cr = pd.DataFrame(pmap("cal", cal.tolist(), th * 8), columns=["gid", "cls", "lab", "votes_for", "voters_annotated", "nb", "bad", "agree", "agree_pfam"])
    run["timing"]["calibration"] = round(time.time() - t, 1)
    run["calibration_pool"] = int(len(calpool))
    bad_total = int(Zr.bad.sum() + Cr.bad.sum())
    if bad_total:
        open(f"{PASS}/stop_reason.txt", "w").write(f"bond gate mismatches {bad_total}\n")
        raise SystemExit("STOP: bond gate")
    Cc = Cr[Cr.cls == "other"].copy()
    Cc["share"] = Cc.votes_for / Cc.voters_annotated
    Cc["vbin"] = Cc.votes_for.map(vbin)
    Cc["sbin"] = Cc.share.map(sbin)
    Cc["length"] = gn.length.to_numpy()[Cc.gid.to_numpy()]
    Cc["lbin"] = Cc.length.map(lbin)
    cells = Cc.groupby(["vbin", "sbin"]).agg(n=("agree", "size"), agree=("agree", "mean"), agree_pfam=("agree_pfam", "mean")).reset_index()
    cells.to_csv(f"{PASS}/calibration_cells.tsv", sep="\t", index=False)
    Cc.groupby("lbin").agg(n=("agree", "size"), agree=("agree", "mean"), agree_pfam=("agree_pfam", "mean")).reset_index().to_csv(
        f"{PASS}/calibration_length.tsv", sep="\t", index=False)
    calsum = dict(sample=int(len(Cr)), called=int(len(Cc)), classes=Cr.cls.value_counts().to_dict(),
                  agree=float(Cc.agree.mean()) if len(Cc) else None, agree_pfam=float(Cc.agree_pfam.mean()) if len(Cc) else None)
    log("calibration", calsum)

    # ---- output rows and the A support column
    t = time.time()
    Out = Zr[Zr.cls == "other"].copy()
    Ar = pd.DataFrame(pmap("A", Out.gid.tolist(), th * 32), columns=["gid", "a_cls", "a_lab", "a_list_len", "a_nb", "a_bad"])
    run["timing"]["a_support"] = round(time.time() - t, 1)
    if int(Ar.a_bad.sum()):
        open(f"{PASS}/stop_reason.txt", "w").write(f"A bond gate mismatches {int(Ar.a_bad.sum())}\n")
        raise SystemExit("STOP: A bond gate")
    Out = Out.merge(Ar, on="gid")
    names = pd.read_csv(S.SRD + "/data/labels.tsv", sep="\t", keep_default_na=False)
    nm = dict(zip(names.label, names.name))
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
    df["tied_keys"] = [";".join(vocab[int(x)] for x in s.split(";")) if s else "" for s in Out.tied]
    loc = []
    sZ, eZ = G["LOCI"]["Z"]
    for x in g_:
        lo = np.searchsorted(sZ, G["cstart"][G["cid"][x]], "left")
        hi = np.searchsorted(sZ, x, "right")
        ss, ee = sZ[lo:hi], eZ[lo:hi]
        m = ee >= x
        j = int(np.argmin(ee[m] - ss[m]))
        loc.append((int(ss[m][j]), int(ee[m][j])))
    df["locus_id"] = [f"Z_{ZOUT}:{s_}-{e_}" for s_, e_ in loc]
    df["locus_size"] = [e_ - s_ + 1 for s_, e_ in loc]
    df["n_bonds"] = Out.n_bonds.to_numpy()
    df["max_bond_lambda"] = np.round(Out.max_bond_lambda.to_numpy(), 2)
    al = [vocab[v] if v >= 0 else "" for v in Out.a_lab]
    df["alignment_label"] = [x if c == "other" else "" for x, c in zip(al, Out.a_cls)]
    df["alignment_lists_same_label"] = np.where(Out.a_list_len.to_numpy() == 0, "no list",
                                                np.where(Out.a_cls.to_numpy() != "other", "no call",
                                                         np.where(df.alignment_label == df.proposed_label_positional_homology, "yes", "no")))
    cell_rate = {(r.vbin, r.sbin): (r.agree, r.n) for r in cells.itertuples()}
    vb = df.votes_for.map(vbin)
    sb = df.vote_share.map(sbin)
    df["confidence_bin"] = [f"{v}|{s}" + ("|sparse" if cell_rate.get((v, s), (0, 0))[1] < 30 else "") for v, s in zip(vb, sb)]
    df["calibration_agreement"] = [round(float(cell_rate.get((v, s), (np.nan, 0))[0]), 4) for v, s in zip(vb, sb)]

    # ---- (c) KO cross-check
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
    df["basis"] = BASIS
    df = df.sort_values("gid").reset_index(drop=True)
    df.to_parquet(f"{PASS}/dpann_slot_rescue_pass.parquet", index=False)

    counts["called"] = int(len(df))
    counts["called_with_own_ko"] = int((df.own_ko != "").sum())
    for k, v in df.alignment_lists_same_label.value_counts().items():
        counts[f"called_alignment_{k.replace(' ', '_')}"] = int(v)
    ka = df.own_ko_agrees.value_counts()
    kt = pd.DataFrame([dict(class_=k, n=int(v)) for k, v in ka.items()])
    kt.to_csv(f"{PASS}/ko_agreement.tsv", sep="\t", index=False)
    pd.DataFrame([dict(key=k, value=v) for k, v in counts.items()]).to_csv(f"{PASS}/counts.tsv", sep="\t", index=False)
    top = df.groupby(["proposed_label_positional_homology", "proposed_label_name"]).size().sort_values(ascending=False).head(50)
    top.rename("n").reset_index().to_csv(f"{PASS}/labels_top50.tsv", sep="\t", index=False)
    bins = pd.cut(df.supporting_groups, [0, 1, 4, 9, 49, 10 ** 9], labels=["1", "2-4", "5-9", "10-49", ">=50"])
    br = df.groupby(bins, observed=False).agg(n=("gid", "size"), median_votes=("votes_for", "median"),
                                              calib_agreement_mean=("calibration_agreement", "mean")).reset_index()
    br.to_csv(f"{PASS}/breadth.tsv", sep="\t", index=False)
    df.groupby("confidence_bin").size().rename("n").reset_index().to_csv(f"{PASS}/confidence_bins.tsv", sep="\t", index=False)
    run["timing"]["total"] = round(time.time() - T0, 1)
    run.update(counts=counts, calibration=calsum, ko_agreement=ka.to_dict())
    json.dump(run, open(f"{PASS}/run.json", "w"), indent=1, default=str)
    pool.close()
    log("done", counts, run["timing"])


if __name__ == "__main__":
    main()
