#!/usr/bin/env python
"""slot_rescue scoring (biotite; predeclared.json): list, slot and combined calls for truth, negative and control genes.

Lists are reconstructed for the query genes and every gene within 3 positions of them:
  Z  ws_calib table, z >= 10.6, own group excluded, ranked by z (vl.z_lists)
  A  MMseqs2 kNN (aln__k1000), identical sequences first, own group excluded, depth 1,000 (vl.aln_lists / expand_knn)
Gate: support of every stored bond touching a query gene, recomputed from these lists, equals stored n - 1 and ng - 1.
Outputs (under $RD/slot_rescue/results): calls.parquet (one row per query row x arm), gate.json, run.json; STATUS is
written by the wrapper.

usage: score.py --threads T [--smoke N]
"""
import argparse
import json
import os
import resource
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
SRD = RD + "/slot_rescue"
EMB = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111"
GROUPS = RD + "/lineagefree/groups/emb_hom_average_t0.75.parquet"
GENES = RD + "/ws_dpann/package/universe/genes.parquet"
ZCUT, KA, WG = 10.6, 1000, 3
OUTS = {"Z": ("L4bp_t0.02", "L4b_t0.02"), "A": ("L4b_t0.02", "L4bp_t0.02")}   # (primary, secondary)
OFFS = np.array([-3, -2, -1, 1, 2, 3], np.int64)
G = {}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def rss():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)


# ------------------------------------------------------------------ list reconstruction (worker functions)
def z_chunk(genes):
    cnt_off, zG, zL, zz = G["zoff"], G["zG"], G["zL"], G["zz"]
    uq, lin, lin_g, g_start = G["uq"], G["lin"], G["lin_of_genome"], G["g_start"]
    ptr, out = [0], []
    for g in genes:
        u = uq[g]
        if u >= 0:
            s, e = cnt_off[u], cnt_off[u + 1]
            z = np.asarray(zz[s:e]).astype(np.float32)
            gg = np.asarray(zG[s:e]).astype(np.int64)
            keep = np.flatnonzero((z >= ZCUT) & (lin_g[gg] != lin[g]))
            o = keep[np.argsort(-z[keep], kind="mergesort")]
            p = g_start[gg[o]] + np.asarray(zL[s:e]).astype(np.int64)[o]
            out.append(p.astype(np.int32))
            ptr.append(ptr[-1] + len(p))
        else:
            ptr.append(ptr[-1])
    return np.array(ptr[1:], np.int64), (np.concatenate(out) if out else np.zeros(0, np.int32))


def a_chunk(genes):
    I, mptr, members, uq, lin = G["I"], G["mptr"], G["members"], G["uq"], G["lin"]
    ptr, out = [0], []
    for a in genes:
        U = uq[a]
        if U < 0:
            ptr.append(ptr[-1])
            continue
        row = np.asarray(I[U])
        neg = np.flatnonzero(row < 0)
        if len(neg):
            row = row[:neg[0]]
        row = row[row != U]
        V = np.r_[U, row].astype(np.int64)
        ln = mptr[V + 1] - mptr[V]
        tot = int(ln.sum())
        base = np.repeat(mptr[V] - np.r_[0, np.cumsum(ln)[:-1]], ln)
        b = members[base + np.arange(tot)]
        b = b[lin[b] != lin[a]][:KA]
        out.append(b.astype(np.int32))
        ptr.append(ptr[-1] + len(b))
    return np.array(ptr[1:], np.int64), (np.concatenate(out) if out else np.zeros(0, np.int32))


def build_lists(arm, need, threads):
    fn = z_chunk if arm == "Z" else a_chunk
    chunks = np.array_split(need, max(1, threads * 8))
    with Pool(threads) as pool:
        res = pool.map(fn, chunks)
    ptr, vals, base = [np.zeros(1, np.int64)], [], 0
    for p, v in res:
        ptr.append(p + base)
        base += len(v)
        vals.append(v)
    return np.concatenate(ptr), np.concatenate(vals)


def get_list(arm, g):
    i = np.searchsorted(G["need"], g)
    if i >= len(G["need"]) or G["need"][i] != g:
        raise KeyError(g)
    p = G["LP"][arm]
    return G["LV"][arm][p[i]:p[i + 1]].astype(np.int64)


def marked_around(partners):
    if len(partners) == 0:
        return np.zeros(0, np.int64)
    n, cid = G["n"], G["cid"]
    x = partners[:, None] + OFFS[None, :]
    ok = (x >= 0) & (x < n)
    xc = np.where(ok, x, 0)
    ok &= cid[xc] == cid[partners][:, None]
    return np.unique(x[ok])


def support(qlist, olist):
    """Genomes where a partner of the query gene lies 1-3 positions from a partner of the other gene (pass_count rule);
    per genome the best-ranked query partner."""
    if len(qlist) == 0 or len(olist) == 0:
        return 0, 0, np.zeros(0, np.int64), np.zeros(0, np.int64)
    m = marked_around(olist)
    idx = np.flatnonzero(np.isin(qlist, m))
    if len(idx) == 0:
        return 0, 0, np.zeros(0, np.int64), np.zeros(0, np.int64)
    gen = G["gc"][qlist[idx]]
    ug, first = np.unique(gen, return_index=True)
    return len(ug), len(np.unique(G["grp"][ug])), ug, idx[first]


# ------------------------------------------------------------------ labels and votes
def expand(P, ptr, val):
    st, en = ptr[P], ptr[P + 1]
    c = en - st
    own = np.repeat(np.arange(len(P)), c)
    idx = np.repeat(st - np.r_[0, np.cumsum(c)[:-1]], c) + np.arange(int(c.sum()))
    return val[idx], own


def right_mask(P, Rids):
    labs, own = expand(P, G["LABP"], G["LABV"])
    r = np.zeros(len(P), bool)
    if len(Rids):
        r[own[np.isin(labs, Rids)]] = True
    return r


def vote(P, right):
    """Returns (class, label id or -1, cR, c_other, n_annotated)."""
    if len(P) == 0:
        return "none_empty", -1, 0, 0, 0
    ann = (G["LABP"][P + 1] - G["LABP"][P]) > 0
    na = int(ann.sum())
    if na == 0:
        return "none_unannotated", -1, 0, 0, 0
    cR = int(right.sum())
    keys, own = expand(P[~right], G["KEYP"], G["KEYV"])
    if len(keys):
        pair = np.unique(own.astype(np.int64) * G["nlab"] + keys)
        k, c = np.unique(pair % G["nlab"], return_counts=True)
        cmax = int(c.max())
        kbest = int(k[c == cmax].min())        # keys tied at the maximum (co-occurring domains): lowest label id
    else:
        cmax, kbest = 0, -1
    if cR > cmax and cR >= 2:
        return "right", -1, cR, cmax, na
    if cmax > cR and cmax >= 2:
        return "other", kbest, cR, cmax, na
    if cR == cmax and cR >= 2:
        return "none_tie", -1, cR, cmax, na
    return "none_lt2", -1, cR, cmax, na


# ------------------------------------------------------------------ per query row
def loci_hits(arm, which, g, flanks):
    s, e = G["LOCI"][(arm, which)]
    cs = G["cstart"][G["cid"][g]]
    lo = np.searchsorted(s, cs, "left")
    hi = np.searchsorted(s, g, "right")
    ss, ee = s[lo:hi], e[lo:hi]
    m = ee >= g
    if not m.any():
        return 0, 0, 0, 0
    ss, ee = ss[m], ee[m]
    size = int((ee - ss + 1).min())
    if len(flanks):
        fa = np.ones(len(ss), bool)
        fy = np.zeros(len(ss), bool)
        for f in flanks:
            inn = (ss <= f) & (ee >= f)
            fa &= inn
            fy |= inn
        return 1, size, int(fa.any()), int(fy.any())
    return 1, size, 0, 0


def work(rows):
    out = []
    cid, n = G["cid"], G["n"]
    for r in rows:
        g = int(r["gid"])
        Rids = r["Rids"]
        flanks = r["flank_gids"]
        rec_all = {}
        for arm in ("Z", "A"):
            P = get_list(arm, g)
            right = right_mask(P, Rids)
            rec = dict(row_id=r["row_id"], arm=arm, list_len=len(P))
            fr = np.flatnonzero(right)
            rec["first_right_rank"] = int(fr[0] + 1) if len(fr) else -1
            rec["n_right_in_list"] = int(len(fr))
            for k in (10, 50):
                c, lab, cR, co, na = vote(P[:k], right[:k])
                rec.update({f"k{k}_class": c, f"k{k}_label": lab, f"k{k}_cR": cR, f"k{k}_cother": co, f"k{k}_nann": na})
            # combined: partners slot-consistent with any gene within 3 positions
            nb = [g + d for d in OFFS if 0 <= g + d < n and cid[g + d] == cid[g]]
            ol = [get_list(arm, k) for k in nb]
            ol = np.concatenate(ol) if ol else np.zeros(0, np.int64)
            mk = marked_around(ol)
            msk = np.isin(P, mk) if len(P) else np.zeros(0, bool)
            Pr, rr = P[msk], right[msk]
            fr = np.flatnonzero(rr)
            rec["comb_len"] = int(len(Pr))
            rec["comb_first_right_rank"] = int(fr[0] + 1) if len(fr) else -1
            c, lab, cR, co, na = vote(Pr[:10], rr[:10])
            rec.update(comb_class=c, comb_label=lab, comb_cR=cR, comb_cother=co, comb_nann=na)
            # loci
            for which, nm in ((0, "pri"), (1, "sec")):
                inl, size, fall, fany = loci_hits(arm, which, g, flanks)
                rec.update({f"inlocus_{nm}": inl, f"locus_size_{nm}": size, f"flank_all_{nm}": fall, f"flank_any_{nm}": fany})
            # slot: stored bonds of g, supporting partners pooled per genome
            bi = G["BOND_IDX"][arm].get(g, [])
            BK, BD, BN, BNG, BL = G["BONDS"][arm]
            best = {}
            nb_ok, gate_bad, bonded_flank, lam_max = 0, 0, 0, 0.0
            for b in bi:
                j, d = int(BK[b]), int(BD[b])
                other = j + d if j == g else j
                ns, ngs, ug, pos = support(P, get_list(arm, other))
                if ns != int(BN[b]) - 1 or ngs != int(BNG[b]) - 1:
                    gate_bad += 1
                nb_ok += 1
                lam_max = max(lam_max, float(BL[b]))
                if other in flanks:
                    bonded_flank = 1
                for G_, p_ in zip(ug.tolist(), pos.tolist()):
                    if G_ not in best or p_ < best[G_]:
                        best[G_] = p_
            rec.update(n_bonds=nb_ok, gate_mismatch=gate_bad, bond_lam_max=lam_max, bonded_to_flank=bonded_flank,
                       slot_n_support=len(best))
            if nb_ok and best:
                pos = np.array(sorted(best.values()), np.int64)
                c, lab, cR, co, na = vote(P[pos], right[pos])
            elif nb_ok:
                c, lab, cR, co, na = "none_nosupport", -1, 0, 0, 0
            else:
                c, lab, cR, co, na = "none_nobond", -1, 0, 0, 0
            rec.update(slotname_class=c, slotname_label=lab, slot_cR=cR, slot_cother=co, slot_nann=na)
            for nm in ("pri", "sec"):
                rec[f"slot_{nm}_class"] = c if rec[f"inlocus_{nm}"] else ("none_notinlocus" if nb_ok else c)
            out.append(rec)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args()
    th = a.threads
    os.makedirs(SRD + "/results", exist_ok=True)
    tag = f"_smoke{a.smoke}" if a.smoke else ""
    run = dict(threads=th, smoke=a.smoke, timing={})

    g = pd.read_parquet(GENES, columns=["gid", "protein_id", "genome_code", "contig_code", "length"])
    assert (g.gid.to_numpy() == np.arange(len(g))).all()
    n = len(g)
    G["n"] = n
    G["gc"] = g.genome_code.to_numpy(np.int64)
    G["cid"] = g.contig_code.to_numpy(np.int64)
    assert (np.diff(G["gc"]) >= 0).all() and (np.diff(G["cid"]) >= 0).all()
    brk = np.flatnonzero(np.diff(G["cid"])) + 1
    c_start = np.r_[0, brk].astype(np.int64)
    G["cstart"] = c_start
    gchange = np.flatnonzero(np.diff(G["gc"])) + 1
    G["g_start"] = np.r_[0, gchange].astype(np.int64)
    lm = pd.read_parquet(GROUPS, columns=["genome_code", "group"]).sort_values("genome_code")
    assert np.array_equal(lm.genome_code.to_numpy(), np.arange(len(G["g_start"])))
    G["lin_of_genome"] = pd.factorize(lm.group.to_numpy())[0].astype(np.int64)
    G["grp"] = G["lin_of_genome"]
    G["lin"] = G["lin_of_genome"][G["gc"]]
    uq = np.load(EMB + "/uid_of_row.npy").astype(np.int64)
    assert len(uq) == n
    G["uq"] = uq
    pid2gid = pd.Series(np.arange(n), index=g.protein_id.to_numpy())
    log("universe", n, "genomes", len(G["g_start"]), "groups", int(G["grp"].max()) + 1)

    # ---- query rows
    T = pd.read_csv(SRD + "/data/truth.tsv", sep="\t", keep_default_na=False)
    C = pd.read_csv(SRD + "/data/controls.tsv", sep="\t", keep_default_na=False)
    T["role"] = "truth"
    C["role"] = "control"
    C["unit"] = C.matched_unit
    C["set"] = "control_" + C.matched_set
    C["flanks"] = ""
    Qr = pd.concat([T[["protein_id", "role", "set", "unit", "R_pfam", "R_ko", "broad_class", "flanks"]],
                    C[["protein_id", "role", "set", "unit", "R_pfam", "R_ko", "broad_class", "flanks"]]], ignore_index=True)
    Qr["row_id"] = np.arange(len(Qr))
    Qr["gid"] = Qr.protein_id.map(pid2gid)
    unm = Qr[Qr.gid.isna()]
    run["unmapped_rows"] = int(len(unm))
    run["unmapped_proteins"] = sorted(set(unm.protein_id))[:50]
    Qr = Qr[Qr.gid.notna()].copy()
    Qr["gid"] = Qr.gid.astype(np.int64)
    fl = Qr.flanks.apply(lambda s: [int(pid2gid[x]) for x in s.split(";") if x and x in pid2gid.index])
    Qr["flank_gids"] = fl
    if a.smoke:
        Qr = Qr.sample(n=min(a.smoke, len(Qr)), random_state=1).sort_values("row_id")
    run["query_rows"] = int(len(Qr))
    run["no_embedding_rows"] = int((uq[Qr.gid.to_numpy()] < 0).sum())

    # ---- labels
    A = pd.read_parquet(SRD + "/data/annot.parquet")
    kp = pd.read_csv(SRD + "/data/ko_proxy.tsv", sep="\t")
    proxy = kp.groupby("ko").pf.apply(sorted).to_dict()
    vocab = sorted({x for v in A.pfam for x in v} | {x for v in A.ko for x in v} | set(kp.pf))
    for s_ in list(Qr.R_pfam) + list(Qr.R_ko):
        vocab += [x for x in str(s_).split(";") if x]
    vocab = sorted(set(vocab))
    lid = {x: i for i, x in enumerate(vocab)}
    G["nlab"] = len(vocab)
    A["gid"] = A.protein_id.map(pid2gid)
    assert A.gid.notna().all()
    A["gid"] = A.gid.astype(np.int64)
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
    Qr["Rids"] = [np.array(sorted({lid[x] for x in (str(p).split(";") + str(k).split(";")) if x}), np.int64)
                  for p, k in zip(Qr.R_pfam, Qr.R_ko)]
    log("labels", len(vocab), "annotated genes", len(A))

    # ---- genes needing lists: queries and every gene within 3 positions on the contig
    qg = np.unique(Qr.gid.to_numpy())
    x = qg[:, None] + np.r_[0, OFFS][None, :]
    ok = (x >= 0) & (x < n)
    xc = np.where(ok, x, 0)
    ok &= G["cid"][xc] == G["cid"][qg][:, None]
    need = np.unique(x[ok])
    G["need"] = need
    run["genes_with_lists_built"] = int(len(need))
    log("query genes", len(qg), "genes needing lists", len(need))

    # ---- Z lists
    t = time.time()
    T_ = RD + "/calib/table/"
    cnt = np.load(T_ + "counts.npy").astype(np.int64)
    G["zoff"] = np.r_[0, np.cumsum(cnt)]
    G["zG"] = np.memmap(T_ + "genome.bin", np.int16, "r")
    G["zL"] = np.memmap(T_ + "local.bin", np.int16, "r")
    G["zz"] = np.memmap(T_ + "z.bin", np.float16, "r")
    G["LP"], G["LV"] = {}, {}
    G["LP"]["Z"], G["LV"]["Z"] = build_lists("Z", need, th)
    run["timing"]["lists_Z"] = round(time.time() - t, 1)
    log("Z lists", len(G["LV"]["Z"]), "entries", run["timing"]["lists_Z"], "s; rss", rss())
    # ---- A lists
    t = time.time()
    G["I"] = np.load(RD + "/vocab_lists/mm/aln__k1000__I.npy", mmap_mode="r")
    okq = uq >= 0
    G["members"] = np.flatnonzero(okq)[np.argsort(uq[okq], kind="stable")].astype(np.int64)
    G["mptr"] = np.r_[0, np.cumsum(np.bincount(uq[okq], minlength=G["I"].shape[0]))].astype(np.int64)
    G["LP"]["A"], G["LV"]["A"] = build_lists("A", need, th)
    run["timing"]["lists_A"] = round(time.time() - t, 1)
    log("A lists", len(G["LV"]["A"]), "entries", run["timing"]["lists_A"], "s; rss", rss())
    for arm in ("Z", "A"):
        ll = np.diff(G["LP"][arm])
        run[f"list_len_{arm}_need_q10_q50_q90"] = np.quantile(ll, [.1, .5, .9]).tolist()

    # ---- bonds and loci
    G["BONDS"], G["BOND_IDX"], G["LOCI"] = {}, {}, {}
    qset = set(qg.tolist())
    for arm in ("Z", "A"):
        z = np.load(f"{RD}/long_loci/nodes/dpann_{arm}/bonds_obs.npz")
        k, d = z["k"].astype(np.int64), z["d"].astype(np.int64)
        G["BONDS"][arm] = (k, d, z["n"], z["ng"], z["lam"])
        sel = np.flatnonzero(np.isin(k, qg) | np.isin(k + d, qg))
        idx = {}
        for b in sel.tolist():
            for e_ in (int(k[b]), int(k[b] + d[b])):
                if e_ in qset:
                    idx.setdefault(e_, []).append(b)
        G["BOND_IDX"][arm] = idx
        run[f"bonds_{arm}"] = dict(stored=int(len(k)), touching_queries=int(len(sel)))
        for which, nm in enumerate(OUTS[arm]):
            L = pd.read_parquet(f"{RD}/long_loci/loci/dpann_{arm}/{nm}.parquet").sort_values(["s", "e"])
            G["LOCI"][(arm, which)] = (L.s.to_numpy(np.int64), L.e.to_numpy(np.int64))
            run[f"loci_{arm}_{nm}"] = int(len(L))
    log("bonds and loci loaded; rss", rss())

    # ---- per-row work
    t = time.time()
    recs = Qr[["row_id", "gid", "Rids", "flank_gids"]].to_dict("records")
    chunks = [recs[i::th * 4] for i in range(th * 4)]
    with Pool(th) as pool:
        res = pool.map(work, chunks)
    out = pd.DataFrame([r for part in res for r in part]).sort_values(["row_id", "arm"])
    run["timing"]["rows"] = round(time.time() - t, 1)
    for c in [c for c in out.columns if c.endswith("_label")]:
        out[c] = [vocab[v] if v >= 0 else "" for v in out[c]]
    meta = Qr.drop(columns=["Rids", "flank_gids"])
    meta["n_flank_gids"] = Qr.flank_gids.apply(len)
    out = meta.merge(out, on="row_id")
    gate = dict(bonds_checked=int(out.n_bonds.sum()), mismatches=int(out.gate_mismatch.sum()),
                rows_with_mismatch=int((out.gate_mismatch > 0).sum()))
    gate["passed"] = gate["mismatches"] == 0 and gate["bonds_checked"] > 0
    json.dump(gate, open(f"{SRD}/results/gate{tag}.json", "w"), indent=1)
    out.to_parquet(f"{SRD}/results/calls{tag}.parquet", index=False)
    run["timing"]["total"] = round(time.time() - T0, 1)
    run["peak_rss_gb_parent"] = rss()
    run["gate"] = gate
    json.dump(run, open(f"{SRD}/results/run{tag}.json", "w"), indent=1, default=str)
    log("done", json.dumps(gate), run["timing"])
    if not gate["passed"]:
        raise SystemExit("GATE FAILED")


if __name__ == "__main__":
    main()
