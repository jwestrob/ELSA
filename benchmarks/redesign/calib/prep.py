#!/usr/bin/env python
"""ws_calib prep (CPU): per-unique-sequence evaluation features, annotation sets, query sample, hubness.

Outputs under $RD/calib/work/prep/:
  uidfeat.npz   length, first gid, genome of first gene, rescue3 token, family genes / genomes, TM-segment count
                (Kyte-Doolittle, 19-residue windows, mean >= 1.6), amino-acid composition (20, float16), valid flag
  ann.npz       CSR sets per uid (ptr / val): pfam family codes, group codes (pfam family + clan, clan offset by
                n_pfam), profile group codes (family-level HH Pfam hits, prob >= 90, cols >= 30, mapped to family +
                clan codes), profile PDB entry codes, CDD superfamily codes, curated role codes; caller bitmask
                (1 cctyper, 2 DefenseFinder, 4 TXSScan); vocabularies in ann_vocab.json
  queries.npz   uid, stratum (0 R, 1 S, 2 N, 3 Y, 4 Yc), caller, ctrl_of, qgid, qgenome
  hub.npz       r10, r50 per uid from the stored zca10 HNSW top-300 (self removed)
Evaluation-only annotations; nothing here enters a method.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
C = RD / "calib"
PKG = RD / "ws_dpann/package"
EMB = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/embed_queue/runs/dpann__esmpp_small__20261111")
TOK = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
OUT = C / "work/prep"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


KD = np.zeros(256, np.float32)
for aa, v in dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9,
                  M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2).items():
    KD[ord(aa)] = v
AAI = np.full(256, -1, np.int32)
for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY"):
    AAI[ord(aa)] = i


@njit(parallel=True, cache=True)
def seqfeat(buf, starts, ends, kd, aai, ntm, comp):
    n = len(starts)
    for k in prange(n):
        s, e = starts[k], ends[k]
        L = e - s
        cnt = np.zeros(20, np.float64)
        tot = 0
        for p in range(s, e):
            j = aai[buf[p]]
            if j >= 0:
                cnt[j] += 1
                tot += 1
        for j in range(20):
            comp[k, j] = cnt[j] / max(tot, 1)
        W = 19
        segs = 0
        inseg = False
        if L >= W:
            acc = 0.0
            for p in range(s, s + W):
                acc += kd[buf[p]]
            for st in range(s, e - W + 1):
                if st > s:
                    acc += kd[buf[st + W - 1]] - kd[buf[st - 1]]
                hit = acc / W >= 1.6
                if hit and not inseg:
                    segs += 1
                inseg = hit
        ntm[k] = segs


@njit(parallel=True, cache=True)
def hub_kernel(D, I, r10, r50):
    n = D.shape[0]
    for i in prange(n):
        s10 = 0.0
        s50 = 0.0
        c = 0
        for j in range(D.shape[1]):
            if I[i, j] < 0:
                continue
            v = float(D[i, j])
            if c < 10:
                s10 += v
            if c < 50:
                s50 += v
            c += 1
            if c >= 50:
                break
        r10[i] = s10 / max(min(c, 10), 1)
        r50[i] = s50 / max(min(c, 50), 1)


def csr(uid, code, n):
    df = pd.DataFrame(dict(u=uid.astype(np.int64), c=code.astype(np.int64))).drop_duplicates().sort_values(["u", "c"])
    ptr = np.zeros(n + 1, np.int64)
    np.add.at(ptr, df.u.to_numpy() + 1, 1)
    return np.cumsum(ptr), df.c.to_numpy(np.int32)


def norm_role(src, name):
    if src == "cctyper":
        return "cct:" + name.split("_")[0]
    if src == "defensefinder_system":
        a, b = name.split("__", 1) if "__" in name else (name, name)
        b = re.sub(r"\.einsi.*$", "", b)
        b = re.sub(r"(_FAM)?_\d+$", "", b)
        return "df:" + a + "__" + b
    return "tx:" + name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    import numba
    numba.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "protein_id", "genome_code", "length"])
    n = len(genes)
    assert (genes.gid.to_numpy() == np.arange(n)).all()
    uq = np.load(EMB / "uid_of_row.npy").astype(np.int64)
    meta = np.load(RD / "softrec/r3/vec/meta.npz")
    assert np.array_equal(meta["uq"].astype(np.int64), uq)
    g2u = pd.read_parquet(RD / "subgenic/gid2uid.parquet")
    log("gid2uid columns", list(g2u.columns))
    gcol = [c for c in g2u.columns if "gid" in c][0]
    ucol = [c for c in g2u.columns if "uid" in c][0]
    chk = np.full(n, -1, np.int64)
    chk[g2u[gcol].to_numpy()] = g2u[ucol].to_numpy()
    assert np.array_equal(chk, uq), "subgenic uid != embedding uid"
    U = int(uq.max()) + 1
    valid = meta["valid"].astype(bool)
    gc = genes.genome_code.to_numpy(np.int32)
    _, first = np.unique(uq, return_index=True)
    assert len(first) == U
    tokdf = pd.read_parquet(TOK)
    tok = np.full(n, -1, np.int64)
    tok[tokdf.gid.to_numpy()] = tokdf.token.to_numpy()
    assert (tok >= 0).all()
    fam_genes = np.bincount(tok)
    key = np.unique(tok * 4096 + gc)
    fam_genomes = np.bincount(key // 4096, minlength=len(fam_genes))
    tok_u = tok[first]
    uni = pd.read_parquet(RD / "subgenic/uniq.parquet", columns=["uid", "length", "rep_gid"])
    assert np.array_equal(uni.uid.to_numpy(), np.arange(U))
    length = uni.length.to_numpy(np.int32)
    log("universe", n, "genes", U, "unique; valid", int(valid.sum()))

    # ---------------------------------------------------------------- sequence features
    raw = np.fromfile(RD / "subgenic/uniq.faa", dtype=np.uint8)
    nl = np.flatnonzero(raw == 10)
    # records: header line '>uid' then one sequence line
    hdr_starts = np.flatnonzero(raw == ord(">"))
    assert len(hdr_starts) == U
    hdr_end = nl[np.searchsorted(nl, hdr_starts)]
    seq_s = hdr_end + 1
    seq_e = nl[np.searchsorted(nl, seq_s)]
    ids = np.array([int(raw[s + 1:e].tobytes()) for s, e in zip(hdr_starts[:5], hdr_end[:5])])
    assert np.array_equal(ids, np.arange(5))
    ntm = np.zeros(U, np.int32)
    comp = np.zeros((U, 20), np.float32)
    seqfeat(raw, seq_s.astype(np.int64), seq_e.astype(np.int64), KD, AAI, ntm, comp)
    slen = (seq_e - seq_s).astype(np.int32)
    log("sequence features; length agreement", float((slen == length).mean()), "ntm q50/q90/q99",
        np.quantile(ntm, [.5, .9, .99]).tolist(), "share >=1 TM", float((ntm >= 1).mean()))
    np.savez(OUT / "uidfeat.npz", length=length, first_gid=first.astype(np.int64), qgenome=gc[first],
             tok=tok_u, fam_genes=fam_genes[tok_u].astype(np.int32), fam_genomes=fam_genomes[tok_u].astype(np.int32),
             ntm=ntm, comp=comp.astype(np.float16), valid=valid)
    del raw

    # ---------------------------------------------------------------- annotations -> uid sets
    pid2gid = pd.Series(genes.gid.to_numpy(), index=genes.protein_id.to_numpy())
    I = C / "inputs"
    seed_lines = (I / "pfam39_seed_ac_cl.txt").read_text().splitlines()
    clan_of, cur = {}, None
    for ln in seed_lines:
        f = ln.split()
        if f[1] == "AC":
            cur = f[2].split(".")[0]
        elif f[1] == "CL" and cur is not None:
            clan_of[cur] = f[2]
    pf = pd.read_parquet(I / "pfam.parquet", columns=["protein_id", "acc"])
    fa = pd.read_parquet(I / "famann.parquet")
    fa = fa[(fa.probability >= 90) & (fa.aligned_cols >= 30)].copy()
    accs = sorted(set(pf.acc) | set(fa.loc[fa.source == "hhsearch_pfam", "hit_accession"].str.split(".").str[0]))
    pcode = {x: i for i, x in enumerate(accs)}
    clans = sorted(set(clan_of.values()))
    ccode = {x: i for i, x in enumerate(clans)}
    NP = len(accs)
    g = pid2gid.reindex(pf.protein_id.to_numpy()).to_numpy()
    miss = int(np.isnan(g).sum())
    pf = pf[~np.isnan(g)]
    pu = uq[g[~np.isnan(g)].astype(np.int64)]
    pc = pf.acc.map(pcode).to_numpy()
    cl = pf.acc.map(lambda x: ccode.get(clan_of.get(x), -1)).to_numpy()
    pfam_ptr, pfam_val = csr(pu, pc, U)
    gu = np.r_[pu, pu[cl >= 0]]
    gv = np.r_[pc, NP + cl[cl >= 0]]
    grp_ptr, grp_val = csr(gu, gv, U)
    log("pfam rows", len(pf), "unmapped", miss, "families", NP, "clans", len(clans), "uids with pfam",
        int((np.diff(pfam_ptr) > 0).sum()))
    # family-level profile hits (base clustering run permissive-e3c50s75)
    pfb = pd.read_parquet(I / "protfam_base.parquet")
    fam_codes = {x: i for i, x in enumerate(pd.unique(pfb.family_id))}
    gb = pid2gid.reindex(pfb.protein_id.to_numpy()).to_numpy()
    assert not np.isnan(gb).any()
    basefam_g = np.full(n, -1, np.int64)
    basefam_g[gb.astype(np.int64)] = pfb.family_id.map(fam_codes).to_numpy()
    fa["fc"] = fa.family_id.map(fam_codes)
    assert fa.fc.notna().all()
    hp = fa[fa.source == "hhsearch_pfam"]
    hacc = hp.hit_accession.str.split(".").str[0]
    hpc = hacc.map(pcode).to_numpy()
    hcl = hacc.map(lambda x: ccode.get(clan_of.get(x), -1)).to_numpy()
    fam_g = np.r_[hp.fc.to_numpy(), hp.fc.to_numpy()[hcl >= 0]]
    fam_v = np.r_[hpc, NP + hcl[hcl >= 0]]
    hb = fa[fa.source == "hhblits_pdb"]
    pdbs = sorted(set(hb.hit_accession.str[:4].str.lower()))
    pdbcode = {x: i for i, x in enumerate(pdbs)}
    fam_pdb_f = hb.fc.to_numpy()
    fam_pdb_v = hb.hit_accession.str[:4].str.lower().map(pdbcode).to_numpy()
    # expand family -> genes -> uids
    fam_df = pd.DataFrame(dict(f=fam_g, v=fam_v)).drop_duplicates()
    gene_df = pd.DataFrame(dict(gid=np.arange(n), f=basefam_g, u=uq))
    m1 = gene_df[gene_df.f.isin(set(fam_df.f))][["u", "f"]].drop_duplicates().merge(fam_df, on="f")
    prof_ptr, prof_val = csr(m1.u.to_numpy(), m1.v.to_numpy(), U)
    pdb_df = pd.DataFrame(dict(f=fam_pdb_f, v=fam_pdb_v)).drop_duplicates()
    m2 = gene_df[gene_df.f.isin(set(pdb_df.f))][["u", "f"]].drop_duplicates().merge(pdb_df, on="f")
    pdb_ptr, pdb_val = csr(m2.u.to_numpy(), m2.v.to_numpy(), U)
    log("profile hits: families with Pfam-HH", fam_df.f.nunique(), "with PDB", pdb_df.f.nunique(), "uids with prof",
        int((np.diff(prof_ptr) > 0).sum()), "uids with pdb", int((np.diff(pdb_ptr) > 0).sum()))
    # CDD superfamilies
    links = pd.read_csv(RD / "cofold/s2/db/cdd/family_superfamily_links", sep="\t", header=None,
                        names=["acc", "cddid", "sf", "sfid"], dtype=str)
    sf_of = dict(zip(links.acc, links.sf))
    cdd = pd.read_parquet(I / "cdd.parquet", columns=["protein_id", "acc", "evalue"])
    cdd = cdd[cdd.evalue <= 0.01]
    sfx = cdd.acc.map(lambda x: sf_of.get(x, "self:" + x))
    sfs = {x: i for i, x in enumerate(pd.unique(sfx))}
    gcd = pid2gid.reindex(cdd.protein_id.to_numpy()).to_numpy()
    ok = ~np.isnan(gcd)
    cdd_ptr, cdd_val = csr(uq[gcd[ok].astype(np.int64)], sfx.map(sfs).to_numpy()[ok], U)
    log("cdd rows E<=0.01", len(cdd), "superfamily codes", len(sfs), "share mapped to cl",
        float(cdd.acc.isin(sf_of.keys()).mean()), "uids with cdd", int((np.diff(cdd_ptr) > 0).sum()))
    # curated roles
    sp = pd.read_parquet(I / "sysprot.parquet")
    sp["role"] = [norm_role(s, nm) for s, nm in zip(sp.system_source, sp.profile_name)]
    roles = sorted(set(sp.role))
    rcode = {x: i for i, x in enumerate(roles)}
    gsp = pid2gid.reindex(sp.protein_id.to_numpy()).to_numpy()
    assert not np.isnan(gsp).any()
    su = uq[gsp.astype(np.int64)]
    role_ptr, role_val = csr(su, sp.role.map(rcode).to_numpy(), U)
    callbit = np.zeros(U, np.int8)
    bits = sp.system_source.map({"cctyper": 1, "defensefinder_system": 2, "txsscan_system": 4}).to_numpy()
    np.bitwise_or.at(callbit, su, bits.astype(np.int8))
    log("roles", len(roles), "uids curated", int((callbit > 0).sum()), "by caller", [int(((callbit & b) > 0).sum()) for b in (1, 2, 4)])
    np.savez(OUT / "ann.npz", pfam_ptr=pfam_ptr, pfam_val=pfam_val, grp_ptr=grp_ptr, grp_val=grp_val,
             prof_ptr=prof_ptr, prof_val=prof_val, pdb_ptr=pdb_ptr, pdb_val=pdb_val, cdd_ptr=cdd_ptr,
             cdd_val=cdd_val, role_ptr=role_ptr, role_val=role_val, callbit=callbit, n_pfam=NP)
    (OUT / "ann_vocab.json").write_text(json.dumps(dict(pfam=accs, clans=clans, roles=roles, n_pdb=len(pdbs),
                                                       n_cdd_sf=len(sfs)), indent=0) + "\n")

    # ---------------------------------------------------------------- query sample (predeclared)
    qR = np.load(RD / "softrec/r3/calib/qids.npy").astype(np.int64)
    chosen = np.zeros(U, bool)
    chosen[qR] = True
    has_pfam = np.diff(pfam_ptr) > 0
    fs = fam_genes[tok_u]
    fbin = np.digitize(fs, [2, 5, 20, 100, 500])
    lt = np.quantile(length[valid], [1 / 3, 2 / 3])
    lbin = np.digitize(length, lt)
    rng = np.random.default_rng([17, 1])
    pool = np.flatnonzero(valid & (length < 100) & ~chosen)
    qS = np.sort(rng.choice(pool, 3000, replace=False)); chosen[qS] = True
    rng = np.random.default_rng([17, 2])
    pool = np.flatnonzero(valid & (fs == 1) & has_pfam & ~chosen)
    qN = np.sort(rng.choice(pool, 3000, replace=False)); chosen[qN] = True
    rng = np.random.default_rng([17, 3])
    qY, cY = [], []
    for b in (1, 2, 4):
        pool = np.flatnonzero(valid & ((callbit & b) > 0) & ~chosen)
        pick = np.sort(rng.choice(pool, min(1000, len(pool)), replace=False))
        chosen[pick] = True
        qY.append(pick); cY.append(np.full(len(pick), b, np.int8))
    qY, cY = np.concatenate(qY), np.concatenate(cY)
    rng = np.random.default_rng([17, 4])
    cand = valid & (callbit == 0)
    qYc = np.full(len(qY), -1, np.int64)
    for i in rng.permutation(len(qY)):
        u = qY[i]
        pool = np.flatnonzero(cand & ~chosen & (fbin == fbin[u]) & (lbin == lbin[u]))
        if len(pool):
            qYc[i] = rng.choice(pool)
            chosen[qYc[i]] = True
    okc = qYc >= 0
    uids = np.r_[qR, qS, qN, qY, qYc[okc]]
    strat = np.r_[np.zeros(len(qR)), np.ones(len(qS)), np.full(len(qN), 2), np.full(len(qY), 3), np.full(int(okc.sum()), 4)].astype(np.int8)
    caller = np.r_[np.zeros(len(qR) + len(qS) + len(qN), np.int8), cY, cY[okc]]
    ctrl_of = np.r_[np.full(len(qR) + len(qS) + len(qN) + len(qY), -1), len(qR) + len(qS) + len(qN) + np.flatnonzero(okc)].astype(np.int64)
    assert len(np.unique(uids)) == len(uids)
    np.savez(OUT / "queries.npz", uid=uids, stratum=strat, caller=caller, ctrl_of=ctrl_of, qgid=first[uids],
             qgenome=gc[first[uids]])
    log("queries", len(uids), "by stratum", np.bincount(strat).tolist(), "Y by caller", [int((cY == b).sum()) for b in (1, 2, 4)],
        "controls matched", int(okc.sum()))

    # ---------------------------------------------------------------- hubness
    D = np.load(RD / "softrec/knn/zca10__k300__D.npy", mmap_mode="r")
    Ik = np.load(RD / "softrec/knn/zca10__k300__I.npy", mmap_mode="r")
    r10 = np.zeros(U, np.float32)
    r50 = np.zeros(U, np.float32)
    B = 200_000
    for s in range(0, U, B):
        d = np.ascontiguousarray(D[s:s + B]).astype(np.float32)
        ii = np.ascontiguousarray(Ik[s:s + B]).astype(np.int64)
        ii = np.where(ii == np.arange(s, s + len(ii))[:, None], -1, ii)
        hub_kernel(d, ii, r10[s:s + B], r50[s:s + B])
    r10[~valid] = 0
    r50[~valid] = 0
    np.savez(OUT / "hub.npz", r10=r10, r50=r50)
    log("hubness r10 q01/q50/q99", np.quantile(r10[valid], [.01, .5, .99]).tolist(), "r50", np.quantile(r50[valid], [.01, .5, .99]).tolist())
    log("done")


if __name__ == "__main__":
    main()
