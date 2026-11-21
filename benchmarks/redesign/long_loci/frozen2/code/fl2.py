#!/usr/bin/env python
"""[frozen run 2: fl.py copied, paths changed to the SR-VP bin set; data key "srvp"]
long_loci frozen run on the lut test set (FROZEN.json): inputs and the PRIMARY configuration, engine comparator.

Method code is imported unchanged (sha256 in FROZEN.json): long_loci ll.py / lw.py / words_ts.py, transfer_spacedust
emb.py kernels (cohesion, groups), engine recurrence.py, boundary blib.py / rules.py. This file only supplies paths.

  meta      emb/meta.npz: uq (package uid of each gid), g_start, g_end, cnt      (login node; seconds)
  uidgate   package uid_of_row equals 06a_unique.py's gid2uid (identical-sequence partition, first-gid numbering)
  zca32     emb/zca10_f32__uid.npy = the stored transform applied to the raw means (slice_lut embed/zca10.py second
            pass: unit -> minus mu -> zero invalid -> @ W -> unit); gate against the stored float16 vectors
  cohesion  emb/kappa_var_tokens.npy (transfer_spacedust emb.py cmd_cohesion body, kernels imported)
  groups    groups/emb_hom_average_t0.75.parquet (transfer_spacedust emb.py cmd_groups body, helpers imported)
  wa        lw.py cmd_build for arm WA with ll.DATA['lut'] and output root $FZ (unchanged code)
  dh        engine nested reading DH (level-2 conserved loci + rule-H cores) from the engine run directory
usage: fl.py CMD --threads T
"""
import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
FZ = RD / "long_loci/frozen2"
PKG = RD / "ws_srvp_bins/package"
E = FZ / "emb"
TOK = FZ / "tok/mmseqs__seq__e3c50s75_rescue3__srvp.parquet"
GROUPS = FZ / "groups/emb_hom_average_t0.75.parquet"
KAPPA_TOK = E / "kappa_var_tokens.npy"
ENGINE_ARM = "srvp__mmseqs__seq__e3c50s75_rescue3"
sys.path.insert(0, str(RD / "long_loci/code"))
sys.path.insert(0, str(RD / "transfer_spacedust/code"))


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 2)


def jdump(obj, fn):
    Path(fn).write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
    log("wrote", fn)


def register_lut():
    """ll.DATA entry for the lut test set (paths only)."""
    import ll
    ll.DATA["srvp"] = dict(genes=PKG / "universe/genes.parquet", tok=TOK, groups=GROUPS, kappa=KAPPA_TOK)
    return ll


# =========================================================================== inputs
def cmd_meta(a):
    E.mkdir(parents=True, exist_ok=True)
    g = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome_code", "contig_code"])
    assert np.array_equal(g.gid.to_numpy(), np.arange(len(g)))
    gc = g.genome_code.to_numpy(np.int64)
    assert (np.diff(gc) >= 0).all() and (np.diff(g.contig_code.to_numpy(np.int64)) >= 0).all()
    uq = np.load(PKG / "embeddings/esmpp_small__uid_of_row.npy").astype(np.int64)
    assert len(uq) == len(g)
    n = len(g)
    ch = np.flatnonzero(np.diff(gc)) + 1
    U = int(uq.max()) + 1
    cnt = np.bincount(uq, minlength=U)
    first = np.unique(uq, return_index=True)[1]
    info = dict(genes=n, genomes=int(gc.max()) + 1, unique=U, uid_numbered_by_first_gid=bool((np.diff(first) > 0).all()))
    np.savez(E / "meta.npz", uq=uq, g_start=np.r_[0, ch].astype(np.int64), g_end=np.r_[ch, n].astype(np.int64), cnt=cnt)
    jdump(info, E / "meta.json")


def cmd_uidgate(a):
    uq = np.load(E / "meta.npz")["uq"].astype(np.int64)
    g2u = pd.read_parquet(FZ / "tok/work/uniq/gid2uid.parquet").sort_values("gid")
    assert np.array_equal(g2u.gid.to_numpy(), np.arange(len(uq)))
    u2 = g2u.uid.to_numpy(np.int64)
    res = dict(equal=bool(np.array_equal(uq, u2)))
    if not res["equal"]:
        # same partition?
        f = pd.DataFrame(dict(a=uq, b=u2)).drop_duplicates()
        res.update(pairs=int(len(f)), a_unique=int(f.a.nunique()), b_unique=int(f.b.nunique()))
    jdump(res, E / "uidgate.json")
    assert res["equal"] or (res["pairs"] == res["a_unique"] == res["b_unique"]), res


def cmd_zca32(a):
    import threadpoolctl
    t0 = time.time()
    z = np.load(PKG / "embeddings/zca10_transform.npz")
    mu, W, valid = z["mu"], z["W"], z["valid"]
    R = np.load(PKG / "embeddings/esmpp_small__mean_raw_f32__uid.npy", mmap_mode="r")
    F16 = np.load(PKG / "embeddings/zca10_f16__uid.npy", mmap_mode="r")
    U, d = R.shape
    out = np.lib.format.open_memmap(E / ".zca10_f32__uid.tmp.npy", mode="w+", dtype=np.float32, shape=(U, d))
    BLK = 200_000
    mx, mincos = 0.0, 1.0
    with threadpoolctl.threadpool_limits(a.threads):
        for s in range(0, U, BLK):
            Y = np.array(R[s:s + BLK], dtype=np.float32)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr[nr == 0] = 1
            Y /= nr
            Y -= mu.astype(np.float32)
            Y[~valid[s:s + BLK]] = 0
            Zb = Y @ W
            nz = np.linalg.norm(Zb, axis=1, keepdims=True)
            nz[nz == 0] = 1
            Zb = (Zb / nz).astype(np.float32)
            out[s:s + BLK] = Zb
            f = np.asarray(F16[s:s + BLK], dtype=np.float32)
            mx = max(mx, float(np.abs(Zb - f).max()))
            mincos = min(mincos, float((Zb * f).sum(1).min()))
    out.flush()
    del out
    Path(E / ".zca10_f32__uid.tmp.npy").rename(E / "zca10_f32__uid.npy")
    res = dict(unique=int(U), dim=int(d), W_dtype=str(W.dtype), mu_dtype=str(mu.dtype), invalid=int((~valid).sum()),
               max_abs_diff_vs_f16=mx, min_cosine_vs_f16=mincos, gate_pass=bool(mx <= 1e-3),
               seconds=round(time.time() - t0, 1), peak_rss_gb=rss_gb())
    jdump(res, E / "zca32.json")
    assert res["gate_pass"], res


def cmd_cohesion(a):
    """transfer_spacedust emb.py cmd_cohesion, body copied with the lut paths; kernels imported unchanged."""
    import numba
    import emb as EM
    numba.set_num_threads(a.threads)
    t0 = time.time()
    g = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome_code", "contig_code", "strand"])
    n = len(g)
    gc = g.genome_code.to_numpy(np.int64)
    cc = g.contig_code.to_numpy(np.int64)
    cid = pd.factorize(cc)[0].astype(np.int64)
    strand = g.strand.to_numpy(np.int64)
    t = pd.read_parquet(TOK)
    tok = np.full(n, -1, np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    T = int(tok.max()) + 1
    X = np.load(E / "zca10_f32__uid.npy")
    uid = np.load(E / "meta.npz")["uq"].astype(np.int64)
    rng = np.random.default_rng(17)
    ga = rng.integers(0, n, 1_000_000)
    gb = rng.integers(0, n, 1_000_000)
    ok = gc[ga] != gc[gb]
    o = np.empty(int(ok.sum()))
    EM.gene_pair_cos(ga[ok], gb[ok], X, uid, o)
    c0 = float(np.nanmean(o))
    A, cnt = EM.side_sums(tok, uid, X, strand, cid, T)
    normA2 = EM.side_norms(A)
    kr = np.empty(n)
    kl = np.empty(n)
    EM.gene_cohesion(tok, uid, X, strand, cid, A, cnt, normA2, c0, kr, kl)
    del A, X
    kappa = np.fmin(kr[:-1], kl[1:])
    same = cid[:-1] == cid[1:]
    kappa = np.r_[np.where(same, kappa, np.nan), np.nan]
    np.save(KAPPA_TOK, kappa.astype(np.float32))
    inside = kappa[:-1][same]
    info = dict(gene_pair_floor=c0, floor_pairs=int(ok.sum()),
                kappa_q05_q25_q50_q75_q95=np.nanquantile(inside, [.05, .25, .5, .75, .95]).tolist(),
                kappa_nan_frac=float(np.isnan(inside).mean()),
                frac_below={str(th): float((inside < th).mean()) for th in (0.02, 0.05, 0.1, 0.2)},
                tokens=T, seconds=round(time.time() - t0, 1), peak_rss_gb=rss_gb())
    jdump(info, E / "cohesion.json")


def cmd_groups(a):
    """transfer_spacedust emb.py cmd_groups, body copied with the lut paths (taxonomy read for description only)."""
    import emb as EM
    t0 = time.time()
    G = FZ / "gtable"
    gm = pd.read_parquet(PKG / "universe/genomes.parquet").sort_values("genome_code").reset_index(drop=True)
    ng = len(gm)
    assert np.array_equal(gm.genome_code.to_numpy(), np.arange(ng))
    z = np.load(G / "sums.npz")
    s_all, n_all, s_hom, n_hom, n_id = (z[k] for k in ("s_all", "n_all", "s_hom", "n_hom", "n_id"))
    n_g = gm.n_genes.to_numpy(np.float64)
    mean = (s_all + EM.FLOOR * (n_g[:, None] - n_all)) / n_g[:, None]
    hom = np.where(n_hom > 0, s_hom / np.maximum(n_hom, 1), EM.FLOOR)
    id95 = n_id / n_g[:, None]
    out = FZ / "groups"
    out.mkdir(parents=True, exist_ok=True)
    for nm, M in (("emb_mean", mean), ("emb_hom", hom), ("emb_id95", id95)):
        np.save(out / f"{nm}.npy", M.astype(np.float32))
    np.save(out / "emb_homn.npy", n_hom.astype(np.int32))
    Z = EM.symmetrise(hom)
    Lk = EM.linkage_of(Z)
    lab = EM.groups_at(Lk, 0.75)
    pd.DataFrame(dict(genome=gm.genome, genome_code=gm.genome_code, group=lab)).to_parquet(GROUPS, index=False)
    sizes = np.bincount(lab)
    iu = np.triu_indices(ng, 1)
    off = Z.copy()
    np.fill_diagonal(off, -np.inf)
    nn = off.max(1)
    ladder = np.round(np.arange(0.5, 0.96, 0.05), 2)
    from sklearn.metrics import adjusted_mutual_info_score as ami
    res = dict(genomes=ng, theta=0.75, groups=int(len(sizes)), largest=int(sizes.max()), singletons=int((sizes == 1).sum()),
               size_q50_q90_q99=np.quantile(sizes, [.5, .9, .99]).tolist(),
               pair_q50_q90_q99_q999=np.quantile(Z[iu], [.5, .9, .99, .999]).tolist(),
               nn_q10_q50_q90=np.quantile(nn, [.1, .5, .9]).tolist(),
               groups_by_theta={str(t): int(groups_at_safe(EM, Lk, t)) for t in ladder},
               ami_family=float(ami(gm.family.fillna("na"), lab)), ami_genus=float(ami(gm.genus.fillna("na"), lab)),
               seconds=round(time.time() - t0, 1))
    jdump(res, out / "groups.json")


def groups_at_safe(EM, Lk, t):
    return EM.groups_at(Lk, t).max() + 1


# =========================================================================== primary
def cmd_wa(a):
    register_lut()
    import lw
    lw.LL = FZ
    ns = argparse.Namespace(data="srvp", arms="WA", w3b=False, suffix="", blocks=4096, threads=a.threads)
    lw.cmd_build(ns)


# =========================================================================== engine DH
def cmd_dh(a):
    import numba
    numba.set_num_threads(a.threads)
    sys.path.insert(0, str(RD / "boundary/code"))
    import blib as B
    import rules as RL
    t0 = time.time()
    rdir = FZ / "engine" / f"{ENGINE_ARM}__w3"
    L = B.Landscape(rdir, universe=PKG / "universe", tokens=TOK)
    bld = RL.Builder(L)
    od = FZ / "loci/srvp_engine"
    od.mkdir(parents=True, exist_ok=True)
    out = {}
    for nm, (s, e) in (("R1", (L.s2, L.e2)), ("DH", bld.build("D", dict(core="H", res=0.3, f=0.5)))):
        s = np.asarray(s, np.int64)
        e = np.asarray(e, np.int64)
        k = e > s
        df = pd.DataFrame(dict(s=s[k], e=e[k])).drop_duplicates().sort_values(["s", "e"]).reset_index(drop=True)
        df.to_parquet(od / f"{nm}.parquet", index=False)
        cov = np.zeros(L.u.n + 1, np.int32)
        np.add.at(cov, df.s.to_numpy(), 1)
        np.add.at(cov, df.e.to_numpy() + 1, -1)
        out[nm] = dict(loci=int(len(df)), genes_in_loci_frac=float((np.cumsum(cov)[:L.u.n] > 0).mean()),
                       span_q50_q90_q99_max=np.quantile(df.e - df.s + 1, [.5, .9, .99, 1]).tolist())
        log(nm, out[nm])
    out["engine_summary"] = {k: v for k, v in L.summary.items() if not isinstance(v, (list, dict)) or k == "modules"}
    out["seconds"] = round(time.time() - t0, 1)
    out["peak_rss_gb"] = rss_gb()
    jdump(out, FZ / "results/dh_srvp.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["meta", "uidgate", "zca32", "cohesion", "groups", "wa", "dh"])
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    {"meta": cmd_meta, "uidgate": cmd_uidgate, "zca32": cmd_zca32, "cohesion": cmd_cohesion, "groups": cmd_groups,
     "wa": cmd_wa, "dh": cmd_dh}[a.cmd](a)


if __name__ == "__main__":
    main()
