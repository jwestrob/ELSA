"""Task 4, step 2 (biotite, Slurm): inter-protein coupling of B with C (and A with B) without folding.

Inputs (work/coup/): aln_A.fa, aln_B.fa, aln_C.fa (deep-dive instance alignments, rows = engine instances),
pairs.json (contact / noncontact column pairs from the 11 passing ABC models, c1x_contacts.py), instances.tsv.
Pairing: rows of the two proteins from the same engine instance (one genome locus).
Columns kept: occupancy >= 0.7 in the paired rows. 21 states (20 aa + gap; other letters -> gap).
Sequence weights: 1 / number of rows with >= 80 % identity over the concatenated kept columns (recomputed after
every shuffle). Neff = sum of weights.
Scores per column pair:
  MIp  weighted mutual information (pseudocount 0.1), average-product corrected within the block scored
       (inter-protein block for X-Y, intra block for X-X)
  DCA  mean-field DCA (pseudocount 0.5) on the concatenated alignment; Frobenius norm of the zero-sum-gauge
       20 x 20 coupling block, APC over the full matrix
Statistics for a block: AUC (contact score > noncontact score), z = (mean contact - mean noncontact) / SD
noncontact, and top-K precision (K = number of contact pairs).
Nulls: (i) global shuffle of the pairing (rows of one protein permuted against the other), (ii) shuffle within
phylum. 1,000 shuffles for MIp, 200 for DCA. One-sided p = (1 + #null >= observed) / (1 + n).
Positive control: intra-C and intra-B contacts vs noncontacts (pairing-independent; Mann-Whitney).
Usage: python c1x_coupling.py WORKDIR NPROC
"""
import json
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
Q = 21
IDX = {a: i for i, a in enumerate(AA)}
W = sys.argv[1]
NPROC = int(sys.argv[2])
RNG = np.random.default_rng(17)


def rf(p):
    d, h = {}, None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            h = line[1:].split()[0]
            d[h] = []
        elif h:
            d[h].append(line)
    return {k: "".join(v) for k, v in d.items()}


def encode(seqs):
    return np.array([[IDX.get(c, 20) for c in s] for s in seqs], dtype=np.int8)


def weights(X, thr=0.8):
    oh = np.eye(Q, dtype=np.float32)[X].reshape(len(X), -1)
    ident = (oh @ oh.T) / X.shape[1]
    return 1.0 / (ident >= thr).sum(1)


def mi_block(X1, X2, w, lam=0.1):
    n = w.sum()
    O1 = np.eye(Q)[X1].reshape(len(X1), -1)
    O2 = np.eye(Q)[X2].reshape(len(X2), -1)
    L1, L2 = X1.shape[1], X2.shape[1]
    Pij = ((O1 * w[:, None]).T @ O2 / n).reshape(L1, Q, L2, Q)
    Pij = (1 - lam) * Pij + lam / (Q * Q)
    Pi = (1 - lam) * ((O1 * w[:, None]).sum(0) / n).reshape(L1, Q) + lam / Q
    Pj = (1 - lam) * ((O2 * w[:, None]).sum(0) / n).reshape(L2, Q) + lam / Q
    MI = (Pij * (np.log(Pij) - np.log(Pi[:, :, None, None]) - np.log(Pj[None, None, :, :]))).sum((1, 3))
    return MI


def apc(M, same=False):
    M = M.copy()
    if same:
        np.fill_diagonal(M, np.nan)
    r = np.nanmean(M, 1, keepdims=True)
    c = np.nanmean(M, 0, keepdims=True)
    a = np.nanmean(M)
    return M - r * c / a


def mfdca(X, w, lam=0.5):
    n, L = X.shape
    O = np.eye(Q)[X][:, :, :Q - 1].reshape(n, -1)          # drop the gap state (reference)
    nw = w.sum()
    fi = (1 - lam) * (O * w[:, None]).sum(0) / nw + lam / Q
    fij = (1 - lam) * ((O * w[:, None]).T @ O) / nw + lam / (Q * Q)
    q1 = Q - 1
    for i in range(L):                                       # within-column block: diagonal single-site freqs
        s = slice(i * q1, (i + 1) * q1)
        fij[s, s] = np.diag(fi[s])
    C = fij - np.outer(fi, fi)
    J = -np.linalg.inv(C)
    F = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            e = np.zeros((Q, Q))
            e[:q1, :q1] = J[i * q1:(i + 1) * q1, j * q1:(j + 1) * q1]
            e = e - e.mean(0, keepdims=True) - e.mean(1, keepdims=True) + e.mean()
            F[i, j] = F[j, i] = np.sqrt((e[:q1, :q1] ** 2).sum())
    np.fill_diagonal(F, np.nan)
    r = np.nanmean(F, 1, keepdims=True)
    return F - r * r.T / np.nanmean(F)


def stats(S, con, non):
    a = np.array([S[i, j] for i, j in con])
    b = np.array([S[i, j] for i, j in non])
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    auc = mannwhitneyu(a, b, alternative="greater").statistic / (len(a) * len(b))
    z = (a.mean() - b.mean()) / b.std()
    allv = np.concatenate([a, b])
    lab = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
    top = lab[np.argsort(-allv)[:len(a)]].mean()
    return dict(auc=float(auc), z=float(z), topK_prec=float(top), n_con=int(len(a)), n_non=int(len(b)),
                base_rate=float(len(a) / (len(a) + len(b))))


class Setup:
    pass


def build(p1, p2, pairs, intra1, intra2):
    al = {k: rf(f"{W}/aln_{k}.fa") for k in "ABC"}
    ids = sorted(set(al[p1]) & set(al[p2]))
    X1 = encode([al[p1][i] for i in ids])
    X2 = encode([al[p2][i] for i in ids])
    k1 = np.where((X1 != 20).mean(0) >= 0.7)[0]
    k2 = np.where((X2 != 20).mean(0) >= 0.7)[0]
    m1 = {c: i for i, c in enumerate(k1)}
    m2 = {c: i for i, c in enumerate(k2)}
    s = Setup()
    s.ids, s.X1, s.X2 = ids, X1[:, k1], X2[:, k2]
    blk = pairs[p1 + p2]
    s.con = [(m1[a], m2[b]) for a, b in blk["contact"] if a in m1 and b in m2]
    s.non = [(m1[a], m2[b]) for a, b in blk["noncontact"] if a in m1 and b in m2]
    s.icon1 = [(m1[a], m1[b]) for a, b in pairs[p1 + p1]["contact"] if a in m1 and b in m1]
    s.inon1 = [(m1[a], m1[b]) for a, b in pairs[p1 + p1]["noncontact"] if a in m1 and b in m1]
    s.icon2 = [(m2[a], m2[b]) for a, b in pairs[p2 + p2]["contact"] if a in m2 and b in m2]
    s.inon2 = [(m2[a], m2[b]) for a, b in pairs[p2 + p2]["noncontact"] if a in m2 and b in m2]
    inst = pd.read_csv(f"{W}/instances.tsv", sep="\t").set_index("instance_id")
    s.phy = inst.loc[ids, "phylum"].values
    return s


def score_mi(X1, X2):
    w = weights(np.concatenate([X1, X2], 1))
    return apc(mi_block(X1, X2, w)), w


def score_dca(X1, X2):
    X = np.concatenate([X1, X2], 1)
    w = weights(X)
    F = mfdca(X, w)
    L1 = X1.shape[1]
    return F[:L1, L1:], F[:L1, :L1], F[L1:, L1:], w


def shuffle(n, phy, within):
    if not within:
        return RNG.permutation(n)
    perm = np.arange(n)
    for p in np.unique(phy):
        ix = np.where(phy == p)[0]
        perm[ix] = ix[RNG.permutation(len(ix))]
    return perm


G = {}


def _null(args):
    kind, perm = args
    s = G["s"]
    X2 = s.X2[perm]
    if kind == "mi":
        S, _ = score_mi(s.X1, X2)
    else:
        S = score_dca(s.X1, X2)[0]
    return stats(S, s.con, s.non)


def run(p1, p2, pairs, n_mi, n_dca):
    s = build(p1, p2, pairs, True, True)
    G["s"] = s
    out = dict(pair=p1 + p2, n_rows=len(s.ids), cols=[s.X1.shape[1], s.X2.shape[1]])
    S_mi, w = score_mi(s.X1, s.X2)
    out["neff80"] = float(w.sum())
    out["mi_obs"] = stats(S_mi, s.con, s.non)
    Sd, D11, D22, _ = score_dca(s.X1, s.X2)
    out["dca_obs"] = stats(Sd, s.con, s.non)
    # positive controls (pairing-independent)
    w1 = weights(s.X1)
    w2 = weights(s.X2)
    out["ctrl_intra_" + p1 + "_mi"] = stats(apc(mi_block(s.X1, s.X1, w1), same=True), s.icon1, s.inon1) if s.icon1 else None
    out["ctrl_intra_" + p2 + "_mi"] = stats(apc(mi_block(s.X2, s.X2, w2), same=True), s.icon2, s.inon2) if s.icon2 else None
    out["ctrl_intra_" + p1 + "_dca"] = stats(D11, s.icon1, s.inon1) if s.icon1 else None
    out["ctrl_intra_" + p2 + "_dca"] = stats(D22, s.icon2, s.inon2) if s.icon2 else None
    n = len(s.ids)
    with Pool(NPROC) as pool:
        for kind, nn in (("mi", n_mi), ("dca", n_dca)):
            for within in (False, True):
                perms = [shuffle(n, s.phy, within) for _ in range(nn)]
                res = pool.map(_null, [(kind, p) for p in perms])
                obs = out[f"{kind}_obs"]
                lab = f"{kind}_null_{'phylum' if within else 'global'}"
                d = {}
                for m in ("auc", "z", "topK_prec"):
                    v = np.array([r[m] for r in res])
                    d[m] = dict(mean=float(v.mean()), sd=float(v.std()), q95=float(np.quantile(v, 0.95)),
                                p=float((1 + (v >= obs[m]).sum()) / (1 + len(v))))
                out[lab] = d
    print(json.dumps(out, indent=1), flush=True)
    return out


if __name__ == "__main__":
    pairs = json.load(open(f"{W}/pairs.json"))
    res = [run("B", "C", pairs, 1000, 200), run("A", "B", pairs, 1000, 200)]
    json.dump(res, open(f"{W}/coupling_results.json", "w"), indent=1)
