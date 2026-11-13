"""Self-test of the locus-space kernels against brute-force Python (tiny random data; seconds)."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lspace as L  # noqa: E402


def sw_ref(C, sa, sb, theta, gap, spen):
    def one(Cm, sbb):
        na, nb = Cm.shape
        H = np.zeros((na + 1, nb + 1))
        for x in range(1, na + 1):
            for y in range(1, nb + 1):
                sub = Cm[x - 1, y - 1] - theta - (spen if (spen > 0 and sa[x - 1] != sbb[y - 1]) else 0)
                H[x, y] = max(0, H[x - 1, y - 1] + sub, H[x - 1, y] - gap, H[x, y - 1] - gap)
        return H.max()
    f = one(C, sb)
    r = one(C[:, ::-1], -sb[::-1])
    return max(f, r)


rng = np.random.default_rng(0)
n_genes = 40
X = rng.normal(size=(n_genes, 16))
X /= np.linalg.norm(X, axis=1, keepdims=True)
G = (X @ X.T).astype(np.float32)
loci = [rng.choice(n_genes, rng.integers(2, 7), replace=False) for _ in range(12)]
lens = np.array([len(l) for l in loci])
ptr = np.r_[0, np.cumsum(lens)].astype(np.int64)
idx = np.concatenate(loci).astype(np.int64)
strand = rng.choice([-1, 1], len(idx)).astype(np.int64)

S = L.chamfer_all(G, ptr, idx)
for i in range(len(loci)):
    for j in range(len(loci)):
        C = G[np.ix_(loci[i], loci[j])]
        ref = 0.5 * (C.max(1).mean() + C.max(0).mean())
        assert abs(S[i, j] - ref) < 1e-5, (i, j, S[i, j], ref)
print("chamfer ok")

for spen in (0.0, 0.3):
    A = L.aln_all(G, ptr, idx, strand, 0.14, 0.3, spen, int(lens.max()))
    for i in range(len(loci)):
        si = strand[ptr[i]:ptr[i + 1]]
        selfi = sw_ref(G[np.ix_(loci[i], loci[i])], si, si, 0.14, 0.3, spen)
        for j in range(len(loci)):
            sj = strand[ptr[j]:ptr[j + 1]]
            selfj = sw_ref(G[np.ix_(loci[j], loci[j])], sj, sj, 0.14, 0.3, spen)
            # forward-only self score in the kernel: compare with the forward self alignment
            def fwd(Cm, sa, sbb):
                na, nb = Cm.shape
                H = np.zeros((na + 1, nb + 1))
                for x in range(1, na + 1):
                    for y in range(1, nb + 1):
                        sub = Cm[x - 1, y - 1] - 0.14 - (spen if (spen > 0 and sa[x - 1] != sbb[y - 1]) else 0)
                        H[x, y] = max(0, H[x - 1, y - 1] + sub, H[x - 1, y] - 0.3, H[x, y - 1] - 0.3)
                return H.max()
            fi = fwd(G[np.ix_(loci[i], loci[i])], si, si)
            fj = fwd(G[np.ix_(loci[j], loci[j])], sj, sj)
            ref = sw_ref(G[np.ix_(loci[i], loci[j])], si, sj, 0.14, 0.3, spen) / math.sqrt(fi * fj)
            assert abs(A[i, j] - ref) < 1e-5, (spen, i, j, A[i, j], ref)
print("alignment ok")


# nn_eval: ties and group exclusion
S = np.array([[1, .5, .5, .9], [.5, 1, .2, .1], [.5, .2, 1, .3], [.9, .1, .3, 1]], np.float32)
key = np.array([0, 1, 1, 0])
grp = np.array([0, 1, 2, 0])
pred, _ = L.nn_eval(S, key, grp, np.array([0]), False)
assert pred[0] == 1, pred          # ref 3 is in the same group; refs 1, 2 tie at .5 with label 1
mods = np.array([5, 5, -1, 5])
pred, _ = L.nn_eval(None, key, grp, np.array([0, 2]), False, mod=mods)
assert pred[0] == 1 and pred[1] == -1, pred
print("nn_eval ok")
