"""Interface residues (CA-CA <= 8 A, both chains pLDDT >= 50) in two-chain co-fold models."""
import numpy as np, sys
def load(p):
    ch = {}
    for l in open(p):
        if l.startswith('ATOM') and l[12:16].strip() == 'CA':
            ch.setdefault(l[21], []).append((int(l[22:26]), np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]), float(l[60:66])))
    return ch
for name in sys.argv[1:]:
    ch = load(f'folds/cx/{name}.pdb'); ks = sorted(ch)
    A, B = ch[ks[0]], ch[ks[1]]
    xa = np.array([x[1] for x in A]); xb = np.array([x[1] for x in B])
    d = np.linalg.norm(xa[:, None] - xb[None], axis=2)
    pa = np.array([x[2] for x in A]); pb = np.array([x[2] for x in B])
    sc = 100 if pa.max() <= 1 else 1
    m = (d <= 8) & (pa[:, None] * sc >= 50) & (pb[None] * sc >= 50)
    ia = sorted({A[i][0] for i in np.where(m.any(1))[0]}); ib = sorted({B[j][0] for j in np.where(m.any(0))[0]})
    def rng(v):
        out, s = [], None
        for i, x in enumerate(v):
            if s is None: s = p = x
            elif x == p + 1: p = x
            else: out.append(f'{s}-{p}'); s = p = x
        if s is not None: out.append(f'{s}-{p}')
        return ','.join(out)
    print(f'{name}: chainA {len(ia)} res [{rng(ia)}] | chainB {len(ib)} res [{rng(ib)}]')
