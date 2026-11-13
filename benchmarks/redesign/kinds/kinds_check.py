#!/usr/bin/env python
"""Pure-python recount of the kind_pass statistics for random queries (observed order and one null order),
with and without the L1 / mask exclusions. Exits non-zero on any mismatch."""
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import kinds as K  # noqa: E402


def relation(sx, sy, y_right):
    if sx == sy:
        return 1 if ((sx > 0 and y_right) or (sx < 0 and not y_right)) else 2
    ls, rs = (sx, sy) if y_right else (sy, sx)
    return 3 if (ls < 0 and rs > 0) else 4


def brute(d, a, b, gene_at, slot_of, ref_ok, excl_own=False, maskrow=None):
    Dmax = int(K.DEPTHS.max())
    Ua, Ub = d.uq[a], d.uq[b]
    rka = {Ua: 0}
    for j in range(Dmax):
        x = d.I[Ua, j]
        if x < 0:
            break
        rka.setdefault(int(x), j + 1)
    rkb = {Ub: 0}
    for j in range(Dmax):
        x = d.I[Ub, j]
        if x < 0:
            break
        rkb.setdefault(int(x), j + 1)
    ga = d.gc[a]
    fa, fb, ma, mb = d.tokF[a], d.tokF[b], d.tokM[a], d.tokM[b]
    rq = relation(d.strand[a], d.strand[b], True)
    rec = []
    for Up, ra in rka.items():
        for t in range(d.mptr[Up], d.mptr[Up + 1]):
            a2 = d.members[t]
            g2 = d.gc[a2]
            if g2 == ga or not ref_ok[g2]:
                continue
            s = slot_of[a2]
            for s2, right in ((s - 1, False), (s + 1, True)):
                if s2 < 0 or s2 >= d.n or d.cid[s2] != d.cid[s]:
                    continue
                b2 = gene_at[s2]
                U2 = d.uq[b2]
                if U2 not in rkb:
                    continue
                m1, m2 = d.tokM[a2], d.tokM[b2]
                if excl_own and (m1 in (ma, mb) or m2 in (ma, mb)):
                    continue
                if maskrow is not None and (maskrow[m1] or maskrow[m2]):
                    continue
                lev = max(ra, rkb[U2])
                f1, f2 = d.tokF[a2], d.tokF[b2]
                xf = f1 not in (fa, fb) and f2 not in (fa, fb)
                xm = m1 not in (ma, mb) and m2 not in (ma, mb)
                ori = relation(d.strand[a2], d.strand[b2], right) == rq
                rec.append((lev, g2, (f1, f2), (m1, m2), xf, xm, ori))
    out = np.zeros((len(K.DEPTHS), 8), np.int64)
    for v, D in enumerate(K.DEPTHS):
        r = [x for x in rec if x[0] <= D]
        out[v] = [len(r), len({x[1] for x in r}), len({x[2] for x in r}), len({x[2] for x in r if x[4]}),
                  len({x[3] for x in r}), len({x[3] for x in r if x[5]}), len({x[2] for x in r if x[4] and x[6]}),
                  len({x[3] for x in r if x[5] and x[6]})]
    homba = rka.get(Ub, Dmax + 1)
    homab = rkb.get(Ua, Dmax + 1)
    return out, (homba, homab)


def main():
    threads = int(sys.argv[1])
    import numba
    numba.set_num_threads(threads)
    d = K.Data()
    rng = np.random.default_rng(5)
    q = rng.choice(len(d.qa), 120, replace=False)
    qa, qb = d.qa[q], d.qb[q]
    ref = np.ones((2, d.ng), np.bool_)
    ref[1, rng.random(d.ng) < 0.3] = False
    qref = (np.arange(len(q)) % 2).astype(np.int32)
    mask = np.zeros((1, d.TM), np.bool_)
    mask[0, rng.choice(d.TM, d.TM // 50, replace=False)] = True
    mid = np.where(np.arange(len(q)) % 3 == 0, 0, -1).astype(np.int32)
    bad = 0
    for r in (-1, 0):
        gene_at, slot_of = d.order(r)
        for mode in ("plain", "own", "mask"):
            kw = {}
            if mode == "own":
                kw = dict(excl_own_m=True)
            if mode == "mask":
                kw = dict(mask=mask, mask_id=mid)
            o, h, ov = d.run(qa, qb, qref, ref, r, threads, label=f"check {mode}", **kw)
            for k in range(len(q)):
                mr = mask[0] if (mode == "mask" and mid[k] == 0) else None
                bo, bh = brute(d, int(qa[k]), int(qb[k]), gene_at, slot_of, ref[qref[k]], mode == "own", mr)
                if not np.array_equal(bo, o[k]) or tuple(h[k]) != bh:
                    bad += 1
                    if bad < 6:
                        print("MISMATCH", r, mode, k, bo.tolist(), o[k].tolist(), bh, h[k].tolist())
            print("checked", r, mode, "nonzero F@1000:", int((o[:, 3, 2] > 0).sum()), "max A:", int(o[:, 3, 0].max()),
                  flush=True)
    print("MISMATCHES", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
