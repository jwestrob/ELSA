"""Task 4, step 3 (biotite, Slurm): stricter pairing-shuffle nulls (within order, within genus) for the B-C and A-B
coupling statistics of c1x_coupling.py, and the top-ranked inter-protein column pairs with their contact status.
Usage: python c1x_coupling2.py WORKDIR NPROC   (imports c1x_coupling from the same directory)
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

import c1x_coupling as cc

W = sys.argv[1]


def shuffle_within(groups, rng):
    perm = np.arange(len(groups))
    for gname in np.unique(groups):
        ix = np.where(groups == gname)[0]
        perm[ix] = ix[rng.permutation(len(ix))]
    return perm


def main():
    pairs = json.load(open(f"{W}/pairs.json"))
    inst = pd.read_csv(f"{W}/instances.tsv", sep="\t").set_index("instance_id")
    out = []
    rng = np.random.default_rng(23)
    for p1, p2 in (("B", "C"), ("A", "B")):
        s = cc.build(p1, p2, pairs, True, True)
        cc.G["s"] = s
        res = dict(pair=p1 + p2)
        S_mi, _ = cc.score_mi(s.X1, s.X2)
        S_dca = cc.score_dca(s.X1, s.X2)[0]
        obs = dict(mi=cc.stats(S_mi, s.con, s.non), dca=cc.stats(S_dca, s.con, s.non))
        res["obs"] = obs
        con = set(s.con)
        for nm, S in (("mi", S_mi), ("dca", S_dca)):
            idx = [(i, j) for i in range(S.shape[0]) for j in range(S.shape[1])]
            top = sorted(idx, key=lambda t: -S[t])[:20]
            k1 = np.where((cc.encode([cc.rf(f"{W}/aln_{p1}.fa")[x] for x in s.ids]) != 20).mean(0) >= 0.7)[0]
            k2 = np.where((cc.encode([cc.rf(f"{W}/aln_{p2}.fa")[x] for x in s.ids]) != 20).mean(0) >= 0.7)[0]
            res[f"top20_{nm}"] = [dict(col1=int(k1[i]), col2=int(k2[j]), score=round(float(S[i, j]), 4),
                                       contact=(i, j) in con, noncontact=(i, j) in set(s.non)) for i, j in top]
        for lvl in ("ord", "genus"):
            groups = inst.loc[s.ids, lvl].fillna("NA").values
            n_movable = int(sum((groups == g).sum() for g in np.unique(groups) if (groups == g).sum() > 1))
            res[f"rows_in_multirow_{lvl}"] = n_movable
            with Pool(int(sys.argv[2])) as pool:
                for kind, nn in (("mi", 1000), ("dca", 100)):
                    perms = [shuffle_within(groups, rng) for _ in range(nn)]
                    rr = pool.map(cc._null, [(kind, p) for p in perms])
                    d = {}
                    for m in ("auc", "z", "topK_prec"):
                        v = np.array([r[m] for r in rr])
                        d[m] = dict(mean=float(v.mean()), q95=float(np.quantile(v, 0.95)),
                                    p=float((1 + (v >= obs[kind][m]).sum()) / (1 + len(v))))
                    res[f"{kind}_null_{lvl}"] = d
        print(json.dumps(res, indent=1), flush=True)
        out.append(res)
    json.dump(out, open(f"{W}/coupling_results2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
