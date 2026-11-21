"""Screen 3 follow-up folds (predeclared_criteria_screen3.json 'margin_controls_completion' and 'refutations'):

1. Margin controls. For every cognate interface (i, j) passing the per-replicate rule in both replicates, the
   controls its two margins need (rep 1: the control replacing j; rep 2: the control replacing i) that the screen
   did not fold (work/s3/need_controls.tsv from s3_analyze.py). Fold id non3f_<rank>_r<rep>_s<slot>. Donor = a chain
   of another pool block's cognate replicate of the same replicate number, sharing no token with the receiving
   block, assigned one-to-one over the requests (Hungarian) on relative length + hydrophobic-fraction difference,
   as in the screen.
2. Monomers. Every chain of every interface passing in both replicates, both replicates: mono3_<rank>_r<rep>_<chain>.
Usage: python s3_followup.py TAG   -> work/s3/fold_manifest3_<TAG>.tsv, work/s3/jobs_s3_<TAG>.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
W3 = HERE / "work" / "s3"
AB = "ABCDEFGH"


def main(tag):
    man = pd.read_csv(W3 / "fold_manifest3.tsv", sep="\t")
    done = set()
    for p in W3.glob("fold_manifest3_*.tsv"):
        done |= set(pd.read_csv(p, sep="\t").fold_id)
    seq, L, H, tok = {}, {}, {}, {}
    for p in sorted(W3.glob("jobs_s3*.jsonl")):
        for line in open(p):
            j = json.loads(line)
            for c in j["chains"]:
                seq[c["gid"]] = c["seq"]
    for _, r in man.iterrows():
        for x, l_, h_, t_ in zip(r.gids.split(";"), str(r.lengths).split(";"), str(r.hyds).split(";"), str(r.tokens).split(";")):
            L[int(x)], H[int(x)], tok[int(x)] = int(l_), float(h_), int(t_)
    need = pd.read_csv(W3 / "need_controls.tsv", sep="\t")
    calls = pd.read_csv(W3 / "s3_calls.tsv", sep="\t")
    cog = man[man.cls == "cognate"]
    rows = []
    for rep in (1, 2):
        nr = need[need.rep == rep].drop_duplicates(["s3_rank", "slot"]).reset_index(drop=True)
        nr = nr[~nr.apply(lambda r: f"non3f_{int(r.s3_rank):03d}_r{rep}_s{int(r.slot)}" in done, axis=1)].reset_index(drop=True)
        if not len(nr):
            continue
        sub = cog[cog.rep == rep].reset_index(drop=True)
        donors = [(int(r.s3_rank), int(x)) for _, r in sub.iterrows() for x in r.gids.split(";")]
        dg = np.array([d[1] for d in donors])
        Ld = np.array([L[x] for x in dg], float)
        Hd = np.array([H[x] for x in dg])
        cost = np.zeros((len(nr), len(donors)))
        recv = []
        for a, r in nr.iterrows():
            gl = [int(x) for x in sub[sub.s3_rank == r.s3_rank].gids.iloc[0].split(";")]
            x = gl[int(r.slot)]
            recv.append((gl, x))
            cost[a] = np.abs(Ld - L[x]) / L[x] + np.abs(Hd - H[x])
            bt = {tok[y] for y in gl}
            bad = np.array([d[0] == r.s3_rank or tok[d[1]] in bt for d in donors])
            cost[a, bad] = 1e6
        ri, cj = linear_sum_assignment(cost)
        for a, b in zip(ri, cj):
            r = nr.iloc[a]
            gl, x = recv[a]
            d = donors[b][1]
            new = list(gl)
            new[int(r.slot)] = d
            base = cog[(cog.s3_rank == r.s3_rank) & (cog.rep == rep)].iloc[0]
            rows.append(dict(fold_id=f"non3f_{int(r.s3_rank):03d}_r{rep}_s{int(r.slot)}", cls="noncognate", s3_rank=int(r.s3_rank),
                             k=int(base.k), rep=rep, type_key=base.type_key, gids=";".join(map(str, new)), genome=base.genome,
                             slot=int(r.slot), match_level=f"followup;donor_rank={donors[b][0]};len_rel_diff={abs(L[d]-L[x])/L[x]:.3f};"
                                                           f"hyd_diff={abs(H[d]-H[x]):.3f}", unrec_class=base.unrec_class))
    pb = calls[calls.pass_both]
    for rank, c in pb.groupby("s3_rank"):
        chains = sorted(set(c.i) | set(c.j))
        for rep in (1, 2):
            base = cog[(cog.s3_rank == rank) & (cog.rep == rep)].iloc[0]
            gl = [int(x) for x in base.gids.split(";")]
            for ch in chains:
                fid = f"mono3_{int(rank):03d}_r{rep}_{AB[ch]}"
                if fid in done:
                    continue
                rows.append(dict(fold_id=fid, cls="monomer", s3_rank=int(rank), k=1, rep=rep, type_key=base.type_key,
                                 gids=str(gl[ch]), genome=base.genome, slot=ch, match_level="", unrec_class=base.unrec_class))
    out = pd.DataFrame(rows)
    if not len(out):
        print("nothing to fold")
        return
    out["lengths"] = out.gids.map(lambda s: ";".join(str(L[int(x)]) for x in s.split(";")))
    out["hyds"] = out.gids.map(lambda s: ";".join(f"{H[int(x)]:.3f}" for x in s.split(";")))
    out["tokens"] = out.gids.map(lambda s: ";".join(str(tok[int(x)]) for x in s.split(";")))
    out["total_aa"] = out.lengths.map(lambda s: sum(int(x) for x in s.split(";")))
    out.to_csv(W3 / f"fold_manifest3_{tag}.tsv", sep="\t", index=False)
    with open(W3 / f"jobs_s3_{tag}.jsonl", "w") as f:
        for _, r in out.iterrows():
            gl = [int(x) for x in r.gids.split(";")]
            f.write(json.dumps({"fold_id": r.fold_id, "cls": r.cls,
                                "chains": [{"id": AB[i], "seq": seq[x], "gid": x} for i, x in enumerate(gl)]}) + "\n")
    print(out.cls.value_counts().to_dict(), "items", len(out))
    print(out[out.cls == "noncognate"][["fold_id", "match_level"]].to_string())


if __name__ == "__main__":
    main(sys.argv[1])
