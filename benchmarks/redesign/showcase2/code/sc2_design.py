#!/usr/bin/env python
"""ws_showcase2 step 3: fold design for the candidates that cleared triage (biotite login node, seconds).

Cognate  sc2c_<rank>_r<rep>: the replicate's k chains in reference order A, B, ...
Swap     sc2s_<rank>_r<rep>_s<slot>: one per block (census rule): odd rank -> replicate 1 with chain index 1 (B)
         replaced; even rank -> replicate 2 with chain index k-2 (A for pairs) replaced. Donor = a chain of another
         cleared block's replicate with the same replicate number, assigned one-to-one (Hungarian) on relative length
         difference + hydrophobic-fraction difference; forbidden donors: same block, a token of the receiving block,
         or whitened (zca10) cosine >= 0.32 to any chain of the receiving block (homologous donors; the earlier
         screens' paralog-swap problem).
Follow-up (written after the screen by --followup): margin controls sc2f_<rank>_r<rep>_s<slot> for every interface
         passing in both replicates (the control its other margin needs, same donor rule over the not-yet-used
         donors), and monomers sc2m_<rank>_r<rep>_<chain> of every chain of such interfaces (zipping rule).
Writes $RD/showcase2/design/: manifest.tsv, jobs.jsonl (or manifest_fu.tsv, jobs_fu.jsonl).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
S = RD / "showcase2"
D = S / "design"
FAA = RD / "ws_dpann/work/proteins_gid.faa"
ZROW = RD / "locus_search/out3/vec/zca10_f32__row.npy"
AB = "ABCDEFGH"
COS_HOM = 0.32


def read_faa(want):
    seqs, keep, h, buf = {}, False, None, []
    with open(FAA) as f:
        for line in f:
            if line[0] == ">":
                if keep:
                    seqs[h] = "".join(buf).rstrip("*")
                h = int(line[1:])
                keep = h in want
                buf = []
            elif keep:
                buf.append(line.strip())
    if keep:
        seqs[h] = "".join(buf).rstrip("*")
    return seqs


def vec(X, gids):
    gids = np.asarray(gids, np.int64)
    o = np.argsort(gids)
    v = np.empty((len(gids), X.shape[1]), np.float32)
    v[o] = X[gids[o]]
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


def assign(reqs, donors, info, X, used=frozenset()):
    """reqs: (block_rank, gid list of receiving block, slot); donors: (block_rank, gid)."""
    L, H, T = info
    dg = np.array([d[1] for d in donors])
    Vd = vec(X, dg)
    cost = np.zeros((len(reqs), len(donors)))
    for a, (rk, gl, s) in enumerate(reqs):
        x = gl[s]
        cost[a] = np.abs(L[dg] - L[x]) / L[x] + np.abs(H[dg] - H[x])
        Vb = vec(X, gl)
        cmax = (Vd @ Vb.T).max(1)
        bad = np.array([d[0] == rk or T[d[1]] in {T[y] for y in gl} or d[1] in used for d in donors]) | (cmax >= COS_HOM)
        cost[a, bad] = 1e6
    ri, cj = linear_sum_assignment(cost)
    out = []
    for a, b in zip(ri, cj):
        assert cost[a, b] < 1e6, "no admissible donor"
        rk, gl, s = reqs[a]
        x, d = gl[s], dg[b]
        out.append((rk, s, d, abs(L[d] - L[x]) / L[x], abs(H[d] - H[x])))
    return out


def main(followup=False):
    D.mkdir(parents=True, exist_ok=True)
    reps = pd.read_csv(S / "data/reps.tsv", sep="\t")
    clear = pd.read_csv(S / "data/triage_blocks_sens.tsv", sep="\t")     # primary rule + wording-sensitivity set
    keep = set(clear[clear.fold_sens].cand)
    reps = reps[reps.cand.isin(keep)].sort_values(["rank", "rep"])
    g = pd.read_parquet(RD / "ws_dpann/package/universe/genes.parquet", columns=["gid", "length", "protein_id"])
    comp = pd.read_parquet(RD / "cofold/s3/data/composition_slim.parquet").set_index("protein_id")
    L = comp.aa_len.reindex(g.protein_id).to_numpy()   # genes.length counts the stop codon in some genomes
    H = comp.hyd.reindex(g.protein_id).to_numpy()
    T = pd.read_parquet(RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet").sort_values("gid").token.to_numpy()
    X = np.load(ZROW, mmap_mode="r")
    blocks = {(int(r["rank"]), int(r.rep)): [int(x) for x in r.gids.split(";")] for _, r in reps.iterrows()}
    ranks = sorted({k[0] for k in blocks})
    rows = []
    if not followup:
        for (rk, rep), gl in blocks.items():
            rows.append(dict(fold_id=f"sc2c_{rk:03d}_r{rep}", cls="cognate", rank=rk, k=len(gl), rep=rep, slot=-1,
                             gids=";".join(map(str, gl)), match=""))
        for rep in (1, 2):
            donors = [(rk, x) for (rk, rp), gl in blocks.items() if rp == rep for x in gl]
            reqs = []
            for rk in ranks:
                if (rk % 2 == 1) != (rep == 1):
                    continue
                gl = blocks[(rk, rep)]
                reqs.append((rk, gl, 1 if rep == 1 else len(gl) - 2))
            for rk, s, d, dl, dh in assign(reqs, donors, (L, H, T), X):
                gl = list(blocks[(rk, rep)])
                gl[s] = int(d)
                rows.append(dict(fold_id=f"sc2s_{rk:03d}_r{rep}_s{s}", cls="swap", rank=rk, k=len(gl), rep=rep, slot=s,
                                 gids=";".join(map(str, gl)), match=f"donor_gid={d};len_rel_diff={dl:.3f};hyd_diff={dh:.3f}"))
        man_name, jobs_name = "manifest.tsv", "jobs.jsonl"
    else:
        man0 = pd.read_csv(D / "manifest.tsv", sep="\t")
        used = {int(m.split(";")[0].split("=")[1]) for m in man0.match.dropna() if m.startswith("donor_gid")}
        need = pd.read_csv(S / "analysis/need_controls.tsv", sep="\t")
        calls = pd.read_csv(S / "analysis/calls.tsv", sep="\t")
        for rep in (1, 2):
            nr = need[need.rep == rep].drop_duplicates(["rank", "slot"])
            if not len(nr):
                continue
            donors = [(rk, x) for (rk, rp), gl in blocks.items() if rp == rep for x in gl]
            reqs = [(int(r["rank"]), blocks[(int(r["rank"]), rep)], int(r.slot)) for _, r in nr.iterrows()]
            for rk, s, d, dl, dh in assign(reqs, donors, (L, H, T), X, used=frozenset(used)):
                gl = list(blocks[(rk, rep)])
                gl[s] = int(d)
                rows.append(dict(fold_id=f"sc2f_{rk:03d}_r{rep}_s{s}", cls="swap", rank=rk, k=len(gl), rep=rep, slot=s,
                                 gids=";".join(map(str, gl)), match=f"followup;donor_gid={d};len_rel_diff={dl:.3f};hyd_diff={dh:.3f}"))
        pb = calls[calls.pass_both]
        for rk, c in pb.groupby("rank"):
            for rep in (1, 2):
                gl = blocks[(int(rk), rep)]
                for ch in sorted(set(c.i) | set(c.j)):
                    rows.append(dict(fold_id=f"sc2m_{int(rk):03d}_r{rep}_{AB[ch]}", cls="monomer", rank=int(rk), k=1, rep=rep,
                                     slot=int(ch), gids=str(gl[ch]), match=""))
        man_name, jobs_name = "manifest_fu.tsv", "jobs_fu.jsonl"
    man = pd.DataFrame(rows)
    man["lengths"] = man.gids.map(lambda s: ";".join(str(int(L[int(x)])) for x in s.split(";")))
    man["hyds"] = man.gids.map(lambda s: ";".join(f"{H[int(x)]:.3f}" for x in s.split(";")))
    man["tokens"] = man.gids.map(lambda s: ";".join(str(int(T[int(x)])) for x in s.split(";")))
    man["total_aa"] = man.lengths.map(lambda s: sum(int(x) for x in s.split(";")))
    man.to_csv(D / man_name, sep="\t", index=False)
    gids = sorted({int(x) for s in man.gids for x in s.split(";")})
    seqs = read_faa(set(gids))
    pid = g.protein_id.to_numpy()
    with open(D / jobs_name, "w") as f:
        for _, r in man.iterrows():
            gl = [int(x) for x in r.gids.split(";")]
            ch = [{"id": AB[i], "seq": seqs[x], "gid": x, "protein_id": pid[x]} for i, x in enumerate(gl)]
            assert all(len(c["seq"]) == L[c["gid"]] for c in ch)
            f.write(json.dumps({"fold_id": r.fold_id, "cls": r.cls, "chains": ch}) + "\n")
    print(man.cls.value_counts().to_dict(), "items", len(man), "max total aa", int(man.total_aa.max()))
    sw = man[man.cls == "swap"]
    if len(sw):
        print(sw.match.str.extract(r"len_rel_diff=([0-9.]+)")[0].astype(float).describe().round(3).to_dict())
        print(sw.match.str.extract(r"hyd_diff=([0-9.]+)")[0].astype(float).describe().round(3).to_dict())


if __name__ == "__main__":
    main(followup=len(sys.argv) > 1 and sys.argv[1] == "--followup")
