#!/usr/bin/env python
"""Co-fold design (predeclared.json, test_F_cofold): classes a (high kappa, uncallable), b (matched low kappa,
uncallable), c (partner swaps of a), d (grammar critic candidates + swaps).

Writes $RD/kinds/cofold/{pool.parquet, manifest.tsv, jobs.jsonl, design.json}.
"""
import argparse
import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
K = RD / "kinds"
OUT = K / "cofold"
FAA = RD / "ws_dpann/work/proteins_gid.faa"
LC_MAX = 0.145
HYD_MAX = 0.45
N_A, N_C = 300, 150
CRITIC = [(1186030, 1186029, "d1"), (1107176, 1107175, "d1b"), (436727, 436728, "d2"), (1186035, 1186034, "d2b"),
          (18915, 18914, "d3"), (644092, 644093, "d4"), (963593, 963594, "d5"), (3363579, 3363580, "d6")]
T0 = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), f"{time.time() - T0:7.0f}s", *a, flush=True)


def entropy_lowcomplex_frac(seq, w=12, k1=2.2):
    """cofold/build_candidates.py: fraction of residues in w-windows with Shannon entropy < k1 bits."""
    n = len(seq)
    if n < w:
        return 0.0
    cov = np.zeros(n, bool)
    c = Counter(seq[:w])
    for i in range(n - w + 1):
        if i:
            out, inn = seq[i - 1], seq[i + w - 1]
            c[out] -= 1
            if c[out] == 0:
                del c[out]
            c[inn] += 1
        h = -sum((v / w) * math.log2(v / w) for v in c.values())
        if h < k1:
            cov[i:i + w] = True
    return float(cov.mean())


def comp(item):
    gid, s = item
    hyd = sum(ch in "ILVFMAWC" for ch in s) / max(len(s), 1)
    return gid, len(s), hyd, entropy_lowcomplex_frac(s), ("X" in s) or ("*" in s)


def read_faa(want):
    seqs, keep, cur = {}, False, None
    with open(FAA) as f:
        for line in f:
            if line[0] == ">":
                cur = int(line[1:].split()[0])
                keep = cur in want
                if keep:
                    seqs[cur] = []
            elif keep:
                seqs[cur].append(line.strip())
    return {k: "".join(v).rstrip("*").upper() for k, v in seqs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    df = pd.read_parquet(K / "main/scores.parquet")
    cid = np.load(K / "prep/cid.npy")
    tokC = np.load(K / "prep/tokC.npy")
    gm = pd.read_parquet(PKG / "universe/genomes.parquet").sort_values("genome_code").reset_index(drop=True)
    genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "protein_id", "genome", "contig",
                                                                     "position_index", "strand", "start", "end"])
    n = len(cid)
    i, j = df.i.to_numpy(), df.j.to_numpy()
    first = np.r_[True, cid[1:] != cid[:-1]]
    last = np.r_[cid[1:] != cid[:-1], True]
    df["edge"] = first[i] | last[j]
    base = ((df.genome_code % 2 == 0) & df.uncallable & ~df.homologous & df.unannotated_both & df.gold_none
            & ~df.edge)
    log("even, uncallable, non-homologous, both unannotated, no gold, not at a contig edge:", int(base.sum()))
    want = set(i[base]) | set(j[base]) | {g for p in CRITIC for g in p[:2]}
    seqs = read_faa(want)
    log("sequences", len(seqs))
    with ProcessPoolExecutor(a.threads) as ex:
        res = list(ex.map(comp, seqs.items(), chunksize=2000))
    cp = pd.DataFrame(res, columns=["gid", "aa", "hyd", "lc", "bad"]).set_index("gid")
    for side, col in (("a", i), ("b", j)):
        sub = cp.reindex(col)
        df[f"len_{side}"] = sub.aa.to_numpy()
        df[f"hyd_{side}"] = sub.hyd.to_numpy()
        df[f"lc_{side}"] = sub.lc.to_numpy()
        df[f"bad_{side}"] = sub.bad.to_numpy()
    pool = base & (df.len_a >= 40) & (df.len_b >= 40) & ((df.len_a + df.len_b) <= 450) \
        & (df.bad_a == False) & (df.bad_b == False) & (df.hyd_a <= HYD_MAX) & (df.hyd_b <= HYD_MAX) \
        & (df.lc_a <= LC_MAX) & (df.lc_b <= LC_MAX)  # noqa: E712
    P = df[pool].copy()
    P["row"] = np.flatnonzero(pool.to_numpy())
    P["tc_a"], P["tc_b"] = tokC[P.i], tokC[P.j]
    for c in ("genus", "family", "ord", "phylum"):
        P[c] = gm[c].to_numpy()[P.genome_code]
    P["tie"] = rng.random(len(P))
    log("pool", len(P), "kappa q50/q90/q99", np.quantile(P.kappa, [.5, .9, .99]).round(3).tolist())
    P.to_parquet(OUT / "pool.parquet", index=False)

    # ---------------------------------------------------------------- class a
    P = P.sort_values(["kappa", "R", "tie"], ascending=[False, False, True])
    used_pair, fam_n, cpair_n, gen_n = set(), Counter(), Counter(), Counter()
    A = []
    for r in P.itertuples():
        fp = (min(r.ta, r.tb), max(r.ta, r.tb))
        cpair = (min(r.tc_a, r.tc_b), max(r.tc_a, r.tc_b))
        if fp in used_pair or fam_n[r.ta] >= 2 or fam_n[r.tb] >= 2 or cpair_n[cpair] >= 5 or gen_n[r.genome_code] >= 3:
            continue
        used_pair.add(fp)
        fam_n[r.ta] += 1
        if r.tb != r.ta:
            fam_n[r.tb] += 1
        cpair_n[cpair] += 1
        gen_n[r.genome_code] += 1
        A.append(r.Index)
        if len(A) == N_A:
            break
    A = P.loc[A].copy()
    A["rank"] = np.arange(1, len(A) + 1)
    log("class a", len(A), "kappa range", float(A.kappa.min()), float(A.kappa.max()), "R median", float(A.R.median()))

    # ---------------------------------------------------------------- class b
    med = float(P.kappa.median())
    low = P[P.kappa <= med].copy()
    low["s_min"] = np.minimum(low.len_a, low.len_b)
    low["s_max"] = np.maximum(low.len_a, low.len_b)
    taken = set(A.i) | set(A.j)
    Brows, levels = [], []
    for r in A.itertuples():
        smin, smax = min(r.len_a, r.len_b), max(r.len_a, r.len_b)
        cand = low[(~low.i.isin(taken)) & (~low.j.isin(taken))
                   & (np.abs(low.s_min - smin) <= 0.2 * smin) & (np.abs(low.s_max - smax) <= 0.2 * smax)]
        pick, lvl = None, None
        for lv, col in (("genome", "genome_code"), ("genus", "genus"), ("family", "family"), ("order", "ord"),
                        ("phylum", "phylum"), ("any", None)):
            c = cand if col is None else cand[cand[col] == getattr(r, col)]
            if col is not None and pd.isna(getattr(r, col)):
                continue
            if len(c):
                d = np.abs(c.s_min - smin) / smin + np.abs(c.s_max - smax) / smax
                c = c.assign(d=d.to_numpy()).sort_values(["d", "tie"])
                pick, lvl = c.index[0], lv
                break
        Brows.append(pick)
        levels.append(lvl)
        if pick is not None:
            taken |= {low.at[pick, "i"], low.at[pick, "j"]}
    B = low.loc[[x for x in Brows if x is not None]].copy()
    B["match_level"] = [lv for x, lv in zip(Brows, levels) if x is not None]
    B["matched_to"] = [int(rk) for x, rk in zip(Brows, A["rank"]) if x is not None]
    log("class b", len(B), "pool median kappa", med, "match levels", Counter(B.match_level))

    # ---------------------------------------------------------------- class c (swaps) and d
    from scipy.optimize import linear_sum_assignment
    chains = pd.DataFrame(dict(gid=np.r_[A.i, A.j], aa=np.r_[A.len_a, A.len_b], hyd=np.r_[A.hyd_a, A.hyd_b],
                               tm=np.r_[A.ta, A.tb], tc=np.r_[A.tc_a, A.tc_b],
                               pair=np.r_[A["rank"], A["rank"]]))
    tokM = np.load(K / "prep/tokM.npy")

    def assign(rows):
        """rows: list of dict(keep_gid, rep_gid, rep_aa, rep_hyd, rep_tc, fams, pair). Returns donor gids."""
        C = np.full((len(rows), len(chains)), 1e6)
        for k, r in enumerate(rows):
            ok = (~chains.tm.isin(r["fams"])) & (chains.tc != r["rep_tc"]) & (chains.pair != r["pair"])
            dl = np.abs(chains.aa - r["rep_aa"]) / r["rep_aa"]
            dh = np.abs(chains.hyd - r["rep_hyd"])
            strict = ok & (dl <= 0.10) & (dh <= 0.05)
            relax = ok & (dl <= 0.25) & (dh <= 0.10)
            cost = (dl + dh).to_numpy()
            C[k] = np.where(strict, cost, np.where(relax, 10 + cost, 1e6))
        ri, ci = linear_sum_assignment(C)
        don = [None] * len(rows)
        tier = [None] * len(rows)
        for x, y in zip(ri, ci):
            if C[x, y] < 1e6:
                don[x] = int(chains.gid.iloc[y])
                tier[x] = "strict" if C[x, y] < 10 else "relaxed"
        return don, tier

    pick = np.sort(rng.choice(len(A), min(N_C, len(A)), replace=False))
    Cin = []
    for t, k in enumerate(pick):
        r = A.iloc[k]
        keepA = (t % 2 == 0)
        rep = "b" if keepA else "a"
        Cin.append(dict(keep_gid=int(r.i if keepA else r.j), rep_gid=int(r.j if keepA else r.i),
                        rep_aa=float(r[f"len_{rep}"]), rep_hyd=float(r[f"hyd_{rep}"]),
                        rep_tc=int(r[f"tc_{rep}"]), fams=[int(r.ta), int(r.tb)], pair=int(r["rank"]),
                        keep_chain="A" if keepA else "B"))
    don, tier = assign(Cin)
    # class d: critic candidates and one swap each (replace B)
    din = []
    for g0, g1, lab in CRITIC:
        s1 = seqs.get(g1)
        h1 = sum(ch in "ILVFMAWC" for ch in s1) / len(s1)
        din.append(dict(keep_gid=g0, rep_gid=g1, rep_aa=float(len(s1)), rep_hyd=h1, rep_tc=int(tokC[g1]),
                        fams=[int(tokM[g0]), int(tokM[g1])], pair=-1, keep_chain="A", label=lab))
    ddon, dtier = assign(din)

    # ---------------------------------------------------------------- manifest + jobs
    gtab = genes.set_index("gid")
    rows, jobs = [], []

    def add(fid, cls, g0, g1, extra):
        s0, s1 = seqs[g0], seqs[g1]
        jobs.append(dict(fold_id=fid, chains=[dict(id="A", seq=s0), dict(id="B", seq=s1)]))
        r = dict(fold_id=fid, cls=cls, gid_A=g0, gid_B=g1, len_A=len(s0), len_B=len(s1), len_sum=len(s0) + len(s1),
                 hyd_A=round(comp((0, s0))[2], 4), hyd_B=round(comp((0, s1))[2], 4),
                 protein_A=gtab.at[g0, "protein_id"], protein_B=gtab.at[g1, "protein_id"], genome=gtab.at[g0, "genome"],
                 tokM_A=int(tokM[g0]), tokM_B=int(tokM[g1]), tokC_A=int(tokC[g0]), tokC_B=int(tokC[g1]))
        r.update(extra)
        rows.append(r)

    for r in A.itertuples():
        add(f"kA_{r.rank:03d}", "a_high", int(r.i), int(r.j), dict(rank=r.rank, row=r.row, kappa=r.kappa, R=r.R, R0=r.R0,
                                                                    eng_L=r.eng_L, singleton=r.singleton_either))
    for r in B.itertuples():
        add(f"kB_{r.matched_to:03d}", "b_low", int(r.i), int(r.j), dict(rank=r.matched_to, row=r.row, kappa=r.kappa,
                                                                         R=r.R, R0=r.R0, eng_L=r.eng_L,
                                                                         singleton=r.singleton_either,
                                                                         match_level=r.match_level))
    for c, d_, t_ in zip(Cin, don, tier):
        if d_ is None:
            continue
        g0, g1 = (c["keep_gid"], d_) if c["keep_chain"] == "A" else (d_, c["keep_gid"])
        add(f"kC_{c['pair']:03d}", "c_swap", g0, g1, dict(rank=c["pair"], swapped_out=c["rep_gid"], donor=d_,
                                                           keep_chain=c["keep_chain"], swap_tier=t_))
    for c, d_, t_ in zip(din, ddon, dtier):
        add(f"kD_{c['label']}", "d_critic", c["keep_gid"], c["rep_gid"], dict(label=c["label"]))
        if d_ is not None:
            add(f"kD_{c['label']}_swap", "d_critic_swap", c["keep_gid"], d_, dict(label=c["label"], donor=d_,
                                                                                swapped_out=c["rep_gid"], swap_tier=t_))
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.tsv", sep="\t", index=False)
    with open(OUT / "jobs.jsonl", "w") as f:
        for jb in jobs:
            f.write(json.dumps(jb) + "\n")
    design = dict(pool=int(len(P)), pool_median_kappa=med, n_items=len(jobs), by_class=man.cls.value_counts().to_dict(),
                  a_kappa_min_max=[float(A.kappa.min()), float(A.kappa.max())],
                  b_kappa_max=float(B.kappa.max()) if len(B) else None,
                  b_match_levels=dict(Counter(B.match_level)),
                  c_tiers=dict(Counter([t for t in tier if t])), c_unassigned=int(sum(d is None for d in don)),
                  len_bins={c: pd.cut(man[man.cls == c].len_sum, [0, 200, 300, 451], right=False).value_counts().sort_index().astype(int).tolist()
                            for c in man.cls.unique()})
    json.dump(design, open(OUT / "design.json", "w"), indent=1, default=str)
    log(design)


if __name__ == "__main__":
    main()
