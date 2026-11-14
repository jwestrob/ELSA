"""Screen 2, step 4: fold manifest (cognate, non-cognate swaps per chain, uncalled windows, positives)
and Biohub jobs, following predeclared_criteria_screen2.json.

Input: work/s2/s2_reps.tsv (replicates), work/s2/fold_ranks.txt (s2_ranks that passed triage, one per line).
Cognate      cog2_<rank>_r<rep>: the replicate's k chains in gene order.
Non-cognate  non2_<rank>_r<rep>_s<slot>: rep 1 replaces every chain except the first, rep 2 every chain
             except the last, one control complex per replaced chain; the replacement is a chain of
             another folded candidate of the same replicate, assigned one-to-one (Hungarian) on relative
             length difference + hydrophobic-fraction difference; a donor that shares a family token
             with the receiving block is forbidden.
Uncalled     unc2_<rank>_r<rep>: k adjacent same-strand dark genes, no X, 60-1,200 aa, consecutive token
             pairs all non-significant, none of the genes in a dark-link run (work/s2/block_instances);
             lengths within 25 % per position and hydrophobic fraction within 0.05 (then 40 % / 0.08),
             either orientation; same contig, then genome, then order, then anywhere; each window used once.
Positive     pos2_k<k>_<caller>_<type>_<n>: k consecutive member genes of one curated gold instance
             (DefenseFinder, TXSScan, CRISPRCasTyper), each 60-1,200 aa, sum <= 2,500, hydrophobic
             fraction <= 0.45, no X (CRISPRCasTyper instances with status cas_putative left out);
             k = 3 (24), 4 (16), 5-6 (8 each when present) and two-gene windows
             with one protein > 700 aa (16); <= 4 per (caller, system type), distinct genomes, round robin.
Writes work/s2/fold_manifest2.tsv and work/s2/jobs_s2.jsonl.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
W = HERE / "work"
W2 = W / "s2"
D = W / "data"
SHARUR = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
N_POS = {3: 24, 4: 16, 5: 8, 6: 8, 2: 16}


def main():
    reps = pd.read_csv(W2 / "s2_reps.tsv", sep="\t")
    ranks = [int(x) for x in open(W2 / "fold_ranks.txt").read().split()]
    reps = reps[reps.s2_rank.isin(ranks)].copy()
    g = pd.read_parquet(W / "genes_dark.parquet")
    comp = pd.read_parquet(D / "composition.parquet")[["protein_id", "hyd", "n_x", "aa_len"]]
    g = g.merge(comp, on="protein_id", how="left")
    assert (g.gid.values == np.arange(len(g))).all()
    L = g.aa_len.to_numpy()
    H = g.hyd.to_numpy()
    tok = g.token.to_numpy()
    gm = pd.read_parquet(D / "genomes.parquet").set_index("genome_code")
    rows = []
    for _, r in reps.iterrows():
        rows.append(dict(fold_id=f"cog2_{r.s2_rank:03d}_r{r.rep}", cls="cognate", s2_rank=r.s2_rank, pool=r.pool, k=r.k,
                         rep=r.rep, type_key=r.type_key, gids=r.gids, genome=r.genome, slot=-1, match_level=""))
    cog = pd.DataFrame(rows)

    # ---- non-cognate swaps ------------------------------------------------------------------
    nc = []
    for rep in (1, 2):
        sub = cog[cog.rep == rep].reset_index(drop=True)
        donors = [(i, s, int(x)) for i, gl in enumerate(sub.gids) for s, x in enumerate(gl.split(";"))]
        reqs = []
        for i, r in sub.iterrows():
            k = int(r.k)
            slots = range(1, k) if rep == 1 else range(0, k - 1)
            for s in slots:
                reqs.append((i, s, int(r.gids.split(";")[s])))
        dg = np.array([d[2] for d in donors])
        cost = np.zeros((len(reqs), len(donors)))
        for a, (i, s, x) in enumerate(reqs):
            block_toks = {tok[int(y)] for y in sub.gids[i].split(";")}
            cost[a] = np.abs(L[dg] - L[x]) / L[x] + np.abs(H[dg] - H[x])
            bad = np.array([d[0] == i or tok[d[2]] in block_toks for d in donors])
            cost[a, bad] = 1e6
        ri, cj = linear_sum_assignment(cost)
        assert (cost[ri, cj] < 1e6).all()
        for a, b in zip(ri, cj):
            i, s, x = reqs[a]
            r = sub.iloc[i]
            new = [int(y) for y in r.gids.split(";")]
            d = donors[b][2]
            new[s] = d
            nc.append(dict(fold_id=f"non2_{r.s2_rank:03d}_r{rep}_s{s}", cls="noncognate", s2_rank=r.s2_rank, pool=r.pool,
                           k=r.k, rep=rep, type_key=r.type_key, gids=";".join(map(str, new)), genome=r.genome, slot=s,
                           match_level=f"donor_rank={sub.s2_rank[donors[b][0]]};len_rel_diff={abs(L[d]-L[x])/L[x]:.3f};"
                                       f"hyd_diff={abs(H[d]-H[x]):.3f}"))
    nc = pd.DataFrame(nc)
    print("non-cognate", len(nc), "max rel length diff",
          nc.match_level.str.extract(r"len_rel_diff=([0-9.]+)")[0].astype(float).describe()[["50%", "90%" if False else "75%", "max"]].to_dict())

    # ---- uncalled windows -------------------------------------------------------------------
    sp = pd.read_parquet(D / "significant_pairs.parquet")
    T = int(tok.max()) + 1
    sig = np.sort(sp.key.values)
    cc = g.contig_code.to_numpy()
    strand = g.strand.to_numpy()
    dark = g.dark.to_numpy()
    inrun = np.zeros(len(g), bool)
    for x in pd.read_parquet(W2 / "block_instances.parquet", columns=["gids"]).gids:
        inrun[list(x)] = True
    okgene = dark & ~inrun & (g.n_x.to_numpy() == 0) & (L >= 60) & (L <= 1200)
    a = np.flatnonzero(cc[:-1] == cc[1:])
    okpair = np.zeros(len(g), bool)          # okpair[i]: genes i, i+1 form an admissible adjacent pair
    key = np.minimum(tok[a], tok[a + 1]) * T + np.maximum(tok[a], tok[a + 1])
    okpair[a] = (okgene[a] & okgene[a + 1] & (strand[a] == strand[a + 1]) & (tok[a] != tok[a + 1])
                 & ~np.isin(key, sig))
    gcode = g.genome_code.to_numpy()
    ordv = gm.loc[gcode, "ord"].to_numpy()
    windows = {}
    for k in sorted(cog.k.unique()):
        st = np.flatnonzero(okpair)
        good = np.ones(len(st), bool)
        for j in range(1, k - 1):
            good &= okpair[np.minimum(st + j, len(g) - 1)] & (cc[np.minimum(st + j, len(g) - 1)] == cc[st])
        st = st[good]
        windows[k] = st
        print(f"uncalled windows k={k}: {len(st):,}")
    used = set()
    un = []
    for _, r in cog.iterrows():
        k = int(r.k)
        x = np.array([int(y) for y in r.gids.split(";")])
        st = windows[k]
        found = None
        for tol_l, tol_h in ((0.25, 0.05), (0.40, 0.08)):
            for level in ("contig", "genome", "order", "any"):
                if level == "contig":
                    cand = st[cc[st] == cc[x[0]]]
                elif level == "genome":
                    cand = st[gcode[st] == gcode[x[0]]]
                elif level == "order":
                    cand = st[ordv[st] == ordv[x[0]]]
                else:
                    cand = st
                if len(cand) == 0:
                    continue
                idx = cand[:, None] + np.arange(k)[None, :]
                best = None
                for orient in (idx, idx[:, ::-1]):
                    dl = (np.abs(L[orient] - L[x]) / L[x]).max(1)
                    dh = np.abs(H[orient] - H[x]).max(1)
                    okm = (dl <= tol_l) & (dh <= tol_h)
                    okm &= np.array([tuple(o) not in used for o in map(tuple, orient)]) if okm.any() else okm
                    if okm.any():
                        s = np.where(okm, dl + dh, np.inf)
                        b = int(np.argmin(s))
                        if best is None or s[b] < best[0]:
                            best = (s[b], orient[b], dl[b], dh[b])
                if best is not None:
                    found = (level, tol_l, best)
                    break
            if found:
                break
        if found is None:
            continue
        level, tol_l, (_, win, dl, dh) = found
        used.add(tuple(win))
        used.add(tuple(win[::-1]))
        un.append(dict(fold_id=f"unc2_{r.s2_rank:03d}_r{r.rep}", cls="uncalled", s2_rank=r.s2_rank, pool=r.pool, k=k,
                       rep=r.rep, type_key=r.type_key, gids=";".join(map(str, win)),
                       genome=gm.at[gcode[win[0]], "genome"], slot=-1,
                       match_level=f"{level};tol={tol_l};len_rel_diff={dl:.3f};hyd_diff={dh:.3f}"))
    un = pd.DataFrame(un)
    print("uncalled", len(un), "of", len(cog), un.match_level.str.split(";").str[:2].str.join(";").value_counts().to_dict())

    # ---- positives -------------------------------------------------------------------------
    gi = pd.read_parquet(D / "instances.parquet")
    gi = gi[(gi.n_members_resolved >= 2) & ~((gi.caller == "CRISPRCasTyper") & (gi.status == "cas_putative"))]
    prow = []
    ks = sorted(set(cog.k.unique()) | {2})
    for _, r in gi.iterrows():
        mg = sorted(int(v) for v in r.member_gids)
        for k in ks:
            for s in range(0, len(mg) - k + 1):
                w = mg[s:s + k]
                if w[-1] - w[0] != k - 1 or len({cc[v] for v in w}) != 1:
                    continue
                lw, hw = L[w], H[w]
                if lw.min() < 60 or lw.max() > 1200 or lw.sum() > 2500 or hw.max() > 0.45 or g.n_x.values[w].sum() > 0:
                    continue
                if k == 2 and lw.max() <= 700:
                    continue
                prow.append(dict(k=k, caller=r.caller, system_type=str(r.system_type), status=r.status,
                                 instance_id=int(r.instance_id), gids=";".join(map(str, w)), genome=r.genome,
                                 genome_code=int(gcode[w[0]])))
    pp = pd.DataFrame(prow).sample(frac=1, random_state=17)
    pos = []
    for k in ks:
        if k not in N_POS:
            continue
        sub = pp[pp.k == k]
        groups = {kk: v for kk, v in sub.groupby(["caller", "system_type"])}
        order = sorted(groups, key=lambda kk: -len(groups[kk]))
        picks, seen = {kk: [] for kk in groups}, {kk: set() for kk in groups}
        tot = 0
        for _round in range(4):
            for kk in order:
                if tot >= N_POS[k]:
                    break
                for _, r in groups[kk].iterrows():
                    if r.genome_code in seen[kk] or r.instance_id in {p.instance_id for p in picks[kk]}:
                        continue
                    picks[kk].append(r)
                    seen[kk].add(r.genome_code)
                    tot += 1
                    break
        for kk, lst in picks.items():
            for n, r in enumerate(lst, 1):
                tag = f"{kk[0][:3]}_{kk[1]}".replace("/", "_").replace(" ", "_")
                pos.append(dict(fold_id=f"pos2_k{k}_{tag}_{n}", cls="positive", s2_rank=-1, pool="positive", k=k, rep=n,
                                type_key=f"{kk[0]}:{kk[1]}", gids=r.gids, genome=r.genome, slot=-1,
                                match_level=f"status={r.status};instance={r.instance_id}"))
    pos = pd.DataFrame(pos)
    print("positives", len(pos), pos.groupby("k").size().to_dict(), pos.type_key.str.split(":").str[0].value_counts().to_dict())

    man = pd.concat([cog, nc, un, pos], ignore_index=True)
    man["lengths"] = man.gids.map(lambda s: ";".join(str(int(L[int(x)])) for x in s.split(";")))
    man["hyds"] = man.gids.map(lambda s: ";".join(f"{H[int(x)]:.3f}" for x in s.split(";")))
    man["tokens"] = man.gids.map(lambda s: ";".join(str(int(tok[int(x)])) for x in s.split(";")))
    man["total_aa"] = man.lengths.map(lambda s: sum(int(x) for x in s.split(";")))
    assert man.total_aa.max() <= 4096
    man.to_csv(W2 / "fold_manifest2.tsv", sep="\t", index=False)
    # sequences
    gids = sorted({int(x) for s in man.gids for x in s.split(";")})
    sh = duckdb.connect(SHARUR, read_only=True)
    sh.execute("SET threads=1")
    pid = g.protein_id.to_numpy()
    sh.register("p", pd.DataFrame({"protein_id": pid[gids]}))
    seq = dict(sh.execute("SELECT q.protein_id, rtrim(q.sequence,'*') FROM proteins q JOIN p USING (protein_id)").fetchall())
    with open(W2 / "jobs_s2.jsonl", "w") as f:
        for _, r in man.iterrows():
            gl = [int(x) for x in r.gids.split(";")]
            chains = [{"id": "ABCDEFGH"[i], "seq": seq[pid[x]], "gid": x} for i, x in enumerate(gl)]
            f.write(json.dumps({"fold_id": r.fold_id, "cls": r.cls, "chains": chains}) + "\n")
    print(man.cls.value_counts().to_dict(), "items", len(man), "unique gids", len(gids), "max aa", int(man.total_aa.max()))


if __name__ == "__main__":
    main()
