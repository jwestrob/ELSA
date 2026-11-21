"""Step 2 design: choose the fold set (cognate replicates, non-cognate, uncalled-neighbour and
positive-control pairs) and write work/fold_manifest.tsv (one row per complex, chain gids in order).

Cognate: the top N_TOP non-redundant candidates (candidates_all.tsv, nonredundant_rank), each as
two instances chosen for maximal taxonomic distance (phylum > class > order > family > genus), with
instance quality (no X, per-protein hydrophobic fraction <= 0.45, lengths within 0.8-1.25 x the family
median, not at a contig edge, Prodigal partial flag 00 or absent, genome completeness).
Non-cognate: replicate 1 keeps chain 1 and takes chain 2 from another candidate's replicate 1;
replicate 2 keeps chain 2 and takes chain 1 from another candidate's replicate 2; partner chosen
by closest length (then hydrophobic fraction), each partner used once per slot.
Uncalled neighbour: an adjacent same-strand pair of dark genes whose token pair is not significant
in the engine run and that lies in no dark-link run, matched to the cognate pair on both lengths
(+-25 %) and hydrophobic fractions (+-0.05); searched on the same contig, then the same genome, then
genomes of the same order; the level used is recorded.
Positive control: adjacent members (d = 1) of one curated gold instance, both 60-700 aa, sum <= 1,400,
hydrophobic fraction <= 0.45, no X; up to 4 per (caller, system type) from distinct genomes, round
robin over types until N_POS.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"
D = W / "data"
SHARUR = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
N_TOP = 150
N_POS = 150
TAX = ["phylum", "class", "ord", "family", "genus"]
RNG = np.random.default_rng(17)


def taxdist(a, b):
    """5 = different phylum, 4 class, 3 order, 2 family, 1 genus, 0 same genus (missing ranks differ)."""
    for lvl, c in zip((5, 4, 3, 2, 1), TAX):
        if pd.isna(a[c]) or pd.isna(b[c]) or a[c] != b[c]:
            return lvl
    return 0


def main():
    g = pd.read_parquet(W / "genes_dark.parquet")
    comp = pd.read_parquet(D / "composition.parquet")[["protein_id", "hyd", "n_x"]]
    g = g.merge(comp, on="protein_id", how="left")
    assert (g.gid.values == np.arange(len(g))).all()
    gm = pd.read_parquet(D / "genomes.parquet")
    for c in TAX:
        gm[c] = gm[c].replace("", np.nan)
    gm = gm.set_index("genome_code")
    ct = pd.read_parquet(D / "contigs.parquet").set_index("contig_code")
    fam = pd.read_parquet(W / "fam_stats.parquet")
    cand = pd.read_csv(W / "candidates_all.tsv", sep="\t")
    top = cand[cand.nonredundant_rank.notna() & (cand.nonredundant_rank <= N_TOP)].sort_values("nonredundant_rank")
    inst = pd.read_parquet(W / "block_instances.parquet")
    inst = inst[inst.type_key.isin(top.type_key)].copy()
    tok = g.token.to_numpy()

    # ---- cognate replicates ----------------------------------------------------------------
    rows = []
    for _, c in top.iterrows():
        tt = [int(x) for x in c.tokens.split(";")]
        flen = [fam.at[t, "fam_len_med"] for t in tt]
        df = inst[inst.type_key == c.type_key].copy()
        q = []
        for _, r in df.iterrows():
            okx = r.n_x == 0
            okh = max(r.hyds) <= 0.45
            okl = all(0.8 * f <= L <= 1.25 * f for L, f in zip(r.lengths, flen))
            oke = r.edge_dist >= 1
            okp = all(p in ("00", "na") for p in r.partials)
            comp_ = 0 if pd.isna(r.completeness) else r.completeness
            q.append(4 * okx + 2 * okh + 2 * okl + 1 * oke + 1 * okp + comp_ / 100)
        df["quality"] = q
        df = df.sort_values("quality", ascending=False)
        # top 60 by quality plus the best instance of every order, so rare deep lineages stay eligible
        df = pd.concat([df.head(60), df.drop_duplicates("ord")]).drop_duplicates("gids" if False else ["genome_code", "pos_start"])
        best, bscore = None, None
        recs = df.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                if recs[i]["genome_code"] == recs[j]["genome_code"]:
                    continue
                s = (taxdist(recs[i], recs[j]), min(recs[i]["quality"], recs[j]["quality"]),
                     recs[i]["quality"] + recs[j]["quality"])
                if bscore is None or s > bscore:
                    best, bscore = (recs[i], recs[j]), s
        for rep, r in enumerate(best, 1):
            rows.append(dict(fold_id=f"cog_{int(c.nonredundant_rank):03d}_r{rep}", cls="cognate",
                             cand_rank=int(c.nonredundant_rank), type_key=c.type_key, rep=rep,
                             gids=";".join(map(str, r["gids"])), genome=r["genome"], phylum=r["phylum"],
                             ord=r["ord"], genus=r["genus"], completeness=r["completeness"],
                             edge_dist=int(r["edge_dist"]), quality=round(r["quality"], 3),
                             replicate_taxdist=bscore[0], lengths=";".join(map(str, r["lengths"])),
                             hyds=";".join(f"{h:.3f}" for h in r["hyds"]), match_level=""))
    cog = pd.DataFrame(rows)
    print("cognate folds", len(cog), "replicate taxdist", cog.groupby("cand_rank").replicate_taxdist.first().value_counts().to_dict())

    L = g.length.to_numpy()
    H = g.hyd.to_numpy()

    # ---- non-cognate controls ---------------------------------------------------------------
    nc_rows = []
    for rep, slot in ((1, 1), (2, 0)):          # rep 1 swaps chain 2; rep 2 swaps chain 1
        sub = cog[cog.rep == rep].reset_index(drop=True)
        donors = [(i, int(x.split(";")[slot])) for i, x in enumerate(sub.gids)]
        # optimal one-to-one assignment (Hungarian) on relative length difference + hydrophobic difference
        from scipy.optimize import linear_sum_assignment
        tg = np.array([int(x.split(";")[slot]) for x in sub.gids])
        dl = np.abs(L[tg][None, :] - L[tg][:, None]) / L[tg][:, None]
        dh = np.abs(H[tg][None, :] - H[tg][:, None])
        cost = dl + dh
        np.fill_diagonal(cost, 1e6)
        ri, cj = linear_sum_assignment(cost)
        assign = dict(zip(ri, cj))
        for i, r in sub.iterrows():
            keep = [int(x) for x in r.gids.split(";")]
            j = assign[i]
            d = donors[j][1]
            relL, dH = dl[i, j], dh[i, j]
            new = list(keep)
            new[slot] = d
            nc_rows.append(dict(fold_id=f"non_{int(r.cand_rank):03d}_r{rep}", cls="noncognate", cand_rank=int(r.cand_rank),
                                type_key=r.type_key, rep=rep, gids=";".join(map(str, new)),
                                genome=r.genome, partner_rank=int(sub.cand_rank[j]),
                                lengths=";".join(str(int(L[x])) for x in new), hyds=";".join(f"{H[x]:.3f}" for x in new),
                                match_level=f"len_rel_diff={relL:.3f};hyd_diff={dH:.3f}"))
    nc = pd.DataFrame(nc_rows)
    print("non-cognate", len(nc), "max rel length diff",
          nc.match_level.str.extract(r"len_rel_diff=([0-9.]+)")[0].astype(float).describe()[["50%", "max"]].to_dict())

    # ---- uncalled-neighbour controls -------------------------------------------------------
    sp = pd.read_parquet(D / "significant_pairs.parquet")
    T = int(tok.max()) + 1
    sig_sorted = np.sort(sp.key.values)
    cc = g.contig_code.to_numpy()
    a = np.flatnonzero(cc[:-1] == cc[1:])
    b = a + 1
    key = np.minimum(tok[a], tok[b]) * T + np.maximum(tok[a], tok[b])
    dark = g.dark.to_numpy()
    strand = g.strand.to_numpy()
    in_run = np.zeros(len(g), bool)
    allinst = pd.read_parquet(W / "block_instances.parquet", columns=["gids"])
    for x in allinst.gids:
        in_run[list(x)] = True
    okp = (dark[a] & dark[b] & (strand[a] == strand[b]) & (tok[a] != tok[b]) & ~np.isin(key, sig_sorted)
           & ~in_run[a] & ~in_run[b] & (g.n_x.to_numpy()[a] == 0) & (g.n_x.to_numpy()[b] == 0)
           & (L[a] >= 60) & (L[b] >= 60) & (L[a] <= 700) & (L[b] <= 700))
    pool = pd.DataFrame(dict(a=a[okp], b=b[okp]))
    pool["contig_code"] = cc[pool.a]
    pool["genome_code"] = g.genome_code.to_numpy()[pool.a]
    pool["ord"] = gm.loc[pool.genome_code, "ord"].to_numpy()
    print("uncalled pool", len(pool))
    by_contig = {k: v for k, v in pool.groupby("contig_code")}
    by_genome = {k: v for k, v in pool.groupby("genome_code")}
    by_ord = {k: v for k, v in pool.groupby("ord")}
    un_rows, used_pairs = [], set()
    for _, r in cog.iterrows():
        x, y = [int(v) for v in r.gids.split(";")]
        gc_ = g.genome_code.values[x]
        found = None
        for level, src in (("contig", by_contig.get(cc[x])), ("genome", by_genome.get(gc_)),
                           ("order", by_ord.get(gm.at[gc_, "ord"]))):
            if src is None:
                continue
            best = None
            for pa, pb in zip(src.a.values, src.b.values):
                if (pa, pb) in used_pairs:
                    continue
                for u, v in ((pa, pb), (pb, pa)):
                    dl = max(abs(L[u] - L[x]) / L[x], abs(L[v] - L[y]) / L[y])
                    dh = max(abs(H[u] - H[x]), abs(H[v] - H[y]))
                    if dl <= 0.25 and dh <= 0.05:
                        s = dl + dh
                        if best is None or s < best[0]:
                            best = (s, u, v, dl, dh)
            if best is not None:
                found = (level, best)
                break
        if found is None:
            continue
        level, (_, u, v, dl, dh) = found
        used_pairs.add((min(u, v), max(u, v)))
        un_rows.append(dict(fold_id=f"unc_{int(r.cand_rank):03d}_r{int(r.rep)}", cls="uncalled", cand_rank=int(r.cand_rank),
                            type_key=r.type_key, rep=int(r.rep), gids=f"{u};{v}",
                            genome=gm.at[g.genome_code.values[u], "genome"],
                            lengths=f"{L[u]};{L[v]}", hyds=f"{H[u]:.3f};{H[v]:.3f}",
                            match_level=f"{level};len_rel_diff={dl:.3f};hyd_diff={dh:.3f}"))
    un = pd.DataFrame(un_rows)
    print("uncalled", len(un), un.match_level.str.split(";").str[0].value_counts().to_dict())

    # ---- positive controls ----------------------------------------------------------------
    gi = pd.read_parquet(D / "instances.parquet")
    gi = gi[gi.n_members_resolved >= 2]
    sh = duckdb.connect(SHARUR, read_only=True)
    sh.execute("SET threads=1")
    prof = sh.execute("SELECT protein_id, system_source, profile_name FROM system_proteins").df()
    pid = g.protein_id.to_numpy()
    prow = []
    for _, r in gi.iterrows():
        mg = sorted(int(v) for v in r.member_gids)
        for u, v in zip(mg[:-1], mg[1:]):
            if v != u + 1 or cc[u] != cc[v]:
                continue
            if not (60 <= L[u] <= 700 and 60 <= L[v] <= 700 and L[u] + L[v] <= 1400 and H[u] <= 0.45 and H[v] <= 0.45
                    and g.n_x.values[u] == 0 and g.n_x.values[v] == 0):
                continue
            prow.append(dict(caller=r.caller, system_type=str(r.system_type), status=r.status, instance_id=int(r.instance_id),
                             u=u, v=v, genome=r.genome, genome_code=int(g.genome_code.values[u])))
    pp = pd.DataFrame(prow)
    print("positive pool", len(pp), pp.groupby("caller").size().to_dict())
    pp = pp.sample(frac=1, random_state=17)
    groups = {k: v for k, v in pp.groupby(["caller", "system_type"])}
    picks, seen_g = {k: [] for k in groups}, {k: set() for k in groups}
    order = sorted(groups, key=lambda k: -len(groups[k]))
    total = 0
    for _round in range(4):
        for k in order:
            if total >= N_POS:
                break
            for _, r in groups[k].iterrows():
                if r.genome_code in seen_g[k] or any(r.u == p.u for p in picks[k]):
                    continue
                picks[k].append(r)
                seen_g[k].add(r.genome_code)
                total += 1
                break
    pos_rows = []
    profmap = prof.groupby("protein_id").profile_name.agg(lambda s: "|".join(sorted(set(s))))
    for k, lst in picks.items():
        for n, r in enumerate(lst, 1):
            pos_rows.append(dict(fold_id=f"pos_{k[0][:3]}_{k[1]}_{n}".replace("/", "_").replace(" ", "_"), cls="positive",
                                 type_key=f"{k[0]}:{k[1]}", rep=n, gids=f"{r.u};{r.v}", genome=r.genome,
                                 lengths=f"{L[r.u]};{L[r.v]}", hyds=f"{H[r.u]:.3f};{H[r.v]:.3f}",
                                 match_level=f"status={r.status};instance={r.instance_id};profiles="
                                             f"{profmap.get(pid[r.u], '')}/{profmap.get(pid[r.v], '')}"))
    pos = pd.DataFrame(pos_rows)
    print("positive", len(pos), pos.type_key.str.split(":").str[0].value_counts().to_dict())

    man = pd.concat([cog, nc, un, pos], ignore_index=True)
    man.to_csv(W / "fold_manifest.tsv", sep="\t", index=False)
    gids = sorted({int(x) for s in man.gids for x in s.split(";")})
    open(W / "fold_gids.txt", "w").write("\n".join(map(str, gids)) + "\n")
    print(man.cls.value_counts().to_dict(), "unique gids", len(gids))


if __name__ == "__main__":
    main()
