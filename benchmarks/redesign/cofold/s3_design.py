"""Screen 3 design (runs serially on the biotite login node): candidate pool, replicates, controls, positives,
Biohub jobs. Rules: predeclared_criteria_screen3.json ('pool', 'replicates', 'controls').

Pool      census types (s3_census.py) with every chain 'no_hit' or 'unknown_only' (all_unrecognised), passing
          every screen filter (pass_fold: dark families, 60-1,200 aa, sum <= 2,500, hydrophobic <= 0.45, low
          complexity <= 0.145, copies <= 1.5, geometry, >= 5 genomes and >= 2 orders), not folded in screens 1-2
          (exact mapped type); ranked by orders, phyla, k (larger first), genomes, block integrity; a type whose
          token set is contained in the token set of a better-ranked selected type is dropped (screen-2 rule);
          top N_MAX.
Replicates  screen-1/2 rule (s2_select.replicates): instance quality, then the pair of instances from different
          genomes with maximal taxonomic distance, then quality.
Cognate   cog3_<rank>_r<rep>: the replicate's k chains in gene order.
Swap      non3_<rank>_r<rep>_s<slot>: one per block (half the earlier ratio). Block ranks alternate the replicate
          (odd rank: replicate 1, even rank: replicate 2). Replicate 1 replaces chain index 1 (B), replicate 2
          replaces chain index k-2 (A for pairs), so the control serves the per-interface margin of the
          adjacent interface (A,B) in replicate 1 and (k-2,k-1) in replicate 2. Donor = a chain of another
          pool block in the same replicate, one-to-one (Hungarian) on relative length + hydrophobic difference;
          donors sharing a token with the receiving block are forbidden.
Uncalled  unc3_<rank>_r<rep>: one per block, in the other replicate; k adjacent same-strand unannotated genes,
          no X, 60-1,200 aa, consecutive token pairs all non-significant, none in a census link run; lengths
          within 25 % per position and hydrophobic fraction within 0.05 (then 40 % / 0.08); same contig, then
          genome, then order, then anywhere; each window used once.
Positive  pos3_k<k>_<caller>_<type>_<n>: 60 windows of k consecutive member genes of one curated gold instance
          (DefenseFinder, TXSScan, CRISPRCasTyper without status cas_putative), each 60-1,200 aa, sum <= 2,500,
          hydrophobic <= 0.45, no X; k distribution proportional to the pool's; <= 4 per (caller, system type),
          distinct genomes, round robin (seed 17).
Writes $RD/cofold/s3/design/: pool.tsv, reps.tsv, fold_manifest3.tsv, jobs_s3.jsonl, design_summary.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
U = RD / "ws_dpann/package/universe"
GOLD = RD / "ws_dpann/package/gold"
ENG = RD / "engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3"
TOKF = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
FAA = RD / "ws_dpann/work/proteins_gid.faa"
S3 = RD / "cofold/s3"
D = S3 / "data"
C = S3 / "census"
OUT = S3 / "design"
TAX = ["phylum", "class", "ord", "family", "genus"]
N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 500
N_POS = 60
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def taxdist(a, b):
    for lvl, c in zip((5, 4, 3, 2, 1), TAX):
        if pd.isna(a[c]) or pd.isna(b[c]) or a[c] != b[c]:
            return lvl
    return 0


def replicates(pool, inst, flen_of):
    rows = []
    for _, c in pool.iterrows():
        tt = [int(x) for x in str(c.tokens).split(";")]
        flen = [flen_of[t] for t in tt]
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
        df = pd.concat([df.head(60), df.drop_duplicates("ord")]).drop_duplicates(["genome_code", "pos_start"])
        recs = df.to_dict("records")
        best, bscore = None, None
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                if recs[i]["genome_code"] == recs[j]["genome_code"]:
                    continue
                s = (taxdist(recs[i], recs[j]), min(recs[i]["quality"], recs[j]["quality"]),
                     recs[i]["quality"] + recs[j]["quality"])
                if bscore is None or s > bscore:
                    best, bscore = (recs[i], recs[j]), s
        for rep, r in enumerate(best, 1):
            rows.append(dict(s3_rank=int(c.s3_rank), type_key=c.type_key, k=int(c.k), rep=rep,
                             gids=";".join(map(str, r["gids"])), genome=r["genome"], phylum=r["phylum"],
                             cls_=r["class"], ord=r["ord"], family=r["family"], genus=r["genus"],
                             completeness=r["completeness"], edge_dist=int(r["edge_dist"]),
                             quality=round(r["quality"], 3), replicate_taxdist=bscore[0],
                             lengths=";".join(map(str, r["lengths"])), hyds=";".join(f"{h:.3f}" for h in r["hyds"]),
                             partials=";".join(r["partials"]), n_x=int(r["n_x"]),
                             intergenic_nt=";".join(map(str, r["intergenic_nt"])),
                             contig_code=int(r["contig_code"]), pos_start=int(r["pos_start"]), pos_end=int(r["pos_end"])))
    return pd.DataFrame(rows).rename(columns={"cls_": "class"})


def read_faa(path, want):
    seqs, keep, h, buf = {}, False, None, []
    with open(path) as f:
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ct = pd.read_parquet(C / "census_types.parquet")
    st = pd.read_parquet(C / "token_status.parquet").set_index("token")
    ok = ct[ct.all_unrecognised & ct.pass_fold & (ct.folded_s12 == "")].copy()
    log(f"census types {len(ct):,}; all chains unrecognised {int(ct.all_unrecognised.sum()):,}; + pass_fold "
        f"{int((ct.all_unrecognised & ct.pass_fold).sum()):,}; + not folded in screens 1-2 {len(ok):,}")
    ok = ok.sort_values(["n_ord", "n_phylum", "k", "n_genomes", "block_integrity"], ascending=False)
    ok["toks"] = ok.tokens.map(lambda s: [int(x) for x in s.split(";")])
    sel, sets = [], []
    for key, tt in zip(ok.type_key, ok.toks):
        s = set(tt)
        if any(s <= t for t in sets):
            continue
        sel.append(key)
        sets.append(s)
    pool = ok[ok.type_key.isin(sel)].copy()
    n_nonred = len(pool)
    pool = pool.head(N_MAX).copy()
    pool["s3_rank"] = np.arange(1, len(pool) + 1)
    pool = pool.drop(columns=["toks"])
    pool.to_csv(OUT / "pool.tsv", sep="\t", index=False)
    log(f"non-redundant {n_nonred}; pool {len(pool)}; by k {pool.k.value_counts().sort_index().to_dict()}; "
        f"by class {pool.unrec_class.value_counts().to_dict()}")

    inst = pd.read_parquet(C / "census_instances.parquet")
    inst = inst[inst.type_key.isin(pool.type_key)]
    flen_of = st.fam_len_med.to_dict()
    reps = replicates(pool, inst, flen_of)
    reps.to_csv(OUT / "reps.tsv", sep="\t", index=False)
    log("replicates", len(reps), "taxdist", reps.groupby("s3_rank").replicate_taxdist.first().value_counts().to_dict())

    # ---- gene table for controls -----------------------------------------------------------------------
    g = pd.read_parquet(U / "genes.parquet", columns=["gid", "protein_id", "genome_code", "contig_code", "strand"])
    fl = pd.read_parquet(D / "prot_flags_slim.parquet", columns=["protein_id", "cur_annot", "in_sys", "derived"])
    comp = pd.read_parquet(D / "composition_slim.parquet")
    g = g.merge(fl, on="protein_id", how="left").merge(comp, on="protein_id", how="left")
    assert (g.gid.values == np.arange(len(g))).all()
    mem = pd.read_parquet(GOLD / "members.parquet")
    g["dark"] = ~(g.cur_annot | g.in_sys | g.derived | g.gid.isin(mem.gid))
    tok = pd.read_parquet(TOKF).sort_values("gid").token.to_numpy()
    # earlier enriched triage (screen 2 + Part A, s2_triage_analyze.py): class per triaged family, carried to the
    # rescue3 token of the family's query gene; a label only (predeclared), never a pool filter
    tri = pd.read_csv(D / "triage_family.tsv", sep="\t")
    tri["tok3"] = tok[tri.query_gid.to_numpy()]
    rank_ = {"none": 0, "weak": 1, "unknown_function": 2, "characterised": 3}
    tri = tri.sort_values("cls", key=lambda s: s.map(rank_)).drop_duplicates("tok3", keep="last")
    tcls = dict(zip(tri.tok3, tri.cls))
    pool["chain_triage_s2"] = pool.tokens.map(lambda s: ";".join(tcls.get(int(x), "-") for x in s.split(";")))
    pool["earlier_triage_characterised"] = pool.chain_triage_s2.str.contains("characterised")
    pool.to_csv(OUT / "pool.tsv", sep="\t", index=False)
    log("earlier triage on pool chains:", pool.chain_triage_s2.str.split(";").explode().value_counts().to_dict(),
        "; blocks with a chain characterised by it:", int(pool.earlier_triage_characterised.sum()))
    L = g.aa_len.to_numpy()
    H = g.hyd.to_numpy()
    NX = g.n_x.to_numpy()
    gm = pd.read_parquet(U / "genomes.parquet").set_index("genome_code")
    gcode = g.genome_code.to_numpy()
    cc = g.contig_code.to_numpy()
    strand = g.strand.to_numpy()
    dark = g.dark.to_numpy()
    T = int(tok.max()) + 1
    sp = pd.read_parquet(ENG / "significant_pairs.parquet", columns=["key"])
    sig = np.sort(sp.key.values)
    a = np.flatnonzero(cc[:-1] == cc[1:])
    key = np.minimum(tok[a], tok[a + 1]) * T + np.maximum(tok[a], tok[a + 1])
    lk = (tok[a] != tok[a + 1]) & dark[a] & dark[a + 1] & np.isin(key, sig)
    inrun = np.zeros(len(g), bool)
    inrun[a[lk]] = True
    inrun[a[lk] + 1] = True

    rows = []
    for _, r in reps.iterrows():
        rows.append(dict(fold_id=f"cog3_{r.s3_rank:03d}_r{r.rep}", cls="cognate", s3_rank=r.s3_rank, k=r.k, rep=r.rep,
                         type_key=r.type_key, gids=r.gids, genome=r.genome, slot=-1, match_level=""))
    cog = pd.DataFrame(rows)
    ctrl_rep = {rk: (1 if rk % 2 == 1 else 2) for rk in pool.s3_rank}

    # ---- swap controls (one per block) -----------------------------------------------------------------
    nc = []
    for rep in (1, 2):
        sub = cog[cog.rep == rep].reset_index(drop=True)
        donors = [(i, s, int(x)) for i, gl in enumerate(sub.gids) for s, x in enumerate(gl.split(";"))]
        reqs = []
        for i, r in sub.iterrows():
            if ctrl_rep[r.s3_rank] != rep:
                continue
            k = int(r.k)
            s = 1 if rep == 1 else k - 2
            reqs.append((i, s, int(r.gids.split(";")[s])))
        dg = np.array([d[2] for d in donors])
        cost = np.zeros((len(reqs), len(donors)))
        for a_, (i, s, x) in enumerate(reqs):
            block_toks = {tok[int(y)] for y in sub.gids[i].split(";")}
            cost[a_] = np.abs(L[dg] - L[x]) / L[x] + np.abs(H[dg] - H[x])
            bad = np.array([d[0] == i or tok[d[2]] in block_toks for d in donors])
            cost[a_, bad] = 1e6
        ri, cj = linear_sum_assignment(cost)
        assert (cost[ri, cj] < 1e6).all()
        for a_, b_ in zip(ri, cj):
            i, s, x = reqs[a_]
            r = sub.iloc[i]
            new = [int(y) for y in r.gids.split(";")]
            d = donors[b_][2]
            new[s] = d
            nc.append(dict(fold_id=f"non3_{r.s3_rank:03d}_r{rep}_s{s}", cls="noncognate", s3_rank=r.s3_rank, k=r.k, rep=rep,
                           type_key=r.type_key, gids=";".join(map(str, new)), genome=r.genome, slot=s,
                           match_level=f"donor_rank={sub.s3_rank[donors[b_][0]]};len_rel_diff={abs(L[d]-L[x])/L[x]:.3f};"
                                       f"hyd_diff={abs(H[d]-H[x]):.3f}"))
    nc = pd.DataFrame(nc)
    log("swap controls", len(nc), nc.match_level.str.extract(r"len_rel_diff=([0-9.]+)")[0].astype(float).describe()[["50%", "max"]].to_dict())

    # ---- uncalled windows (one per block, the other replicate) -----------------------------------------
    okgene = dark & ~inrun & (NX == 0) & (L >= 60) & (L <= 1200)
    okpair = np.zeros(len(g), bool)
    okpair[a] = okgene[a] & okgene[a + 1] & (strand[a] == strand[a + 1]) & (tok[a] != tok[a + 1]) & ~np.isin(key, sig)
    ordv = gm.loc[gcode, "ord"].to_numpy()
    windows = {}
    for k in sorted(cog.k.unique()):
        s_ = np.flatnonzero(okpair)
        good = np.ones(len(s_), bool)
        for j in range(1, k - 1):
            jj = np.minimum(s_ + j, len(g) - 1)
            good &= okpair[jj] & (cc[jj] == cc[s_])
        windows[k] = s_[good]
        log(f"uncalled windows k={k}: {len(windows[k]):,}")
    used = set()
    un = []
    for _, r in cog.iterrows():
        if ctrl_rep[r.s3_rank] == r.rep:
            continue
        k = int(r.k)
        x = np.array([int(y) for y in r.gids.split(";")])
        st_ = windows[k]
        found = None
        for tol_l, tol_h in ((0.25, 0.05), (0.40, 0.08)):
            for level in ("contig", "genome", "order", "any"):
                if level == "contig":
                    cand = st_[cc[st_] == cc[x[0]]]
                elif level == "genome":
                    cand = st_[gcode[st_] == gcode[x[0]]]
                elif level == "order":
                    cand = st_[ordv[st_] == ordv[x[0]]]
                else:
                    cand = st_
                if len(cand) == 0:
                    continue
                idx = cand[:, None] + np.arange(k)[None, :]
                best = None
                for orient in (idx, idx[:, ::-1]):
                    dl = (np.abs(L[orient] - L[x]) / L[x]).max(1)
                    dh = np.abs(H[orient] - H[x]).max(1)
                    okm = (dl <= tol_l) & (dh <= tol_h)
                    if okm.any():
                        okm &= np.array([tuple(o) not in used for o in map(tuple, orient)])
                    if okm.any():
                        sc = np.where(okm, dl + dh, np.inf)
                        bb = int(np.argmin(sc))
                        if best is None or sc[bb] < best[0]:
                            best = (sc[bb], orient[bb], dl[bb], dh[bb])
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
        un.append(dict(fold_id=f"unc3_{r.s3_rank:03d}_r{r.rep}", cls="uncalled", s3_rank=r.s3_rank, k=k, rep=r.rep,
                       type_key=r.type_key, gids=";".join(map(str, win)), genome=gm.at[gcode[win[0]], "genome"], slot=-1,
                       match_level=f"{level};tol={tol_l};len_rel_diff={dl:.3f};hyd_diff={dh:.3f}"))
    un = pd.DataFrame(un)
    log("uncalled", len(un), un.match_level.str.split(";").str[:2].str.join(";").value_counts().to_dict())

    # ---- positives ---------------------------------------------------------------------------------------
    gi = pd.read_parquet(GOLD / "instances.parquet")
    gi = gi[(gi.n_members_resolved >= 2) & ~((gi.caller == "CRISPRCasTyper") & (gi.status == "cas_putative"))]
    kd = pool.k.value_counts(normalize=True).sort_index()
    npos = {int(k): int(round(N_POS * f)) for k, f in kd.items()}
    npos[int(kd.idxmax())] += N_POS - sum(npos.values())
    prow = []
    for _, r in gi.iterrows():
        mg = sorted(int(v) for v in r.member_gids)
        for k in npos:
            if npos[k] == 0:
                continue
            for s in range(0, len(mg) - k + 1):
                w = mg[s:s + k]
                if w[-1] - w[0] != k - 1 or len({cc[v] for v in w}) != 1:
                    continue
                lw, hw = L[w], H[w]
                if lw.min() < 60 or lw.max() > 1200 or lw.sum() > 2500 or hw.max() > 0.45 or NX[w].sum() > 0:
                    continue
                prow.append(dict(k=k, caller=r.caller, system_type=str(r.system_type), status=r.status,
                                 instance_id=int(r.instance_id), gids=";".join(map(str, w)), genome=r.genome,
                                 genome_code=int(gcode[w[0]])))
    pp = pd.DataFrame(prow).sample(frac=1, random_state=17)
    pos = []
    for k in sorted(npos):
        sub = pp[pp.k == k]
        groups = {kk: v for kk, v in sub.groupby(["caller", "system_type"])}
        order = sorted(groups, key=lambda kk: -len(groups[kk]))
        picks, seen = {kk: [] for kk in groups}, {kk: set() for kk in groups}
        tot = 0
        for _round in range(4):
            for kk in order:
                if tot >= npos[k]:
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
                pos.append(dict(fold_id=f"pos3_k{k}_{tag}_{n}", cls="positive", s3_rank=-1, k=k, rep=n,
                                type_key=f"{kk[0]}:{kk[1]}", gids=r.gids, genome=r.genome, slot=-1,
                                match_level=f"status={r.status};instance={r.instance_id}"))
    pos = pd.DataFrame(pos)
    log("positives", len(pos), pos.groupby("k").size().to_dict(), pos.type_key.str.split(":").str[0].value_counts().to_dict())

    man = pd.concat([cog, nc, un, pos], ignore_index=True)
    man["lengths"] = man.gids.map(lambda s: ";".join(str(int(L[int(x)])) for x in s.split(";")))
    man["hyds"] = man.gids.map(lambda s: ";".join(f"{H[int(x)]:.3f}" for x in s.split(";")))
    man["tokens"] = man.gids.map(lambda s: ";".join(str(int(tok[int(x)])) for x in s.split(";")))
    man["total_aa"] = man.lengths.map(lambda s: sum(int(x) for x in s.split(";")))
    assert man.total_aa.max() <= 4096
    unc_cls = pool.set_index("s3_rank").unrec_class.to_dict()
    man["unrec_class"] = man.s3_rank.map(unc_cls).fillna("")
    etc = pool.set_index("s3_rank").earlier_triage_characterised.to_dict()
    man["earlier_triage_characterised"] = man.s3_rank.map(etc).fillna(False)
    man.to_csv(OUT / "fold_manifest3.tsv", sep="\t", index=False)
    gids = sorted({int(x) for s in man.gids for x in s.split(";")})
    seqs = read_faa(FAA, set(gids))
    assert len(seqs) == len(gids)
    pid = g.protein_id.to_numpy()
    with open(OUT / "jobs_s3.jsonl", "w") as f:
        for _, r in man.iterrows():
            gl = [int(x) for x in r.gids.split(";")]
            chains = [{"id": "ABCDEFGH"[i], "seq": seqs[x], "gid": x, "protein_id": pid[x]} for i, x in enumerate(gl)]
            assert all(len(c["seq"]) == L[c["gid"]] for c in chains)
            f.write(json.dumps({"fold_id": r.fold_id, "cls": r.cls, "chains": chains}) + "\n")
    summ = dict(pool=len(pool), nonredundant_available=n_nonred, items=len(man), by_cls=man.cls.value_counts().to_dict(),
                by_k=pool.k.value_counts().sort_index().to_dict(), by_unrec_class=pool.unrec_class.value_counts().to_dict(),
                max_total_aa=int(man.total_aa.max()), unique_gids=len(gids))
    json.dump(summ, open(OUT / "design_summary.json", "w"), indent=1, default=int)
    log(json.dumps(summ, default=int))


if __name__ == "__main__":
    main()
