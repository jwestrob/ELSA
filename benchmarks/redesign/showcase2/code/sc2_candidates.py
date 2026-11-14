#!/usr/bin/env python
"""ws_showcase2 step 1: candidate blocks from (a) the locus-space uncalled dense clusters and (b) the kinds co-fold
passers, with replicate selection. Runs on the biotite login node (pandas/numpy, <= 4 threads; a few minutes).

Sources
 (a) locus space: clusters of the MEAN map (resolution 10) with >= 10 cores and no curated-caller member
     (cluster_table_MEAN.called == False; 306 clusters). Realisations = cores of the cluster with span k (2-5).
     Candidate = (cluster, k). Reference family set = the canonical rescue3 token tuple of span k seen in the most
     genome groups (ties: most cores); reference core = its best-quality core. Each other core is oriented to the
     reference (token match, else the orientation with the higher summed whitened cosine) and kept as a
     realisation only if every position has whitened (zca10) cosine >= 0.32 to the reference position (the kinds /
     calib homology level), so a candidate holds one architecture.
 (b) kinds: the 21 passers that cleared the split-gene and disorder-zipping rules (kinds/results/cofold/passers.tsv
     minus split_gene_artefact minus zipping.tsv flags). Realisations = support pairs at list depth 1,000 exactly as
     the kinds statistic defines them (a' carries a sequence in N(a) = own sequence + zca10 top-1,000, b' adjacent to
     a' carries a sequence in N(b); query genome excluded). Chain A = the a-side.

Eligibility of a realisation as a replicate (both sources)
  every protein 40-400 aa, sum <= 700 aa; >= 1 unannotated gene (annotation_flags); no X; hydrophobic fraction <=
  0.45; SEG-trigger low-complexity fraction <= 0.145 (screen filters); Prodigal partial '00'; >= 1 gene from the
  contig end; each length within 0.8-1.25x of the realisations' median at that position; its unordered rescue3
  family set was never folded before (work/earlier_folds.tsv: screens 1-3 cognate + uncalled windows, deep dive,
  kinds classes a/b/d); for kinds also outside the passer's own genome group.
Kinds realisations are kept only when both chains have whitened cosine >= 0.32 to the passer's chains (the same
  homology-level rule as the locus-space positions), so replicates are realisations of the same block.
Replicates: shortlist the most typical realisation per genome group (g075, homolog-level cosine 0.75; minimum
  per-position cosine to the reference, then completeness), up to 30 groups; per-position global identity (BLOSUM62,
  -10/-0.5, identical / shorter length) for every pair in different groups; among pairs whose maximum per-position
  identity is <= 0.60 (ceiling) take tier A (both replicates at cosine >= 0.50 to the reference at every position)
  before tier B (>= 0.32), and within a tier the lowest maximum identity; ties: higher minimum completeness.
Removal: candidates containing a token of the A-B-C operon or its known variants (81718, 393054, 130016, 247277,
  399854, 190823); locus-space candidates whose reference family set had a two-replicate cognate test earlier
  (screens 1-3, deep dive); candidates with fewer than 2 eligible genome groups or no pair under the ceiling;
  duplicates (a candidate sharing a replicate gene or its reference family set with a better-ranked one).
Ranking: genome groups of the realisations, then distinct family sets (unordered rescue3 sets); phyla reported only.
Outputs ($RD/showcase2/data/): candidates_all.tsv, candidates.tsv (kept, ranked), reps.tsv, realisations.parquet.
"""
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
PKG = RD / "ws_dpann/package"
OUT = RD / "showcase2/data"
MAP = RD / "locus_space/out/map"
TOKF = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
ZROW = RD / "locus_search/out3/vec/zca10_f32__row.npy"
KNN = RD / "softrec/knn/zca10__k1000__I.npy"
PREP = RD / "kinds/prep"
FAA = RD / "ws_dpann/work/proteins_gid.faa"
ABC_TOKENS = {81718, 393054, 130016, 247277, 399854, 190823}
COS_MIN, COS_TYPICAL, ID_MAX, LC_MAX, HYD_MAX = 0.32, 0.50, 0.60, 0.145, 0.45
MAXLEN, MAXSUM, MINLEN = 400, 700, 40
SHORT_EACH, SHORT_SUM = 300, 450
N_SHORT = 30
T0 = time.time()
al = Align.PairwiseAligner(mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"),
                           open_gap_score=-10, extend_gap_score=-0.5)


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def ident(a, b):
    if a == b:
        return 1.0
    aln = al.align(a, b)[0]
    n = sum(1 for i, j in zip(*aln.indices) if i >= 0 and j >= 0 and a[i] == b[j])
    return n / min(len(a), len(b))


def lowcomplex(seq, w=12, k1=2.2):
    n = len(seq)
    if n < w:
        return 0.0
    cov = np.zeros(n, bool)
    c = Counter(seq[:w])
    for i in range(n - w + 1):
        if i:
            o, inn = seq[i - 1], seq[i + w - 1]
            c[o] -= 1
            if c[o] == 0:
                del c[o]
            c[inn] += 1
        if -sum((v / w) * math.log2(v / w) for v in c.values()) < k1:
            cov[i:i + w] = True
    return float(cov.mean())


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


class U:
    pass


def load():
    u = U()
    g = pd.read_parquet(PKG / "universe/genes.parquet",
                        columns=["gid", "protein_id", "genome_code", "contig_code", "position_index", "strand", "length"])
    assert (g.gid.to_numpy() == np.arange(len(g))).all()
    u.n = len(g)
    u.pid = g.protein_id.to_numpy()
    u.gc = g.genome_code.to_numpy(np.int64)
    u.cc = g.contig_code.to_numpy(np.int64)
    u.pos = g.position_index.to_numpy(np.int64)
    u.strand = g.strand.to_numpy()
    u.L = g.length.to_numpy(np.int64)
    ncont = g.groupby("contig_code").size()
    u.edge = np.minimum(u.pos, ncont.reindex(u.cc).to_numpy() - 1 - u.pos)
    t = pd.read_parquet(TOKF)
    tok = np.full(u.n, -1, np.int64)
    tok[t.gid.to_numpy()] = t.token.to_numpy()
    u.tok = tok
    fl = pd.read_parquet(PKG / "annotation_flags.parquet", columns=["gid", "unannotated"]).sort_values("gid")
    u.unann = fl.unannotated.to_numpy()
    comp = pd.read_parquet(RD / "cofold/s3/data/composition_slim.parquet").set_index("protein_id")
    u.hyd = comp.hyd.reindex(u.pid).to_numpy()
    u.nx = comp.n_x.reindex(u.pid).to_numpy()
    pf = pd.read_parquet(RD / "cofold/s3/data/prot_flags_slim.parquet", columns=["protein_id", "partial"]).set_index("protein_id")
    u.partial = pf.partial.reindex(u.pid).fillna("na").to_numpy()
    gm = pd.read_parquet(PKG / "universe/genomes.parquet").set_index("genome_code").sort_index()
    u.gm = gm
    gr = pd.read_parquet(RD / "lineagefree/groups/emb_hom_average_t0.75.parquet").set_index("genome_code").sort_index()
    u.group = gr.group.reindex(np.arange(len(gm))).to_numpy()
    mem = pd.read_parquet(PKG / "gold/members.parquet", columns=["gid"])
    u.curated = np.zeros(u.n, bool)
    u.curated[mem.gid.to_numpy()] = True
    u.X = np.load(ZROW, mmap_mode="r")
    return u


def vecs(u, gids):
    gids = np.asarray(gids, np.int64)
    o = np.argsort(gids)
    v = np.empty((len(gids), u.X.shape[1]), np.float32)
    v[o] = u.X[gids[o]]
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v


def earlier_sets(u):
    e = pd.read_csv(OUT / "earlier_folds.tsv", sep="\t")
    e["set"] = e.gids.map(lambda s: frozenset(int(u.tok[int(x)]) for x in str(s).split(";")))
    folded = set(e.set)
    tested = set(e[e.cls.isin(["cognate", "abc", "cognate_posthoc"]) & ~e.source.eq("kinds")].set)
    return folded, tested


# ------------------------------------------------------------------------------------------ realisation tables
def locus_space(u):
    C = pd.read_parquet(MAP / "cores_clusters_MEAN.parquet", columns=["gid_s", "gid_e", "span", "cluster"])
    G = pd.read_parquet(MAP / "cluster_table_MEAN.parquet")
    dense = G[(G.cores >= 10) & (~G.called)]
    C = C[C.cluster.isin(dense.cluster) & C.span.between(2, 5)].copy()
    rows = []
    for (cl, k), sub in C.groupby(["cluster", "span"]):
        recs = []
        for s, e in zip(sub.gid_s, sub.gid_e):
            gl = np.arange(s, e + 1)
            tt = tuple(u.tok[gl].tolist())
            fwd = tt <= tt[::-1]
            recs.append(dict(gids=gl if fwd else gl[::-1], ct=tt if fwd else tt[::-1]))
        if len(recs) < 2:
            continue
        df = pd.DataFrame(recs)
        df["genome_code"] = [u.gc[g[0]] for g in df.gids]
        df["group"] = u.group[df.genome_code.to_numpy()]
        if df.group.nunique() < 2:
            continue
        ctg = df.groupby("ct").agg(groups=("group", "nunique"), n=("group", "size")).sort_values(["groups", "n"], ascending=False)
        ref_ct = ctg.index[0]
        ref_rows = df[df.ct == ref_ct]
        ref = ref_rows.gids.iloc[0]
        rv = vecs(u, ref)
        allg = np.concatenate(df.gids.to_numpy())
        V = vecs(u, allg).reshape(len(df), k, -1)
        keep, cosmin, flip = [], [], []
        for i, (gl, ct) in enumerate(zip(df.gids, df.ct)):
            f = V[i] @ rv.T
            fw, rvs = float(np.trace(f)), float(np.trace(f[::-1]))
            if ct == ref_ct:
                o = False
            else:
                o = rvs > fw
            d = np.diag(f[::-1] if o else f)
            cosmin.append(float(d.min()))
            flip.append(o)
            keep.append(d.min() >= COS_MIN)
        df["flip"] = flip
        df["cos_min"] = cosmin
        df = df[np.array(keep)].copy()
        df["gids"] = [g[::-1] if o else g for g, o in zip(df.gids, df.flip)]
        if df.group.nunique() < 2:
            continue
        for _, r in df.iterrows():
            rows.append(dict(cand=f"ls_c{cl}_k{k}", source="locus_space", cluster=int(cl), k=int(k),
                             gids=r.gids.tolist(), ref_set=";".join(map(str, ref_ct)), cos_min=r.cos_min, level=-1))
    out = pd.DataFrame(rows)
    log("locus-space realisations", len(out), "candidates", out.cand.nunique())
    return out, dense


def kinds(u):
    p = pd.read_csv(RD / "kinds/passers/passers.tsv", sep="\t")
    zip_ = pd.read_csv(OUT / "kinds_zipping.tsv", sep="\t")      # copy of kinds/results/cofold/zipping.tsv
    flag = set(zip_[zip_.zipping_flag].fold_id)
    p = p[p.cls.eq("a_high") & ~p.split_gene_artefact & ~p.fold_id.isin(flag)].copy()
    log("kinds passers kept", len(p), sorted(p.fold_id))
    uq = np.load(PREP / "uq.npy")
    mptr = np.load(PREP / "mptr.npy")
    members = np.load(PREP / "members.npy")
    cid = np.load(PREP / "cid.npy")
    I = np.load(KNN, mmap_mode="r")
    rows = []
    for r in p.itertuples():
        a, b = int(r.gid_A), int(r.gid_B)
        la = np.r_[uq[a], np.asarray(I[uq[a], :1000])]
        lb = np.r_[uq[b], np.asarray(I[uq[b], :1000])]
        rank_b = {int(x): i for i, x in enumerate(lb) if x >= 0}
        ga = u.gc[a]
        seen = set()
        for ra, Up in enumerate(la):
            if Up < 0:
                break
            for t in range(mptr[Up], mptr[Up + 1]):
                a2 = int(members[t])
                if u.gc[a2] == ga:
                    continue
                for s2 in (a2 - 1, a2 + 1):
                    if s2 < 0 or s2 >= u.n or cid[s2] != cid[a2]:
                        continue
                    rb = rank_b.get(int(uq[s2]))
                    if rb is None or (a2, s2) in seen:
                        continue
                    seen.add((a2, s2))
                    rows.append(dict(cand=f"kd_{r.fold_id}", source="kinds", cluster=-1, k=2, gids=[a2, s2],
                                     ref_set=f"{u.tok[a]};{u.tok[b]}", cos_min=np.nan, level=max(ra, rb),
                                     passer_group=int(u.group[ga])))
    out = pd.DataFrame(rows)
    gl = np.array(out.gids.tolist(), np.int64)
    va, vb = vecs(u, gl[:, 0]), vecs(u, gl[:, 1])
    ref = {f"kd_{r.fold_id}": (int(r.gid_A), int(r.gid_B)) for r in p.itertuples()}
    qa = vecs(u, [ref[c][0] for c in out.cand])
    qb = vecs(u, [ref[c][1] for c in out.cand])
    out["cos_A"] = (va * qa).sum(1)
    out["cos_B"] = (vb * qb).sum(1)
    out["cos_min"] = np.minimum(out.cos_A, out.cos_B)
    n0 = len(out)
    out = out[out.cos_min >= COS_MIN].drop(columns=["cos_A", "cos_B"]).reset_index(drop=True)
    log("kinds realisations", n0, "with both chains at cosine >=", COS_MIN, "to the passer:", len(out),
        "candidates", out.cand.nunique())
    return out, p


# ------------------------------------------------------------------------------------------ candidates
def annotate(u, R, folded):
    R = R.copy()
    R["k"] = R.gids.map(len)
    R["genome_code"] = R.gids.map(lambda g: int(u.gc[g[0]]))
    R["group"] = u.group[R.genome_code.to_numpy()]
    R["phylum"] = u.gm.phylum.to_numpy()[R.genome_code.to_numpy()]
    R["fset"] = R.gids.map(lambda g: frozenset(int(u.tok[x]) for x in g))
    R["tokens"] = R.gids.map(lambda g: ";".join(str(int(u.tok[x])) for x in g))
    R["lengths"] = R.gids.map(lambda g: [int(u.L[x]) for x in g])
    R["any_unann"] = R.gids.map(lambda g: bool(u.unann[g].any()))
    R["n_unann"] = R.gids.map(lambda g: int(u.unann[g].sum()))
    R["curated"] = R.gids.map(lambda g: bool(u.curated[g].any()))
    R["folded_before"] = R.fset.isin(folded)
    R["abc"] = R.fset.map(lambda s: bool(s & ABC_TOKENS))
    gg = np.concatenate(R.gids.to_numpy())
    R["ok_basic"] = R.gids.map(lambda g: bool((u.L[g] >= MINLEN).all() and (u.L[g] <= MAXLEN).all()
                                               and u.L[g].sum() <= MAXSUM and (u.nx[g] == 0).all()
                                               and (u.hyd[g] <= HYD_MAX).all() and all(x in ("00", "na") for x in u.partial[g])
                                               and min(u.edge[g[0]], u.edge[g[-1]]) >= 1))
    R["completeness"] = u.gm.completeness.to_numpy()[R.genome_code.to_numpy()]
    return R


def choose_pair(u, sub, seqs, prefer_level):
    k = int(sub.k.iloc[0])
    med = np.median(np.vstack(sub.lengths.to_numpy()), axis=0)
    ok = sub.ok_basic & sub.any_unann & ~sub.folded_before & ~sub.curated
    ok &= sub.lengths.map(lambda L: all(0.8 * m <= x <= 1.25 * m for x, m in zip(L, med)))
    if "passer_group" in sub and sub.passer_group.notna().any():
        ok &= sub.group != sub.passer_group
    pool = sub[ok].copy()
    if pool.group.nunique() < 2:
        return None, dict(eligible=len(pool), eligible_groups=int(pool.group.nunique()), reason="lt2_eligible_groups")
    pool = pool.sort_values(["cos_min", "completeness"], ascending=False).drop_duplicates("group").head(N_SHORT)
    pool["lc"] = pool.gids.map(lambda g: max(lowcomplex(seqs[int(x)]) for x in g))
    pool = pool[pool.lc <= LC_MAX]
    if pool.group.nunique() < 2:
        return None, dict(eligible=len(pool), eligible_groups=int(pool.group.nunique()), reason="lt2_after_lowcomplexity")
    recs = pool.to_dict("records")
    best, bkey, nunder = None, None, 0
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            if recs[i]["group"] == recs[j]["group"]:
                continue
            ids = []
            for x, y in zip(recs[i]["gids"], recs[j]["gids"]):
                ids.append(ident(seqs[int(x)], seqs[int(y)]))
                if ids[-1] > ID_MAX:
                    break
            if max(ids) > ID_MAX:
                continue
            nunder += 1
            tierA = min(recs[i]["cos_min"], recs[j]["cos_min"]) >= COS_TYPICAL
            key = (0 if tierA else 1, max(ids), -min(recs[i]["completeness"], recs[j]["completeness"]))
            if bkey is None or key < bkey:
                best, bkey = (recs[i], recs[j], ids), key
    info = dict(eligible=int(ok.sum()), eligible_groups=int(sub[ok].group.nunique()), shortlist=len(recs), pairs_under_ceiling=nunder)
    if best is None:
        info["reason"] = "no_pair_under_identity_ceiling"
        return None, info
    info["reason"] = ""
    info["pair_tier"] = "A_cos_ge_%.2f" % COS_TYPICAL if bkey[0] == 0 else "B_cos_ge_%.2f" % COS_MIN
    return best, info


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    u = load()
    log("universe loaded")
    folded, tested = earlier_sets(u)
    log("earlier folded family sets", len(folded), "two-replicate tested", len(tested))
    ls, dense = locus_space(u)
    kd, passers = kinds(u)
    R = annotate(u, pd.concat([ls, kd], ignore_index=True), folded)
    R.drop(columns=["fset"]).assign(gids=R.gids.map(lambda g: ";".join(map(str, g))),
                                    lengths=R.lengths.map(lambda g: ";".join(map(str, g)))) \
        .to_parquet(OUT / "realisations.parquet", index=False)
    elig = R.ok_basic & R.any_unann & ~R.folded_before & ~R.curated
    seqs = read_faa({int(x) for g in R[elig].gids for x in g})
    log("sequences read", len(seqs))
    cands, reps = [], []
    for cand, sub in R.groupby("cand", sort=False):
        k = int(sub.k.iloc[0])
        src = sub.source.iloc[0]
        ref = frozenset(int(x) for x in sub.ref_set.iloc[0].split(";"))
        med = np.median(np.vstack(sub.lengths.to_numpy()), axis=0)
        row = dict(cand=cand, source=src, cluster=int(sub.cluster.iloc[0]), k=k, ref_set=sub.ref_set.iloc[0],
                   realisations=len(sub), genomes=int(sub.genome_code.nunique()), groups=int(sub.group.nunique()),
                   famsets=int(sub.fset.nunique()), phyla=int(sub.phylum.nunique()),
                   phyla_list=";".join(f"{a}:{b}" for a, b in sub.phylum.value_counts().items()),
                   len_med=";".join(str(int(m)) for m in med), sum_med=int(med.sum()),
                   short=bool((med <= SHORT_EACH).all() and med.sum() <= SHORT_SUM),
                   unann_frac=round(float(sub.n_unann.sum() / (len(sub) * k)), 3),
                   realisations_folded_before=int(sub.folded_before.sum()),
                   famsets_folded_before=int(sub[sub.folded_before].fset.nunique()),
                   ref_tested_before=bool(ref in tested and src == "locus_space"),
                   abc=bool(sub.abc.any() or (ref & ABC_TOKENS)))
        if row["abc"]:
            row["status"] = "removed_abc_operon"
        elif row["ref_tested_before"]:
            row["status"] = "removed_ref_set_folded_with_replicates"
        else:
            best, info = choose_pair(u, sub, seqs, prefer_level=(src == "kinds"))
            row.update(info)
            if best is None:
                row["status"] = "removed_" + info["reason"]
            else:
                r1, r2, ids = best
                row["status"] = "kept"
                row["rep_identity"] = ";".join(f"{x:.3f}" for x in ids)
                row["rep_identity_max"] = round(max(ids), 3)
                for rep, r in ((1, r1), (2, r2)):
                    gm = u.gm.loc[r["genome_code"]]
                    reps.append(dict(cand=cand, rep=rep, gids=";".join(map(str, r["gids"])),
                                     protein_ids=";".join(u.pid[x] for x in r["gids"]),
                                     tokens=r["tokens"], lengths=";".join(map(str, r["lengths"])),
                                     hyds=";".join(f"{u.hyd[x]:.3f}" for x in r["gids"]),
                                     unann=";".join(str(int(u.unann[x])) for x in r["gids"]),
                                     genome=gm.genome, group=int(r["group"]), phylum=gm.phylum, cls_=gm["class"],
                                     ord=gm.ord, family=gm.family, genus=gm.genus, completeness=gm.completeness,
                                     contig_code=int(u.cc[r["gids"][0]]), pos=f"{u.pos[min(r['gids'])]}-{u.pos[max(r['gids'])]}",
                                     strands="".join("+" if str(u.strand[x]) in ("1", "+") else "-" for x in r["gids"]),
                                     level=r.get("level", -1), cos_min=r.get("cos_min", np.nan)))
        cands.append(row)
    C = pd.DataFrame(cands)
    Rp = pd.DataFrame(reps).rename(columns={"cls_": "class"})
    # dedupe kept candidates: shared replicate gene or shared reference family set with a better-ranked candidate
    C = C.sort_values(["groups", "famsets", "genomes"], ascending=False).reset_index(drop=True)
    seen_g, seen_ref = set(), set()
    for i, r in C.iterrows():
        if r.status != "kept":
            continue
        gl = set(int(x) for s in Rp[Rp.cand == r.cand].gids for x in s.split(";"))
        refset = frozenset(int(x) for x in r.ref_set.split(";"))
        if gl & seen_g or refset in seen_ref:
            C.at[i, "status"] = "removed_duplicate"
            continue
        seen_g |= gl
        seen_ref.add(refset)
    C["rank"] = np.nan
    kept = C.status.eq("kept")
    C.loc[kept, "rank"] = np.arange(1, int(kept.sum()) + 1)
    C.to_csv(OUT / "candidates_all.tsv", sep="\t", index=False)
    K = C[kept].copy()
    Rp = Rp[Rp.cand.isin(K.cand)].merge(K[["cand", "rank"]], on="cand")
    K.to_csv(OUT / "candidates.tsv", sep="\t", index=False)
    Rp.to_csv(OUT / "reps.tsv", sep="\t", index=False)
    log("candidates", len(C), C.status.value_counts().to_dict())
    log("kept by source", K.source.value_counts().to_dict(), "short", int(K.short.sum()), "k", K.k.value_counts().to_dict())
    pd.set_option("display.width", 250)
    print(K[["rank", "cand", "k", "groups", "famsets", "phyla", "genomes", "len_med", "short", "unann_frac",
             "rep_identity", "famsets_folded_before"]].to_string(index=False))


if __name__ == "__main__":
    main()
