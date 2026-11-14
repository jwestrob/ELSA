#!/usr/bin/env python3
"""Coordinator follow-ups on the split side, fragments and giant tilings (existing outputs only).
1. split loci (adjacent Y1, Y2 in sample genomes): nt gap classes, same vs shifted frame, joint coverage
   of the X homolog (best X per locus = largest union coverage), X-gap between the two regions;
2. per-genome concentration of split genes and fragments vs completeness, contigs, N50; by phylum/order;
3. per family (token of the X homolog): share of member sample genomes carrying the split form,
   overall and within GTDB order;
4. giant tilings (>= 2 adjacent smaller genes in another genome): broken-call-like vs separate genes;
5. repair estimate: genes merged by joining broken-call-like split loci, fragments that would join a
   family token under a shorter-sequence coverage rule (>= 80 % of the shorter, E <= 1e-5), gold instances touched.
"""
import json, os, sys
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr
W, P, TOK = sys.argv[1], sys.argv[2], sys.argv[3]
R = f"{W}/results"; T = os.environ.get("SLURM_CPUS_ON_NODE", "2")
con = duckdb.connect(); con.execute(f"SET threads={T}; SET temp_directory='{W}/tmp/duck'")
con.execute(f"ATTACH '{W}/tmp/comp.duckdb' AS comp (READ_ONLY)")
con.execute(f"ATTACH '{W}/tmp/frag.duckdb' AS frag (READ_ONLY)")
G = con.execute(f"""SELECT g.gid, g.genome_code, g.contig_code, g.position_index pos, g.strand, g.start, g."end" stop, t.token, m.uid
   FROM '{P}/universe/genes.parquet' g JOIN '{TOK}' t USING (gid) JOIN '{W}/gid2uid.parquet' m USING (gid) ORDER BY gid""").fetchnumpy()
NG = len(G["gid"]); strand = G["strand"].astype(int); start = G["start"].astype(np.int64); stop = G["stop"].astype(np.int64)
pos = G["pos"].astype(np.int64); token = G["token"].astype(np.int64); genome = G["genome_code"].astype(np.int64); contig = G["contig_code"]
U = con.execute(f"SELECT uid, length, rep_gid FROM '{W}/uniq.parquet' ORDER BY uid").fetchnumpy()
ulen = U["length"]; tok_u = token[U["rep_gid"]]
gen = con.execute(f"SELECT genome_code, phylum, ord, family, genus, completeness, contamination, n_contigs FROM '{P}/universe/genomes.parquet'").df()
n50 = con.execute(f"""SELECT genome, arg_min(length_nt, c) n50 FROM (SELECT genome, length_nt,
     sum(length_nt) OVER (PARTITION BY genome ORDER BY length_nt DESC) c, sum(length_nt) OVER (PARTITION BY genome) tot
     FROM '{P}/universe/contigs.parquet') WHERE c >= tot / 2 GROUP BY genome""").df()
gname = con.execute(f"SELECT genome, genome_code FROM '{P}/universe/genomes.parquet'").df()
gen = gen.merge(gname.merge(n50, on="genome"), on="genome_code", how="left")
samp = np.loadtxt(f"{W}/lists/sample_genomes.txt", dtype=np.int64)
ins = np.isin(genome, samp)
cf = np.load(f"{R}/composite_flags.npz"); ff = np.load(f"{R}/fragment_flags.npz")
Yadj, frag = cf["Y_adj"], ff["frag"]
gold_mem = con.execute(f"""SELECT DISTINCT m.instance_id, m.gid, i.caller FROM '{P}/gold/members.parquet' m
   JOIN '{P}/gold/instances.parquet' i USING (instance_id) WHERE i.n_members_resolved >= 2""").df()
res = {}

def frame_same(a, b):
    return np.where(strand[a] > 0, (start[a] - start[b]) % 3 == 0, (stop[a] - stop[b]) % 3 == 0)
def nt_gap(a, b):
    first = start[a] <= start[b]
    return np.where(first, start[b] - stop[a] - 1, start[a] - stop[b] - 1)
def gap_class(g):
    return np.select([g < 0, g <= 50, g <= 200], ["overlap", "0-50", "51-200"], ">200")

# ---------------- 1. split loci
con.execute(f"CREATE TABLE gy AS SELECT x, B, least(y1, y2) ya, greatest(y1, y2) yb FROM '{R}/composite_groups_y.parquet' WHERE adj")
lx = con.execute(f"""SELECT gy.ya, gy.yb, gy.B, gy.x, r1.qs s1, r1.qe e1, r2.qs s2, r2.qe e2, u.length xlen
   FROM gy JOIN comp.yrows2 r1 ON r1.x = gy.x AND r1.ygid = gy.ya JOIN comp.yrows2 r2 ON r2.x = gy.x AND r2.ygid = gy.yb
   JOIN '{W}/uniq.parquet' u ON u.uid = gy.x""").df()
ov = np.maximum(0, np.minimum(lx.e1, lx.e2) - np.maximum(lx.s1, lx.s2) + 1)
lx["jcov"] = ((lx.e1 - lx.s1 + 1) + (lx.e2 - lx.s2 + 1) - ov) / lx.xlen
nfirst_a = lx.s1 <= lx.s2
lx["xgap"] = np.where(nfirst_a, lx.s2 - lx.e1 - 1, lx.s1 - lx.e2 - 1)
lx["nterm"] = np.where(nfirst_a, lx.ya, lx.yb); lx["cterm"] = np.where(nfirst_a, lx.yb, lx.ya)
lx["n_x"] = lx.groupby(["ya", "yb"]).x.transform("size")
loc = lx.sort_values("jcov", ascending=False).drop_duplicates(["ya", "yb"]).reset_index(drop=True)
loc["tok_x"] = tok_u[loc.x.values]
a, b = loc.nterm.values, loc.cterm.values
loc["same_strand"] = strand[a] == strand[b]
loc["collinear"] = loc.same_strand & np.where(strand[a] > 0, pos[a] < pos[b], pos[a] > pos[b])
loc["dpos"] = np.abs(pos[a] - pos[b]); loc["gap_nt"] = nt_gap(a, b); loc["same_frame"] = frame_same(a, b)
loc["in_sample"] = ins[a]
loc["broken_like"] = loc.collinear & (loc.dpos == 1) & (loc.gap_nt <= 50) & (loc.xgap <= 10)
loc.to_parquet(f"{R}/split_loci.parquet")
L = loc[loc.in_sample]
d1 = L[L.dpos == 1]
res["1_split_loci_sample"] = int(len(L)); res["1_dpos_counts"] = {int(k): int(v) for k, v in L.dpos.value_counts().sort_index().items()}
res["1_collinear_frac"] = float(L.collinear.mean())
gc = pd.crosstab(gap_class(d1.gap_nt.values), d1.same_frame.values)
res["1_dpos1_gap_class_x_same_frame"] = {f"{k}|{'same' if c else 'shifted'}_frame": int(gc.loc[k, c]) for k in gc.index for c in gc.columns}
res["1_dpos1_gap_class_counts"] = {k: int(v) for k, v in pd.Series(gap_class(d1.gap_nt.values)).value_counts().items()}
res["1_dpos1_same_frame_frac"] = float(d1.same_frame.mean())
# baseline: every co-oriented neighbour pair in sample genomes
i = np.flatnonzero(ins[:-1] & (contig[:-1] == contig[1:]) & (strand[:-1] == strand[1:]))
gb = nt_gap(i, i + 1); fb = frame_same(i, i + 1)
res["1_baseline_neighbours_n"] = int(len(i))
res["1_baseline_gap_class_counts"] = {k: int(v) for k, v in pd.Series(gap_class(gb)).value_counts().items()}
res["1_baseline_same_frame_frac"] = float(fb.mean())
res["1_baseline_same_frame_frac_gap_le_50"] = float(fb[gb <= 50].mean())
res["1_joint_cov_of_X_q10_q25_q50_q75_q90"] = L.jcov.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
res["1_xgap_aa_q10_q25_q50_q75_q90"] = L.xgap.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
res["1_frac_xgap_le_10aa"] = float((L.xgap <= 10).mean()); res["1_frac_jcov_ge_0.8"] = float((L.jcov >= 0.8).mean())
res["1_broken_like_frac"] = float(L.broken_like.mean())
bl = L[L.broken_like]
res["1_broken_like_same_frame_frac"] = float(bl.same_frame.mean())
res["1_broken_like_overlap_frac"] = float((bl.gap_nt < 0).mean())

# ---------------- 2. per-genome concentration
df = pd.DataFrame({"genome_code": genome[ins], "y": Yadj[ins], "f": frag[ins]})
bl_genes = np.zeros(NG, bool); bl_genes[bl.ya.values] = True; bl_genes[bl.yb.values] = True
df["ybl"] = bl_genes[ins]
pg = df.groupby("genome_code").agg(genes=("y", "size"), y=("y", "sum"), f=("f", "sum"), ybl=("ybl", "sum")).reset_index().merge(gen, on="genome_code")
for v in ("y", "f", "ybl"):
    pg[f"{v}_rate"] = pg[v] / pg.genes
    srt = pg.sort_values(f"{v}_rate", ascending=False)
    res[f"2_{v}_rate_per_genome_q10_q50_q90"] = pg[f"{v}_rate"].quantile([0.1, 0.5, 0.9]).tolist()
    res[f"2_{v}_share_in_top10pct_genomes"] = float(srt[v].head(len(srt) // 10).sum() / srt[v].sum())
    for c in ("completeness", "contamination", "n_contigs", "n50"):
        r = spearmanr(pg[f"{v}_rate"], pg[c], nan_policy="omit"); res[f"2_spearman_{v}_rate_vs_{c}"] = [round(float(r.correlation), 3), float(r.pvalue)]
pg.to_csv(f"{R}/per_genome_rates.tsv", sep="\t", index=False)
for lvl in ("phylum", "ord"):
    t = pg.groupby(lvl).agg(genomes=("genes", "size"), genes=("genes", "sum"), split=("y", "sum"), broken_like=("ybl", "sum"),
                            frag=("f", "sum")).reset_index()
    t["split_rate"] = t.split / t.genes; t["broken_like_rate"] = t.broken_like / t.genes; t["frag_rate"] = t.frag / t.genes
    t.sort_values("genes", ascending=False).to_csv(f"{R}/concentration_by_{lvl}.tsv", sep="\t", index=False)
    t3 = t[t.genomes >= 3]
    res[f"2_{lvl}_split_rate_q10_q50_q90_(>=3 genomes)"] = t3.split_rate.quantile([0.1, 0.5, 0.9]).tolist()
    res[f"2_{lvl}_frag_rate_q10_q50_q90_(>=3 genomes)"] = t3.frag_rate.quantile([0.1, 0.5, 0.9]).tolist()
    res[f"2_{lvl}_n_with_ge3_genomes"] = int(len(t3))

# ---------------- 3. family split fraction (family = token of the X homolog with the best joint coverage)
Ls = L[["ya", "yb", "B", "tok_x", "broken_like"]].rename(columns={"tok_x": "F"})
split_gb = Ls[["F", "B"]].drop_duplicates()
piece = pd.concat([Ls[["F", "ya"]].rename(columns={"ya": "gid"}), Ls[["F", "yb"]].rename(columns={"yb": "gid"})])
fams = set(Ls.F.unique())
sg = pd.DataFrame({"gid": np.flatnonzero(ins), "F": token[ins], "B": genome[ins]})
sg = sg[sg.F.isin(fams)]
sg = sg.merge(piece.drop_duplicates(), on=["F", "gid"], how="left", indicator=True)
intact_gb = sg[sg._merge == "left_only"][["F", "B"]].drop_duplicates()
mem = pd.concat([split_gb.assign(split=1), intact_gb.assign(split=0)]).groupby(["F", "B"]).split.max().reset_index()
mem = mem.merge(gen[["genome_code", "ord"]], left_on="B", right_on="genome_code")
fam = mem.groupby("F").agg(members=("B", "size"), split_genomes=("split", "sum")).reset_index()
fam["frac"] = fam.split_genomes / fam.members
both = fam[(fam.split_genomes >= 1) & (fam.split_genomes < fam.members)]
res["3_families_with_split_form"] = int(len(fam)); res["3_families_with_both_forms"] = int(len(both))
res["3_split_frac_per_family_q10_q25_q50_q75_q90_(both forms)"] = both.frac.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
lf = Ls.merge(fam, on="F")
bins = pd.cut(lf.frac, [-0.01, 0.1, 0.5, 0.9999, 1.0], labels=["<=0.1", "0.1-0.5", "0.5-<1", "1 (no intact copy in sample)"])
res["3_split_loci_by_family_split_frac"] = {str(k): int(v) for k, v in bins.value_counts().sort_index().items()}
fo = mem.groupby(["F", "ord"]).agg(m=("B", "size"), s=("split", "sum")).reset_index(); fo["fo"] = fo.s / fo.m
lo = Ls.merge(gen[["genome_code", "ord"]], left_on="B", right_on="genome_code").merge(fo, on=["F", "ord"])
lo3 = lo[lo.m >= 3]
res["3_split_loci_with_>=3_member_genomes_in_order"] = int(len(lo3))
res["3_frac_of_those_split_in_>=80pct_of_order_genomes"] = float((lo3.fo >= 0.8).mean())
res["3_frac_of_those_split_in_<=20pct_of_order_genomes"] = float((lo3.fo <= 0.2).mean())
lfb = Ls.merge(fam, on="F")
res["3_broken_like_loci_family_split_frac_q25_q50_q75"] = lfb[lfb.broken_like].frac.quantile([0.25, 0.5, 0.75]).tolist()
res["3_not_broken_like_loci_family_split_frac_q25_q50_q75"] = lfb[~lfb.broken_like].frac.quantile([0.25, 0.5, 0.75]).tolist()

# ---------------- 4. giant tilings
gt = con.execute("SELECT q, B, c, pos, ygid, qs, qe FROM frag.gtile ORDER BY q, B, c, pos").df()
best = {}
for (q, B, c), s in gt.groupby(["q", "B", "c"], sort=False):
    p = s.pos.values
    for r in np.split(np.arange(len(s)), np.flatnonzero(np.diff(p) > 3) + 1):
        iv = sorted(zip(s.qe.values[r], s.qs.values[r], s.ygid.values[r]))
        k, last, cov, used, pcs = 0, -10 ** 9, 0, set(), []
        for qe_, qs_, y in iv:
            if y in used: continue
            if qs_ > last - 20:
                cov += (qe_ - max(qs_, last + 1) + 1) if qs_ <= last else (qe_ - qs_ + 1)
                k += 1; last = qe_; used.add(y); pcs.append((qs_, y))
        bk = best.get(q)
        if bk is None or k > bk[0] or (k == bk[0] and cov > bk[1]):
            best[q] = (k, cov, [y for _, y in sorted(pcs)])
rows = []
for q, (k, cov, pcs) in best.items():
    if k < 2: continue
    pcs = np.array(pcs)
    ss = bool((strand[pcs] == strand[pcs[0]]).all())
    order = np.diff(pos[pcs]); col = ss and (bool((order > 0).all()) if strand[pcs[0]] > 0 else bool((order < 0).all()))
    adj = bool((np.abs(order) == 1).all())
    gaps = nt_gap(pcs[:-1], pcs[1:]); fr = frame_same(pcs[:-1], pcs[1:])
    rows.append((q, k, cov / ulen[q], ss, col, adj, int(gaps.max()), int(gaps.min()), bool(fr.all()), bool((~fr).any())))
gtl = pd.DataFrame(rows, columns=["uid", "k", "cov_frac", "same_strand", "collinear", "all_adjacent", "max_gap_nt", "min_gap_nt", "all_same_frame", "any_shifted"])
gtl["broken_like"] = gtl.collinear & gtl.all_adjacent & (gtl.max_gap_nt <= 50)
gtl["separate_like"] = gtl.collinear & (gtl.min_gap_nt > 50)
gtl.to_parquet(f"{R}/giant_tilings.parquet")
res["4_giants_tiled_k_ge2"] = int(len(gtl)); res["4_collinear"] = int(gtl.collinear.sum())
res["4_broken_like_(collinear, adjacent, all gaps<=50nt)"] = int(gtl.broken_like.sum())
res["4_broken_like_any_shifted_frame"] = int((gtl.broken_like & gtl.any_shifted).sum())
res["4_separate_like_(collinear, all gaps>50nt)"] = int(gtl.separate_like.sum())
res["4_other_(mixed gaps, non-collinear or mixed strand)"] = int((~gtl.broken_like & ~gtl.separate_like).sum())
res["4_max_gap_nt_q25_q50_q75"] = gtl.max_gap_nt.quantile([0.25, 0.5, 0.75]).tolist()

# ---------------- 5. repair estimate
scale = NG / ins.sum()
blS = L[L.broken_like]
par = {}
def find(x):
    while par.get(x, x) != x: x = par[x]
    return x
for ya, yb in zip(blS.ya.values, blS.yb.values):
    ra, rb = find(ya), find(yb)
    if ra != rb: par[ra] = rb
genes_bl = set(blS.ya.values) | set(blS.yb.values)
comps = len({find(g) for g in genes_bl})
res["5_scale_factor_all_genes_over_sample_genes"] = float(scale)
res["5_broken_like_genes_sample"] = int(len(genes_bl)); res["5_merged_genes_sample"] = int(comps)
res["5_gene_records_removed_sample"] = int(len(genes_bl) - comps)
res["5_broken_like_genes_all_est"] = float(len(genes_bl) * scale)
res["5_all_split_genes_sample"] = int(len(set(L.ya.values) | set(L.yb.values)))
# fragments that would join a family token (they have a partner covering >= 80 % of them at E <= 1e-5 by definition)
fg = pd.DataFrame({"gid": np.flatnonzero(frag)}); fg["uid"] = G["uid"][fg.gid.values]; fg["tok"] = token[fg.gid.values]
con.register("fg", fg)
tk_df = pd.DataFrame({"uid": np.arange(len(tok_u)), "tk": tok_u}); con.register("tk", tk_df)
fj = con.execute("""SELECT fg.gid, bool_or(tk.tk = fg.tok) shares FROM fg JOIN frag.fp ON fp.f = fg.uid AND fp.fr
   JOIN tk ON tk.uid = fp.p GROUP BY fg.gid""").df()
join = np.zeros(NG, bool); join[fj.gid.values[~fj.shares.values]] = True
single = np.bincount(token - token.min())[token - token.min()] == 1
res["5_fragments_sample"] = int((frag & ins).sum())
res["5_fragments_that_would_join_a_family_token_sample"] = int((join & ins).sum())
res["5_of_which_singletons_now"] = int((join & ins & single).sum())
res["5_fragments_join_all_est"] = float((join & ins).sum() * scale)
# gold instances touched
bl_all = loc[loc.broken_like]
blg = np.zeros(NG, bool); blg[bl_all.ya.values] = True; blg[bl_all.yb.values] = True
gm = gold_mem.copy(); gm["bl"] = blg[gm.gid.values]; gm["fj"] = join[gm.gid.values]; gm["fr"] = frag[gm.gid.values]
gi = gm.groupby(["instance_id", "caller"]).agg(bl=("bl", "max"), fj=("fj", "max"), fr=("fr", "max")).reset_index()
gi["any"] = gi.bl | gi.fj
res["5_gold_instances_ge2"] = int(len(gi))
res["5_gold_touched_by_split_merge"] = int(gi.bl.sum()); res["5_gold_touched_by_fragment_join"] = int(gi.fj.sum())
res["5_gold_touched_any"] = int(gi["any"].sum()); res["5_gold_touched_any_frac"] = float(gi["any"].mean())
res["5_gold_touched_by_caller"] = gi.groupby("caller")[["bl", "fj", "any"]].sum().astype(int).reset_index().to_dict("records")
json.dump(res, open(f"{R}/interp2.json", "w"), indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
print(json.dumps(res, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
