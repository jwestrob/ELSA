"""Step 1: dark, recurrent, operon-like, foldable blocks of 2-4 adjacent genes from the ELSA
engine v1.5 DPANN run (permissive sequence families e3c50s75, window 3, genome-shuffle null).

Inputs (copied from biotite into work/data/, see REPORT.md):
  genes.parquet, genomes.parquet, contigs.parquet          package universe
  annotation_flags.parquet, members.parquet                package annotation flags, gold members
  sharur_extra_flags.parquet                               extract_sharur_flags.py (cctyper, derived atoms, partial)
  composition.parquet                                      composition.py (hydrophobic fraction, X, md5)
  mmseqs__seq__e3c50s75__dpann.parquet                     engine tokens
  significant_pairs.parquet, token_modules.parquet,
  loci_all_genomes.parquet                                 engine results_dpann_v5/.../w3

Definitions
  dark gene    no row in any annotation source of the Sharur DB (annotations: pfam, kofam, kegg,
               defensefinder, defensefinder_system, txsscan, txsscan_system, hyddb, hyddb_subgroup;
               system_proteins incl. cctyper; derived _locus/_validation atoms) and no gold membership.
  dark family  token whose members are >= 98 % dark genes (column fam_dark_frac).
  link         adjacent genes (d = 1, same contig), both dark genes of dark families, token pair
               significant in the engine run.
  run          maximal chain of links on a contig; block instance = window of k = 2..4 consecutive
               genes inside a run; block type = canonical ordered token tuple.

Outputs (work/): block_instances.parquet, candidates.tsv (all types with >= 3 genomes, all columns),
fam_seqstats.parquet.
"""
import json
import math
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"
D = W / "data"
SHARUR = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
DARK_FAM = 0.98
KMAX = 4
# 90th percentile of the SEG-trigger fraction over 3,000 random annotated DPANN proteins (median 0.026)
LC_MAX = 0.145
TAX = ["phylum", "class", "ord", "family", "genus"]


def entropy_lowcomplex_frac(seq, w=12, k1=2.2):
    """Fraction of residues covered by w-residue windows with Shannon entropy < k1 bits (SEG trigger)."""
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


def main():
    g = pd.read_parquet(W / "genes_dark.parquet")          # built by the inline step recorded in REPORT.md
    comp = pd.read_parquet(D / "composition.parquet")
    g = g.merge(comp[["protein_id", "hyd", "n_x", "seq_md5", "aa_len"]], on="protein_id", how="left")
    assert (g.gid.values == np.arange(len(g))).all() and g.hyd.notna().all()
    gm = pd.read_parquet(D / "genomes.parquet")
    for c in TAX:
        gm[c] = gm[c].replace("", np.nan)
    ct = pd.read_parquet(D / "contigs.parquet")[["contig_code", "n_genes", "length_nt"]]
    g = g.merge(ct, on="contig_code", how="left")
    assert (g.gid.values == np.arange(len(g))).all()
    tok = g.token.to_numpy()
    T = int(tok.max()) + 1

    # ---- family (token) statistics ---------------------------------------------------------
    fam = g.groupby("token").agg(fam_n=("gid", "size"), fam_ndark=("dark", "sum"), fam_genomes=("genome_code", "nunique"),
                                 fam_len_med=("length", "median"), fam_hyd_med=("hyd", "median"))
    fam["fam_dark_frac"] = fam.fam_ndark / fam.fam_n
    fam["fam_copies_mean"] = fam.fam_n / fam.fam_genomes
    per_genome = g.groupby(["token", "genome_code"]).size()
    fam["fam_copies_max"] = per_genome.groupby(level=0).max()
    darkfam = np.zeros(T, bool)
    darkfam[fam.index.values] = fam.fam_dark_frac.values >= DARK_FAM

    sp = pd.read_parquet(D / "significant_pairs.parquet")
    assert (sp.key.values == sp.lo.values * T + sp.hi.values).all()
    sig_sorted = np.sort(sp.key.values)
    sp_by_key = sp.set_index("key")
    tm = pd.read_parquet(D / "token_modules.parquet").set_index("token").module

    cc = g.contig_code.to_numpy()
    dark = g.dark.to_numpy()
    a = np.flatnonzero(cc[:-1] == cc[1:])
    b = a + 1
    ta, tb = tok[a], tok[b]
    key = np.minimum(ta, tb) * T + np.maximum(ta, tb)
    ok = (ta != tb) & dark[a] & dark[b] & darkfam[ta] & darkfam[tb] & np.isin(key, sig_sorted)
    link = np.zeros(len(g), bool)
    link[a[ok]] = True
    inrun = np.zeros(len(g), bool)
    inrun[a[ok]] = True
    inrun[b[ok]] = True
    start = inrun & ~np.r_[False, link[:-1]]
    rid = np.cumsum(start) - 1
    rid[~inrun] = -1
    run_start = np.flatnonzero(start)
    run_len = np.bincount(rid[rid >= 0])
    print(f"links {ok.sum():,}; runs {len(run_len):,}; genes in runs {inrun.sum():,}", flush=True)

    # ---- windows (block instances) ----------------------------------------------------------
    rows = []
    for s0, L in zip(run_start, run_len):
        for k in range(2, KMAX + 1):
            for s in range(s0, s0 + L - k + 1):
                gids = list(range(s, s + k))
                tt = tuple(int(tok[x]) for x in gids)
                if len(set(tt)) < k:
                    continue
                rev = tuple(reversed(tt)) < tt
                if rev:
                    tt, gids = tuple(reversed(tt)), list(reversed(gids))
                rows.append((tt, k, gids, rev, int(L)))
    inst = pd.DataFrame(rows, columns=["type", "k", "gids", "reversed", "run_len"])
    inst["type_key"] = inst["type"].map(lambda t: "-".join(map(str, t)))
    gg = g.set_index("gid")
    first = inst.gids.map(lambda x: x[0]).to_numpy()
    inst["genome_code"] = g.genome_code.to_numpy()[first]
    inst["contig_code"] = cc[first]
    lo_g = inst.gids.map(min).to_numpy()
    hi_g = inst.gids.map(max).to_numpy()
    pos = g.position_index.to_numpy()
    ngc = g.n_genes.to_numpy()
    inst["pos_start"] = pos[lo_g]
    inst["pos_end"] = pos[hi_g]
    inst["contig_n_genes"] = ngc[lo_g]
    inst["edge_dist"] = np.minimum(pos[lo_g], ngc[hi_g] - 1 - pos[hi_g])
    strand = g.strand.to_numpy()
    st_arr = inst.gids.map(lambda x: tuple(int(strand[i]) for i in x))
    inst["same_strand"] = st_arr.map(lambda s: len(set(s)) == 1)
    s_nt, e_nt = g.start.to_numpy(), g.end.to_numpy()
    inst["intergenic_nt"] = [[int(s_nt[i + 1] - e_nt[i] - 1) for i in range(lo, hi)] for lo, hi in zip(lo_g, hi_g)]
    inst["lengths"] = inst.gids.map(lambda x: [int(g.length.values[i]) for i in x])
    inst["hyds"] = inst.gids.map(lambda x: [round(float(g.hyd.values[i]), 4) for i in x])
    inst["n_x"] = inst.gids.map(lambda x: int(sum(g.n_x.values[i] for i in x)))
    part = g.partial.fillna("na").to_numpy()
    inst["partials"] = inst.gids.map(lambda x: [part[i] for i in x])
    inst = inst.merge(gm[["genome_code", "genome", "completeness", "contamination"] + TAX], on="genome_code", how="left")
    print(f"window instances {len(inst):,}; types {inst.type_key.nunique():,}", flush=True)

    # ---- engine conserved loci covering the instance --------------------------------------
    con = duckdb.connect()
    con.execute("SET threads=1")
    con.register("ii", inst[["type_key", "genome", "contig_code", "pos_start", "pos_end"]].reset_index())
    con.register("ctab", pd.read_parquet(D / "contigs.parquet")[["contig_code", "contig"]])
    q = con.execute(f"""
        WITH loc AS (SELECT genome, contig, start, "end", level FROM '{D}/loci_all_genomes.parquet' WHERE kind='conserved')
        SELECT ii.index, max(loc.level) AS max_level
        FROM ii JOIN ctab USING (contig_code)
        JOIN loc ON loc.genome = ii.genome AND loc.contig = ctab.contig
                 AND loc.start <= ii.pos_start AND loc."end" >= ii.pos_end
        GROUP BY ii.index""").df().set_index("index")
    inst["in_conserved_locus"] = inst.index.isin(q.index)
    inst["locus_max_level"] = q.max_level.reindex(inst.index).to_numpy()

    # ---- aggregate per type ------------------------------------------------------------------
    def agg(df):
        r = dict(k=int(df.k.iloc[0]), n_instances=len(df), n_genomes=df.genome_code.nunique())
        for c in TAX:
            r[f"n_{c}"] = df[c].nunique()
        r["n_genomes_no_phylum"] = int(df.loc[df.phylum.isna(), "genome_code"].nunique())
        r["phyla"] = ";".join(f"{p}:{n}" for p, n in df.groupby("phylum").genome_code.nunique().sort_values(ascending=False).items())
        r["same_strand_frac"] = float(df.same_strand.mean())
        ig = np.array(df.intergenic_nt.tolist())
        r["intergenic_med_nt"] = ";".join(str(int(np.median(ig[:, j]))) for j in range(ig.shape[1]))
        r["intergenic_max_med_nt"] = int(np.max(np.median(ig, axis=0)))
        ln = np.array(df.lengths.tolist())
        r["len_med"] = ";".join(str(int(np.median(ln[:, j]))) for j in range(ln.shape[1]))
        r["len_sum_med"] = int(np.median(ln.sum(1)))
        r["edge_frac"] = float((df.edge_dist == 0).mean())
        r["completeness_med"] = float(df.completeness.median())
        r["in_conserved_locus_frac"] = float(df.in_conserved_locus.mean())
        return pd.Series(r)

    cand = inst.groupby("type_key")[[c for c in inst.columns if c != "type_key"]].apply(agg)
    cand = cand[cand.n_genomes >= 3].copy()
    print(f"types with >= 3 genomes: {len(cand):,}", flush=True)
    toks = cand.index.map(lambda s: [int(x) for x in s.split("-")])
    cand["tokens"] = list(toks)

    # family columns per position
    for col, src in [("fam_dark_frac", "fam_dark_frac"), ("fam_copies_mean", "fam_copies_mean"),
                     ("fam_copies_max", "fam_copies_max"), ("fam_genomes", "fam_genomes"), ("fam_n", "fam_n"),
                     ("fam_len_med", "fam_len_med"), ("fam_hyd_med", "fam_hyd_med")]:
        cand[col] = [";".join(f"{fam.at[t, src]:.3g}" for t in tt) for tt in cand.tokens]
    cand["fam_dark_frac_min"] = [min(fam.at[t, "fam_dark_frac"] for t in tt) for tt in cand.tokens]
    cand["fam_copies_mean_max"] = [max(fam.at[t, "fam_copies_mean"] for t in tt) for tt in cand.tokens]
    cand["fam_hyd_med_max"] = [max(fam.at[t, "fam_hyd_med"] for t in tt) for tt in cand.tokens]
    cand["fam_len_med_min"] = [min(fam.at[t, "fam_len_med"] for t in tt) for tt in cand.tokens]
    cand["fam_len_med_max"] = [max(fam.at[t, "fam_len_med"] for t in tt) for tt in cand.tokens]
    cand["modules"] = [";".join(str(int(tm.get(t, -1))) for t in tt) for tt in cand.tokens]

    # engine pair statistics for consecutive pairs (G, L = genera, NP, J, logP via P)
    def pairstats(tt):
        out = []
        for x, y in zip(tt[:-1], tt[1:]):
            r = sp_by_key.loc[min(x, y) * T + max(x, y)]
            out.append((int(r.G), int(r.L), int(r.NP), float(r.J), float(r.P)))
        return out
    ps = cand.tokens.map(pairstats)
    cand["pair_G"] = ps.map(lambda v: ";".join(str(p[0]) for p in v))
    cand["pair_J_min"] = ps.map(lambda v: min(p[3] for p in v))
    cand["pair_P_max"] = ps.map(lambda v: max(p[4] for p in v))

    # ---- gap-0 conservation and block integrity --------------------------------------------
    need_pairs = sorted({(min(x, y), max(x, y)) for tt in cand.tokens for x, y in zip(tt[:-1], tt[1:])})
    need_keys = np.array([x * T + y for x, y in need_pairs], dtype=np.int64)
    gc = g.genome_code.to_numpy()
    recs = []
    for d in (1, 2, 3):
        l = np.flatnonzero(cc[:-d] == cc[d:])
        k2 = np.minimum(tok[l], tok[l + d]) * T + np.maximum(tok[l], tok[l + d])
        m = np.isin(k2, need_keys)
        recs.append(pd.DataFrame(dict(key=k2[m], genome=gc[l[m]], d=d)))
    co = pd.concat(recs)
    w3 = co.groupby("key").genome.nunique()
    adj = co[co.d == 1].groupby("key").genome.nunique()
    # genomes carrying both tokens anywhere
    need_tok = np.unique(np.array(need_pairs).ravel())
    tg = g[np.isin(tok, need_tok)].groupby("token").genome_code.agg(lambda s: set(s.tolist()))
    both = {x * T + y: len(tg[x] & tg[y]) for x, y in need_pairs}

    def gap0(tt):
        v = []
        for x, y in zip(tt[:-1], tt[1:]):
            kk = min(x, y) * T + max(x, y)
            v.append((adj.get(kk, 0), w3.get(kk, 0), both[kk]))
        return v
    gv = cand.tokens.map(gap0)
    cand["gap0_frac_min"] = gv.map(lambda v: min(a_ / w_ if w_ else 0 for a_, w_, _ in v))
    cand["coloc_frac_min"] = gv.map(lambda v: min(w_ / b_ if b_ else 0 for _, w_, b_ in v))
    cand["genomes_all_tokens"] = [len(set.intersection(*[tg[t] for t in tt])) for tt in cand.tokens]
    cand["block_integrity"] = cand.n_genomes / cand.genomes_all_tokens

    # ---- low complexity on family members (sequences from the Sharur DB, read-only) ---------
    cand_tok = np.unique(np.concatenate(cand.tokens.map(np.array).to_numpy()))
    mem = g.loc[np.isin(tok, cand_tok), ["gid", "protein_id", "token"]]
    sh = duckdb.connect(SHARUR, read_only=True)
    sh.execute("SET threads=1")
    sh.register("mem", mem[["protein_id"]])
    seqs = sh.execute("SELECT p.protein_id, rtrim(p.sequence,'*') AS seq FROM proteins p JOIN mem USING (protein_id)").df()
    mem = mem.merge(seqs, on="protein_id")
    mem["lc_frac"] = mem.seq.map(entropy_lowcomplex_frac)
    fam_lc = mem.groupby("token").lc_frac.median()
    mem[["gid", "token", "lc_frac"]].to_parquet(W / "member_lowcomplex.parquet")
    cand["fam_lc_med_max"] = [max(float(fam_lc[t]) for t in tt) for tt in cand.tokens]

    # ---- filters and rank ---------------------------------------------------------------------
    cand["pass_dark"] = cand.fam_dark_frac_min >= DARK_FAM
    cand["pass_len"] = (cand.fam_len_med_min >= 60) & (cand.fam_len_med_max <= 700) & (cand.len_sum_med <= 1400)
    cand["pass_hyd"] = cand.fam_hyd_med_max <= 0.45
    cand["pass_lowcomplex"] = cand.fam_lc_med_max <= LC_MAX
    cand["pass_copies"] = cand.fam_copies_mean_max <= 1.5
    cand["pass_geometry"] = (cand.same_strand_frac >= 0.8) & (cand.gap0_frac_min >= 0.5) & (cand.intergenic_max_med_nt <= 150)
    cand["pass_recurrence"] = (cand.n_genomes >= 5) & (cand.n_ord >= 2)
    cand["pass_all"] = cand[[c for c in cand.columns if c.startswith("pass_") and c != "pass_all"]].all(axis=1)
    # default rank: deep divergence first (orders, then families), then genomes, then integrity
    cand = cand.sort_values(["pass_all", "n_ord", "n_family", "n_genomes", "block_integrity"],
                            ascending=[False, False, False, False, False])
    # non-redundant selection: a family token serves one candidate (the best-ranked)
    used, nr = set(), []
    for key_, tt, pa in zip(cand.index, cand.tokens, cand.pass_all):
        if pa and not (set(tt) & used):
            nr.append(key_)
            used |= set(tt)
    cand["nonredundant_rank"] = pd.Series(np.arange(1, len(nr) + 1), index=nr).reindex(cand.index)
    cand["rank"] = np.arange(1, len(cand) + 1)
    cand.index.name = "type_key"
    cand["tokens"] = cand.tokens.map(lambda t: ";".join(map(str, t)))
    cand.reset_index().to_csv(W / "candidates_all.tsv", sep="\t", index=False)
    inst.to_parquet(W / "block_instances.parquet")
    fam.to_parquet(W / "fam_stats.parquet")
    summ = dict(links=int(ok.sum()), runs=int(len(run_len)), genes_in_runs=int(inrun.sum()),
                window_instances=int(len(inst)), types=int(inst.type_key.nunique()), types_ge3_genomes=int(len(cand)),
                pass_counts={c: int(cand[c].sum()) for c in cand.columns if c.startswith("pass_")},
                nonredundant=len(nr))
    json.dump(summ, open(W / "candidates_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
