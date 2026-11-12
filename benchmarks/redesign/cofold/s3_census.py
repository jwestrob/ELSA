"""Census of conserved unannotated gene blocks called by the ELSA recurrence engine on DPANN
(engine v1.5, rescue3 token run, results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3), with
the remote-homology status of every chain family. Runs serially (one thread) on the biotite login node.

Block construction (screens 1-2, build_candidates.py, with the dark-family rule moved to a column):
  unannotated gene  no row in any curated Sharur annotation source (every `annotations` source except
                    'cdd'), no system_proteins row (incl. cctyper), no derived _locus/_validation atom, no
                    curated gold-instance membership (live database of 12 Nov 2026; identical to the
                    October set: 1,398,957 proteins)
  link              adjacent genes on one contig, both unannotated, distinct tokens, token pair significant
                    in the engine run (P <= 1e-4 and L >= 2)
  run / window      maximal chain of links; block instance = window of k = 2..8 consecutive genes of a run
                    with k distinct tokens; block type = canonical ordered token tuple
  census            block types in >= 3 genomes
Screen filters, kept as columns (screen 2 values): pass_darkfam (every chain family >= 98 % unannotated
members), pass_len2 (family medians 60-1,200 aa, median summed length <= 2,500 aa), pass_hyd (family median
hydrophobic fraction <= 0.45), pass_lowcomplex (family median SEG-trigger fraction <= 0.145), pass_copies
(members / genomes <= 1.5), pass_geometry (same strand >= 0.8, gap-0 fraction >= 0.5, median intergenic
<= 150 nt), pass_recurrence (>= 5 genomes, >= 2 orders). Lengths are true sequence lengths (Sharur
sequence without '*'), not universe `length` (which includes the stop for 1.23 M genes).

Chain-family status (rules in predeclared_criteria_screen3.json, 'census_status_rules'; order of recognition):
  annotated                family fails the dark-family rule (< 98 % of members unannotated)
  resembles_characterised  the searched permissive-e3c50s75 family sharing the token family's representative
                           (Sharur family_annotations_via semantics) has a family_resemblance row (HHsearch
                           Pfam / HHblits PDB100, probability >= 90 % over >= 30 columns) to an entry
                           without unknown-function wording
  not_in_tier              no searched counterpart (counterpart in < 5 genomes or not Pfam-dark, or no
                           family with the same representative); its own status, with a CDD sub-status
  cdd_only                 a CDD model without unknown-function wording on >= 2 members and >= 10 % of members
  unknown_only             only unknown-function entries meet the tier rule and/or the CDD family rule
  no_hit                   searched, nothing meets the tier rule or the CDD family rule
Ordering for least / most recognised: not_in_tier < no_hit < unknown_only < cdd_only <
resembles_characterised < annotated.

Writes $RD/cofold/s3/census/: token_status.parquet, census_types.parquet, census_instances.parquet,
gene_census.parquet (genes of engine loci), census_summary.json, and prints the tables.
"""
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
U = RD / "ws_dpann/package/universe"
GOLD = RD / "ws_dpann/package/gold"
ENG = RD / "engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3"
TOKF = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet"
OLDTOKF = RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75__dpann.parquet"
FAA = RD / "ws_dpann/work/proteins_gid.faa"
S3 = RD / "cofold/s3"
D = S3 / "data"
OUT = S3 / "census"
SEARCHED = "permissive-e3c50s75"
TOKENRUN = "permissive-e3c50s75-rescue3"
KMAX = 8
DARK_FAM = 0.98
LC_MAX = 0.145
TAX = ["phylum", "class", "ord", "family", "genus"]
RULE_SOURCES = ("hhsearch_pfam", "hhsearch_pdb", "hhblits_pdb")
STATUS_ORDER = ["not_in_tier", "no_hit", "unknown_only", "cdd_only", "resembles_characterised", "annotated"]
SRANK = {s: i for i, s in enumerate(STATUS_ORDER)}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ low complexity (vectorised SEG trigger)
def lc_frac_ref(seq, w=12, k1=2.2):
    """Reference implementation from build_candidates.py."""
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


def lc_frac(seq, w=12, k1=2.2):
    n = len(seq)
    if n < w:
        return 0.0
    a = np.frombuffer(seq.encode(), dtype=np.uint8) - 65
    oh = np.zeros((n + 1, 26), np.int16)
    oh[np.arange(1, n + 1), a] = 1
    cs = np.cumsum(oh, axis=0)
    cnt = (cs[w:] - cs[:-w]).astype(float) / w
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(cnt > 0, cnt * np.log2(cnt), 0.0), axis=1)
    low = h < k1 - 1e-12
    d = np.zeros(n + 1, np.int32)
    idx = np.flatnonzero(low)
    np.add.at(d, idx, 1)
    np.add.at(d, idx + w, -1)
    return float((np.cumsum(d[:n]) > 0).mean())


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


def canon(tt, gids):
    tt = tuple(tt)
    if tuple(reversed(tt)) < tt:
        return tuple(reversed(tt)), list(reversed(gids)), True
    return tt, list(gids), False


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=1")

    # ---- gene table -------------------------------------------------------------------------------
    g = pd.read_parquet(U / "genes.parquet")
    assert (g.gid.values == np.arange(len(g))).all()
    fl = pd.read_parquet(D / "prot_flags_slim.parquet")
    comp = pd.read_parquet(D / "composition_slim.parquet")
    pf = pd.read_parquet(D / "prot_fams.parquet")[["protein_id", "fam_e3", "fam_r3"]]
    g = g.merge(fl, on="protein_id", how="left").merge(comp, on="protein_id", how="left").merge(pf, on="protein_id", how="left")
    assert (g.gid.values == np.arange(len(g))).all() and g.cur_annot.notna().all() and g.hyd.notna().all()
    tok = pd.read_parquet(TOKF).sort_values("gid")
    assert (tok.gid.values == np.arange(len(g))).all()
    g["token"] = tok.token.values
    oldtok = pd.read_parquet(OLDTOKF).sort_values("gid")
    g["token_e3"] = oldtok.token.values
    mem = pd.read_parquet(GOLD / "members.parquet")
    g["gold"] = g.gid.isin(mem.gid)
    g["dark"] = ~(g.cur_annot | g.in_sys | g.derived | g.gold)
    log(f"genes {len(g):,}; unannotated {int(g.dark.sum()):,} ({g.dark.mean():.4f}); "
        f"cur_annot {int(g.cur_annot.sum()):,} in_sys {int(g.in_sys.sum()):,} derived {int(g.derived.sum()):,} gold {int(g.gold.sum()):,}")
    # token <-> Sharur rescue3 family must be one-to-one
    tf = g.groupby("token").fam_r3.nunique()
    ft = g.groupby("fam_r3").token.nunique()
    log(f"tokens {len(tf):,}; tokens with >1 Sharur rescue3 family: {int((tf > 1).sum())}; families with >1 token: {int((ft > 1).sum())}")
    assert (tf == 1).all() and (ft == 1).all()

    gm = pd.read_parquet(U / "genomes.parquet")
    for c in TAX:
        gm[c] = gm[c].replace("", np.nan)
    ct = pd.read_parquet(U / "contigs.parquet")[["contig_code", "n_genes", "length_nt"]]
    g = g.merge(ct, on="contig_code", how="left")
    assert (g.gid.values == np.arange(len(g))).all()
    tokv = g.token.to_numpy()
    T = int(tokv.max()) + 1

    # ---- family (token) statistics -------------------------------------------------------------------
    fam = g.groupby("token").agg(fam_n=("gid", "size"), fam_ndark=("dark", "sum"), fam_genomes=("genome_code", "nunique"),
                                 fam_len_med=("aa_len", "median"), fam_hyd_med=("hyd", "median"),
                                 fam_r3=("fam_r3", "first"), n_pfam=("src_pfam", "sum"), n_ko=("src_ko", "sum"),
                                 n_df=("src_df", "sum"), n_txss=("src_txss", "sum"), n_sys=("in_sys", "sum"),
                                 n_gold=("gold", "sum"))
    fam["fam_dark_frac"] = fam.fam_ndark / fam.fam_n
    fam["fam_copies_mean"] = fam.fam_n / fam.fam_genomes
    fam["fam_copies_max"] = g.groupby(["token", "genome_code"]).size().groupby(level=0).max()
    darkfam = np.zeros(T, bool)
    darkfam[fam.index.values] = fam.fam_dark_frac.values >= DARK_FAM

    sp = pd.read_parquet(ENG / "significant_pairs.parquet")
    assert (sp.key.values == sp.lo.values * T + sp.hi.values).all()
    sig_sorted = np.sort(sp.key.values)
    sp_by_key = sp.set_index("key")
    tm = pd.read_parquet(ENG / "token_modules.parquet").set_index("token").module

    # ---- links, runs, windows ------------------------------------------------------------------------
    cc = g.contig_code.to_numpy()
    dark = g.dark.to_numpy()
    a = np.flatnonzero(cc[:-1] == cc[1:])
    b = a + 1
    ta, tb = tokv[a], tokv[b]
    key = np.minimum(ta, tb) * T + np.maximum(ta, tb)
    sig = np.isin(key, sig_sorted)
    ok = (ta != tb) & dark[a] & dark[b] & sig
    ok_df = ok & darkfam[ta] & darkfam[tb]
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
    adj_pairs = int(len(a))
    log(f"adjacent pairs {adj_pairs:,}; significant {int(sig.sum()):,}; links (unannotated genes) {int(ok.sum()):,} "
        f"(of which both families dark {int(ok_df.sum()):,}); runs {len(run_len):,}; genes in runs {int(inrun.sum()):,}")

    W_s, W_k, W_key, W_rev, W_L = [], [], [], [], []
    for s0, L in zip(run_start, run_len):
        tl = [int(x) for x in tokv[s0:s0 + L]]
        for k in range(2, min(KMAX, L) + 1):
            for i in range(0, L - k + 1):
                tt = tuple(tl[i:i + k])
                if len(set(tt)) < k:
                    continue
                rt = tuple(reversed(tt))
                rev = rt < tt
                W_s.append(s0 + i)
                W_k.append(k)
                W_key.append("-".join(map(str, rt if rev else tt)))
                W_rev.append(rev)
                W_L.append(int(L))
    inst = pd.DataFrame(dict(type_key=W_key, k=np.array(W_k, np.int8), s=np.array(W_s, np.int64), reversed=W_rev,
                             run_len=np.array(W_L, np.int32)))
    del W_s, W_k, W_key, W_rev, W_L
    inst["genome_code"] = g.genome_code.to_numpy()[inst.s.to_numpy()]
    ng = inst.groupby("type_key").genome_code.nunique()
    n_windows, n_types = len(inst), len(ng)
    log(f"window instances {n_windows:,}; types {n_types:,}; types in >= 3 genomes {int((ng >= 3).sum()):,}")
    inst = inst[inst.type_key.isin(ng.index[ng >= 3])].reset_index(drop=True)
    inst["gids"] = [list(range(s, s + k))[::-1] if rv else list(range(s, s + k))
                    for s, k, rv in zip(inst.s, inst.k, inst.reversed)]
    inst = inst.drop(columns=["s"])
    inst["k"] = inst.k.astype(int)
    first = inst.gids.map(lambda x: x[0]).to_numpy()
    inst["contig_code"] = cc[first]
    lo_g = inst.gids.map(min).to_numpy()
    hi_g = inst.gids.map(max).to_numpy()
    pos = g.position_index.to_numpy()
    ngc = g.n_genes.to_numpy()
    inst["pos_start"] = pos[lo_g]
    inst["pos_end"] = pos[hi_g]
    inst["gid_lo"] = lo_g
    inst["gid_hi"] = hi_g
    inst["contig_n_genes"] = ngc[lo_g]
    inst["edge_dist"] = np.minimum(pos[lo_g], ngc[hi_g] - 1 - pos[hi_g])
    strand = g.strand.to_numpy()
    inst["same_strand"] = inst.gids.map(lambda x: len({int(strand[i]) for i in x}) == 1)
    s_nt, e_nt = g.start.to_numpy(), g.end.to_numpy()
    inst["intergenic_nt"] = [[int(s_nt[i + 1] - e_nt[i] - 1) for i in range(lo, hi)] for lo, hi in zip(lo_g, hi_g)]
    L_aa = g.aa_len.to_numpy()
    H = g.hyd.to_numpy()
    NX = g.n_x.to_numpy()
    inst["lengths"] = inst.gids.map(lambda x: [int(L_aa[i]) for i in x])
    inst["hyds"] = inst.gids.map(lambda x: [round(float(H[i]), 4) for i in x])
    inst["n_x"] = inst.gids.map(lambda x: int(sum(NX[i] for i in x)))
    part = g.partial.fillna("na").to_numpy()
    inst["partials"] = inst.gids.map(lambda x: [part[i] for i in x])
    inst = inst.merge(gm[["genome_code", "genome", "completeness", "contamination"] + TAX], on="genome_code", how="left")

    # engine conserved loci covering the instance (any level, and the default level-2 reading)
    con.register("ii", inst[["genome", "contig_code", "pos_start", "pos_end"]].reset_index())
    con.register("ctab", pd.read_parquet(U / "contigs.parquet")[["contig_code", "contig"]])
    q = con.execute(f"""
        WITH loc AS (SELECT genome, contig, start, "end", level FROM '{ENG}/loci_all_genomes.parquet' WHERE kind='conserved')
        SELECT ii.index, max(loc.level) AS max_level, bool_or(loc.level = 2) AS in_l2
        FROM ii JOIN ctab USING (contig_code)
        JOIN loc ON loc.genome = ii.genome AND loc.contig = ctab.contig
                 AND loc.start <= ii.pos_start AND loc."end" >= ii.pos_end
        GROUP BY ii.index""").df().set_index("index")
    inst["in_conserved_locus"] = inst.index.isin(q.index)
    inst["locus_max_level"] = q.max_level.reindex(inst.index).to_numpy()
    log(f"instances of >= 3-genome types {len(inst):,}; inside an engine conserved locus {inst.in_conserved_locus.mean():.4f}")

    # ---- per-type aggregation ------------------------------------------------------------------------
    def agg(df):
        r = dict(k=int(df.k.iloc[0]), n_instances=len(df), n_genomes=df.genome_code.nunique())
        for c in TAX:
            r[f"n_{c}"] = df[c].nunique()
        r["phyla"] = ";".join(f"{p}:{n}" for p, n in df.groupby("phylum").genome_code.nunique().sort_values(ascending=False).items())
        r["orders"] = ";".join(f"{p}:{n}" for p, n in df.groupby("ord").genome_code.nunique().sort_values(ascending=False).head(12).items())
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
    cand["tokens"] = [[int(x) for x in s.split("-")] for s in cand.index]
    log(f"types aggregated {len(cand):,}")
    for col in ("fam_dark_frac", "fam_copies_mean", "fam_genomes", "fam_n", "fam_len_med", "fam_hyd_med"):
        cand[col] = [";".join(f"{fam.at[t, col]:.3g}" for t in tt) for tt in cand.tokens]
    cand["fam_dark_frac_min"] = [min(fam.at[t, "fam_dark_frac"] for t in tt) for tt in cand.tokens]
    cand["fam_copies_mean_max"] = [max(fam.at[t, "fam_copies_mean"] for t in tt) for tt in cand.tokens]
    cand["fam_hyd_med_max"] = [max(fam.at[t, "fam_hyd_med"] for t in tt) for tt in cand.tokens]
    cand["fam_len_med_min"] = [min(fam.at[t, "fam_len_med"] for t in tt) for tt in cand.tokens]
    cand["fam_len_med_max"] = [max(fam.at[t, "fam_len_med"] for t in tt) for tt in cand.tokens]
    cand["modules"] = [";".join(str(int(tm.get(t, -1))) for t in tt) for tt in cand.tokens]

    def pairstats(tt):
        out = []
        for x, y in zip(tt[:-1], tt[1:]):
            r = sp_by_key.loc[min(x, y) * T + max(x, y)]
            out.append((int(r.G), int(r.L), int(r.NP), float(r.J), float(r.P)))
        return out
    ps = cand.tokens.map(pairstats)
    cand["pair_G"] = ps.map(lambda v: ";".join(str(p[0]) for p in v))
    cand["pair_L"] = ps.map(lambda v: ";".join(str(p[1]) for p in v))
    cand["pair_J_min"] = ps.map(lambda v: min(p[3] for p in v))
    cand["pair_P_max"] = ps.map(lambda v: max(p[4] for p in v))

    need_pairs = sorted({(min(x, y), max(x, y)) for tt in cand.tokens for x, y in zip(tt[:-1], tt[1:])})
    need_keys = np.array([x * T + y for x, y in need_pairs], dtype=np.int64)
    gc = g.genome_code.to_numpy()
    recs = []
    for d in (1, 2, 3):
        l = np.flatnonzero(cc[:-d] == cc[d:])
        k2 = np.minimum(tokv[l], tokv[l + d]) * T + np.maximum(tokv[l], tokv[l + d])
        m = np.isin(k2, need_keys)
        recs.append(pd.DataFrame(dict(key=k2[m], genome=gc[l[m]], d=d)))
    co = pd.concat(recs)
    w3 = co.groupby("key").genome.nunique()
    adj = co[co.d == 1].groupby("key").genome.nunique()
    need_tok = np.unique(np.array(need_pairs).ravel())
    tg = g[np.isin(tokv, need_tok)].groupby("token").genome_code.agg(lambda s: set(s.tolist()))
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
    log("geometry and integrity done")

    # ---- low complexity (family members' sequences, biotite FASTA) ------------------------------------
    cand_tok = np.unique(np.concatenate([np.array(t) for t in cand.tokens]))
    memg = g.loc[np.isin(tokv, cand_tok), ["gid", "token"]]
    seqs = read_faa(FAA, set(memg.gid.tolist()))
    log(f"member sequences read {len(seqs):,}")
    rng = np.random.default_rng(17)
    test = rng.choice(memg.gid.to_numpy(), size=min(300, len(memg)), replace=False)
    dif = max(abs(lc_frac(seqs[x]) - lc_frac_ref(seqs[x])) for x in test)
    log(f"vectorised low-complexity check on 300 members: max |diff| = {dif:.2e}")
    assert dif < 1e-9
    memg["lc"] = [lc_frac(seqs[x]) for x in memg.gid]
    fam_lc = memg.groupby("token").lc.median()
    cand["fam_lc_med_max"] = [max(float(fam_lc[t]) for t in tt) for tt in cand.tokens]
    log("low complexity done")

    # ---- screen filters as columns -------------------------------------------------------------------
    cand["pass_darkfam"] = cand.fam_dark_frac_min >= DARK_FAM
    cand["pass_len1"] = (cand.fam_len_med_min >= 60) & (cand.fam_len_med_max <= 700) & (cand.len_sum_med <= 1400)
    cand["pass_len2"] = (cand.fam_len_med_min >= 60) & (cand.fam_len_med_max <= 1200) & (cand.len_sum_med <= 2500)
    cand["pass_hyd"] = cand.fam_hyd_med_max <= 0.45
    cand["pass_lowcomplex"] = cand.fam_lc_med_max <= LC_MAX
    cand["pass_copies"] = cand.fam_copies_mean_max <= 1.5
    cand["pass_geometry"] = (cand.same_strand_frac >= 0.8) & (cand.gap0_frac_min >= 0.5) & (cand.intergenic_max_med_nt <= 150)
    cand["pass_recurrence"] = (cand.n_genomes >= 5) & (cand.n_ord >= 2)
    cand["pass_fold"] = cand[["pass_darkfam", "pass_len2", "pass_hyd", "pass_lowcomplex", "pass_copies", "pass_geometry",
                              "pass_recurrence"]].all(axis=1)

    # ---- chain-family status ---------------------------------------------------------------------------
    fs = pd.read_parquet(D / "fam_stats.parquet")
    fs_r3 = fs[fs.clustering_run == TOKENRUN].set_index("family_id")
    fs_e3 = fs[fs.clustering_run == SEARCHED]
    assert fs_e3.rep_protein_id.is_unique and fs_r3.rep_protein_id.is_unique
    e3_by_rep = fs_e3.set_index("rep_protein_id")
    tstat = pd.read_parquet(D / "tier_status.parquet")
    tstat = tstat[(tstat.clustering_run == SEARCHED) & (tstat.status == "ok")].set_index("family_id")
    th = pd.read_parquet(D / "tier_hits.parquet")
    rule = th[th.source.isin(RULE_SOURCES) & (th.probability >= 90) & (th.aligned_cols >= 30)]
    best_char = (rule[~rule.unknown_function].sort_values(["probability", "aligned_cols"], ascending=False)
                 .drop_duplicates("family_id").set_index("family_id"))
    best_unk = (rule[rule.unknown_function].sort_values(["probability", "aligned_cols"], ascending=False)
                .drop_duplicates("family_id").set_index("family_id"))
    best_any = (th[th.source.isin(RULE_SOURCES)].sort_values(["probability", "aligned_cols"], ascending=False)
                .drop_duplicates("family_id").set_index("family_id"))

    st = fam.loc[cand_tok, ["fam_n", "fam_genomes", "fam_dark_frac", "fam_r3", "fam_len_med", "n_pfam", "n_ko", "n_df",
                            "n_txss", "n_sys", "n_gold"]].copy()
    st["rep_r3"] = fs_r3.rep_protein_id.reindex(st.fam_r3).to_numpy()
    st["e3_counterpart"] = [e3_by_rep.family_id.get(r) if r in e3_by_rep.index else None for r in st.rep_r3]
    st["e3_members"] = [int(e3_by_rep.n_members.get(r)) if r in e3_by_rep.index else 0 for r in st.rep_r3]
    st["e3_genomes"] = [int(e3_by_rep.n_genomes.get(r)) if r in e3_by_rep.index else 0 for r in st.rep_r3]
    st["e3_pfam_dark"] = [bool(e3_by_rep.pfam_dark.get(r)) if r in e3_by_rep.index else False for r in st.rep_r3]
    st["in_tier"] = st.e3_counterpart.isin(tstat.index)
    for lab, src in (("char", best_char), ("unk", best_unk), ("best", best_any)):
        sub = src.reindex(st.e3_counterpart)
        st[f"tier_{lab}_hit"] = (sub.hit_name.fillna("") + " " + sub.hit_description.fillna("").str[:60]).str.strip().to_numpy()
        st[f"tier_{lab}_acc"] = sub.hit_accession.to_numpy()
        st[f"tier_{lab}_src"] = sub.source.to_numpy()
        st[f"tier_{lab}_prob"] = sub.probability.to_numpy()
        st[f"tier_{lab}_cols"] = sub.aligned_cols.to_numpy()
    st["tier_char"] = st.tier_char_prob.notna()
    st["tier_unk"] = st.tier_unk_prob.notna()
    # member-based view: every searched family among the token's members (sensitivity only)
    mm = g.loc[np.isin(tokv, cand_tok), ["token", "fam_e3"]]
    mm["searched"] = mm.fam_e3.isin(tstat.index)
    mm["char"] = mm.fam_e3.isin(best_char.index)
    mm["unk"] = mm.fam_e3.isin(best_unk.index)
    mg_ = mm.groupby("token").agg(frac_members_searched=("searched", "mean"), frac_members_char_family=("char", "mean"),
                                  frac_members_unk_family=("unk", "mean"), n_e3_families=("fam_e3", "nunique"))
    st = st.join(mg_)
    # CDD family rule
    cdd = pd.read_parquet(D / "cdd_prot.parquet", columns=["protein_id", "accession", "evalue"])
    cm = pd.read_parquet(D / "cdd_models.parquet").set_index("accession")
    pid2tok = g.loc[np.isin(tokv, cand_tok)].set_index("protein_id").token
    cdd = cdd[cdd.protein_id.isin(pid2tok.index)].copy()
    cdd["token"] = pid2tok.reindex(cdd.protein_id).to_numpy()
    cdd["unknown"] = cm.unknown_function.reindex(cdd.accession).fillna(False).to_numpy()
    cmem = cdd.groupby("token").protein_id.nunique()
    cmem_char = cdd[~cdd.unknown].groupby("token").protein_id.nunique()
    ta_ = cdd.groupby(["token", "accession"]).agg(n=("protein_id", "nunique"), emin=("evalue", "min")).reset_index()
    ta_ = ta_.merge(st[["fam_n"]], left_on="token", right_index=True)
    ta_["frac"] = ta_.n / ta_.fam_n
    ta_["unknown"] = cm.unknown_function.reindex(ta_.accession).fillna(False).to_numpy()
    ta_["meets"] = (ta_.n >= 2) & (ta_.frac >= 0.10)
    tm_ = ta_[ta_.meets].sort_values(["n", "emin"], ascending=[False, True])
    cc_char = tm_[~tm_.unknown].drop_duplicates("token").set_index("token")
    cc_unk = tm_[tm_.unknown].drop_duplicates("token").set_index("token")
    st["cdd_members_any"] = cmem.reindex(st.index).fillna(0).astype(int)
    st["cdd_members_char"] = cmem_char.reindex(st.index).fillna(0).astype(int)
    for lab, src in (("char", cc_char), ("unk", cc_unk)):
        sub = src.reindex(st.index)
        st[f"cdd_{lab}"] = sub.accession.notna().to_numpy()
        st[f"cdd_{lab}_acc"] = sub.accession.to_numpy()
        st[f"cdd_{lab}_name"] = cm.name.reindex(sub.accession).to_numpy()
        st[f"cdd_{lab}_desc"] = cm.description.reindex(sub.accession).str[:80].to_numpy()
        st[f"cdd_{lab}_n"] = sub.n.to_numpy()
        st[f"cdd_{lab}_frac"] = sub.frac.to_numpy()

    def status(r):
        if r.fam_dark_frac < DARK_FAM:
            return "annotated"
        if r.tier_char:
            return "resembles_characterised"
        if not r.in_tier:
            return "not_in_tier"
        if r.cdd_char:
            return "cdd_only"
        if r.tier_unk or r.cdd_unk:
            return "unknown_only"
        return "no_hit"
    st["status"] = st.apply(status, axis=1)
    st["not_in_tier_cdd"] = np.where(st.status == "not_in_tier",
                                     np.where(st.cdd_char, "cdd_char", np.where(st.cdd_unk, "cdd_unknown", "cdd_none")), "")
    st["not_in_tier_reason"] = np.where(st.status != "not_in_tier", "",
                                        np.where(st.e3_counterpart.isna(), "no_counterpart_same_rep",
                                                 np.where(~st.e3_pfam_dark, "counterpart_not_pfam_dark",
                                                          np.where(st.e3_genomes < 5, "counterpart_lt5_genomes", "other"))))
    st.index.name = "token"
    st.reset_index().to_parquet(OUT / "token_status.parquet")
    log("token status:", st.status.value_counts().to_dict())
    log("not_in_tier reasons:", st.not_in_tier_reason[st.status == "not_in_tier"].value_counts().to_dict())

    # ---- block-level status ----------------------------------------------------------------------------
    sdict = st.status.to_dict()
    cand["chain_status"] = [";".join(sdict[t] for t in tt) for tt in cand.tokens]
    cand["least_status"] = [min((sdict[t] for t in tt), key=SRANK.get) for tt in cand.tokens]
    cand["most_status"] = [max((sdict[t] for t in tt), key=SRANK.get) for tt in cand.tokens]
    cand["all_unrecognised"] = [all(sdict[t] in ("no_hit", "unknown_only") for t in tt) for tt in cand.tokens]
    cand["unrec_class"] = np.where(~cand.all_unrecognised, "",
                                   np.where([all(sdict[t] == "no_hit" for t in tt) for tt in cand.tokens], "all_no_hit",
                                            "with_unknown_only"))
    cand["any_member_char_family"] = [any(st.at[t, "frac_members_char_family"] > 0 for t in tt) for tt in cand.tokens]
    cand["tier_best_prob_max"] = [np.nanmax([st.at[t, "tier_best_prob"] if pd.notna(st.at[t, "tier_best_prob"]) else 0 for t in tt])
                                  for tt in cand.tokens]

    # ---- screens 1-2 folded block types mapped to rescue3 tokens ----------------------------------------
    m1 = pd.read_csv(D / "fold_manifest.tsv", sep="\t")
    p1 = pd.read_csv(D / "per_candidate_folds.tsv", sep="\t").set_index("cand_rank")
    m2 = pd.read_csv(D / "fold_manifest2.tsv", sep="\t")
    b2 = pd.read_csv(D / "s2_blocks.tsv", sep="\t")
    b2s = b2[b2["mode"] == "single"].set_index("s2_rank")
    folded = []
    for _, r in m1[m1.cls == "cognate"].iterrows():
        gl = sorted(int(x) for x in str(r.gids).split(";"))
        tt, _, _ = canon([int(tokv[x]) for x in gl], gl)
        rk = int(r.cand_rank)
        folded.append(dict(screen="s1", rank=rk, rep=int(r.rep), old_type=r.type_key, new_type="-".join(map(str, tt)),
                           outcome=("hit" if p1.at[rk, "hit_predeclared"] else ("c123_both" if p1.at[rk, "c123_both"] else "no")),
                           min_cog_iptm=float(p1.at[rk, "min_cog_iptm"])))
    for _, r in m2[(m2.cls == "cognate") & m2.s2_rank.isin(b2s.index)].iterrows():
        gl = sorted(int(x) for x in str(r.gids).split(";"))
        tt, _, _ = canon([int(tokv[x]) for x in gl], gl)
        rk = int(r.s2_rank)
        x = b2s.loc[rk]
        folded.append(dict(screen="s2", rank=rk, rep=int(r.rep), old_type=r.type_key, new_type="-".join(map(str, tt)),
                           outcome=("full_block" if x.full_block_cofold else ("interface_hit" if x.interface_hit_block else "no")),
                           min_cog_iptm=float(x.best_min_pair_iptm)))
    fo = pd.DataFrame(folded)
    fo.to_csv(OUT / "folded_s12_mapped.tsv", sep="\t", index=False)
    lab = fo.groupby("new_type").apply(lambda d: ";".join(sorted({f"{s}:{k}:{o}" for s, k, o in zip(d.screen, d["rank"], d.outcome)})))
    cand["folded_s12"] = lab.reindex(cand.index).fillna("").to_numpy()
    fpairs = set()
    for t in fo.new_type:
        tt = [int(x) for x in t.split("-")]
        fpairs |= {(min(x, y), max(x, y)) for x, y in zip(tt[:-1], tt[1:])}
    cand["shares_pair_with_folded"] = [any((min(x, y), max(x, y)) in fpairs for x, y in zip(tt[:-1], tt[1:])) for tt in cand.tokens]
    log(f"screens 1-2 folded: {fo.groupby('screen').new_type.nunique().to_dict()} distinct mapped types; "
        f"census types matching one: {int((cand.folded_s12 != '').sum())}")

    cand.index.name = "type_key"
    cand2 = cand.copy()
    cand2["tokens"] = cand2.tokens.map(lambda t: ";".join(map(str, t)))
    cand2.reset_index().to_parquet(OUT / "census_types.parquet")
    inst.to_parquet(OUT / "census_instances.parquet")

    # ---- gene-level census over engine loci (default reading: conserved level 2) -----------------------
    loc = con.execute(f"""SELECT l.genome, l.contig, l.start, l."end", l.level FROM '{ENG}/loci_all_genomes.parquet' l
                          WHERE kind='conserved'""").df()
    cmap = pd.read_parquet(U / "contigs.parquet")[["genome", "contig", "first_gid"]]
    loc = loc.merge(cmap, on=["genome", "contig"])
    in_l2 = np.zeros(len(g) + 1, np.int32)
    in_any = np.zeros(len(g) + 1, np.int32)
    for lv, arr in ((2.0, in_l2), (None, in_any)):
        sub = loc if lv is None else loc[loc.level == lv]
        s0 = (sub.first_gid + sub.start).to_numpy()
        e0 = (sub.first_gid + sub["end"]).to_numpy() + 1
        np.add.at(arr, s0, 1)
        np.add.at(arr, e0, -1)
    in_l2 = np.cumsum(in_l2[:-1]) > 0
    in_any = np.cumsum(in_any[:-1]) > 0
    cat = cand[["least_status", "most_status", "all_unrecognised", "pass_fold", "pass_recurrence"]]
    inst2 = inst[["type_key", "gids"]].merge(cat, left_on="type_key", right_index=True)
    gflag = {}
    for name, sel in (("any_block", np.ones(len(inst2), bool)),
                      ("block_all_unrecognised", inst2.all_unrecognised.to_numpy()),
                      ("block_all_unrecognised_rec", (inst2.all_unrecognised & inst2.pass_recurrence).to_numpy()),
                      ("block_least_not_in_tier", (inst2.least_status == "not_in_tier").to_numpy()),
                      ("block_most_recognised_ge_cdd", inst2.most_status.map(SRANK).ge(SRANK["cdd_only"]).to_numpy())):
        arr = np.zeros(len(g), bool)
        for x in inst2.gids[sel]:
            arr[list(x)] = True
        gflag[name] = arr
    gc_ = pd.DataFrame(dict(gid=g.gid, dark=dark, in_l2=in_l2, in_any_conserved=in_any, **gflag))
    gc_.to_parquet(OUT / "gene_census.parquet")
    gs = {}
    for den in ("in_l2", "in_any_conserved"):
        dmask = gc_[den].to_numpy()
        gs[den] = dict(genes=int(dmask.sum()), unannotated=int((dmask & dark).sum()),
                       **{k: int((dmask & v).sum()) for k, v in gflag.items()},
                       **{f"frac_{k}": round(float((dmask & v).sum() / dmask.sum()), 5) for k, v in gflag.items()})
    log("gene-level:", json.dumps(gs))

    # ---- summary tables -------------------------------------------------------------------------------
    summ = dict(adjacent_pairs=adj_pairs, links=int(ok.sum()), links_darkfam=int(ok_df.sum()), runs=int(len(run_len)),
                genes_in_runs=int(inrun.sum()), windows=n_windows, types_all=n_types,
                types_ge3=int(len(cand)), instances_ge3=int(len(inst)),
                status_tokens=st.status.value_counts().to_dict(), gene_level=gs,
                pass_counts={c: int(cand[c].sum()) for c in cand.columns if c.startswith("pass_")})
    json.dump(summ, open(OUT / "census_summary.json", "w"), indent=1, default=int)
    pd.set_option("display.width", 250)
    for col in ("least_status", "most_status"):
        print(cand.groupby(col).agg(types=("k", "size"), instances=("n_instances", "sum")).reindex(STATUS_ORDER).to_string())
    print(pd.crosstab(cand.least_status, cand.most_status).reindex(index=STATUS_ORDER, columns=STATUS_ORDER).to_string())
    print(cand.groupby(["all_unrecognised", "pass_fold"]).size().to_string())
    print(json.dumps(summ, indent=1, default=int))
    log("done")


if __name__ == "__main__":
    main()
