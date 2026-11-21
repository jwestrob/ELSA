#!/usr/bin/env python3
"""Loci homologous to the M. extorquens AM1 lut cluster in a GTDB slice (workstream slice_lut).

usage: 05_loci.py CONFIG_JSON
CONFIG keys: universe (dir with genes/genomes/contigs/gene_flags/contig_extra parquet), fwd, rev, self (convertalis TSVs),
  globdb_rows (parquet: genome_id, gene_idx, source, accession, function, e_value), globdb_counts (parquet), gene_offset
  (int, GlobDB gene_idx = file_index + offset; null = estimate from pfam_check), pfam_check (TSV, optional),
  pfam_check_universe (dir, optional), comps (ordered list of AM1 names counted as cluster components), labels
  ({AM1 name: short label}), out (dir), threads, null_reps, seed.

Gene assignment tiers (each gene gets at most one component):
  RB   reverse best: the gene's best hit in the AM1 proteome (MMseqs2, bits) is the component, and the forward hit of
       that component to the gene has E <= 1e-5.
  ANY  forward hit E <= 1e-3; component = the forward hit with the highest bits.
Locus definitions (components counted among CONFIG comps only; flanking AM1 genes are reported, not counted):
  main   RB genes, consecutive component genes <= 3 intervening genes apart, >= 3 distinct components
  strict RB genes, <= 1 intervening gene, >= 5 distinct components including the TonB-dependent receptor and >= 1 ABC
         transporter gene
  loose  ANY genes, <= 5 intervening genes, >= 2 distinct components
Null: each component's assigned genes are shifted by an independent random offset within their genome (circular over
the genome's gene order; contig membership taken at the new position), loci recalled; NULL_REPS replicates.
"""
import json, os, sys, time
import numpy as np, pandas as pd, duckdb

cfg = json.load(open(sys.argv[1]))
OUT = cfg["out"]; os.makedirs(OUT, exist_ok=True)
U = cfg["universe"]; T = cfg.get("threads", 8)
COMPS = cfg["comps"]; LAB = cfg["labels"]
TBDR = cfg["tbdr"]; ABC = cfg["abc"]
t0 = time.time()
def log(*a): print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)
con = duckdb.connect()
con.execute(f"SET threads={T}; SET preserve_insertion_order=false")
cols = "{'q':'VARCHAR','t':'VARCHAR','evalue':'DOUBLE','bits':'DOUBLE','fident':'DOUBLE','alnlen':'INTEGER','qstart':'INTEGER','qend':'INTEGER','qlen':'INTEGER','tstart':'INTEGER','tend':'INTEGER','tlen':'INTEGER'}"
rd = lambda p: f"read_csv('{p}', delim='\t', header=false, columns={cols})"
con.execute(f"CREATE TABLE selfb AS SELECT q AS comp, max(bits) AS self_bits FROM {rd(cfg['self'])} WHERE q = t GROUP BY 1")
con.execute(f"""CREATE TABLE fwd AS SELECT q AS comp, t::BIGINT AS gid, evalue, bits, fident, qstart, qend, qlen, tstart, tend, tlen,
       (qend - qstart + 1) / qlen::DOUBLE AS qcov, (tend - tstart + 1) / tlen::DOUBLE AS tcov
    FROM {rd(cfg['fwd'])} WHERE evalue <= 1e-3""")
con.execute("CREATE TABLE fwd1 AS SELECT * FROM fwd QUALIFY row_number() OVER (PARTITION BY gid, comp ORDER BY bits DESC) = 1")
con.execute(f"""CREATE TABLE rev AS SELECT q::BIGINT AS gid, t AS am1, bits AS rbits, evalue AS revalue, fident AS rfident
    FROM {rd(cfg['rev'])} QUALIFY row_number() OVER (PARTITION BY q ORDER BY bits DESC, evalue, t) = 1""")
con.execute(f"CREATE VIEW genes AS SELECT * FROM read_parquet('{U}/genes.parquet')")
con.execute(f"CREATE VIEW flags AS SELECT * FROM read_parquet('{U}/gene_flags.parquet')")
con.execute(f"CREATE TABLE gm AS SELECT * FROM read_parquet('{U}/genomes.parquet')")
con.execute(f"""CREATE TABLE cx AS SELECT c.contig_code, c.n_genes AS contig_genes, c.first_gid, x.max_end_nt, x.first_partial, x.last_partial
               FROM read_parquet('{U}/contigs.parquet') c JOIN read_parquet('{U}/contig_extra.parquet') x USING (contig_code)""")
comp_sql = ",".join(repr(c) for c in COMPS)
# gene assignments
con.execute(f"""CREATE TABLE asg AS
  WITH anyb AS (SELECT * FROM fwd1 QUALIFY row_number() OVER (PARTITION BY gid ORDER BY bits DESC, comp) = 1)
  SELECT a.gid, a.comp AS comp_any, a.evalue AS e_any, a.bits AS bits_any,
         CASE WHEN r.am1 IS NOT NULL AND f.gid IS NOT NULL AND f.evalue <= 1e-5 THEN r.am1 END AS comp_rb,
         r.am1 AS rev_best, r.rbits AS rev_bits
  FROM anyb a LEFT JOIN rev r USING (gid)
  LEFT JOIN fwd1 f ON f.gid = a.gid AND f.comp = r.am1""")
log("assigned genes", con.execute("SELECT count(*), count(comp_rb) FROM asg").fetchone())
# per-gene detail for the assigned component (RB if present else ANY)
con.execute("""CREATE TABLE gdet AS SELECT s.gid, coalesce(s.comp_rb, s.comp_any) AS comp, s.comp_rb IS NOT NULL AS is_rb,
       f.fident, f.qcov, f.tcov, f.evalue, f.bits, f.bits / sb.self_bits AS bsr, s.rev_best, s.rev_bits, s.comp_any
    FROM asg s JOIN fwd1 f ON f.gid = s.gid AND f.comp = coalesce(s.comp_rb, s.comp_any) LEFT JOIN selfb sb ON sb.comp = f.comp""")
# position arrays for every gene of the slice
G = con.execute("SELECT gid, genome_code, contig_code, position_index FROM genes ORDER BY gid").fetchnumpy()
gid_all = G["gid"]; assert np.array_equal(gid_all, np.arange(len(gid_all)))
g_genome = G["genome_code"].astype(np.int64); g_contig = G["contig_code"].astype(np.int64); g_pos = G["position_index"].astype(np.int64)
NG = len(gid_all)
gstart = np.r_[0, np.flatnonzero(np.diff(g_genome)) + 1]; gcode = g_genome[gstart]
gfirst = np.zeros(gcode.max() + 1, np.int64); gn = np.zeros(gcode.max() + 1, np.int64)
gfirst[gcode] = gstart; gn[gcode] = np.diff(np.r_[gstart, NG])
log("genes", NG, "genomes", len(gcode))
comp_idx = {c: i for i, c in enumerate(COMPS)}
POPC = np.array([bin(i).count("1") for i in range(1 << len(COMPS))], np.int64)
A = con.execute(f"SELECT gid, comp_rb, comp_any FROM asg").df()

DEFS = {"main": dict(tier="rb", gap=3, mincomp=3, need=None),
        "strict": dict(tier="rb", gap=1, mincomp=5, need="tbdr_abc"),
        "loose": dict(tier="any", gap=5, mincomp=2, need=None)}


def tier_genes(tier):
    col = "comp_rb" if tier == "rb" else "comp_any"
    d = A[A[col].isin(COMPS)]
    return d.gid.values.astype(np.int64), np.array([comp_idx[c] for c in d[col].values], np.int64)


def call(gids, cidx, gap, mincomp, need):
    """runs of component genes on one contig with <= gap intervening genes; returns list of (gid list, comp idx list)."""
    if len(gids) == 0:
        return []
    o = np.lexsort((g_pos[gids], g_contig[gids]))
    gg, cc = gids[o], cidx[o]
    ctg, pos = g_contig[gg], g_pos[gg]
    brk = np.r_[True, (ctg[1:] != ctg[:-1]) | (pos[1:] - pos[:-1] > gap + 1)]
    run = np.cumsum(brk) - 1
    nrun = run[-1] + 1
    # distinct components per run (bitmask), vectorised
    mask = np.zeros(nrun, np.int64)
    np.bitwise_or.at(mask, run, (1 << cc).astype(np.int64))
    ndist = POPC[mask]
    ok = ndist >= mincomp
    if need == "tbdr_abc":
        abc = 0
        for a in ABC:
            abc |= 1 << comp_idx[a]
        ok &= ((mask >> comp_idx[TBDR]) & 1).astype(bool) & ((mask & abc) != 0)
    starts = np.flatnonzero(brk); ends = np.r_[starts[1:], len(gg)]
    return [(gg[s:e], cc[s:e]) for s, e in zip(starts[ok], ends[ok])]


loci = {}
for name, d in DEFS.items():
    gids, cidx = tier_genes(d["tier"])
    loci[name] = call(gids, cidx, d["gap"], d["mincomp"], d["need"])
    log(name, "component genes", len(gids), "loci", len(loci[name]), "genomes", len({int(g_genome[l[0][0]]) for l in loci[name]}))

# null: independent circular shift per component within each genome
rng = np.random.default_rng(cfg.get("seed", 17))
NR = cfg.get("null_reps", 200)
null = {n: [] for n in DEFS}; null_gen = {n: [] for n in DEFS}
for name, d in DEFS.items():
    gids, cidx = tier_genes(d["tier"])
    gm_ = g_genome[gids]; rel = gids - gfirst[gm_]; n_ = gn[gm_]
    for r in range(NR):
        off = rng.integers(0, 1 << 40, size=(len(COMPS), gcode.max() + 1))
        new = gfirst[gm_] + (rel + off[cidx, gm_]) % n_
        Lr = call(new, cidx, d["gap"], d["mincomp"], d["need"])
        null[name].append(len(Lr)); null_gen[name].append(len({int(g_genome[l[0][0]]) for l in Lr}))
    log(name, "null loci mean", np.mean(null[name]), "max", np.max(null[name]))

# locus table
rows = []; members = []
lid = 0
for name in DEFS:
    for gg, cc in loci[name]:
        rows.append(dict(definition=name, locus_id=f"{name}_{lid:05d}", first_gid=int(gg.min()), last_gid=int(gg.max()),
                         n_component_genes=len(gg), comps=sorted({COMPS[i] for i in cc.tolist()}, key=COMPS.index)))
        for g in gg.tolist():
            members.append((f"{name}_{lid:05d}", int(g)))
        lid += 1
L = pd.DataFrame(rows)
M = pd.DataFrame(members, columns=["locus_id", "gid"])
con.register("L0", L); con.register("M0", M)
con.execute("""CREATE TABLE L AS SELECT L0.*, g1.genome, g1.genome_code, g1.contig, g1.contig_code, g1.position_index AS start_pos,
       g2.position_index AS end_pos, g2.position_index - g1.position_index + 1 AS span_genes, g1.start AS start_nt, g2."end" AS end_nt,
       cx.contig_genes, g1.position_index AS genes_to_contig_start, cx.contig_genes - 1 - g2.position_index AS genes_to_contig_end,
       cx.first_partial, cx.last_partial, cx.max_end_nt
    FROM L0 JOIN genes g1 ON g1.gid = L0.first_gid JOIN genes g2 ON g2.gid = L0.last_gid JOIN cx ON cx.contig_code = g1.contig_code""")
log("locus rows", con.execute("SELECT count(*) FROM L").fetchone())

# GlobDB gene index offset
off = cfg.get("gene_offset")
val = {}
if cfg.get("pfam_check"):
    pu = cfg["pfam_check_universe"]
    con.execute(f"""CREATE TABLE pc AS SELECT split_part(g.genome, '.', 1) AS genome_id, f.file_index, split_part(c.acc, '.', 1) AS acc
        FROM read_csv('{cfg['pfam_check']}', delim='\t', header=false, columns={{'gid':'BIGINT','model':'VARCHAR','acc':'VARCHAR',
             'e':'DOUBLE','score':'DOUBLE','dscore':'DOUBLE','af':'INTEGER','at':'INTEGER'}}) c
        JOIN read_parquet('{pu}/genes.parquet') g ON g.gid = c.gid JOIN read_parquet('{pu}/gene_flags.parquet') f ON f.gid = c.gid
        GROUP BY ALL""")
    con.execute(f"""CREATE TABLE gr AS SELECT DISTINCT genome_id, gene_idx, split_part(accession, '.', 1) AS acc FROM read_parquet('{cfg['globdb_rows']}')
        WHERE source = 'Pfam' AND split_part(accession, '.', 1) IN (SELECT DISTINCT acc FROM pc)
          AND genome_id IN (SELECT DISTINCT genome_id FROM pc)""")
    for o in (-1, 0, 1, 2):
        n_pc, n_hit = con.execute(f"""SELECT count(*), count(gr.gene_idx) FROM pc LEFT JOIN gr ON gr.genome_id = pc.genome_id
            AND gr.acc = pc.acc AND gr.gene_idx = pc.file_index + {o}""").fetchone()
        n_gr, n_hit2 = con.execute(f"""SELECT count(*), count(pc.file_index) FROM gr LEFT JOIN pc ON gr.genome_id = pc.genome_id
            AND gr.acc = pc.acc AND gr.gene_idx = pc.file_index + {o}""").fetchone()
        val[o] = dict(own_hits=n_pc, own_hits_found_in_globdb=n_hit, globdb_rows=n_gr, globdb_rows_found_in_own=n_hit2)
    best = max(val, key=lambda o: val[o]["own_hits_found_in_globdb"])
    if off is None:
        off = best
    # per-genome agreement at the chosen offset
    pg = con.execute(f"""SELECT pc.genome_id, count(*) AS n, count(gr.gene_idx) AS found FROM pc LEFT JOIN gr ON gr.genome_id = pc.genome_id
        AND gr.acc = pc.acc AND gr.gene_idx = pc.file_index + {off} GROUP BY 1""").df()
    pg["frac"] = pg.found / pg.n
    val["per_genome_frac_q01_q05_q50"] = pg.frac.quantile([0.01, 0.05, 0.5]).tolist()
    val["genomes_below_0.9"] = int((pg.frac < 0.9).sum()); val["genomes_checked"] = int(len(pg))
    pg.to_csv(f"{OUT}/globdb_index_check_per_genome.tsv", sep="\t", index=False)
val["offset_used"] = off
# gene counts GTDB vs GlobDB
gc = con.execute(f"""SELECT count(*) AS genomes, sum((c.max_gene_idx + 1 - {off} = g.n_genes)::INT) AS same_count,
       sum((c.genome_id IS NULL)::INT) AS absent
    FROM gm g LEFT JOIN read_parquet('{cfg['globdb_counts']}') c ON c.genome_id = split_part(g.genome, '.', 1)""").fetchone()
val["genomes"], val["max_gene_idx_matches_gene_count"], val["genomes_absent_from_globdb_counts"] = gc
json.dump(val, open(f"{OUT}/globdb_index_check.json", "w"), indent=1, default=str)
log("GlobDB index check", val)

# MDH KOs anywhere in each genome (gid level)
MDH = cfg["mdh_kos"]
if cfg.get("mdh_hits") and os.path.exists(cfg["mdh_hits"]):
    # direct KOfam search of every slice protein (10_annot.py mdh), positions exact
    con.execute(f"""CREATE TABLE mdh AS SELECT g.gid, g.genome, g.contig_code, g.position_index, h.ko
        FROM read_parquet('{cfg['mdh_hits']}') h JOIN genes g USING (gid)""")
    MDH_SRC = "direct KOfam hmmsearch (ko_list thresholds)"
else:
    con.execute(f"""CREATE TABLE mdh AS SELECT g.gid, g.genome, g.contig_code, g.position_index, r.accession AS ko
        FROM (SELECT * FROM read_parquet('{cfg['mdh_rows']}') WHERE source = 'KOfam' AND accession IN ({','.join(repr(k) for k in MDH)})) r
        JOIN (SELECT g.gid, g.genome, g.contig_code, g.position_index, split_part(g.genome, '.', 1) AS genome_id, f.file_index
              FROM genes g JOIN flags f USING (gid)) g ON g.genome_id = r.genome_id AND g.file_index = r.gene_idx - {off}""")
    MDH_SRC = "GlobDB rows joined on gene index (invalid when the index check fails)"
# genome-level GlobDB presence of the MDH KOs (no gene positions needed)
con.execute(f"""CREATE TABLE gmdh AS SELECT genome_id, accession AS ko, count(DISTINCT gene_idx) AS n
    FROM read_parquet('{cfg['mdh_rows']}') WHERE source = 'KOfam' AND accession IN ({','.join(repr(k) for k in MDH)}) GROUP BY 1, 2""")
# locus genes with 5 flanking genes each side
FL = cfg.get("flank", 5)
con.execute(f"""CREATE TABLE lg AS SELECT L.locus_id, L.definition, g.gid, g.position_index - L.start_pos AS rel_pos,
       CASE WHEN g.position_index < L.start_pos OR g.position_index > L.end_pos THEN 'flank'
            WHEN M0.gid IS NOT NULL THEN 'component' ELSE 'internal' END AS role,
       g.genome, g.contig, g.position_index, g.strand, g.start, g."end", g.length, g.protein_id, f.partial, f.file_index
    FROM L JOIN genes g ON g.contig_code = L.contig_code AND g.position_index BETWEEN L.start_pos - {FL} AND L.end_pos + {FL}
    JOIN flags f ON f.gid = g.gid LEFT JOIN M0 ON M0.locus_id = L.locus_id AND M0.gid = g.gid""")
if cfg.get("own_annot") and os.path.exists(cfg["own_annot"]):
    con.execute(f"CREATE TABLE ann AS SELECT DISTINCT ON (gid) gid, pfam, kofam FROM read_parquet('{cfg['own_annot']}')")
    ANN_SRC = "direct Pfam 38.1 GA + KOfam (ko_list thresholds) on locus genes"
else:
    con.execute(f"""CREATE TABLE ann AS SELECT f.gid, a.pfam, a.kofam FROM (
        SELECT r.genome_id, r.gene_idx - {off} AS file_index,
           string_agg(DISTINCT CASE WHEN source = 'Pfam' THEN split_part(accession, '.', 1) || ':' || function END, '; ') AS pfam,
           string_agg(DISTINCT CASE WHEN source = 'KOfam' THEN accession || ':' || function END, '; ') AS kofam
        FROM read_parquet('{cfg['globdb_rows']}') r
        SEMI JOIN (SELECT DISTINCT split_part(genome, '.', 1) AS genome_id, file_index + {off} AS gene_idx FROM lg) k
          ON k.genome_id = r.genome_id AND k.gene_idx = r.gene_idx
        WHERE source IN ('Pfam', 'KOfam') GROUP BY 1, 2) a
        JOIN (SELECT DISTINCT lg.gid, split_part(lg.genome, '.', 1) AS genome_id, lg.file_index FROM lg) f USING (genome_id, file_index)""")
    ANN_SRC = "GlobDB rows joined on gene index (invalid when the index check fails)"
con.execute("""CREATE TABLE lg2 AS SELECT lg.*, d.comp AS best_component, d.is_rb AS reverse_best, d.fident, d.qcov, d.tcov, d.evalue, d.bits,
       d.bsr, d.rev_best AS am1_best_hit, d.rev_bits AS am1_best_bits, a.pfam, a.kofam
    FROM lg LEFT JOIN gdet d USING (gid) LEFT JOIN ann a USING (gid)
    ORDER BY locus_id, rel_pos""")
lg = con.execute("SELECT * FROM lg2").df()
lg["label"] = lg.best_component.map(LAB)
# locus attributes: architecture in AM1 orientation
Ld = con.execute("SELECT * FROM L JOIN gm USING (genome, genome_code) ORDER BY locus_id").df()
arch, strands, present, missing, orient, nflip, mdh_near, mdh_genome, lanm = [], [], [], [], [], [], [], [], []
mdh_df = con.execute("SELECT * FROM mdh").df()
lg_by = {k: v for k, v in lg[lg.role != "flank"].groupby("locus_id")}
mdh_by_contig = {k: v for k, v in mdh_df.groupby("contig_code")}
mdh_by_genome = {k: v for k, v in mdh_df.groupby("genome")}
empty = mdh_df.iloc[0:0]
for r in Ld.itertuples():
    x = lg_by[r.locus_id].sort_values("rel_pos")
    comp_rows = x[x.role == "component"]
    # orientation: AM1 genes are all on + in the order META1_1778 -> 1787; flip if the majority of component genes are on -
    flip = (comp_rows.strand.values == -1).sum() > (comp_rows.strand.values == 1).sum()
    xs = x.iloc[::-1] if flip else x
    toks = []
    for y in xs.itertuples():
        s = (-y.strand if flip else y.strand)
        lab = LAB.get(y.best_component, "?") if y.role == "component" else "."
        toks.append(("" if s == 1 else "-") + lab)
    arch.append(" ".join(toks)); orient.append(-1 if flip else 1)
    nflip.append(int(sum(1 for t in toks if t.startswith("-") and t != "-.")))
    pres = [c for c in COMPS if c in set(comp_rows.best_component)]
    present.append(",".join(LAB[c] for c in pres)); missing.append(",".join(LAB[c] for c in COMPS if c not in pres))
    lanm.append(cfg["lanm"] in set(comp_rows.best_component))
    mc = mdh_by_contig.get(r.contig_code, empty)
    mm = mc[(mc.position_index >= r.start_pos - 25) & (mc.position_index <= r.end_pos + 25)]
    mdh_near.append(";".join(f"{k}@{p - r.start_pos}" for k, p in zip(mm.ko, mm.position_index)))
    mg = mdh_by_genome.get(r.genome, empty)
    mdh_genome.append(",".join(f"{k}x{n}" for k, n in mg.ko.value_counts().sort_index().items()))
Ld["architecture"] = arch; Ld["orientation"] = orient; Ld["n_components_opposite_strand"] = nflip
Ld["components_present"] = present; Ld["components_missing"] = missing; Ld["lanm_like_in_locus"] = lanm
Ld["mdh_ko_within_25_genes"] = mdh_near; Ld["mdh_ko_in_genome"] = mdh_genome
Ld["comps"] = Ld.comps.map(lambda v: ",".join(LAB[c] for c in v))
Ld["at_contig_edge_le2"] = (Ld.genes_to_contig_start <= 2) | (Ld.genes_to_contig_end <= 2)
Ld["copies_in_genome"] = Ld.groupby(["definition", "genome"]).locus_id.transform("count")
keep = ["locus_id", "definition", "genome", "gtdb_id", "phylum", "class", "ord", "family", "genus", "species", "completeness",
        "contamination", "contig", "contig_genes", "start_pos", "end_pos", "span_genes", "start_nt", "end_nt", "n_component_genes",
        "comps", "components_present", "components_missing", "architecture", "orientation", "n_components_opposite_strand",
        "lanm_like_in_locus", "mdh_ko_within_25_genes", "mdh_ko_in_genome", "genes_to_contig_start", "genes_to_contig_end",
        "first_partial", "last_partial", "at_contig_edge_le2", "copies_in_genome"]
Ld[keep].to_csv(f"{OUT}/loci.tsv", sep="\t", index=False)
lg.to_csv(f"{OUT}/locus_genes.tsv", sep="\t", index=False)
# genome-level table: every slice genome, RB component genes per component, loci per definition, MDH KOs
gsum = con.execute("""SELECT g.genome, g.genome_code, g.gtdb_id, g.phylum, g.class, g.ord, g.family, g.genus, g.species,
       g.completeness, g.contamination, g.n_genes, g.n_contigs FROM gm g ORDER BY genome_code""").df()
for c in COMPS:
    s = A[A.comp_rb == c].gid.values
    cnt = np.bincount(g_genome[s], minlength=gcode.max() + 1)
    gsum[f"rb_{LAB[c]}"] = cnt[gsum.genome_code.values]
for name in DEFS:
    cnt = np.bincount([int(g_genome[l[0][0]]) for l in loci[name]], minlength=gcode.max() + 1)
    gsum[f"loci_{name}"] = cnt[gsum.genome_code.values]
mk = mdh_df.groupby(["genome", "ko"]).size().unstack(fill_value=0)
gk = con.execute("SELECT * FROM gmdh").df().pivot_table(index="genome_id", columns="ko", values="n", fill_value=0)
gid9 = gsum.genome.str.split(".").str[0]
for k in MDH:
    gsum[f"ko_{k}"] = gsum.genome.map(mk[k] if k in mk.columns else {}).fillna(0).astype(int)
    gsum[f"globdb_{k}"] = gid9.map(gk[k] if k in gk.columns else {}).fillna(0).astype(int)
gsum.to_csv(f"{OUT}/genomes.tsv", sep="\t", index=False)
# per-gene RB assignment table (all component genes, for HMM building and inspection)
con.execute(f"COPY (SELECT d.*, g.genome, g.contig_code, g.position_index, g.strand, g.length FROM gdet d JOIN genes g USING (gid) "
            f"WHERE d.comp IN ({comp_sql}) OR d.comp_any IN ({comp_sql})) TO '{OUT}/component_genes.parquet' (FORMAT parquet)")
ctrl = {n: dict(observed_loci=len(loci[n]), observed_genomes=len({int(g_genome[l[0][0]]) for l in loci[n]}),
                null_loci_mean=float(np.mean(null[n])), null_loci_q95=float(np.quantile(null[n], 0.95)), null_loci_max=int(np.max(null[n])),
                null_genomes_mean=float(np.mean(null_gen[n])), null_reps=NR, **DEFS[n]) for n in DEFS}
json.dump(dict(controls=ctrl, config=cfg, annotation_source=ANN_SRC, mdh_source=MDH_SRC, seconds=round(time.time() - t0, 1)), open(f"{OUT}/loci_summary.json", "w"), indent=1, default=str)
log("done", json.dumps(ctrl, default=str))
