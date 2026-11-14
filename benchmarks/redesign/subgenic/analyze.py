#!/usr/bin/env python3
"""Sub-genic sizing on DPANN (sampled design): composites, fragments, token loss, gold impact, giants.

Hit pool HQ (unique-sequence space, E <= 1e-5, local, no coverage filter, self excluded):
  hits/S.parquet  queries = non-giant unique sequences of 356 random genomes + gold neighbourhoods
  hits/B.parquet  queries = all giant unique sequences (>= 2,046 aa)
Every per-gene quantity below uses only the hits of that gene's own sequence (X side, fragments,
token sharing) or of the gene and its neighbours within 3 positions (Y side), so it is exact for:
  sample genes (all genes of the 356 genomes) and gold member genes (neighbours are queries).
Gene fractions are estimated on the sample genes with a genome-level bootstrap (2,000 draws).
Thresholds come from code/predeclared.json (fixed before measurement).
"""
import argparse, json, os, time
import numpy as np, pandas as pd
import duckdb, numba as nb

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--pkg", required=True)
ap.add_argument("--tokens", required=True)
ap.add_argument("--scores", required=True)
ap.add_argument("--stages", default="composite,fragment,tokens,gold,giants")
ap.add_argument("--max-seqs", type=int, default=8000)
ap.add_argument("--db", default="analyze.duckdb", help="DuckDB file name under tmp/ (one per concurrent job)")
a = ap.parse_args()
W = a.work; P = a.pkg; R = f"{W}/results"; os.makedirs(R, exist_ok=True)
T = int(os.environ.get("SLURM_CPUS_ON_NODE", "4"))
nb.set_num_threads(T)
PD = json.load(open(f"{W}/code/predeclared.json"))
CP, FP, GP = PD["composite"], PD["fragment"], PD["giant"]
RMIN, OV, MAXD, YC = CP["x_region_min_aa"], CP["max_region_overlap_aa"], CP["adjacent_max_position_diff"], CP["y_coverage_min"]
os.makedirs(f"{W}/tmp/duck", exist_ok=True)
con = duckdb.connect(f"{W}/tmp/{a.db}")
con.execute(f"SET threads={T}; SET temp_directory='{W}/tmp/duck'; SET preserve_insertion_order=false")
t0 = time.time()
def log(*x): print(f"[{time.time()-t0:8.1f}s]", *x, flush=True)
def jd(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
def dump(name, obj): json.dump(obj, open(f"{R}/{name}.json", "w"), indent=1, default=jd)
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]
STAGES = set(a.stages.split(","))

# ---------------------------------------------------------------- universe
con.execute(f"""CREATE OR REPLACE TABLE genes AS
  SELECT g.gid, g.genome_code, g.contig_code, g.position_index pos, g.strand, g.start, g."end" AS stop,
         m.uid, c.n_genes, c.length_nt
  FROM '{P}/universe/genes.parquet' g JOIN '{W}/gid2uid.parquet' m USING (gid)
  JOIN '{P}/universe/contigs.parquet' c USING (contig_code) ORDER BY gid""")
con.execute(f"CREATE OR REPLACE TABLE uniq AS SELECT * FROM '{W}/uniq.parquet'")
con.execute(f"CREATE OR REPLACE TABLE genomes AS SELECT genome_code, phylum, ord, completeness FROM '{P}/universe/genomes.parquet'")
con.execute(f"CREATE OR REPLACE TABLE tok AS SELECT gid, token FROM '{a.tokens}'")
G = con.execute("SELECT * FROM genes ORDER BY gid").fetchnumpy()
NG = len(G["gid"]); assert np.array_equal(G["gid"], np.arange(NG))
genome_of = G["genome_code"].astype(np.int32); contig_of = G["contig_code"].astype(np.int64)
pos_of = G["pos"].astype(np.int32); strand_of = G["strand"].astype(np.int8); uid_of = G["uid"].astype(np.int64)
edge = (G["start"] <= 3) | (G["stop"] >= G["length_nt"] - 2)
terminal = (G["pos"] == 0) | (G["pos"] == G["n_genes"] - 1)
token_of = con.execute("SELECT token FROM tok ORDER BY gid").fetchnumpy()["token"].astype(np.int64)
_t = token_of - token_of.min(); tok_size = np.bincount(_t)[_t]
U = con.execute("SELECT uid, length, n_genomes, genome_code_first, rep_gid FROM uniq ORDER BY uid").fetchnumpy()
ulen = U["length"]; NU = len(ulen)
tok_u = token_of[U["rep_gid"]]
SG = con.execute(f"SELECT gid, in_sample, in_gold_nbhd FROM '{W}/lists/S_genes.parquet'").fetchnumpy()
in_sample = np.zeros(NG, np.bool_); in_sample[SG["gid"]] = SG["in_sample"]
in_nbhd = np.zeros(NG, np.bool_); in_nbhd[SG["gid"]] = SG["in_gold_nbhd"]
in_S = in_sample | in_nbhd
gold_mem = np.zeros(NG, np.bool_)
gold_mem[con.execute(f"""SELECT DISTINCT gid FROM '{P}/gold/members.parquet' JOIN '{P}/gold/instances.parquet' USING (instance_id)
                         WHERE n_members_resolved >= 2""").fetchnumpy()["gid"]] = True
qu = np.zeros(NU, np.bool_)
qu[np.loadtxt(f"{W}/lists/S.uid", dtype=np.int64)] = True
qu[np.loadtxt(f"{W}/lists/giants.uid", dtype=np.int64)] = True
assert qu[uid_of[in_S]].all(), "a gene of the query set has a sequence that was not searched"
measured_y = in_sample | gold_mem
sample_genomes = np.loadtxt(f"{W}/lists/sample_genomes.txt", dtype=np.int64)
NSG = int(in_sample.sum())
log("universe", NG, "genes;", NU, "unique;", NSG, "sample genes;", int(gold_mem.sum()), "gold member genes;", int(qu.sum()), "query uids")
con.register("qu_df", pd.DataFrame({"uid": np.flatnonzero(qu)}))
con.execute("CREATE OR REPLACE TABLE qset AS SELECT uid FROM qu_df")
con.execute(f"""CREATE OR REPLACE VIEW HQ AS SELECT q::BIGINT q, t::BIGINT t, evalue, bits, alnlen::INT alnlen,
    qstart::INT qstart, qend::INT qend, qlen::INT qlen, tstart::INT tstart, tend::INT tend, tlen::INT tlen,
    (qend::INT-qstart::INT+1)/qlen::DOUBLE qcov, (tend::INT-tstart::INT+1)/tlen::DOUBLE tcov
  FROM (SELECT * FROM read_parquet('{W}/hits/S_*.parquet') UNION ALL SELECT * FROM '{W}/hits/B.parquet') WHERE q<>t""")

# genome-level bootstrap helper on sample genes
sg_idx = np.searchsorted(sample_genomes, genome_of)
sg_ok = in_sample
def per_genome(mask):
    return np.bincount(sg_idx[sg_ok & mask], minlength=len(sample_genomes)).astype(float)
NGEN_GENES = per_genome(np.ones(NG, np.bool_))
BOOT = np.random.default_rng(17).integers(0, len(sample_genomes), size=(2000, len(sample_genomes)))
def boot_ratio(num_counts, den_counts=None):
    den = NGEN_GENES if den_counts is None else den_counts
    est = num_counts.sum() / den.sum()
    b = num_counts[BOOT].sum(1) / den[BOOT].sum(1)
    return {"est": float(est), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
def boot_diff(c1, c2):
    est = (c1.sum() - c2.sum()) / NGEN_GENES.sum()
    b = (c1[BOOT].sum(1) - c2[BOOT].sum(1)) / NGEN_GENES[BOOT].sum(1)
    return {"est": float(est), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}

# ---------------------------------------------------------------- composite scan kernel
@nb.njit(parallel=True, cache=True)
def scan_groups(gs, ygid, qs, qe, cont, pos, strand, maxov, maxd,
                g_adj, g_non, g_col, g_d, g_y1, g_y2, row_adj, row_non):
    for gi in nb.prange(len(gs) - 1):
        a0 = gs[gi]; b0 = gs[gi + 1]
        best = 1 << 30
        for i in range(a0, b0):
            yi = ygid[i]; ci = cont[yi]; pi = pos[yi]
            for j in range(i + 1, b0):
                yj = ygid[j]
                if yi == yj:
                    continue
                ov = min(qe[i], qe[j]) - max(qs[i], qs[j]) + 1
                if ov > maxov:
                    continue
                cj = cont[yj]; pj = pos[yj]
                d = abs(pi - pj)
                if ci == cj and d <= maxd:
                    g_adj[gi] = True; row_adj[i] = True; row_adj[j] = True
                    c = False
                    if strand[yi] == strand[yj]:
                        fi = qs[i] < qs[j]
                        if strand[yi] > 0:
                            c = fi == (pi < pj)
                        else:
                            c = fi == (pi > pj)
                    rank = (0 if c else 100) + d
                    if rank < best:
                        best = rank; g_col[gi] = c; g_d[gi] = d; g_y1[gi] = yi; g_y2[gi] = yj
                else:
                    g_non[gi] = True; row_non[i] = True; row_non[j] = True

def run_scan(arr, cont, pos):
    n = len(arr["gs"]) - 1; m = len(arr["ygid"])
    out = dict(g_adj=np.zeros(n, np.bool_), g_non=np.zeros(n, np.bool_), g_col=np.zeros(n, np.bool_),
               g_d=np.full(n, -1, np.int32), g_y1=np.full(n, -1, np.int64), g_y2=np.full(n, -1, np.int64),
               row_adj=np.zeros(m, np.bool_), row_non=np.zeros(m, np.bool_))
    scan_groups(arr["gs"], arr["ygid"], arr["qs"], arr["qe"], cont, pos, strand_of, OV, MAXD, *out.values())
    return out

def load_groups(table):
    arr = con.execute(f"SELECT x, B, ygid, qs, qe FROM {table} ORDER BY x, B, ygid").fetchnumpy()
    arr = {k: np.asarray(v) for k, v in arr.items()}
    arr["qs"] = arr["qs"].astype(np.int32); arr["qe"] = arr["qe"].astype(np.int32); arr["ygid"] = arr["ygid"].astype(np.int64)
    brk = np.flatnonzero((np.diff(arr["x"]) != 0) | (np.diff(arr["B"]) != 0)) + 1
    arr["gs"] = np.r_[0, brk, len(arr["x"])].astype(np.int64)
    arr["gx"] = arr["x"][arr["gs"][:-1]]; arr["gB"] = arr["B"][arr["gs"][:-1]]
    return arr

def x_gene_flags(gx, gB):
    """X genes: genes of uid gx in genome A != gB, restricted to genes whose uid is a query."""
    df = pd.DataFrame({"x": gx, "B": gB}).drop_duplicates()
    con.register("xg", df)
    r = con.execute("""SELECT g.gid, count(*) ev FROM xg JOIN genes g ON g.uid = xg.x AND g.genome_code <> xg.B
                       GROUP BY g.gid""").fetchnumpy()
    con.unregister("xg")
    f = np.zeros(NG, np.bool_); f[r["gid"]] = True
    ev = np.zeros(NG, np.int64); ev[r["gid"]] = r["ev"]
    return f & qu[uid_of], ev

def shuffled_positions(seed):
    rng = np.random.default_rng(seed)
    starts = np.r_[0, np.flatnonzero(np.diff(genome_of)) + 1, NG]
    perm = np.empty(NG, np.int64)
    for i in range(len(starts) - 1):
        s0, s1 = starts[i], starts[i + 1]
        perm[s0:s1] = s0 + rng.permutation(s1 - s0)
    return contig_of[perm], pos_of[perm]              # gene gid now sits in slot perm[gid]

flags = {}
if "composite" in STAGES:
    # X side: query = X (fused), target genes = Y candidates in any genome B
    con.execute(f"""CREATE OR REPLACE TABLE xrows AS
      SELECT h.q x, g.genome_code B, g.gid ygid, h.qstart qs, h.qend qe
      FROM (SELECT q, t, qstart, qend FROM HQ
            WHERE (qend-qstart+1) >= {RMIN} AND (qend-qstart+1) <= qlen - ({RMIN} - {OV}) AND tcov >= {YC}) h
      JOIN genes g ON g.uid = h.t JOIN uniq u ON u.uid = h.q
      WHERE NOT (u.n_genomes = 1 AND u.genome_code_first = g.genome_code)""")
    con.execute("""CREATE OR REPLACE TABLE xrows2 AS SELECT r.* FROM xrows r
      JOIN (SELECT x, B FROM xrows GROUP BY x, B HAVING count(DISTINCT ygid) >= 2) k USING (x, B)""")
    # Y side: query = Y (split piece, gene in S), target = X anywhere
    con.register("ins_df", pd.DataFrame({"gid": np.flatnonzero(in_S)}))
    con.execute(f"""CREATE OR REPLACE TABLE yrows AS
      SELECT h.t x, g.genome_code B, g.gid ygid, h.tstart qs, h.tend qe
      FROM (SELECT q, t, tstart, tend FROM HQ
            WHERE (tend-tstart+1) >= {RMIN} AND (tend-tstart+1) <= tlen - ({RMIN} - {OV}) AND qcov >= {YC}) h
      JOIN genes g ON g.uid = h.q
      JOIN uniq u ON u.uid = h.t
      WHERE g.gid IN (SELECT gid FROM ins_df) AND NOT (u.n_genomes = 1 AND u.genome_code_first = g.genome_code)""")
    con.execute("""CREATE OR REPLACE TABLE yrows2 AS SELECT r.* FROM yrows r
      JOIN (SELECT x, B FROM yrows GROUP BY x, B HAVING count(DISTINCT ygid) >= 2) k USING (x, B)""")
    nx = con.execute("SELECT (SELECT count(*) FROM xrows), (SELECT count(*) FROM xrows2), (SELECT count(*) FROM yrows), (SELECT count(*) FROM yrows2)").fetchone()
    log("rows X all/pairable, Y all/pairable", nx)
    AX = load_groups("xrows2"); AY = load_groups("yrows2")
    log("groups X", len(AX["gx"]), "Y", len(AY["gx"]))
    res = {"rows": {"x_candidate": nx[0], "x_pairable": nx[1], "y_candidate": nx[2], "y_pairable": nx[3]},
           "groups": {"x": int(len(AX["gx"])), "y": int(len(AY["gx"]))}}

    def evaluate(cont, pos, tag):
        ox = run_scan(AX, cont, pos); oy = run_scan(AY, cont, pos)
        out = {"ox": ox, "oy": oy}
        out["X_adj"], out["X_adj_ev"] = x_gene_flags(AX["gx"][ox["g_adj"]], AX["gB"][ox["g_adj"]])
        out["X_non"], _ = x_gene_flags(AX["gx"][ox["g_non"]], AX["gB"][ox["g_non"]])
        nonx = ox["g_non"] & ~ox["g_adj"]
        out["X_nonx"], _ = x_gene_flags(AX["gx"][nonx], AX["gB"][nonx])
        ya = np.zeros(NG, np.bool_); ya[AY["ygid"][oy["row_adj"]]] = True
        yn = np.zeros(NG, np.bool_); yn[AY["ygid"][oy["row_non"]]] = True
        rg = np.repeat(np.arange(len(AY["gx"])), np.diff(AY["gs"]))
        ynx = np.zeros(NG, np.bool_); ynx[AY["ygid"][oy["row_non"] & ~oy["g_adj"][rg]]] = True
        out["Y_adj"], out["Y_non"], out["Y_nonx"] = ya & measured_y, yn & in_sample, ynx & in_sample
        log(tag, "X_adj sample", int((out["X_adj"] & in_sample).sum()), "Y_adj sample", int((out["Y_adj"] & in_sample).sum()))
        return out

    obs = evaluate(contig_of, pos_of, "observed")
    flags.update({k: obs[k] for k in ("X_adj", "X_non", "X_nonx", "Y_adj", "Y_non", "Y_nonx")})
    def inv_counts(o, kind):
        X, Y = o[f"X_{kind}"], o[f"Y_{kind}"]
        return per_genome(X), per_genome(Y), per_genome(X | Y)
    blocks = {}
    for kind in ("adj", "non", "nonx"):
        cx, cy, ci = inv_counts(obs, kind)
        blocks[kind] = ci
        res[{"adj": "adjacent", "non": "nonadjacent_incl", "nonx": "nonadjacent_only"}[kind]] = {
            "x_genes_sample": int(cx.sum()), "y_genes_sample": int(cy.sum()), "involved_genes_sample": int(ci.sum()),
            "x_frac": boot_ratio(cx), "y_frac": boot_ratio(cy), "involved_frac": boot_ratio(ci)}
    res["sample_genes"] = NSG; res["sample_genomes"] = int(len(sample_genomes))
    res["excess_vs_nonadjacent_incl"] = boot_diff(blocks["adj"], blocks["non"])
    res["excess_vs_nonadjacent_only"] = boot_diff(blocks["adj"], blocks["nonx"])
    # within-genome shuffle null
    sh_inv = []; res["shuffle"] = []
    for seed in (17, 18, 19):
        cs, ps = shuffled_positions(seed)
        o = evaluate(cs, ps, f"shuffle{seed}")
        cx, cy, ci = inv_counts(o, "adj"); sh_inv.append(ci)
        res["shuffle"].append({"seed": seed, "x_genes_sample": int(cx.sum()), "y_genes_sample": int(cy.sum()),
                               "involved_genes_sample": int(ci.sum()), "involved_frac": float(ci.sum() / NSG),
                               "groups_adjacent_x": int(o["ox"]["g_adj"].sum()), "groups_adjacent_y": int(o["oy"]["g_adj"].sum()),
                               "collinear_frac_x": float(o["ox"]["g_col"][o["ox"]["g_adj"]].mean()) if o["ox"]["g_adj"].any() else None})
    res["excess_vs_shuffle"] = boot_diff(blocks["adj"], np.mean(sh_inv, 0))
    ox, oy = obs["ox"], obs["oy"]
    res["groups_adjacent_x"] = int(ox["g_adj"].sum()); res["groups_adjacent_y"] = int(oy["g_adj"].sum())
    res["groups_nonadjacent_x"] = int(ox["g_non"].sum())
    res["adjacent_over_shuffle_groups_x"] = res["groups_adjacent_x"] / max(np.mean([s["groups_adjacent_x"] for s in res["shuffle"]]), 1)
    # properties of the best adjacent pair (X side groups whose X is a sample gene's sequence)
    xs_uid = np.zeros(NU, np.bool_); xs_uid[uid_of[in_sample]] = True
    m = ox["g_adj"] & xs_uid[AX["gx"]]
    d = ox["g_d"][m]
    res["adjacent_best_pair_x_side"] = {
        "groups": int(m.sum()), "collinear_frac": float(ox["g_col"][m].mean()) if m.any() else None,
        "dpos_counts": {int(k): int(v) for k, v in zip(*np.unique(d, return_counts=True))},
        "y1_y2_same_token_frac": float((token_of[ox["g_y1"][m]] == token_of[ox["g_y2"][m]]).mean()) if m.any() else None}
    # events per sample X gene, and the phyla of the genomes holding the split pieces
    xe = obs["X_adj_ev"]
    res["x_side_events_sample"] = int(xe[in_sample].sum())
    res["x_side_events_per_composite_gene_median"] = float(np.median(xe[in_sample & obs["X_adj"]])) if (in_sample & obs["X_adj"]).any() else None
    # truncation diagnostic: queries whose hit count approaches max-seqs
    hc = con.execute("SELECT q, count(*) n FROM HQ GROUP BY q").fetchnumpy()
    nh = np.zeros(NU, np.int64); nh[hc["q"]] = hc["n"]
    trunc_u = (nh >= 0.8 * a.max_seqs) & (ulen < GP["min_length_aa"])
    tr = trunc_u[uid_of]
    res["truncation"] = {"query_uids_hits_ge_0.8_maxseqs": int(trunc_u.sum()), "sample_genes_affected": int((tr & in_sample).sum()),
                         "x_adj_rate_trunc": float(obs["X_adj"][tr & in_sample].mean()) if (tr & in_sample).any() else None,
                         "x_adj_rate_not_trunc": float(obs["X_adj"][~tr & in_sample].mean())}
    # taxonomy (sample genes)
    tx = pd.DataFrame({"x": obs["X_adj"], "inv": obs["X_adj"] | obs["Y_adj"], "invn": obs["X_non"] | obs["Y_non"],
                       "shuf": np.zeros(NG, bool), "genome_code": genome_of})[in_sample]
    con.register("tx", tx)
    for lvl in ("phylum", "ord"):
        con.execute(f"""SELECT {lvl}, count(DISTINCT genome_code) genomes, count(*) genes, sum(x::int) x_adj, sum(inv::int) involved_adj,
              sum(invn::int) involved_non, sum(x::int)/count(*) x_rate, sum(inv::int)/count(*) involved_rate,
              (sum(inv::int)-sum(invn::int))/count(*) excess_rate_vs_non
              FROM tx JOIN genomes USING (genome_code) GROUP BY 1 ORDER BY genes DESC""").df().to_csv(f"{R}/composite_by_{lvl}.tsv", sep="\t", index=False)
    con.unregister("tx")
    # phylum pairs of X-side adjacent events (genome A of sample X gene, genome B of the split pair)
    gxa = AX["gx"][ox["g_adj"]]; gba = AX["gB"][ox["g_adj"]]
    con.register("ev", pd.DataFrame({"x": gxa, "B": gba}))
    con.register("sm", pd.DataFrame({"gid": np.flatnonzero(in_sample)}))
    con.execute("""SELECT ga.phylum phylum_fused, gb.phylum phylum_split, count(*) events, count(DISTINCT g.gid) x_genes
        FROM ev JOIN genes g ON g.uid = ev.x AND g.genome_code <> ev.B
        JOIN genomes ga ON ga.genome_code = g.genome_code JOIN genomes gb ON gb.genome_code = ev.B
        WHERE g.gid IN (SELECT gid FROM sm)
        GROUP BY 1, 2 ORDER BY events DESC""").df().to_csv(f"{R}/composite_event_phylum_pairs.tsv", sep="\t", index=False)
    con.unregister("ev"); con.unregister("sm")
    pd.DataFrame({"x": AX["gx"], "B": AX["gB"], "adj": ox["g_adj"], "non": ox["g_non"], "collinear": ox["g_col"],
                  "dpos": ox["g_d"], "y1": ox["g_y1"], "y2": ox["g_y2"]}).to_parquet(f"{R}/composite_groups_x.parquet")
    pd.DataFrame({"x": AY["gx"], "B": AY["gB"], "adj": oy["g_adj"], "non": oy["g_non"], "collinear": oy["g_col"],
                  "dpos": oy["g_d"], "y1": oy["g_y1"], "y2": oy["g_y2"]}).to_parquet(f"{R}/composite_groups_y.parquet")
    np.savez_compressed(f"{R}/composite_flags.npz", **{k: flags[k] for k in ("X_adj", "X_non", "X_nonx", "Y_adj", "Y_non", "Y_nonx")})
    dump("composite", res)
    log("composite", json.dumps({k: res[k] for k in ("adjacent", "nonadjacent_incl", "excess_vs_nonadjacent_incl", "excess_vs_shuffle")}, default=jd))

# ---------------------------------------------------------------- fragments
if "fragment" in STAGES:
    sc, pc = FP["self_coverage_min"], FP["partner_coverage_max"]
    con.register("ins_df", pd.DataFrame({"gid": np.flatnonzero(in_S)}))
    con.execute(f"""CREATE OR REPLACE TABLE fp AS
      SELECT q f, t p, (qcov >= {sc} AND tcov <= {pc}) fr, qcov, tcov FROM HQ WHERE qcov >= 0.7 OR tcov >= 0.7""")
    con.execute("""CREATE OR REPLACE TABLE fgene AS
      SELECT g.gid, count(*) n_part, sum(fp.fr::int) n_frag
      FROM fp JOIN uniq u ON u.uid = fp.p JOIN genes g ON g.uid = fp.f
      WHERE g.gid IN (SELECT gid FROM ins_df) AND NOT (u.n_genomes = 1 AND u.genome_code_first = g.genome_code)
      GROUP BY g.gid""")
    fg = con.execute("SELECT gid, n_part, n_frag FROM fgene").fetchnumpy()
    n_part = np.zeros(NG, np.int64); n_frag = np.zeros(NG, np.int64)
    n_part[fg["gid"]] = fg["n_part"]; n_frag[fg["gid"]] = fg["n_frag"]
    frag = (n_part >= FP["min_partners"]) & (n_frag > FP["majority_frac_gt"] * n_part)
    evaluable = n_part >= FP["min_partners"]
    flags["frag"] = frag; flags["frag_evaluable"] = evaluable
    res = {"sample_genes": NSG, "fragment_genes_sample": int((frag & in_sample).sum()),
           "fragment_frac_sample": boot_ratio(per_genome(frag)),
           "evaluable_genes_sample": int((evaluable & in_sample).sum()),
           "fragment_frac_of_evaluable_sample": boot_ratio(per_genome(frag), per_genome(evaluable)),
           "genes_with_any_frag_partner_sample": int(((n_frag > 0) & in_sample).sum())}
    cls = np.where(edge, "edge", np.where(terminal, "terminal_not_edge", "interior"))
    rows = []
    for c in ("edge", "terminal_not_edge", "interior"):
        m = (cls == c) & in_sample; me = m & evaluable; k = int((frag & m).sum())
        rows.append({"class": c, "genes": int(m.sum()), "evaluable": int(me.sum()), "fragments": k,
                     "rate_all": k / max(m.sum(), 1), "rate_evaluable": k / max(me.sum(), 1), "rate_evaluable_ci95": wilson(k, int(me.sum()))})
    res["by_position"] = rows
    comp = con.execute("SELECT genome_code, completeness FROM genomes ORDER BY genome_code").fetchnumpy()
    cmap = np.full(genome_of.max() + 1, np.nan); cmap[comp["genome_code"]] = comp["completeness"]; cg = cmap[genome_of]
    crow = []
    for lo, hi in [(0, 50), (50, 70), (70, 90), (90, 101)]:
        mb = (cg >= lo) & (cg < hi) & in_sample
        for c in ("edge", "interior", "all"):
            m = mb & evaluable & ((cls == c) if c != "all" else True); k = int((frag & m).sum())
            crow.append({"completeness": f"[{lo},{hi})", "class": c, "genomes": int(len(np.unique(genome_of[mb]))),
                         "evaluable": int(m.sum()), "fragments": k, "rate": k / max(m.sum(), 1), "ci95": wilson(k, int(m.sum()))})
    res["by_completeness"] = crow
    L = ulen[uid_of]
    res["length_median_fragment"] = float(np.median(L[frag & in_sample])) if (frag & in_sample).any() else None
    res["length_median_evaluable"] = float(np.median(L[evaluable & in_sample]))
    # the partners that make a gene a fragment: how much longer, and how often the fragment sits at an edge
    r = con.execute("""SELECT quantile_cont(u2.length/u1.length::DOUBLE, [0.25,0.5,0.75]) FROM fp
        JOIN uniq u1 ON u1.uid = fp.f JOIN uniq u2 ON u2.uid = fp.p WHERE fp.fr""").fetchone()
    res["partner_to_fragment_length_ratio_q25_q50_q75"] = r[0]
    tx = pd.DataFrame({"frag": frag, "ev": evaluable, "genome_code": genome_of})[in_sample]
    con.register("tx", tx)
    con.execute("""SELECT phylum, count(DISTINCT genome_code) genomes, count(*) genes, sum(ev::int) evaluable, sum(frag::int) fragments,
                   sum(frag::int)/nullif(sum(ev::int),0) rate_evaluable FROM tx JOIN genomes USING (genome_code)
                   GROUP BY 1 ORDER BY genes DESC""").df().to_csv(f"{R}/fragment_by_phylum.tsv", sep="\t", index=False)
    con.unregister("tx")
    np.savez_compressed(f"{R}/fragment_flags.npz", frag=frag, evaluable=evaluable, n_part=n_part, n_frag=n_frag)
    dump("fragment", res); log("fragment", json.dumps(res, default=jd)[:2500])

def load_flags():
    if "X_adj" not in flags and os.path.exists(f"{R}/composite_flags.npz"):
        flags.update(dict(np.load(f"{R}/composite_flags.npz")))
    if "frag" not in flags and os.path.exists(f"{R}/fragment_flags.npz"):
        z = np.load(f"{R}/fragment_flags.npz"); flags["frag"] = z["frag"]; flags["frag_evaluable"] = z["evaluable"]

# ---------------------------------------------------------------- token loss
if "tokens" in STAGES:
    load_flags()
    tu = con.execute("SELECT count(*) FROM (SELECT g.uid FROM genes g JOIN tok t USING (gid) GROUP BY g.uid HAVING count(DISTINCT t.token) > 1)").fetchone()[0]
    con.register("tu", pd.DataFrame({"uid": np.arange(NU), "tk": tok_u}))
    sh = con.execute("""SELECT fp.f u, bool_or(a.tk = b.tk) s, count(*) n FROM fp JOIN tu a ON a.uid = fp.f JOIN tu b ON b.uid = fp.p
                        GROUP BY fp.f""").fetchnumpy()
    has_u = np.zeros(NU, np.bool_); sh_u = np.zeros(NU, np.bool_); has_u[sh["u"]] = True; sh_u[sh["u"]] = sh["s"]
    # a gene's own token may differ from its uid representative's token: compare gene token with partner tokens
    shared = sh_u[uid_of]; haspart = has_u[uid_of]; single = tok_size == 1
    def tl(m):
        n = int(m.sum())
        return {"genes": n, "singleton_frac": float(single[m].mean()) if n else None,
                "shares_token_with_an_aligned_partner_frac": float(shared[m].mean()) if n else None,
                "lost_frac": float((single | ~shared)[m].mean()) if n else None}
    S_ = in_sample
    res = {"uids_with_more_than_one_token": int(tu), "singleton_frac_all_genes": float(single.mean()),
           "definition": "lost = singleton token, or no aligned partner (E<=1e-5, one side covered >= 0.7) in the same token",
           "fragments_sample": tl(flags["frag"] & S_), "composite_x_sample": tl(flags["X_adj"] & S_),
           "composite_y_sample": tl(flags["Y_adj"] & S_),
           "baseline_sample_evaluable_not_composite_not_fragment": tl(S_ & flags["frag_evaluable"] & ~flags["frag"] & ~flags["X_adj"] & ~flags["Y_adj"]),
           "all_sample_genes_with_a_partner": tl(S_ & haspart)}
    # X shares a token with the split pieces of its own adjacent events?
    gx = pd.read_parquet(f"{R}/composite_groups_x.parquet"); gx = gx[gx.adj]
    con.register("ga", gx[["x", "B", "y1", "y2"]])
    con.register("sm", pd.DataFrame({"gid": np.flatnonzero(in_sample)}))
    xs = con.execute("""SELECT g.gid, bool_or(tx.token = t1.token OR tx.token = t2.token) s
        FROM ga JOIN genes g ON g.uid = ga.x AND g.genome_code <> ga.B
        JOIN tok tx ON tx.gid = g.gid JOIN tok t1 ON t1.gid = ga.y1 JOIN tok t2 ON t2.gid = ga.y2
        WHERE g.gid IN (SELECT gid FROM sm) GROUP BY g.gid""").fetchnumpy()
    res["composite_x_sample_shares_token_with_its_split_pair_frac"] = float(xs["s"].mean()) if len(xs["s"]) else None
    gy = pd.read_parquet(f"{R}/composite_groups_y.parquet"); gy = gy[gy.adj]
    con.register("gb", gy[["x", "y1", "y2"]])
    ys = con.execute("""SELECT y gid, bool_or(ty.token = tu.tk) s FROM (SELECT x, y1 y FROM gb UNION ALL SELECT x, y2 FROM gb) z
        JOIN tok ty ON ty.gid = z.y JOIN tu ON tu.uid = z.x WHERE z.y IN (SELECT gid FROM sm) GROUP BY y""").fetchnumpy()
    res["composite_y_sample_shares_token_with_fused_x_frac"] = float(ys["s"].mean()) if len(ys["s"]) else None
    con.register("fgd", pd.DataFrame({"gid": np.flatnonzero(flags["frag"] & in_sample)}))
    fs = con.execute("""SELECT g.gid, bool_or(t1.token = tu.tk) s FROM fgd JOIN genes g USING (gid)
        JOIN fp ON fp.f = g.uid AND fp.fr JOIN tok t1 ON t1.gid = g.gid JOIN tu ON tu.uid = fp.p GROUP BY g.gid""").fetchnumpy()
    res["fragment_sample_shares_token_with_a_longer_partner_frac"] = float(fs["s"].mean()) if len(fs["s"]) else None
    for v in ("tu", "ga", "sm", "gb", "fgd"): con.unregister(v)
    dump("tokens", res); log("tokens", json.dumps(res, default=jd))

# ---------------------------------------------------------------- gold
if "gold" in STAGES:
    load_flags()
    gf = pd.DataFrame({"gid": np.arange(NG), "x_adj": flags["X_adj"], "y_adj": flags["Y_adj"], "frag": flags["frag"],
                       "x_non": flags["X_non"], "x_nonx": flags["X_nonx"], "edge": edge, "terminal": terminal})
    con.register("gf", gf)
    inst = con.execute(f"""
      WITH m AS (SELECT instance_id, bool_or(x_adj) x_adj, bool_or(y_adj) y_adj, bool_or(frag) frag,
                        bool_or(x_non) x_non, bool_or(x_nonx) x_nonx, bool_or(edge) any_edge, bool_or(terminal) any_terminal
                 FROM '{P}/gold/members.parquet' JOIN gf USING (gid) GROUP BY instance_id)
      SELECT i.instance_id, i.caller, i.system_type, i.n_members_resolved n, i.start_idx, i.end_idx, c.n_genes contig_genes,
             m.* EXCLUDE (instance_id), s.f1, s.coverage, s.boundary_precision, s.covered50, s.localized, s.control_mean_f1
      FROM '{P}/gold/instances.parquet' i JOIN m ON m.instance_id = i.instance_id
      LEFT JOIN '{a.scores}' s ON s.instance_id = i.instance_id
      JOIN '{P}/universe/contigs.parquet' c ON c.contig_code = i.contig_code
      WHERE i.n_members_resolved >= {PD['gold']['min_members']}""").df()
    con.unregister("gf")
    inst["comp"] = inst.x_adj | inst.y_adj
    inst["affected"] = inst.comp | inst.frag
    inst["near_end"] = (inst.start_idx < 3) | (inst.end_idx > inst.contig_genes - 4)
    inst["size_bin"] = pd.cut(inst.n, [1, 3, 6, 10 ** 6], labels=["2-3", "4-6", "7+"])
    inst["poor"] = inst.f1 < 0.5
    inst.to_parquet(f"{R}/gold_instances_flags.parquet")
    res = {"instances_ge2": int(len(inst)), "with_scores": int(inst.f1.notna().sum()),
           "gate_b_threshold_instances": int(np.ceil(PD["gate"]["b_gold_instance_fraction_min"] * len(inst)))}
    for col in ("affected", "comp", "frag", "x_adj", "y_adj", "x_non", "x_nonx", "any_edge", "near_end"):
        res[f"n_{col}"] = int(inst[col].sum()); res[f"frac_{col}"] = float(inst[col].mean())
    res["by_caller"] = inst.groupby("caller").agg(n=("affected", "size"), affected=("affected", "sum"), comp=("comp", "sum"),
                                                  frag=("frag", "sum"), x_non=("x_non", "sum"), near_end=("near_end", "sum"),
                                                  mean_f1=("f1", "mean")).reset_index().to_dict("records")
    rng = np.random.default_rng(17)
    def strat_test(df, col, strata, nperm=5000):
        df = df[df.f1.notna()]
        sid = df.groupby(strata, observed=True).ngroup().values; S = int(sid.max()) + 1
        lab = df[col].values.astype(bool); f1 = df.f1.values.astype(float); poor = df.poor.values.astype(bool)
        num = den = 0.0
        for s in range(S):
            m = sid == s; n_ = m.sum()
            num += (lab & poor & m).sum() * (~lab & ~poor & m).sum() / n_
            den += (lab & ~poor & m).sum() * (~lab & poor & m).sum() / n_
        def mdiff(l):
            lf = l.astype(float)
            c1 = np.bincount(sid, lf, S); c0 = np.bincount(sid, 1 - lf, S)
            s1 = np.bincount(sid, f1 * lf, S); s0 = np.bincount(sid, f1 * (1 - lf), S)
            ok = (c1 > 0) & (c0 > 0)
            return float(((s1[ok] / c1[ok] - s0[ok] / c0[ok]) * c1[ok]).sum() / c1[ok].sum()) if ok.any() else np.nan
        obs = mdiff(lab)
        order = np.argsort(sid, kind="stable"); bnd = np.r_[0, np.flatnonzero(np.diff(sid[order])) + 1, len(sid)]
        ls = lab[order]; perm = np.empty(nperm)
        for k in range(nperm):
            lp = ls.copy()
            for i in range(len(bnd) - 1):
                lp[bnd[i]:bnd[i + 1]] = rng.permutation(lp[bnd[i]:bnd[i + 1]])
            out = np.empty_like(lab); out[order] = lp; perm[k] = mdiff(out)
        return {"n_scored": int(len(df)), "n_affected_scored": int(lab.sum()),
                "mh_odds_ratio_poor": float(num / den) if den > 0 else None,
                "poor_rate_affected": float(poor[lab].mean()) if lab.any() else None, "poor_rate_unaffected": float(poor[~lab].mean()),
                "mean_f1_affected": float(f1[lab].mean()) if lab.any() else None, "mean_f1_unaffected": float(f1[~lab].mean()),
                "stratified_mean_f1_diff": obs, "perm_p_two_sided": float((np.abs(perm) >= abs(obs)).mean()),
                "perm_null_sd": float(np.nanstd(perm)), "n_strata": S}
    res["tests"] = {}
    for col in ("affected", "comp", "frag", "x_non"):
        res["tests"][f"{col}__caller_x_size"] = strat_test(inst, col, ["caller", "size_bin"])
        res["tests"][f"{col}__caller_x_size_x_nearend"] = strat_test(inst, col, ["caller", "size_bin", "near_end"])
    dump("gold", res); log("gold", json.dumps(res, default=jd)[:3000])

# ---------------------------------------------------------------- giants
if "giants" in STAGES:
    load_flags()
    gmin, aln_min = GP["min_length_aa"], GP["local_homolog_min_aln"]
    giants = np.flatnonzero(ulen >= gmin); gg = np.flatnonzero(ulen[uid_of] >= gmin)
    con.register("gg", pd.DataFrame({"gid": gg}))
    hh = con.execute(f"""
      WITH gq AS (SELECT g.gid, g.uid, g.genome_code A FROM gg JOIN genes g USING (gid)),
      hb AS (SELECT q, t, tlen FROM '{W}/hits/B.parquet' WHERE q<>t AND alnlen >= {aln_min})
      SELECT gq.gid, gq.uid, count(DISTINCT hb.t) n_t,
             bool_or(u.n_genomes > 1 OR u.genome_code_first <> gq.A) other_genome,
             bool_or((u.n_genomes > 1 OR u.genome_code_first <> gq.A) AND hb.tlen >= {gmin}) other_genome_giant,
             bool_or((u.n_genomes > 1 OR u.genome_code_first <> gq.A) AND hb.tlen < {gmin}) other_genome_small
      FROM gq LEFT JOIN hb ON hb.q = gq.uid LEFT JOIN uniq u ON u.uid = hb.t
      GROUP BY gq.gid, gq.uid""").df()
    ident = con.execute("SELECT g.gid, u.n_genomes > 1 ident FROM gg JOIN genes g USING (gid) JOIN uniq u ON u.uid = g.uid").df()
    hh = hh.merge(ident, on="gid")
    for c in ("other_genome", "other_genome_giant", "other_genome_small"):
        hh[c] = hh[c].fillna(False).astype(bool)
    hh["any_other"] = hh.other_genome | hh.ident
    cc = con.execute(f"SELECT q, count(DISTINCT t) n FROM '{W}/hits/C.parquet' WHERE q<>t AND alnlen >= {aln_min} GROUP BY q").fetchnumpy()
    n_gg = np.zeros(NU, np.int64); n_gg[cc["q"]] = cc["n"]
    nB = con.execute(f"SELECT q, count(*) n FROM '{W}/hits/B.parquet' WHERE q<>t GROUP BY q").fetchnumpy()
    res = {"giant_unique": int(len(giants)), "giant_genes": int(len(gg)),
           "genes_with_local_homolog_other_genome_incl_identical": int(hh.any_other.sum()),
           "frac_genes_with_local_homolog_other_genome_incl_identical": float(hh.any_other.mean()),
           "genes_with_identical_copy_other_genome": int(hh.ident.sum()),
           "genes_with_nonidentical_homolog_other_genome": int(hh.other_genome.sum()),
           "frac_genes_nonidentical_homolog_other_genome": float(hh.other_genome.mean()),
           "genes_partner_giant_other_genome": int(hh.other_genome_giant.sum()),
           "genes_partner_small_other_genome": int(hh.other_genome_small.sum()),
           "genes_partner_small_only": int((hh.other_genome_small & ~hh.other_genome_giant).sum()),
           "unique_with_giant_partner_in_C": int((n_gg[giants] > 0).sum()),
           "B_hits_per_giant_q50_q90_max": [float(np.median(nB["n"])), float(np.percentile(nB["n"], 90)), int(nB["n"].max())],
           "B_giants_with_hits_ge_0.8_maxseqs": int((nB["n"] >= 0.8 * 20000).sum())}
    uh = hh.groupby("uid").agg(any_other=("any_other", "max"), other=("other_genome", "max"),
                               og=("other_genome_giant", "max"), osm=("other_genome_small", "max")).reset_index()
    res["unique_frac_with_local_homolog_other_genome_incl_identical"] = float(uh.any_other.mean())
    res["unique_frac_nonidentical_homolog_other_genome"] = float(uh.other.mean())
    res["unique_frac_giant_partner_other_genome"] = float(uh.og.mean())
    res["unique_frac_small_partner_only"] = float((uh.osm & ~uh.og).mean())
    # tiling of each giant by runs of adjacent smaller genes in another genome (split-in-synteny for giants)
    con.execute(f"""CREATE OR REPLACE TABLE gtile AS
      SELECT h.q, g.genome_code B, g.contig_code c, g.pos, g.gid ygid, h.qstart::INT qs, h.qend::INT qe
      FROM '{W}/hits/B.parquet' h JOIN genes g ON g.uid = h.t JOIN uniq u ON u.uid = h.q
      WHERE h.q<>h.t AND (h.tend::INT-h.tstart::INT+1) >= {YC}*h.tlen AND (h.qend::INT-h.qstart::INT+1) >= {RMIN}
        AND h.tlen < {gmin} AND NOT (u.n_genomes = 1 AND u.genome_code_first = g.genome_code)""")
    any1 = con.execute("SELECT q, max(qe-qs+1) cov FROM gtile GROUP BY q").df()
    # rows that have another partner gene within MAXD positions on the same contig (candidate runs of >= 2)
    tdf = con.execute(f"""SELECT * FROM (SELECT *, lag(pos) OVER w lp, lead(pos) OVER w np_ FROM gtile
        WINDOW w AS (PARTITION BY q, B, c ORDER BY pos, ygid))
        WHERE (pos - lp <= {MAXD}) OR (np_ - pos <= {MAXD}) ORDER BY q, B, c, pos""").df()
    log("giant tiling rows", len(tdf))
    best = {int(q): (1, int(cv)) for q, cv in zip(any1.q, any1["cov"])}
    for (q, B, c), s in tdf.groupby(["q", "B", "c"], sort=False):
        p = s.pos.values
        runs = np.split(np.arange(len(s)), np.flatnonzero(np.diff(p) > MAXD) + 1)
        for r in runs:
            iv = sorted(zip(s.qe.values[r], s.qs.values[r], s.ygid.values[r]))
            k, last_end, cov, used = 0, -10 ** 9, 0, set()
            for qe_, qs_, y in iv:              # max non-overlapping regions (overlap <= OV), one region per gene
                if y in used: continue
                if qs_ > last_end - OV:
                    cov += (qe_ - max(qs_, last_end + 1) + 1) if qs_ <= last_end else (qe_ - qs_ + 1)
                    k += 1; last_end = qe_; used.add(y)
            bk = best.get(q, (0, 0))
            if k > bk[0] or (k == bk[0] and cov > bk[1]):
                best[q] = (k, cov)
    tiles = pd.DataFrame([(q, k, cv) for q, (k, cv) in best.items()], columns=["uid", "k", "cov_aa"])
    tiles = pd.DataFrame({"uid": giants}).merge(tiles, on="uid", how="left").fillna({"k": 0, "cov_aa": 0})
    tiles["len"] = ulen[tiles.uid.values]; tiles["cov_frac"] = tiles.cov_aa / tiles["len"]
    res["tiling_k_counts_unique"] = {int(k): int(v) for k, v in tiles.k.value_counts().sort_index().items()}
    res["unique_split_in_synteny_k_ge2"] = int((tiles.k >= 2).sum())
    res["unique_split_in_synteny_k_ge3"] = int((tiles.k >= 3).sum())
    res["tiled_cov_frac_q25_q50_q75_k_ge2"] = tiles.cov_frac[tiles.k >= 2].quantile([0.25, 0.5, 0.75]).tolist() if (tiles.k >= 2).any() else None
    tiles.to_parquet(f"{R}/giant_tiling.parquet"); hh.to_parquet(f"{R}/giant_genes_homologs.parquet")
    if "X_adj" in flags:
        res["giant_genes_composite_x_adjacent"] = int(flags["X_adj"][gg].sum())
    if os.path.exists(f"{W}/rep/aln.tsv"):
        rp = con.execute(f"""SELECT * FROM read_csv('{W}/rep/aln.tsv', delim='\t', header=false,
            columns={{'qn':'VARCHAR','tn':'VARCHAR','evalue':'DOUBLE','bits':'DOUBLE','qstart':'INT','qend':'INT','qlen':'INT',
                      'tstart':'INT','tend':'INT','tlen':'INT','alnlen':'INT','pident':'DOUBLE'}})""").df()
        sp = rp.qn.str.rsplit("_", n=1)
        rp["seg0"] = sp.str[1].astype(int); rp["parent"] = sp.str[0]
        assert (rp.parent == rp.tn).all()
        rp["shift"] = rp.tstart - (rp.seg0 + rp.qstart)
        res["repeat_alignments_total"] = int(len(rp)); res["repeat_self_alignments"] = int((rp["shift"] == 0).sum())
        rp = rp[rp["shift"] != 0]
        lens = dict(line.split() for line in open(f"{W}/rep/parents.len"))
        lens = {k: int(v) for k, v in lens.items()}
        out = {}
        for thr in (1e-3, 1e-5):
            s = rp[rp.evalue <= thr]; cover = {}
            for name, d in s.groupby("parent"):
                L = lens[name]; m = np.zeros(L + 2, np.int32)
                for x0, x1, y0, y1 in zip(d.seg0 + d.qstart, d.seg0 + d.qend, d.tstart, d.tend):
                    m[x0] += 1; m[x1 + 1] -= 1; m[y0] += 1; m[y1 + 1] -= 1
                cover[name] = float((np.cumsum(m)[1:L + 1] > 0).mean())
            for kind, pre in (("giants", "g"), ("length_control_1000_2045aa", "c"), ("window_shuffled_giants", "s")):
                names = [n_ for n_ in lens if n_.startswith(pre)]
                v = np.array([cover.get(n_, 0.0) for n_ in names])
                out[f"{kind}_E{thr:g}"] = {"n": len(v), "frac_any_repeat": float((v > 0).mean()),
                                         "frac_ge10pct": float((v >= 0.1).mean()), "frac_ge25pct": float((v >= 0.25).mean()),
                                         "frac_ge50pct": float((v >= 0.5).mean()), "median_repeat_frac": float(np.median(v)),
                                         "mean_repeat_frac": float(v.mean())}
            if thr == 1e-5:
                pd.DataFrame({"parent": list(cover), "repeat_frac": list(cover.values())}).to_parquet(f"{R}/repeat_frac.parquet")
                sg = s[s.parent.str.startswith("g")]
                out["giant_repeat_abs_shift_q25_q50_q75_E1e-05"] = np.percentile(np.abs(sg["shift"]), [25, 50, 75]).tolist() if len(sg) else None
        res["internal_repeats"] = out
    af = con.execute(f"SELECT avg(has_pfam::int), avg(unannotated::int) FROM '{P}/annotation_flags.parquet' JOIN gg USING (gid)").fetchone()
    res["giant_genes_frac_with_any_pfam"] = af[0]; res["giant_genes_frac_unannotated"] = af[1]
    con.execute("""SELECT phylum, count(*) giant_genes, count(DISTINCT g.genome_code) genomes FROM gg JOIN genes g USING (gid)
                   JOIN genomes USING (genome_code) GROUP BY 1 ORDER BY 2 DESC""").df().to_csv(f"{R}/giants_by_phylum.tsv", sep="\t", index=False)
    con.unregister("gg")
    dump("giants", res); log("giants", json.dumps(res, default=jd)[:3000])
log("done")
