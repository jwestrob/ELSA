"""Operational truth for synton-recovery modules taken from the Spicy LanMs notes, computed on the lut collection
(1,032 GTDB r232 representatives) from Pfam content, gene order, strand and nucleotide spacing only.

No curated caller is involved: every module is an OBSERVED domain-and-arrangement pattern, never a system name.
Inputs (read-only): package_k8 universe tables, lut whole-proteome Pfam (Aksha 0.2.3, Pfam 38.1, --cut_ga),
gold lut tables (only for the context census and the lanmodulin spacing check).
Outputs (working directory): module_instances.parquet / .tsv, module_counts.tsv (observed, genomes, genera, shift
null), lanm_context_census.tsv, element_counts.tsv, modules.json.

Element classes (protein level, Pfam GA rows):
  EFLOOP   EF-hand clan rows (names EF-hand_*) merged when envelopes overlap = one loop
  LANM     lanmodulin-like: length <= 200 aa, >= 2 loops, median loop-start spacing 20-29 aa, no non-EF Pfam row
  EFS      small EF-hand protein: length <= 300 aa, >= 2 loops (the notes' Type A / B 'LanM' used EF-hand_5/6/7/8)
  PQQDH    PQQ or PQQ_2 row, length 500-700 aa (the notes' operational 'xoxF': PQQ domain, 500-700 aa)
  partner classes by Pfam name as listed in CLASSES; a PQQDH protein is never a partner of itself.
Module rules: see MODULES. 'adj' = neighboring genes (no gene between); 'chain' = every gene from the first to the
last member on one strand with every intergenic gap <= max_ig (the notes' co-orientation protocol: same contig, same
strand, < 500 bp); 'div' = head-to-head (left gene on -, right gene on +); 'win' = within k genes on one contig, no
strand rule (used only where the notes never applied the protocol).
Shift null (slice_lut control): every element class shifted by an independent random circular offset within each
genome's gene order (labels move onto the genes at the new positions, with their strand and coordinates); modules
recounted; 100 replicates, seed 17.
"""
import json
import sys
import time

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PKG = RD + "/slice_lut/testset/package_k8"
DATASET = sys.argv[1] if len(sys.argv) > 1 else "lut"          # lut | spicy
NREP = int(sys.argv[2]) if len(sys.argv) > 2 else 100
PFX = "" if DATASET == "lut" else "spicy_"
t0 = time.time()
log = lambda *a: print(f"{time.time() - t0:7.1f}s", *a, flush=True)

if DATASET == "lut":
    genes = pd.read_parquet(PKG + "/universe/genes.parquet",
                            columns=["gid", "genome", "genome_code", "contig_code", "position_index", "strand", "length", "start", "end"])
    gm = pd.read_parquet(PKG + "/universe/genomes.parquet", columns=["genome", "genome_code", "tax", "layer"]).sort_values("genome_code")
    gm["genus"] = gm.tax.str.extract(r"g__([^;]*)")[0]
    gm["family"] = gm.tax.str.extract(r"f__([^;]*)")[0]
    P = pd.read_parquet(RD + "/annot_gpu/lut/PFAM.parquet", columns=["gid", "name", "env_from", "env_to"])
else:
    # earlier Spicy LanMs Sharur database (prodigal genes with coordinates, Pfam rows from its own annotation), READ_ONLY
    import duckdb
    con = duckdb.connect("/groups/banfield/users/jwestrob/bin/Sharur/data/spicy_lams/sharur.duckdb", read_only=True)
    genes = con.execute("""select protein_id, bin_id as genome, contig_id, gene_index, case when strand='+' then 1 else -1 end as strand,
                           sequence_length as length, start, end_coord as "end" from proteins
                           order by bin_id, contig_id, gene_index""").fetchdf()
    genes["gid"] = np.arange(len(genes))
    genes["genome_code"] = pd.factorize(genes.genome)[0]
    genes["contig_code"] = pd.factorize(genes.genome + "|" + genes.contig_id)[0]
    tax = pd.read_csv("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/spicy_lams/all_genomes_taxonomy.tsv", sep="\t")
    tax["key"] = tax.assembly_acc.str.replace(r"^GC[AF]_", "", regex=True)
    gm = genes.drop_duplicates("genome_code")[["genome", "genome_code"]].sort_values("genome_code").reset_index(drop=True)
    gm["key"] = gm.genome.str.extract(r"^GC[AF]_(\d+\.\d+)")[0]
    gm = gm.merge(tax.drop_duplicates("key")[["key", "genus", "family", "source"]], on="key", how="left")
    gm["layer"] = gm.source.fillna("unknown")
    pid2gid = pd.Series(genes.gid.to_numpy(), index=genes.protein_id.to_numpy())
    P = con.execute("select protein_id, name, start_aa as env_from, end_aa as env_to from annotations where source='pfam'").fetchdf()
    P["gid"] = pid2gid.reindex(P.protein_id.to_numpy()).to_numpy()
    P = P[P.gid.notna()].astype({"gid": np.int64})[["gid", "name", "env_from", "env_to"]]
N = len(genes)
assert (genes.gid.to_numpy() == np.arange(N)).all()
CC = genes.contig_code.to_numpy(np.int64)
ST = genes.strand.to_numpy(np.int64)
SNT = genes.start.to_numpy(np.int64)
ENT = genes["end"].to_numpy(np.int64)
LEN = genes.length.to_numpy(np.int64)
GC = genes.genome_code.to_numpy(np.int64)
G0 = genes.groupby("genome_code").gid.min().reindex(gm.genome_code).to_numpy(np.int64)
GN = genes.groupby("genome_code").size().reindex(gm.genome_code).to_numpy(np.int64)
log("loaded", DATASET, N, "genes,", len(P), "Pfam rows,", len(gm), "genomes")

# ---------------------------------------------------------------- EF-hand loops
ef = P[P.name.str.startswith("EF-hand")].sort_values(["gid", "env_from"])
loop_n, loop_gap = {}, {}
for gid, g in ef.groupby("gid"):
    starts, cur_e = [], -1
    for s, e in zip(g.env_from.to_numpy(), g.env_to.to_numpy()):
        if s > cur_e:
            starts.append(s)
        cur_e = max(cur_e, e)
    loop_n[gid] = len(starts)
    d = np.diff(starts)
    # lanmodulin ladder: every loop-start gap within 4.5 aa of 24.5 x k (k = 1-3; Pfam often misses a loop)
    loop_gap[gid] = bool(len(d) > 0 and all(min(abs(x - 24.5 * k) for k in (1, 2, 3)) <= 4.5 for x in d))
nonef = set(P.gid[~P.name.str.startswith("EF-hand")])
efg = np.fromiter(loop_n.keys(), np.int64)
nl = np.fromiter(loop_n.values(), np.int64)
ladder = np.array([loop_gap[g] for g in efg])
oth = np.isin(efg, list(nonef))
LANM = efg[(LEN[efg] >= 100) & (LEN[efg] <= 200) & (nl >= 2) & ladder & ~oth]
EFS = efg[(LEN[efg] <= 300) & (nl >= 2)]
LANMW = efg[(LEN[efg] >= 100) & (LEN[efg] <= 200) & ~oth]           # wide: >= 1 loop, no ladder rule
EF1 = efg[LEN[efg] <= 300]                                             # any small EF-hand protein, >= 1 loop
if DATASET == "lut":
    _m = pd.read_parquet(PKG + "/gold/lut_members.parquet")
    _rbM = set(_m.gid[(_m.component == "M") & (_m.caller == "lut_homology_main")])
else:
    _rbM = set()
log("LANM", len(LANM), "proteins; lut lanM members captured", len(_rbM & set(LANM.tolist())), "of", len(_rbM),
    "; EFS", len(EFS), "captures", len(_rbM & set(EFS.tolist())),
    "; small EF proteins (<=200 aa, >=2 loops) off the ladder", int(((LEN[efg] <= 200) & (nl >= 2) & ~ladder).sum()))

byname = P.groupby("name").gid.unique()
def has(*names):
    out = [byname[n] for n in names if n in byname.index]
    return np.unique(np.concatenate(out)) if out else np.empty(0, np.int64)

PQQDH = has("PQQ", "PQQ_2")
PQQDH = PQQDH[(LEN[PQQDH] >= 500) & (LEN[PQQDH] <= 700)]
notpqq = lambda a: np.setdiff1d(a, PQQDH)
CLASSES = {
    "LANM": LANM, "LANMW": LANMW, "EF1": EF1, "EFS": EFS, "PQQDH": PQQDH,
    "CBB3": notpqq(has("Cytochrome_CBB3")), "SBP3": has("SBP_bac_3"), "RIESKE": has("Rieske"),
    "LACB": notpqq(has("Lactamase_B")), "C14": has("Peptidase_C14"), "COXG": has("COXG"), "GFA": notpqq(has("GFA")),
    "ANK": has("Ank", "Ank_2", "Ank_4", "Ank_5"), "DCTM": has("DctM"), "DCTQ": has("DctQ"), "DCTP": has("DctP"),
    "MMSB": np.intersect1d(has("NAD_binding_2"), has("NAD_binding_11")), "MOCOBD": has("MoCoBD_1", "MoCoBD_2"),
    "CNNM": has("CNNM"), "SBP5": has("SBP_bac_5"), "D2158": has("DUF2158"),
    "SIG": has("Sigma70_r2", "Sigma70_r4_2", "Sigma70_r4", "Sigma70_ECF"),
    "S58": has("Peptidase_S58"), "R3P": has("Ribul_P_3_epim"), "MASE4": has("MASE4"),
    "LDC": has("Lysine_decarbox"), "THDPS": has("THDPS_N_2", "Hexapep"), "M20": has("Peptidase_M20"),
    "METRES": has("Metal_resist"), "CCME": has("CcmE"), "MPHOS": has("Metallophos"), "D6163": has("DUF6163"),
    "TANN": has("Tannase"), "DAO": has("DAO"), "ADHS": has("adh_short"), "D4394": has("DUF4394"),
    "YCAO": has("YcaO"), "TFUA": has("TfuA"), "SOXYZ": np.intersect1d(has("SoxY"), has("SoxZ")),
    "HK3": has("HisKA_3"), "GERE": has("GerE"), "ABC": has("ABC_tran"),
    "TONB": has("Plug", "TonB_dep_Rec_b-barrel"), "RR": has("Response_reg"),
}
CLASSES["MPHOS"] = CLASSES["MPHOS"][(LEN[CLASSES["MPHOS"]] >= 200) & (LEN[CLASSES["MPHOS"]] <= 300)]
log("classes", {k: len(v) for k, v in CLASSES.items()})

# ---------------------------------------------------------------- module rules
# (name, kind, spec, earlier record)
MODULES = [
    ("M01_pqqdh_cbb3", "pair", dict(a="PQQDH", b="CBB3", strand="same", max_ig=500), "xoxF + Cytochrome_CBB3: 450 xoxF / 296 genomes, mean 67 bp; validated co-oriented"),
    ("M02_pqqdh_sbp3", "chain", dict(a="PQQDH", b="SBP3", k=1, max_ig=500), "xoxF + SBP_bac_3: 288 / 275 genomes, mean 282 bp; co-oriented neighbor"),
    ("M03_pqqdh_rieske", "pair", dict(a="PQQDH", b="RIESKE", strand="same", max_ig=500), "xoxF + Rieske: 184 / 175 genomes, 111 bp; validated"),
    ("M04_pqqdh_lactamaseB", "pair", dict(a="PQQDH", b="LACB", strand="same", max_ig=500), "xoxF + Lactamase_B: 86 / 86, 344 bp; validated"),
    ("M05_pqqdh_peptidaseC14", "pair", dict(a="PQQDH", b="C14", strand="same", max_ig=500), "xoxF + Peptidase_C14: 78 / 78, 73 bp; validated"),
    ("M06_pqqdh_ankyrin", "pair", dict(a="PQQDH", b="ANK", strand="same", max_ig=500), "xoxF + Ankyrin: 67 / 59, 114 bp; validated"),
    ("M07_pqqdh_coxg", "pair", dict(a="PQQDH", b="COXG", strand="same", max_ig=500), "xoxF + COXG: 58 / 58, 341 bp; validated"),
    ("M08_pqqdh_gfa", "pair", dict(a="PQQDH", b="GFA", strand="same", max_ig=500), "xoxF + GFA (CBB3-fused xoxF): 54 / 54, 180 bp; validated"),
    ("M09_pqqdh_dctMQP", "trap", dict(max_ig=500), "xoxF + DctM-DctQ-DctP: 43 / 43, 341 bp; 41/43 complete; validated"),
    ("M10_pqqdh_mmsB", "pair", dict(a="PQQDH", b="MMSB", strand="same", max_ig=500), "xoxF + mmsB (K00074): 90 genomes, 99% co-oriented"),
    ("M11_lanm_pqqdh_divergent", "pair", dict(a="LANM", b="PQQDH", strand="div", max_ig=500), "LanM | xoxF: opposite strands 150/151, ~340 bp; 209 genomes 99.1% divergent (REVISED from operon)"),
    ("M12_lanm_cnnm", "pair", dict(a="LANM", b="CNNM", strand="same", max_ig=200), "LanM + CNNM/CorC: 18/18 adjacent, co-oriented, 110-200 bp; STRONGLY SURVIVES"),
    ("M13_sbp5_lanm", "chain", dict(a="SBP5", b="LANM", k=1, max_ig=500), "SBP_bac_5 + LanM: 33/34 co-oriented, 29/33 chain-operonic, 32 genomes; SURVIVES"),
    ("M14_duf2158_lanmW", "pair", dict(a="D2158", b="LANMW", strand="same", max_ig=500), "DUF2158 at -1 of LanM, co-oriented: 86 of 235 DUF2158 within +-5 of LanM; protocol not reported (wide EF anchor: the LanM here carries one EF-hand_5 row)"),
    ("M14t_duf2158_lanmW_sigma", "win3", dict(a="D2158", b="LANMW", c="SIG", kb=1, kc=3), "DUF2158 -> LanM -> ECF Sigma70 triplet"),
    ("M15_typeA_core", "win3", dict(a="EFS", b="S58", c="R3P", kb=2, kc=2), "Type A (EF-hand, Peptidase_S58, R3P, MASE4/PAS sensor, ABC): 325 genomes of 785; protocol not applied"),
    ("M15w_typeA_core_EF1", "win3", dict(a="EF1", b="S58", c="R3P", kb=2, kc=2), "Type A with any EF-hand protein (>= 1 loop) at the LanM position"),
    ("M15s_typeA_scaffold", "win2", dict(a="S58", b="R3P", kb=2), "Type A scaffold without the EF-hand gene: Peptidase_S58 + R3P within 2 genes"),
    ("M16_typeB_core", "win3", dict(a="EFS", b="LDC", c="THDPS", kb=2, kc=3), "Type B (EF-hand, Lysine_decarbox, THDPS/Hexapep, M20): 288 genomes; protocol not applied"),
    ("M17_metalresist_lanm_ccmE", "win3", dict(a="EFS", b="METRES", c="CCME", kb=2, kc=3), "EF-hand_5 -> Metal_resist -> CcmE: 10 Mesorhizobium genomes (ELSA cluster); protocol not applied"),
    ("M17w_metalresist_EF1_ccmE", "win3", dict(a="EF1", b="METRES", c="CCME", kb=2, kc=3), "M17 with any EF-hand protein"),
    ("M18_metallophos_lanm", "pair", dict(a="MPHOS", b="LANM", strand="same", max_ig=200), "Metallophos at -1 of LanM, co-oriented, 53-157 bp: 5 Methylopilaceae"),
    ("M19_duf6163_lanm", "exact", dict(a="EFS", b="D6163", d=5), "DUF6163 exactly 5 genes from LanM, co-oriented: 5 loci"),
    ("M20_tannase_lanm", "win3", dict(a="LANM", b="TANN", c="DAO", kb=4, kc=6), "Tannase + LanM cassette (ABC permease, DAO, adh_short, CRP): 2 unique genomes"),
    ("M21_duf4394_cassette", "win3", dict(a="EFS", b="D4394", c="SIG", kb=6, kc=6), "DUF4394 + ECF-sigma + anti-sigma + EF-hand_5 + GST + ResIII: 5 perfect of 10 bins"),
    ("M22_ycaO_tfuA_lanm", "win3", dict(a="EFS", b="YCAO", c="TFUA", kb=10, kc=10), "YcaO + TfuA near LanM: co-oriented 2/14, operonic 0/14; FAILS (proximity only)"),
    ("M11w_lanmW_pqqdh_divergent", "pair", dict(a="LANMW", b="PQQDH", strand="div", max_ig=500), "sensitivity of M11 with the wide EF anchor"),
    ("M12w_lanmW_cnnm", "pair", dict(a="LANMW", b="CNNM", strand="same", max_ig=200), "sensitivity of M12 with the wide EF anchor"),
    ("M13w_sbp5_lanmW", "chain", dict(a="SBP5", b="LANMW", k=1, max_ig=500), "sensitivity of M13 with the wide EF anchor"),
    ("M23_lanm_narL_tcs", "win3", dict(a="LANM", b="HK3", c="GERE", kb=3, kc=3), "HisKA_3 sensor kinase (12%) + GerE regulator (17%) flank the cassette; protocol not applied"),
]


def masks(cls):
    m = {}
    for k, v in cls.items():
        x = np.zeros(N, bool)
        x[v] = True
        m[k] = x
    return m


IG = np.full(N, 10 ** 9, np.int64)      # IG[i] = gap between gene i and i+1 (bp), huge across contigs
same_c = CC[1:] == CC[:-1]
IG[:-1] = np.where(same_c, SNT[1:] - ENT[:-1] - 1, 10 ** 9)


def chain_break(max_ig):
    """brk[i] = 1 if boundary i|i+1 breaks a same-strand chain (other contig, other strand or gap > max_ig)."""
    b = np.ones(N, np.int64)
    b[:-1] = ~(same_c & (ST[1:] == ST[:-1]) & (IG[:-1] <= max_ig))
    return np.concatenate([[0], np.cumsum(b)])          # csum[i] = breaks in boundaries 0..i-1


def shifted(a, d):
    """b[i] = a[i + d] (False out of range)."""
    b = np.zeros(N, bool)
    if d > 0:
        b[:-d] = a[d:]
    elif d < 0:
        b[-d:] = a[:d]
    else:
        b[:] = a
    return b


def samec(d):
    b = np.zeros(N, bool)
    if d > 0:
        b[:-d] = CC[d:] == CC[:-d]
    elif d < 0:
        b[-d:] = CC[-d:] == CC[:d]
    else:
        b[:] = True
    return b


SAMEC = {d: samec(d) for d in range(-10, 11)}
CSUM = {mi: chain_break(mi) for mi in (200, 500)}


TIMING = {}


def detect(M):
    """returns {module: list of member-gid arrays} (before merging)."""
    out = {}
    for name, kind, sp, _ in MODULES:
        tm = time.time()
        inst = []
        if kind == "pair":
            A, B = M[sp["a"]], M[sp["b"]]
            i = np.arange(N - 1)
            ok = same_c & (IG[:-1] <= sp["max_ig"])
            if sp["strand"] == "same":
                ok &= ST[1:] == ST[:-1]
                hit = ok & ((A[:-1] & B[1:]) | (B[:-1] & A[1:]))
            else:   # divergent, head-to-head
                ok &= (ST[:-1] == -1) & (ST[1:] == 1)
                hit = ok & ((A[:-1] & B[1:]) | (B[:-1] & A[1:]))
            inst = [np.array([x, x + 1]) for x in i[hit]]
        elif kind == "chain":
            A, B = M[sp["a"]], M[sp["b"]]
            cs = CSUM[sp["max_ig"]]
            for d in range(1, sp["k"] + 2):
                i = np.arange(N - d)
                ok = (cs[i + d] - cs[i]) == 0
                hit = ok & ((A[:-d] & B[d:]) | (B[:-d] & A[d:]))
                inst += [np.array([x, x + d]) for x in i[hit]]
        elif kind == "exact":
            A, B = M[sp["a"]], M[sp["b"]]
            d = sp["d"]
            i = np.arange(N - d)
            ok = SAMEC[d][:-d] & (ST[d:] == ST[:-d])
            hit = ok & ((A[:-d] & B[d:]) | (B[:-d] & A[d:]))
            inst = [np.array([x, x + d]) for x in i[hit]]
        elif kind == "trap":
            P_, Mm, Q, Pp = M["PQQDH"], M["DCTM"], M["DCTQ"], M["DCTP"]
            cs = CSUM[sp["max_ig"]]
            i = np.arange(N - 1)
            pq = same_c & (ST[1:] == ST[:-1]) & (IG[:-1] <= sp["max_ig"]) & ((P_[:-1] & Mm[1:]) | (Mm[:-1] & P_[1:]))
            for x in i[pq]:
                m = x + 1 if Mm[x + 1] else x
                lo, hi = max(m - 4, 0), min(m + 4, N - 1)
                qs = [j for j in range(lo, hi + 1) if Q[j] and cs[max(j, m)] - cs[min(j, m)] == 0]
                ps = [j for j in range(lo, hi + 1) if Pp[j] and cs[max(j, m)] - cs[min(j, m)] == 0]
                if qs and ps:
                    inst.append(np.unique([x, x + 1, qs[0], ps[0]]))
        elif kind == "win2":
            A, B = M[sp["a"]], M[sp["b"]]
            for x in np.flatnonzero(A):
                bs = [x + d for d in range(-sp["kb"], sp["kb"] + 1) if d and 0 <= x + d < N and CC[x + d] == CC[x] and B[x + d]]
                if bs:
                    inst.append(np.unique([x, min(bs, key=lambda j: abs(j - x))]))
        elif kind == "win3":
            A, B, C = M[sp["a"]], M[sp["b"]], M[sp["c"]]
            anchors = np.flatnonzero(A)
            for x in anchors:
                bs = [x + d for d in range(-sp["kb"], sp["kb"] + 1) if d and 0 <= x + d < N and CC[x + d] == CC[x] and B[x + d]]
                cs_ = [x + d for d in range(-sp["kc"], sp["kc"] + 1) if d and 0 <= x + d < N and CC[x + d] == CC[x] and C[x + d]]
                if bs and cs_:
                    b = min(bs, key=lambda j: abs(j - x))
                    c = min([j for j in cs_ if j != b] or [None], key=lambda j: abs(j - x) if j is not None else 99)
                    if c is not None:
                        inst.append(np.unique([x, b, c]))
        out[name] = inst
        TIMING[name] = TIMING.get(name, 0.0) + time.time() - tm
    return out


def merge(inst):
    """union of instances sharing a gene (same module)."""
    if not inst:
        return []
    inst = sorted(inst, key=lambda a: a.min())
    out, cur = [], set(inst[0].tolist())
    for a in inst[1:]:
        if set(a.tolist()) & cur or a.min() <= max(cur):
            cur |= set(a.tolist())
        else:
            out.append(np.array(sorted(cur)))
            cur = set(a.tolist())
    out.append(np.array(sorted(cur)))
    return out


M0 = masks(CLASSES)
obs = {k: merge(v) for k, v in detect(M0).items()}
log("observed", {k: len(v) for k, v in obs.items()})
log("timing", {k: round(v, 2) for k, v in TIMING.items()})

rows = []
iid = 0
for name, kind, sp, rec in MODULES:
    for mem in obs[name]:
        g = GC[mem[0]]
        rows.append(dict(instance_id=iid, module=name, genome=gm.genome.iloc[g], genome_code=int(g), genus=gm.genus.iloc[g],
                         family=gm.family.iloc[g], layer=gm.layer.iloc[g], contig_code=int(CC[mem[0]]),
                         first_gid=int(mem.min()), last_gid=int(mem.max()), span_genes=int(mem.max() - mem.min() + 1),
                         n_members=int(len(mem)), members=mem.tolist(), strands="".join("+" if ST[x] > 0 else "-" for x in mem),
                         lengths=",".join(str(int(LEN[x])) for x in mem)))
        iid += 1
I = pd.DataFrame(rows)
assert (I.contig_code == CC[I.last_gid]).all()
I.to_parquet(PFX + "module_instances.parquet", index=False)
I.drop(columns=["members"]).to_csv(PFX + "module_instances.tsv", sep="\t", index=False)

# counts without the null first (written again with the null below)
_pre = []
for name, kind, sp, rec in MODULES:
    x = I[I.module == name]
    _pre.append(dict(module=name, instances=len(x), genomes=x.genome.nunique(), genera=x.genus.nunique(),
                     families=x.family.nunique(), top_genera=";".join(f"{a}:{b}" for a, b in x.genus.value_counts().head(5).items()),
                     earlier_record=rec))
pd.DataFrame(_pre).to_csv(PFX + "module_counts_nonull.tsv", sep="\t", index=False)
log("wrote counts without null")

# ---------------------------------------------------------------- shift null
rng = np.random.default_rng(17)
genome_of = GC
null = {name: [] for name, *_ in MODULES}
for r in range(NREP):
    cl = {}
    for k, v in CLASSES.items():
        off = rng.integers(0, GN)                       # one offset per genome for this class
        g = genome_of[v]
        cl[k] = G0[g] + (v - G0[g] + off[g]) % GN[g]
    det = detect(masks(cl))
    for name in null:
        null[name].append(len(merge(det[name])))
    if r % 10 == 0:
        log("null rep", r, {k: round(v, 1) for k, v in TIMING.items() if v > 2})
cnt = []
for name, kind, sp, rec in MODULES:
    x = I[I.module == name]
    nv = np.asarray(null[name])
    cnt.append(dict(module=name, rule=kind + " " + json.dumps(sp), instances=len(x), genomes=x.genome.nunique(),
                    genera=x.genus.nunique(), families=x.family.nunique(),
                    top_genera=";".join(f"{a}:{b}" for a, b in x.genus.value_counts().head(5).items()),
                    null_mean=float(nv.mean()), null_max=int(nv.max()), null_reps=NREP,
                    obs_over_null=float(len(x) / nv.mean()) if nv.mean() > 0 else None, earlier_record=rec))
C = pd.DataFrame(cnt)
C.to_csv(PFX + "module_counts.tsv", sep="\t", index=False)
pd.DataFrame([dict(element=k, proteins=len(v), genomes=len(np.unique(GC[v]))) for k, v in CLASSES.items()]).to_csv(
    PFX + "element_counts.tsv", sep="\t", index=False)
print(C[["module", "instances", "genomes", "genera", "null_mean", "null_max", "top_genera"]].to_string())

# ---------------------------------------------------------------- LanM context census (earlier 'locus types')
inlut = np.zeros(N, bool)
rbM = set()
if DATASET == "lut":
    mem = pd.read_parquet(PKG + "/gold/lut_members.parquet")
    mem = mem[mem.caller == "lut_homology_main"]
    ins = pd.read_parquet(PKG + "/gold/lut_instances.parquet")
    ins = ins[ins.caller == "lut_homology_main"]
else:
    ins = pd.DataFrame(dict(member_gids=[], member_components=[], instance_id=[]))
    mem = pd.DataFrame(dict(gid=[], component=[]))
for a, b in zip(ins.member_gids.map(min), ins.member_gids.map(max)):
    inlut[a:b + 1] = True
rbM = set(mem.gid[mem.component == "M"])


def near(cls, x, k):
    return any(0 <= x + d < N and CC[x + d] == CC[x] and M0[cls][x + d] for d in range(-k, k + 1) if d)


cen = []
for x in LANM:
    cen.append(dict(gid=int(x), genome=gm.genome.iloc[GC[x]], genus=gm.genus.iloc[GC[x]], layer=gm.layer.iloc[GC[x]],
                    in_lut_locus_span=bool(inlut[x]), lut_lanM_member=x in rbM,
                    pqqdh_5=near("PQQDH", x, 5), abc_5=near("ABC", x, 5), tonb_5=near("TONB", x, 5), rr_5=near("RR", x, 5),
                    cnnm_5=near("CNNM", x, 5), lactB_5=near("LACB", x, 5), soxyz_5=near("SOXYZ", x, 5), soxyz_10=near("SOXYZ", x, 10),
                    sbp5_5=near("SBP5", x, 5), hk3_5=near("HK3", x, 5), gere_5=near("GERE", x, 5)))
CE = pd.DataFrame(cen)
CE["isolated"] = ~(CE.pqqdh_5 | CE.abc_5 | CE.tonb_5)
CE.to_csv(PFX + "lanm_context_census.tsv", sep="\t", index=False)
summ = {c: [int(CE[c].sum()), len(CE)] for c in CE.columns if CE[c].dtype == bool}
summ["lanm_genomes"] = int(CE.genome.nunique())
summ["lut_lanM_members_total"] = len(rbM)
summ["lut_lanM_members_that_are_LANM"] = int(len(rbM & set(LANM.tolist())))
# co-orientation of lut lanM members with their nearest lut component (earlier protocol)
co = []
for _, r in ins.iterrows():
    g = np.asarray(r.member_gids)
    c = np.asarray(r.member_components)
    for x in g[c == "M"]:
        others = g[c != "M"]
        if len(others) == 0:
            continue
        y = others[np.argmin(np.abs(others - x))]
        lo, hi = min(x, y), max(x, y)
        same = bool((ST[lo:hi + 1] == ST[x]).all())
        maxig = int(IG[lo:hi].max())
        co.append(dict(instance_id=int(r.instance_id), lanM=int(x), nearest=int(y), comp=c[g == y][0], dist=int(hi - lo),
                       same_strand_chain=same, max_ig=maxig, operonic=same and maxig < 500))
CO = pd.DataFrame(co, columns=["instance_id", "lanM", "nearest", "comp", "dist", "same_strand_chain", "max_ig", "operonic"])
CO.to_csv(PFX + "lut_lanM_coorientation.tsv", sep="\t", index=False)
summ["lut_lanM_coorientation"] = dict(n=len(CO), same_strand_chain=int(CO.same_strand_chain.sum()),
                                      operonic_lt500=int(CO.operonic.sum()),
                                      nearest_component=CO.comp.value_counts().to_dict(), dist=CO.dist.value_counts().to_dict())
json.dump(dict(classes={k: len(v) for k, v in CLASSES.items()}, census=summ, nrep=NREP), open(PFX + "modules.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
log("done")
