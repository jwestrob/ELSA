"""Larger truth modules on lut, as fixed in modules_large_rule.json (Pfam, order, strand, spacing; two flagged token
families). Reads no method output. Writes large_instances.parquet/.tsv, large_counts.tsv, boundary_instances.parquet,
x3_extension.json. Usage: x14_large.py [NREP]"""
import json
import sys
import time

import numpy as np
import pandas as pd

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PKG = RD + "/slice_lut/testset/package_k8"
FZ = RD + "/long_loci/frozen"
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 50
t0 = time.time()
log = lambda *a: print(f"{time.time() - t0:7.1f}s", *a, flush=True)

g = pd.read_parquet(PKG + "/universe/genes.parquet", columns=["gid", "genome", "genome_code", "contig_code", "strand", "length", "start", "end"])
N = len(g)
CC = g.contig_code.to_numpy(); ST = g.strand.to_numpy(); S = g.start.to_numpy(); E = g["end"].to_numpy()
L = g.length.to_numpy(); GC = g.genome_code.to_numpy()
gm = pd.read_parquet(PKG + "/universe/genomes.parquet", columns=["genome", "genome_code", "tax", "layer"]).sort_values("genome_code")
gm["genus"] = gm.tax.str.extract(r"g__([^;]*)")[0]
grp = pd.read_parquet(FZ + "/groups/emb_hom_average_t0.75.parquet")
GRP = grp.set_index("genome_code").group.reindex(gm.genome_code).to_numpy()
G0 = g.groupby("genome_code").gid.min().reindex(gm.genome_code).to_numpy()
GN = g.groupby("genome_code").size().reindex(gm.genome_code).to_numpy()
P = pd.read_parquet(RD + "/annot_gpu/lut/PFAM.parquet", columns=["gid", "name"])
TOK = pd.read_parquet(FZ + "/tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet").token.to_numpy()
byname = P.groupby("name").gid.unique()
log("loaded")


def has(*names):
    out = [byname[n] for n in names if n in byname.index]
    return np.unique(np.concatenate(out)) if out else np.empty(0, np.int64)


lenin = lambda a, lo, hi: a[(L[a] >= lo) & (L[a] <= hi)]
PQQDH = lenin(has("PQQ", "PQQ_2"), 500, 700)
nopq = lambda a: np.setdiff1d(a, PQQDH)
ef = P[P.name.str.startswith("EF-hand")].gid.unique()
nonef = P[~P.name.str.startswith("EF-hand")].gid.unique()
LANMW = lenin(np.setdiff1d(ef, nonef), 100, 200)
CL = {
    "PQQDH": PQQDH, "CBB3": nopq(has("Cytochrome_CBB3")), "CYTC": nopq(has("Cytochrom_C")), "SBP3": has("SBP_bac_3"),
    "MDH": has("MDH"), "RHOD": has("Rhodanese"), "NMT1": has("NMT1"), "BPD1": has("BPD_transp_1"), "ABC": has("ABC_tran"),
    "RR": has("Response_reg"), "MOXR": has("bpMoxR", "AAA_5", "AAA_3"), "DUF58": has("DUF58"), "VWA": has("VWA", "VWA_2"),
    "PKC2": has("Polyketide_cyc2"), "MOXZ": has("MoxZ"), "PQQA": has("PqqA"), "PQQB": lenin(has("Lactamase_B", "Lactamase_B_2"), 250, 400),
    "PQQC": has("TENA_THI-4"), "PQQD": has("PqqD"), "PQQE": lenin(has("Radical_SAM", "SPASM", "Fer4_12", "Fer4_14"), 300, 450),
    "ORF6": np.flatnonzero(TOK == 298295), "ORF7": np.flatnonzero(TOK == 361272), "RIESKE": has("Rieske"),
    "MOCO": has("MoCoBD_1", "MoCoBD_2"), "MCH": has("MCH"), "FAE": has("Fae"),
    "H4X": has("GHMP_kinases_N", "Mpt_N", "ATP-grasp_3", "CitG", "RimK"),
    "L6513": has("DUF6513"), "L447": has("DUF447_N"), "LAAK": has("AA_kinase"), "LPK": has("Polyketide_cyc2", "4HFCP_synth"),
    "LFX": has("DUF3280", "4HFCP_synth", "AA_kinase", "Flavoprotein", "DUF6513", "DUF447_N", "Polyketide_cyc2"),
    "S58": has("Peptidase_S58"), "PBP6": has("Peripla_BP_6"), "BPD2": has("BPD_transp_2"), "LDC": has("Lysine_decarbox"),
    "M20": has("Peptidase_M20"), "THDPS": has("THDPS_N_2", "Hexapep"),
    "TBX": has("Formyl_trans_N", "Pep_deformylase", "PseudoU_synth_2", "EamA", "AA_kinase"),
    "ACAD": has("Acyl-CoA_dh_1", "Acyl-CoA_dh_M", "Acyl-CoA_dh_N"), "TGL": has("Transgly", "Transpeptidase"),
    "LANMW": LANMW, "D3280": has("DUF3280"), "SBP5": has("SBP_bac_5"), "YCAO": has("YcaO"), "TFUA": has("TfuA"),
}
CL["CBC"] = np.union1d(CL["CBB3"], CL["CYTC"])
log("classes", {k: len(v) for k, v in CL.items()})

IG = np.full(N, 10 ** 9); IG[:-1] = np.where(CC[1:] == CC[:-1], S[1:] - E[:-1] - 1, 10 ** 9)
same = np.zeros(N, bool); same[:-1] = (CC[1:] == CC[:-1]) & (ST[1:] == ST[:-1]) & (IG[:-1] <= 500)
CH = np.concatenate([[0], np.cumsum(~same[:-1])])
CS = pd.Series(np.arange(N)).groupby(CH).min().to_numpy(); CE = pd.Series(np.arange(N)).groupby(CH).max().to_numpy()


def masks(cl):
    m = {}
    for k, v in cl.items():
        x = np.zeros(N, bool); x[v] = True; m[k] = x
    return m


def near(M, cls, x, lo, hi, k=1):
    ys = [y for y in range(lo, hi + 1) if y != x and M[cls][y]]
    return sorted(ys, key=lambda y: abs(y - x))[:k] if k else ys


def chain_mod(M, anchor, window, req, nearest=(), allc=(), exclude=()):
    out = []
    for x in np.flatnonzero(M[anchor]):
        c = CH[x]; a, b = CS[c], CE[c]
        if any(M[e][a:b + 1].any() for e in exclude):
            continue
        lo, hi = max(a, x - window), min(b, x + window)
        if not all(len(near(M, r, x, lo, hi, 0)) >= k for r, k in req.items()):
            continue
        mem = {x}
        for cl in nearest:
            mem |= set(near(M, cl, x, lo, hi, 1))
        for cl in allc:
            mem |= set(near(M, cl, x, lo, hi, 0))
        out.append(np.array(sorted(mem)))
    return out


def merge(inst):
    if not inst:
        return []
    inst = sorted(inst, key=lambda a: a.min())
    out, cur = [], set(inst[0].tolist())
    for a in inst[1:]:
        if set(a.tolist()) & cur or a.min() <= max(cur):
            cur |= set(a.tolist())
        else:
            out.append(np.array(sorted(cur))); cur = set(a.tolist())
    out.append(np.array(sorted(cur)))
    return out


def window_mod(M, anchor, offs, req, allc, minmem):
    out = []
    for x in np.flatnonzero(M[anchor]):
        ys = [x + d * (ST[x] if offs[2] else 1) for d in range(offs[0], offs[1] + 1) if d]
        ys = [y for y in ys if 0 <= y < N and CC[y] == CC[x]]
        cnt = {r: sum(M[r][y] for y in ys) for r in req}
        if not all(cnt[r] >= k for r, k in req.items()):
            continue
        mem = {x} | {y for y in ys for c in allc if M[c][y]}
        if len(mem) >= minmem:
            out.append(np.array(sorted(mem)))
    return out


def detect(M):
    D = {}
    i = np.arange(N - 1)
    pair = (CC[1:] == CC[:-1]) & (ST[1:] == ST[:-1]) & (IG[:-1] <= 500) & ((M["PQQDH"][:-1] & M["CBB3"][1:]) | (M["CBB3"][:-1] & M["PQQDH"][1:]))
    D["X0_xoxF_CBB3"] = [np.array([x, x + 1]) for x in i[pair]]
    D["X1_xoxF_CBB3_SBP3"] = chain_mod(M, "PQQDH", 4, {"CBB3": 1, "SBP3": 1}, nearest=("CBB3", "SBP3"), exclude=("MDH",))
    x2r = []
    for m in D["X1_xoxF_CBB3_SBP3"]:
        x = [y for y in m if M["PQQDH"][y]][0]; c = CH[x]
        r = near(M, "RHOD", x, max(CS[c], x - 5), min(CE[c], x + 5), 1)
        if r:
            x2r.append(np.array(sorted(set(m.tolist()) | set(r))))
    D["X2r_xoxF_CBB3_SBP3_Rhodanese"] = x2r
    D["X2t_xoxF_CBB3_ABCtransporter"] = chain_mod(M, "PQQDH", 8, {"CBB3": 1, "NMT1": 1, "BPD1": 1, "ABC": 1},
                                                  nearest=("CBB3", "SBP3", "RR", "NMT1", "BPD1", "ABC"), exclude=("MDH",))
    D["MX0_mxaF_J_G_I"] = chain_mod(M, "PQQDH", 5, {"MDH": 1, "SBP3": 1, "CBC": 1}, nearest=("MDH", "SBP3", "CBC"))
    D["MX1_mxa_cluster"] = chain_mod(M, "PQQDH", 14, {"MDH": 1, "DUF58": 1, "MOXR": 1, "VWA": 1},
                                     allc=("SBP3", "CBB3", "CYTC", "MDH", "MOXR", "DUF58", "VWA", "PKC2", "MOXZ"))
    D["PQ0_pqqBCDE"] = chain_mod(M, "PQQC", 3, {"PQQB": 1, "PQQE": 1}, allc=("PQQA", "PQQB", "PQQD", "PQQE"))
    pq1 = []
    for m in chain_mod(M, "PQQC", 6, {"PQQB": 1, "PQQE": 1, "ORF7": 1, "ORF6": 1}, allc=("ORF6", "ORF7")):
        x = [y for y in m if M["PQQC"][y]][0]; c = CH[x]
        core = near(M, "PQQA", x, max(CS[c], x - 3), min(CE[c], x + 3), 0) + near(M, "PQQB", x, max(CS[c], x - 3), min(CE[c], x + 3), 0) + \
            near(M, "PQQD", x, max(CS[c], x - 3), min(CE[c], x + 3), 0) + near(M, "PQQE", x, max(CS[c], x - 3), min(CE[c], x + 3), 0)
        pq1.append(np.array(sorted(set(m.tolist()) | set(core))))
    D["PQ1_pqq_orf7_orf6"] = pq1
    D["H4_H4MPT_cluster"] = chain_mod(M, "MCH", 8, {"FAE": 1, "H4X": 2}, allc=("FAE", "H4X"))
    D["LF_lutA_flank_chain"] = chain_mod(M, "L6513", 6, {"L447": 1, "LAAK": 1, "LPK": 1}, allc=("LFX",))
    D["TA_typeA_ABC_S58"] = window_mod(M, "S58", (-8, -1, True), {"PBP6": 1, "ABC": 2, "BPD2": 1}, ("PBP6", "ABC", "BPD2"), 5)
    D["TB_typeB_core"] = window_mod(M, "LDC", (-6, 6, False), {"M20": 1, "THDPS": 1, "TBX": 2}, ("M20", "THDPS", "TBX"), 5)
    return {k: merge(v) for k, v in D.items()}


M0 = masks(CL)
obs = detect(M0)
log("observed", {k: len(v) for k, v in obs.items()})

# lut modules from the frozen truth
ins = pd.read_parquet(PKG + "/gold/lut_instances.parquet")
ins = ins[ins.caller == "lut_homology_main"]
LUTMAP = {"3_transporter_module_without_receptor": "LUT_T", "2_receptor_plus_transporter": "LUT_RT",
          "1_complete_AtoI": "LUT_C", "1_complete_AtoI_M": "LUT_C"}
lut_inner = {}
for _, r in ins.iterrows():
    m = LUTMAP.get(r.system_type)
    if m:
        mg = np.asarray(r.member_gids, np.int64); comp = np.asarray(r.member_components, dtype=object)
        obs.setdefault(m, []).append(np.sort(mg))
        lut_inner[(m, int(mg.min()))] = np.sort(mg[np.isin(comp, list("ABCEFG"))]).tolist()

rows, iid = [], 0
for name, insts in obs.items():
    fam = "lut flank" if name.startswith("LF") else ("lut" if name.startswith("LUT") else
          {"X": "xoxF", "M": "mxa", "P": "pqq", "H": "H4MPT", "T": "Type A/B"}[name[0]])
    for mem in insts:
        gc = GC[mem[0]]
        rows.append(dict(instance_id=iid, module=name, family=fam, genome=gm.genome.iat[gc], genome_code=int(gc),
                         genus=gm.genus.iat[gc], layer=gm.layer.iat[gc], group=int(GRP[gc]), contig_code=int(CC[mem[0]]),
                         first_gid=int(mem.min()), last_gid=int(mem.max()), span_genes=int(mem.max() - mem.min() + 1),
                         n_members=int(len(mem)), members=[int(v) for v in mem], tokens=[int(TOK[v]) for v in mem],
                         one_chain=bool(len(set(CH[mem])) == 1),
                         inner=lut_inner.get((name, int(mem.min())))))
        iid += 1
I = pd.DataFrame(rows)
I.to_parquet("large_instances.parquet", index=False)
I.drop(columns=["members", "tokens", "inner"]).to_csv("large_instances.tsv", sep="\t", index=False)
log("instances", I.module.value_counts().to_dict())

# X3: orf6 / orf7 relative to X1 instances
x3 = {"ORF6": {"in_chain": 0, "opposite_strand": 0, "same_strand_other_chain": 0, "absent": 0},
      "ORF7": {"in_chain": 0, "opposite_strand": 0, "same_strand_other_chain": 0, "absent": 0}}
for m in obs["X1_xoxF_CBB3_SBP3"]:
    x = [y for y in m if M0["PQQDH"][y]][0]
    for k in ("ORF6", "ORF7"):
        ys = [y for y in range(x - 10, x + 11) if 0 <= y < N and CC[y] == CC[x] and M0[k][y]]
        if not ys:
            x3[k]["absent"] += 1
        else:
            y = min(ys, key=lambda y: abs(y - x))
            x3[k]["in_chain" if CH[y] == CH[x] else ("opposite_strand" if ST[y] != ST[x] else "same_strand_other_chain")] += 1
json.dump(x3, open("x3_extension.json", "w"), indent=1)
log("X3", x3)

# boundary negatives
B = []
def addb(t, X, Y):
    X, Y = sorted(set(int(v) for v in X)), sorted(set(int(v) for v in Y))
    gc = GC[X[0]]
    B.append(dict(btype=t, genome=gm.genome.iat[gc], group=int(GRP[gc]), genus=gm.genus.iat[gc], contig_code=int(CC[X[0]]), X=X, Y=Y))
i = np.arange(N - 1)
hh = (CC[1:] == CC[:-1]) & (IG[:-1] <= 500) & (ST[:-1] == -1) & (ST[1:] == 1) & \
     ((M0["LANMW"][:-1] & M0["PQQDH"][1:]) | (M0["PQQDH"][:-1] & M0["LANMW"][1:]))
for x in i[hh]:
    lm, pq = (x, x + 1) if M0["LANMW"][x] else (x + 1, x)
    c = CH[pq]
    Y = [pq] + near(M0, "CBB3", pq, max(CS[c], pq - 3), min(CE[c], pq + 3), 1) + near(M0, "SBP3", pq, max(CS[c], pq - 3), min(CE[c], pq + 3), 1)
    addb("B1_lanM_xoxF_head_to_head", [lm], Y)
for d in (1, 2):
    for x in np.flatnonzero(M0["SBP5"]):
        for y in (x - d, x + d):
            if 0 <= y < N and CC[y] == CC[x] and M0["LANMW"][y] and CH[x] == CH[y]:
                lo, hi = min(x, y), max(x, y)
                t = [z for z in range(lo - 7, hi + 8) if 0 <= z < N and CC[z] == CC[x] and M0["TGL"][z] and ST[z] != ST[x]]
                if t:
                    addb("B5_sbp5_lanM_transglycosylase", [x, y], [min(t, key=lambda z: min(abs(z - lo), abs(z - hi)))])
for x in np.flatnonzero(M0["LANMW"]):
    for z in range(x - 10, x + 11):
        if 0 <= z < N - 1 and CC[z] == CC[x] and M0["YCAO"][z] and ST[z] != ST[x]:
            tf = [w for w in (z - 1, z + 1) if CC[w] == CC[z] and M0["TFUA"][w]]
            if tf:
                addb("B6_ycaO_tfuA_lanM", [x], [z, tf[0]]); break
gap = (CC[1:] == CC[:-1]) & (ST[1:] == ST[:-1]) & (IG[:-1] > 500) & ((M0["PQQDH"][:-1] & M0["ACAD"][1:]) | (M0["ACAD"][:-1] & M0["PQQDH"][1:]))
for x in i[gap]:
    pq, ac = (x, x + 1) if M0["PQQDH"][x] else (x + 1, x)
    addb("B7_xoxF_acylCoAdh_gap", [pq], [ac])
rp = (CC[1:] == CC[:-1]) & (ST[1:] == ST[:-1]) & (IG[:-1] <= 500) & ((M0["PQQDH"][:-1] & M0["RIESKE"][1:]) | (M0["RIESKE"][:-1] & M0["PQQDH"][1:]))
for x in i[rp]:
    pq = x if M0["PQQDH"][x] else x + 1
    mo = [z for z in range(pq - 4, pq + 5) if 0 <= z < N and CC[z] == CC[pq] and M0["MOCO"][z] and ST[z] != ST[pq]]
    if mo:
        addb("B2_rieske_xoxF_moco", [x, x + 1], mo)
for m in obs["X1_xoxF_CBB3_SBP3"]:
    x = [y for y in m if M0["PQQDH"][y]][0]
    ys = [y for y in range(x - 10, x + 11) if 0 <= y < N and CC[y] == CC[x] and (M0["ORF6"][y] or M0["ORF7"][y]) and ST[y] != ST[x]]
    if ys:
        addb("B3_xoxFGJ_orf6_orf7", m, ys)
for _, r in ins.iterrows():
    mg = np.asarray(r.member_gids, np.int64)
    for end, step in ((mg.min(), -1), (mg.max(), 1)):
        ds = [end + step * k for k in (1, 2) if 0 <= end + step * k < N and CC[end + step * k] == CC[end]
              and M0["D3280"][end + step * k] and ST[end + step * k] != ST[end]]
        if ds:
            d3 = ds[0]; c = CH[d3]
            Y = [d3] + near(M0, "LFX", d3, max(CS[c], d3 - 6), min(CE[c], d3 + 6), 0)
            addb("B4_lutA_DUF3280_flank", mg, Y)
            break
BD = pd.DataFrame(B)
BD.to_parquet("boundary_instances.parquet", index=False)
log("boundary", BD.btype.value_counts().to_dict())

# shift null
H = pd.read_parquet(FZ + "/score_in/homologs.parquet", columns=["gid", "component"])
RBC = {c: np.sort(v.gid.to_numpy()) for c, v in H.groupby("component")}


def lut_main(comp_gids):
    lab = np.full(N, "", dtype=object)
    for c, v in comp_gids.items():
        lab[v] = c
    gs = np.sort(np.concatenate(list(comp_gids.values())))
    n_all = n_cpl = 0
    run = [gs[0]]
    def close(run):
        nonlocal n_all, n_cpl
        cs = set(lab[run])
        if len(cs) >= 3:
            n_all += 1
            n_cpl += int(set("ABCDEFGHI") <= cs)
    for v in gs[1:]:
        if CC[v] == CC[run[-1]] and v - run[-1] <= 4:
            run.append(v)
        else:
            close(run); run = [v]
    close(run)
    return n_all, n_cpl


rng = np.random.default_rng(17)
null = {k: [] for k in obs}
lutnull = []
for r in range(NREP):
    cl = {}
    for k, v in CL.items():
        off = rng.integers(0, GN); gg = GC[v]
        cl[k] = G0[gg] + (v - G0[gg] + off[gg]) % GN[gg]
    det = detect(masks(cl))
    for k in det:
        null[k].append(len(det[k]))
    rc = {}
    for c, v in RBC.items():
        off = rng.integers(0, GN); gg = GC[v]
        rc[c] = G0[gg] + (v - G0[gg] + off[gg]) % GN[gg]
    lutnull.append(lut_main(rc))
    if r % 10 == 0:
        log("null rep", r)
obs_lut = lut_main(RBC)
cnt = []
for name in obs:
    x = I[I.module == name]
    nv = null.get(name)
    if name.startswith("LUT"):
        nmean = float(np.mean([a for a, b in lutnull])) if name != "LUT_C" else float(np.mean([b for a, b in lutnull]))
        nmax = int(np.max([a for a, b in lutnull])) if name != "LUT_C" else int(np.max([b for a, b in lutnull]))
        nnote = "lut main rule on shifted reverse-best genes (all classes together; LUT_C = complete)"
    else:
        nmean, nmax, nnote = float(np.mean(nv)), int(np.max(nv)), "element classes shifted"
    q = x.n_members.quantile([0, .5, 1]).tolist()
    sp = x.span_genes.quantile([.5, .9]).tolist()
    cnt.append(dict(module=name, family=x.family.iat[0], instances=len(x), genomes=x.genome.nunique(), genera=x.genus.nunique(),
                    genome_groups=x.group.nunique(), largest_group_share=round(x.group.value_counts(normalize=True).iat[0], 3),
                    members_min_median_max="/".join(str(int(v)) for v in q), span_median_q90="/".join(str(int(v)) for v in sp),
                    one_chain_share=round(x.one_chain.mean(), 3), null_mean=round(nmean, 2), null_max=nmax, null_reps=NREP, null_note=nnote))
C = pd.DataFrame(cnt)
C.to_csv("large_counts.tsv", sep="\t", index=False)
print(C.drop(columns=["null_note"]).to_string())
json.dump(dict(lut_main_rule_on_RB_genes_observed=obs_lut, x3=x3), open("large_misc.json", "w"), indent=1, default=int)
log("done")
