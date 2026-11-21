"""POST HOC scoring of frozen lut outputs on the larger modules (large_instances.parquet), nested-node test, genome-group
strata (with the first-pass two-gene modules) and boundary negatives (boundary_instances.parquet). Frozen definitions:
ribosomal_check recovered / resolved on instance spans, score_dpann_loci F1 / excess on members (200 rigid
translations, seed sha256('17:' + id)), complete-cluster held / matched / nested rule. Chance for recovered, resolved,
boundary outcomes = the same 200 translations; nested chance = the first 50 of them. Usage: x15_score_large.py THREADS"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
FZ = RD / "long_loci/frozen"
PKG = RD / "slice_lut/testset/package_k8"
sys.path.insert(0, str(RD / "ws_dpann/package/scorer"))
sys.path.insert(0, str(RD / "ribosomal/code"))
import score_dpann_loci as S  # noqa: E402
import ribosomal_check as RC  # noqa: E402

THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
NCTL = 200
OUT = {"SEC": FZ / "loci/lut_Z/L4bp_t0.02.parquet", "PRI": FZ / "loci/lut_WA/L4b_t0.2.parquet",
       "DH": FZ / "loci/lut_engine/DH.parquet", "SYN": FZ / "synton/runs/faithful/post/loci.parquet"}
t0 = time.time()
log = lambda *a: print(f"{time.time() - t0:7.1f}s", *a, flush=True)
genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome", "contig", "contig_code", "position_index"])
contigs = pd.read_parquet(PKG / "universe/contigs.parquet").sort_values("contig_code")
FG = np.zeros(int(contigs.contig_code.max()) + 1, np.int64); FG[contigs.contig_code.to_numpy()] = contigs.first_gid.to_numpy()
NG = np.zeros_like(FG); NG[contigs.contig_code.to_numpy()] = contigs.n_genes.to_numpy()
GEN, CON, POS, CCA = genes.genome.to_numpy(), genes.contig.to_numpy(), genes.position_index.to_numpy(), genes.contig_code.to_numpy()
TOK = pd.read_parquet(FZ / "tok/mmseqs__seq__e3c50s75_rescue3__lut.parquet").token.to_numpy()
grp = pd.read_parquet(FZ / "groups/emb_hom_average_t0.75.parquet")
GRPG = dict(zip(grp.genome, grp.group))


def origins(key, cc, width):
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"17:{key}".encode()).digest()[:8], "big"))
    return FG[cc] + rng.integers(0, max(NG[cc] - width, 1), size=NCTL)


def load_loci(label, path):
    df = pd.read_parquet(path)
    if label == "SYN":
        s = FG[df.contig_code.to_numpy(np.int64)] + df.start.to_numpy(np.int64)
        e = FG[df.contig_code.to_numpy(np.int64)] + df.end.to_numpy(np.int64)
    else:
        s, e = df.s.to_numpy(np.int64), df.e.to_numpy(np.int64)
    k = e > s
    se = pd.DataFrame(dict(s=s[k], e=e[k])).drop_duplicates().sort_values(["s", "e"]).reset_index(drop=True)
    return se.s.to_numpy(np.int64), se.e.to_numpy(np.int64)


def boot(v):
    v = np.asarray(v, float)
    idx = np.random.default_rng(17).integers(0, len(v), size=(10000, len(v)))
    m = v[idx].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


I = pd.read_parquet("large_instances.parquet")
I["mem"] = I.members.map(lambda v: np.asarray(v, np.int64))
I["member_positions"] = [(m - FG[c]).tolist() for m, c in zip(I.mem, I.contig_code)]
I["contig"] = CON[I.first_gid.to_numpy()]
I["start_idx"] = POS[I.first_gid.to_numpy()]
I["end_idx"] = POS[I.last_gid.to_numpy()]
I["caller"] = "spicy_large"; I["system_type"] = I.module; I["n_members_resolved"] = I.n_members
I["resolved"] = True; I["order_trusted"] = True
sh = Path("shadow_large"); (sh / "gold").mkdir(parents=True, exist_ok=True)
if not (sh / "universe").exists():
    (sh / "universe").symlink_to(PKG / "universe")
I[["instance_id", "caller", "system_type", "genome", "contig", "contig_code", "start_idx", "end_idx", "span_genes",
   "member_positions", "n_members_resolved", "resolved", "order_trusted"]].to_parquet(sh / "gold/instances.parquet", index=False)
QS = np.vstack([origins(i, c, w) for i, c, w in zip(I.instance_id, I.contig_code, I.span_genes - 1)])
QE = QS + (I.last_gid - I.first_gid).to_numpy()[:, None]

# nested pairs: (inner module or 'lut_inner', outer module)
NP = [("X0_xoxF_CBB3", "X1_xoxF_CBB3_SBP3"), ("X1_xoxF_CBB3_SBP3", "X2r_xoxF_CBB3_SBP3_Rhodanese"),
      ("X1_xoxF_CBB3_SBP3", "X2t_xoxF_CBB3_ABCtransporter"), ("MX0_mxaF_J_G_I", "MX1_mxa_cluster"),
      ("PQ0_pqqBCDE", "PQ1_pqq_orf7_orf6"), ("lut transporter genes", "LUT_RT"), ("lut transporter genes", "LUT_C")]
pairs = []
for inner, outer in NP:
    for _, o in I[I.module == outer].iterrows():
        if inner.startswith("lut"):
            inn = np.asarray(o.inner if o.inner is not None else [], np.int64)
        else:
            cand = I[(I.module == inner) & (I.contig_code == o.contig_code) & (I.first_gid >= o.first_gid) & (I.last_gid <= o.last_gid)]
            if not len(cand):
                continue
            inn = cand.mem.iloc[int(np.argmax(cand.n_members.to_numpy()))]
        if len(inn) == 0 or set(inn.tolist()) == set(o.mem.tolist()):
            continue
        pairs.append(dict(pair=f"{inner} in {outer}", outer_id=int(o.instance_id), contig_code=int(o.contig_code),
                          outer=o.mem, inner=inn))
PR = pd.DataFrame(pairs)
log("nested pairs", PR.pair.value_counts().to_dict())


def nested_eval(s, e, maxlen, outer, inner):
    a0, b0 = outer.min(), outer.max()
    lo = np.searchsorted(s, a0 - maxlen + 1); hi = np.searchsorted(s, b0, side="right")
    ss, ee = s[lo:hi], e[lo:hi]
    k = ee >= a0; ss, ee = ss[k], ee[k]
    if len(ss) == 0:
        return False, False, False
    ins = (outer[None, :] >= ss[:, None]) & (outer[None, :] <= ee[:, None])
    cov = ins.sum(1) / len(outer); span = ee - ss + 1
    o = np.lexsort((span, -cov))[0]
    held = cov[o] >= 0.8
    mi = (inner[None, :] >= ss[:, None]) & (inner[None, :] <= ee[:, None])
    mm = (mi.sum(1) / len(inner) >= 0.8) & (mi.sum(1) / span >= 0.8)
    nst = bool(held and (mm & (ss >= ss[o]) & (ee <= ee[o])).any())
    return bool(held), bool(mm.any()), nst


# family genome groups (strata)
def family_groups(df):
    out = np.zeros(len(df), int)
    for m, sub in df.groupby("module"):
        idx = sub.index.to_numpy(); toks = [set(t) for t in sub.tokens]; grs = sub.group.to_numpy()
        for a in range(len(idx)):
            gs = {grs[b] for b in range(len(idx)) if len(toks[a] & toks[b]) / max(len(toks[a] | toks[b]), 1) >= 0.5}
            out[np.searchsorted(df.index.to_numpy(), idx[a])] = len(gs)
    return out


I["family_groups"] = family_groups(I)
F = pd.read_parquet("module_instances.parquet")          # first-pass modules
F["tokens"] = F.members.map(lambda v: [int(TOK[x]) for x in v]); F["group"] = F.genome.map(GRPG)
F["family_groups"] = family_groups(F.reset_index(drop=True))
FP = pd.read_parquet("score_per_instance.parquet").merge(F[["instance_id", "family_groups", "n_members"]], on="instance_id")

BD = pd.read_parquet("boundary_instances.parquet")
BD["x_s"] = BD.X.map(min); BD["x_e"] = BD.X.map(max); BD["y_s"] = BD.Y.map(min); BD["y_e"] = BD.Y.map(max)
BD["u_s"] = BD[["x_s", "y_s"]].min(1); BD["u_e"] = BD[["x_e", "y_e"]].max(1)
BO = np.vstack([origins(f"b{i}", c, w) for i, c, w in zip(range(len(BD)), BD.contig_code, BD.u_e - BD.u_s)]) - BD.u_s.to_numpy()[:, None]

rows, nrows, brows, per_all = [], [], [], []
for lab, path in OUT.items():
    s, e = load_loci(lab, path)
    ls = dict(a=s, b=e, mod=np.full(len(s), -1, np.int64), maxlen=int((e - s + 1).max()), n=int(len(s)))
    rc = RC.score(ls, I.first_gid.to_numpy(np.int64), I.last_gid.to_numpy(np.int64))
    ct = RC.score(ls, QS.ravel(), QE.ravel())
    crec = ct.recovered.to_numpy().reshape(len(I), NCTL).mean(1); cres = ct.resolved.to_numpy().reshape(len(I), NCTL).mean(1)
    df = pd.DataFrame(dict(genome=GEN[s], contig=CON[s], start=POS[s], end=POS[e], module_id=np.full(len(s), -1)))
    per, _, _ = S.score(sh, df, controls=NCTL, seed=17, threads=THREADS, min_members=2)
    per = per.set_index("instance_id").reindex(I.instance_id)
    p = pd.DataFrame(dict(instance_id=I.instance_id, module=I.module, output=lab, n_members=I.n_members, family_groups=I.family_groups,
                          recovered=rc.recovered.to_numpy(), resolved=rc.resolved.to_numpy(), chance_recovered=crec, chance_resolved=cres,
                          f1=per.f1.to_numpy(), control_f1=per.control_mean_f1.to_numpy()))
    p["excess"] = p.f1 - p.control_f1
    per_all.append(p)
    for m, g in p.groupby("module", sort=False):
        lo, hi = boot(g.excess)
        rows.append(dict(module=m, output=lab, instances=len(g), recovered=int(g.recovered.sum()), resolved=int(g.resolved.sum()),
                         chance_recovered=round(g.chance_recovered.sum(), 1), chance_resolved=round(g.chance_resolved.sum(), 1),
                         f1=round(g.f1.mean(), 3), control_f1=round(g.control_f1.mean(), 3), excess=round(g.excess.mean(), 3),
                         excess_lo=round(lo, 3), excess_hi=round(hi, 3)))
    # nested
    for pr, g in PR.groupby("pair", sort=False):
        res = np.array([nested_eval(s, e, ls["maxlen"], o, i) for o, i in zip(g.outer, g.inner)])
        ch = []
        for k, (o, i, oid) in enumerate(zip(g.outer, g.inner, g.outer_id)):
            orr = origins(f"n{oid}", int(g.contig_code.iat[k]), int(o.max() - o.min())) - o.min()
            ch.append(np.mean([nested_eval(s, e, ls["maxlen"], o + d, i + d)[2] for d in orr[:50]]))
        nrows.append(dict(pair=pr, output=lab, outer_instances=len(g), outer_held=int(res[:, 0].sum()), inner_matched=int(res[:, 1].sum()),
                          inner_nested_in_holding_locus=int(res[:, 2].sum()), chance_nested=round(float(np.sum(ch)), 1)))
    # boundary
    for t, g in BD.groupby("btype", sort=False):
        ix = g.index.to_numpy()
        def rs(a, b):
            return RC.score(ls, a.astype(np.int64), b.astype(np.int64)).resolved.to_numpy()
        jo = rs(g.u_s.to_numpy(), g.u_e.to_numpy())
        sx = rs(g.x_s.to_numpy(), g.x_e.to_numpy()); sy = rs(g.y_s.to_numpy(), g.y_e.to_numpy())
        off = BO[ix]                                            # offsets relative to the union start
        base = off
        cjo = rs((base + g.u_s.to_numpy()[:, None]).ravel(), (base + g.u_e.to_numpy()[:, None]).ravel()).reshape(len(g), NCTL).mean(1)
        cx = rs((base + g.x_s.to_numpy()[:, None]).ravel(), (base + g.x_e.to_numpy()[:, None]).ravel()).reshape(len(g), NCTL)
        cy = rs((base + g.y_s.to_numpy()[:, None]).ravel(), (base + g.y_e.to_numpy()[:, None]).ravel()).reshape(len(g), NCTL)
        brows.append(dict(btype=t, output=lab, instances=len(g), genome_groups=g.group.nunique(), joined=int(jo.sum()),
                          separated=int((sx & sy).sum()), joined_only=int((jo & ~(sx & sy)).sum()),
                          separated_only=int(((sx & sy) & ~jo).sum()), chance_joined=round(float(cjo.sum()), 1),
                          chance_separated=round(float((cx & cy).mean(1).sum()), 1)))
    log(lab, "done")

R = pd.DataFrame(rows); R.to_csv("large_scores.tsv", sep="\t", index=False)
NR = pd.DataFrame(nrows); NR.to_csv("large_nested.tsv", sep="\t", index=False)
BR = pd.DataFrame(brows); BR.to_csv("boundary_scores.tsv", sep="\t", index=False)
PA = pd.concat(per_all); PA.to_parquet("large_score_per_instance.parquet", index=False)

# strata: family genome groups 1 / 2 / >=3, two-gene (first pass, members == 2) and larger (>= 3 members)
FP["output"] = FP.output.map({"SECONDARY_embedding_list_tree_Z_L4bp_t0.02": "SEC", "PRIMARY_family_bond_tree_WA_L4b_t0.2": "PRI",
                              "engine_DH": "DH", "synton_faithful": "SYN"})
two = FP[FP.n_members == 2][["output", "family_groups", "recovered", "resolved", "chance_recovered", "chance_resolved"]].assign(size="2 genes (first pass)")
big = PA[PA.n_members >= 3][["output", "family_groups", "recovered", "resolved", "chance_recovered", "chance_resolved"]].assign(size=">= 3 genes (larger set)")
ST_ = pd.concat([two, big])
ST_["stratum"] = ST_.family_groups.clip(upper=3).map({1: "1 group", 2: "2 groups", 3: ">= 3 groups"})
SS = ST_.groupby(["size", "stratum", "output"]).agg(instances=("recovered", "size"), recovered=("recovered", "sum"), resolved=("resolved", "sum"),
                                                     chance_recovered=("chance_recovered", "sum"), chance_resolved=("chance_resolved", "sum")).reset_index()
SS["chance_recovered"] = SS.chance_recovered.round(1); SS["chance_resolved"] = SS.chance_resolved.round(1)
SS.to_csv("strata_by_family_groups.tsv", sep="\t", index=False)
json.dump(dict(post_hoc=True, controls=NCTL, seconds=round(time.time() - t0, 1)), open("large_scores.json", "w"), indent=1)
pd.set_option("display.width", 250)
print(R.to_string()); print(NR.to_string()); print(BR.to_string()); print(SS.to_string())
