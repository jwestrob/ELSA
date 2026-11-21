"""Probe 2: EF-hand loops = EF-hand clan Pfam rows merged when their envelopes overlap; spacing = median distance
between consecutive loop starts. Compare lanM reverse-best members of lut loci with other small EF-hand proteins.
Read-only; prints only."""
import numpy as np
import pandas as pd
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PKG = RD + "/slice_lut/testset/package_k8"
P = pd.read_parquet(RD + "/annot_gpu/lut/PFAM.parquet", columns=["gid", "name", "env_from", "env_to"])
genes = pd.read_parquet(PKG + "/universe/genes.parquet", columns=["gid", "genome", "length"])
npf = P[~P.name.str.startswith("EF-hand")].gid.value_counts()
ef = P[P.name.str.startswith("EF-hand")].sort_values(["gid", "env_from"])
rows = []
for gid, g in ef.groupby("gid"):
    loops, cur_s, cur_e = [], None, None
    for s, e in zip(g.env_from.values, g.env_to.values):
        if cur_s is None or s > cur_e:
            if cur_s is not None:
                loops.append(cur_s)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    loops.append(cur_s)
    gap = float(np.median(np.diff(loops))) if len(loops) > 1 else np.nan
    rows.append((gid, len(loops), gap))
st = pd.DataFrame(rows, columns=["gid", "n_loops", "gap"]).merge(genes, on="gid")
st["other_pfam"] = st.gid.map(npf).fillna(0).astype(int)
mem = pd.read_parquet(PKG + "/gold/lut_members.parquet")
lanm = set(mem.gid[(mem.component == "M") & (mem.caller == "lut_homology_main")])
st["lanM"] = st.gid.isin(lanm)
sm = st[(st.length <= 250)]
print(pd.crosstab(pd.cut(sm.gap, [0, 18, 21, 23, 25, 27, 29, 31, 33, 36, 40, 60, 400]), sm.lanM).to_string())
print(pd.crosstab(sm.n_loops, sm.lanM).to_string())
for nm, q in [("strict: len<=200, loops>=2, gap 20-29, no other Pfam", (st.length <= 200) & (st.n_loops >= 2) & st.gap.between(20, 29) & (st.other_pfam == 0)),
              ("broad: len<=300, loops>=2", (st.length <= 300) & (st.n_loops >= 2))]:
    x = st[q]
    print(nm, "proteins", len(x), "genomes", x.genome.nunique(), "lanM members captured", int(x.lanM.sum()), "/", len(lanm))
print("lanM members missed by strict:", st[st.lanM & ~((st.length <= 200) & (st.n_loops >= 2) & st.gap.between(20, 29) & (st.other_pfam == 0))][["gid", "n_loops", "gap", "length", "other_pfam"]].to_string())
