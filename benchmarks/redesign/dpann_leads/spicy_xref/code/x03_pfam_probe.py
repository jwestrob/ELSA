"""Probe: which Pfam models the module definitions need are present in the lut whole-proteome Pfam table, and how
the spacing of EF-hand domain hits separates lanmodulin (lanM reverse-best homologs inside lut loci) from other
small EF-hand proteins. Read-only; prints only."""
import numpy as np
import pandas as pd
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
PKG = RD + "/slice_lut/testset/package_k8"
P = pd.read_parquet(RD + "/annot_gpu/lut/PFAM.parquet", columns=["gid", "accession", "name", "env_from", "env_to"])
genes = pd.read_parquet(PKG + "/universe/genes.parquet", columns=["gid", "length"])
names = ["EF-hand_1", "EF-hand_5", "EF-hand_6", "EF-hand_7", "EF-hand_8", "EF-hand_9", "EF-hand_10", "EF-hand_11",
         "PQQ", "PQQ_2", "PQQ_3", "Cytochrome_CBB3", "SBP_bac_3", "Rieske", "Rieske_2", "Lactamase_B", "Lactamase_B_2",
         "Lactamase_B_3", "Peptidase_C14", "COXG", "GFA", "DctM", "DctQ", "DctP", "Ank", "Ank_2", "Ank_4", "Ank_5",
         "NAD_binding_2", "NAD_binding_11", "CNNM", "CBS", "CorC_HlyC", "SBP_bac_5", "DUF2158", "Sigma70_r2",
         "Sigma70_r4_2", "Sigma70_r4", "Peptidase_S58", "MASE4", "MASE3", "PAS", "PAS_3", "PAS_4", "PAS_9",
         "Ribul_P_3_epim", "Pkinase", "Usp", "Metallophos", "KTSC", "Peptidase_M20", "THDPS_N_2", "Lysine_decarbox",
         "Hexapep", "Metal_resist", "CcmE", "DUF6163", "DUF3141", "Methyltransf_3", "ECH_1", "ALAD", "Tannase", "DAO",
         "adh_short", "HTH_Crp_2", "cNMP_binding", "BPD_transp_1", "DUF4394", "GST_N", "GST_N_3", "GST_C", "ResIII",
         "YcaO", "TfuA", "SoxY", "SoxZ", "DUF2155", "HisKA_3", "GerE", "Response_reg", "HATPase_c", "HisKA", "HAMP",
         "Trans_reg_C", "DUF6894", "Peripla_BP_6", "Lactonase", "ABC_tran", "ABC2_membrane", "Plug", "TonB_dep_Rec_b-barrel",
         "TonB_dep_Rec", "DUF3280", "RskA", "zf-HC2", "Transglycosylas", "Transpeptidase", "MoCoBD_1", "MoCoBD_2",
         "Ald_Xan_dh_C", "Fer2", "Fer2_2", "Condensation", "Lys_Orn_oxgnase", "DUF2950", "DUF3300", "Cu-oxidase_4",
         "TctC", "AraC_binding", "HTH_18", "Glyco_transf_61", "DUF1236", "TAL_FSA", "PdxJ", "DUF899", "MerR_1", "VWA", "AAA_5"]
cnt = P[P.name.isin(names)].groupby("name").gid.nunique()
print("present (proteins):", cnt.to_dict())
print("absent:", [n for n in names if n not in cnt.index])
print("names containing MoCoBD/MASE/Transgly/Cu-ox/TonB/Rieske:",
      sorted(set(P.name[P.name.str.contains("MoCoBD|MASE|Transgly|Cu-ox|TonB|Rieske|Sigma70|Ald_Xan|Tannase|SoxY|SoxZ")])))
ef = P[P.name.str.startswith("EF-hand")].merge(genes, on="gid")
ef = ef.sort_values(["gid", "env_from"])
g = ef.groupby("gid")
st = pd.DataFrame(dict(n_ef=g.size(), n_ef5=g.name.apply(lambda s: (s == "EF-hand_5").sum()),
                       length=g.length.first(),
                       med_gap=g.env_from.apply(lambda s: float(np.median(np.diff(s.values))) if len(s) > 1 else np.nan)))
mem = pd.read_parquet(PKG + "/gold/lut_members.parquet")
lanm = set(mem.gid[(mem.component == "M") & (mem.caller == "lut_homology_main")])
st["lanM_member"] = st.index.isin(lanm)
print("lanM members:", len(lanm), "with EF-hand rows:", int(st.lanM_member.sum()))
print(st[st.lanM_member].describe().to_string())
small = st[(st.length <= 250) & (st.n_ef >= 2)]
print("small (<=250 aa) >=2 EF rows:", len(small), "; med_gap histogram (lanM members vs others):")
bins = [0, 20, 24, 26, 28, 30, 32, 34, 36, 40, 60, 400]
print(pd.crosstab(pd.cut(small.med_gap, bins), small.lanM_member).to_string())
print("n_ef distribution in lanM members:", st[st.lanM_member].n_ef.value_counts().to_dict())
print("EF domain names in lanM members:", ef[ef.gid.isin(lanm)].name.value_counts().to_dict())
