#!/usr/bin/env python
"""galt step 10: genome-level and protein-level feature calls for the locus and distribution questions (all 3,564 DPANN
genomes). Every feature is an OBSERVED database row (source named); no feature is a curated system call.
  GT3_GS      Pfam PF05693 Glycogen_syn, or KO K00693 (GYS, EC 2.4.1.11), or CDD cd03793 GT3_GSY2-like
  GT5_GS      Pfam PF08323 Glyco_transf_5 AND CDD best hit naming a GT5 / glycogen synthase model (cd03791 GT5_Glycogen
              _synthase_DULL1-like, PRK00654 glgA, TIGR02095 glgA), or KO K00703 / K20812 / K13679, and no GT3 call
  GT4_GS_KO   KO K16150 (glycogen synthase, EC 2.4.1.11, KEGG; a GT4-type entry) as a separate column
  GH57        Pfam PF03065;  GDE_C Pfam PF06202;  GH15 Pfam PF00723;  GP  glycogen phosphorylase K00688 or PF00343
  glgC        KO K00975;  UGP  KO K00963 or Pfam PF01704 (UDPGP) ;  USP KO K12447
  NTP_transferase  Pfam PF00483 (sugar-1-phosphate nucleotidyltransferase fold); the CDD best hit names the type
  ArchGlmU    CDD TIGR03992 Arch_glmU;  galK K00849;  galE K01784;  galM K01785;  PGM K01835 / K15778 / Pfam PF02878
  GMPP        K00966 / K16881 / K00971 / K16011;  rmlA K00973
Writes data/genome_features.tsv (bin x feature counts) and data/protein_features.parquet (protein x feature)."""
from pathlib import Path
import duckdb
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
c = duckdb.connect()
c.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
c.execute("SET threads=1")
PF = {"PF05693": "pf_GT3", "PF08323": "pf_GT5", "PF03065": "GH57", "PF06202": "GDE_C", "PF00723": "GH15", "PF00343": "pf_GP",
      "PF01704": "pf_UGP", "PF00483": "NTP_transferase", "PF02878": "pf_PGM", "PF19027": "DUF5752"}
KO = {"K00693": "ko_GT3", "K00703": "ko_GT5", "K20812": "ko_GT5", "K13679": "ko_GT5", "K16150": "GT4_GS_KO", "K00688": "ko_GP",
      "K00975": "glgC", "K00963": "ko_UGP", "K12447": "USP", "K00849": "galK", "K01784": "galE", "K01785": "galM",
      "K01835": "ko_PGM", "K15778": "ko_PGM", "K00966": "GMPP", "K16881": "GMPP", "K00971": "GMPP", "K16011": "GMPP",
      "K00973": "rmlA", "K00965": "ko_GalT", "K16148": "glgM", "K00700": "glgB", "K16149": "glgB", "K02438": "glgX",
      "K01214": "glgX", "K01178": "SGA1", "K07405": "ko_GH57amy", "K00705": "malQ", "K22451": "malQ"}
CDD = {"cd03793": "cdd_GT3", "cd03791": "cdd_GT5", "PRK00654": "cdd_GT5", "TIGR02095": "cdd_GT5", "TIGR03992": "ArchGlmU",
       "PRK05293": "cdd_glgC", "cd02508": "cdd_glgC", "cd04181": "cdd_NTPtrans", "cd02540": "cdd_GlmU", "cd02538": "cdd_RmlA",
       "cd02541": "cdd_UGPase_prok", "cd06428": "cdd_eIF2B_like"}
pfa = "','".join(PF); koa = "','".join(KO); cda = "','".join(CDD)
a = c.execute(f"""SELECT p.protein_id, p.bin_id, p.contig_id, p.gene_index, a.source, a.accession FROM d.annotations a
   JOIN d.proteins p USING(protein_id)
   WHERE (a.source='pfam' AND a.accession IN ('{pfa}')) OR (a.source IN ('kofam','kegg') AND a.accession IN ('{koa}'))""").fetchdf()
r = c.execute(f"""SELECT p.protein_id, p.bin_id, p.contig_id, p.gene_index, 'cdd' source, r.accession FROM d.protein_resemblance r
   JOIN d.proteins p USING(protein_id) WHERE r.accession IN ('{cda}')""").fetchdf()
X = pd.concat([a, r], ignore_index=True)
X["feat"] = [PF.get(x) if s == "pfam" else (KO.get(x) if s in ("kofam", "kegg") else CDD.get(x)) for s, x in zip(X.source, X.accession)]
P = X.pivot_table(index=["protein_id", "bin_id", "contig_id", "gene_index"], columns="feat", values="accession", aggfunc="size",
                  fill_value=0).reset_index()
for col in set(PF.values()) | set(KO.values()) | set(CDD.values()):
    if col not in P.columns:
        P[col] = 0
P["GT3_GS"] = ((P.pf_GT3 + P.ko_GT3 + P.cdd_GT3) > 0).astype(int)
P["GT5_GS"] = (((P.pf_GT5 > 0) & (P.cdd_GT5 > 0)) | (P.ko_GT5 > 0)).astype(int) * (1 - P.GT3_GS)
P["GP"] = ((P.pf_GP + P.ko_GP) > 0).astype(int)
P["UGP"] = ((P.pf_UGP + P.ko_UGP) > 0).astype(int)
P["PGM"] = ((P.pf_PGM + P.ko_PGM) > 0).astype(int)
P.to_parquet(H / "data/protein_features.parquet", index=False)
feats = ["GT3_GS", "GT5_GS", "GT4_GS_KO", "GH57", "GDE_C", "GH15", "GP", "glgC", "UGP", "USP", "NTP_transferase", "ArchGlmU",
         "galK", "galE", "galM", "PGM", "GMPP", "rmlA", "glgB", "glgX", "SGA1", "glgM", "malQ", "DUF5752", "cdd_glgC"]
Gf = P.groupby("bin_id")[feats].apply(lambda d: (d > 0).sum()).reset_index()
bins = pd.read_csv(H / "data/genomes.tsv", sep="\t")
Gf = bins.merge(Gf, on="bin_id", how="left").fillna({f: 0 for f in feats})
C = pd.read_csv(H / "data/census_proteins.tsv", sep="\t")
for k, m in [("n_named", C.named & ~C.lab76), ("n_lab76", C.lab76), ("n_lab76_named", C.lab76 & C.named),
             ("n_lab76_unann", C.lab76 & ~C.named), ("n_lead15", C.lead15), ("n_set64", C.set64)]:
    Gf[k] = Gf.bin_id.map(C[m].groupby("bin_id").size()).fillna(0).astype(int)
Gf.to_csv(H / "data/genome_features.tsv", sep="\t", index=False)
print(len(P), "feature proteins;", len(Gf), "genomes")
print((Gf[feats] > 0).sum().to_string())
