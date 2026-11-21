"""Print schemas of the lut package tables and the whole-proteome Pfam table (read-only)."""
import pandas as pd
RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
P = RD + "/slice_lut/testset/package_k8"
for f in ["universe/genes.parquet", "universe/genomes.parquet", "universe/contigs.parquet",
          "gold/lut_instances.parquet", "gold/lut_members.parquet"]:
    d = pd.read_parquet(P + "/" + f)
    print("==", f, d.shape)
    print(d.head(3).T.to_string()[:2500])
d = pd.read_parquet(RD + "/annot_gpu/lut/PFAM.parquet")
print("== PFAM", d.shape)
print(d.head(5).T.to_string())
for f in ["loci/lut_WA/L4b_t0.2.parquet", "loci/lut_Z/L4bp_t0.02.parquet"]:
    x = pd.read_parquet(RD + "/long_loci/frozen/" + f)
    print("==", f, x.shape, list(x.columns)); print(x.head(3).to_string())
