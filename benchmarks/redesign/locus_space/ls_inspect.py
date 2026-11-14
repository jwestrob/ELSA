"""Quick schema inspection (login node, one thread)."""
import pyarrow.parquet as pq
import sys

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
paths = [
    f"{RD}/ws_dpann/package/universe/genes.parquet",
    f"{RD}/ws_dpann/package/universe/genomes.parquet",
    f"{RD}/ws_dpann/package/gold/instances.parquet",
    f"{RD}/ws_dpann/package/gold/members.parquet",
    f"{RD}/engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet",
    f"{RD}/engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3/token_modules.parquet",
    f"{RD}/engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3/loci_all_genomes.parquet",
    f"{RD}/engine/results_dpann_repair/dpann__mmseqs__seq__e3c50s75_rescue3__w3/significant_pairs.parquet",
    f"{RD}/boundary/loci/rescue3/DH_coreH_res0.3_f0.5.parquet",
    f"{RD}/boundary/loci/rescue3/H_res0.3_f0.5.parquet",
    f"{RD}/lineagefree/groups/emb_hom_average_t0.75.parquet",
    f"{RD}/lineagefree/groups/fam_jac_average_t0.2.parquet",
    f"{RD}/lineagefree/groups/genome_map.parquet",
]
for p in paths:
    try:
        f = pq.ParquetFile(p)
        print("==", p.replace(RD, "$RD"), f.metadata.num_rows)
        print("  ", f.schema_arrow)
        t = f.read_row_group(0).slice(0, 3).to_pandas()
        print(t.to_string()[:1500])
    except Exception as e:
        print("ERR", p, e)
