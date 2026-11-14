import duckdb, sys
RD="/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
P=RD+"/ws_dpann/package"
con=duckdb.connect(); con.execute("SET threads=2")
def q(s):
    print(">>",s[:200]); print(con.execute(s).fetchdf().to_string()); print()
q(f"DESCRIBE SELECT * FROM '{P}/universe/genes.parquet'")
q(f"DESCRIBE SELECT * FROM '{P}/universe/genomes.parquet'")
q(f"DESCRIBE SELECT * FROM '{P}/universe/contigs.parquet'")
q(f"DESCRIBE SELECT * FROM '{P}/gold/instances.parquet'")
q(f"DESCRIBE SELECT * FROM '{RD}/engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3/score_min2/instance_scores.parquet'")
q(f"DESCRIBE SELECT * FROM '{RD}/engine/results_dpann_v5/dpann__mmseqs__seq__e3c50s75__w3/loci_all_genomes.parquet'")
q(f"SELECT count(*) n, sum((length>=2046)::int) giants, max(length) mx, min(length) mn, quantile_cont(length,[0.01,0.1,0.5,0.9,0.99]) q, sum((length<160)::int) lt160, sum(length) res FROM '{P}/universe/genes.parquet'")
q(f"SELECT * FROM '{P}/universe/genomes.parquet' LIMIT 2")
q(f"SELECT caller, count(*) n, sum((n_members_resolved>=2)::int) ge2 FROM '{P}/gold/instances.parquet' GROUP BY 1")
