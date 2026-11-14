"""Schema of census tables and the Sharur DPANN copy (login node, one thread)."""
import duckdb
import pyarrow.parquet as pq

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
for p in ("census_instances", "census_types", "gene_census"):
    f = pq.ParquetFile(f"{RD}/cofold/s3/census/{p}.parquet")
    print("==", p, f.metadata.num_rows)
    print(f.schema_arrow)
    print(f.read_row_group(0).slice(0, 3).to_pandas().to_string()[:1200])
con = duckdb.connect(f"{RD}/ws_dpann/input/sharur.duckdb", read_only=True, config={"threads": 1})
print(con.execute("select table_name from information_schema.tables order by 1").fetchall())
for t in ("annotations", "proteins"):
    print(t, con.execute(f"select column_name, data_type from information_schema.columns where table_name='{t}'").fetchall())
print(con.execute("select source, count(*) from annotations group by 1 order by 2 desc").fetchall())
print(con.execute("select * from annotations limit 3").fetchdf().to_string())
