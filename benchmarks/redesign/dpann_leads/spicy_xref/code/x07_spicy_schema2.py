"""Schema of proteins / annotations in the earlier Spicy LanMs Sharur database (READ_ONLY)."""
import duckdb
con = duckdb.connect("/groups/banfield/users/jwestrob/bin/Sharur/data/spicy_lams/sharur.duckdb", read_only=True)
for t in ["proteins", "annotations"]:
    print(con.execute(f"describe {t}").fetchdf()[["column_name", "column_type"]].to_string())
print(con.execute("select * exclude (sequence) from proteins limit 3").fetchdf().T.to_string())
print(con.execute("select source, count(*) n, count(distinct protein_id) p from annotations group by 1").fetchdf().to_string())
print(con.execute("select * from annotations where source ilike '%pfam%' limit 3").fetchdf().T.to_string())
