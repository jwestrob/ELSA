"""List tables and the genome table of the earlier Spicy LanMs Sharur database (READ_ONLY)."""
import duckdb
DB = "/groups/banfield/users/jwestrob/bin/Sharur/data/spicy_lams/sharur.duckdb"
con = duckdb.connect(DB, read_only=True)
print(con.execute("select table_name, estimated_size from duckdb_tables() order by 1").fetchdf().to_string())
for t in ["bins", "contigs"]:
    try:
        print(t, con.execute(f"describe {t}").fetchdf()[["column_name", "column_type"]].to_string())
        print(con.execute(f"select count(*) from {t}").fetchone())
    except Exception as ex:
        print(t, ex)
print(con.execute("select * from bins limit 3").fetchdf().T.to_string()[:3000])
