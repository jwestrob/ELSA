"""Size of the earlier Spicy LanMs database for a cost estimate: proteins, unique sequences, residues; overall and for
the LanM-source assemblies (source lanm / both in all_genomes_taxonomy.tsv). READ_ONLY; prints only."""
import duckdb, pandas as pd
con = duckdb.connect("/groups/banfield/users/jwestrob/bin/Sharur/data/spicy_lams/sharur.duckdb", read_only=True)
tax = pd.read_csv("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/spicy_lams/all_genomes_taxonomy.tsv", sep="\t")
tax["key"] = tax.assembly_acc.str.replace(r"^GC[AF]_", "", regex=True)
con.register("tax", tax[["key", "source"]].drop_duplicates("key"))
print(con.execute("""with p as (select regexp_extract(bin_id, '^GC[AF]_([0-9]+\\.[0-9]+)', 1) as key, sequence, sequence_length from proteins)
  select coalesce(t.source,'unknown') as src, count(*) as proteins, count(distinct md5(p.sequence)) as unique_seq,
         sum(p.sequence_length) as residues, count(distinct p.key) as genomes
  from p left join tax t using(key) group by rollup(src) order by 1""").fetchdf().to_string())
