"""Diagnostics on the earlier database: Pfam name spellings for classes whose counts looked odd, and the
neighborhoods of DUF2158 and Peptidase_S58 genes (which EF-hand rows sit within 5 genes). READ_ONLY; prints only."""
import duckdb
con = duckdb.connect("/groups/banfield/users/jwestrob/bin/Sharur/data/spicy_lams/sharur.duckdb", read_only=True)
q = lambda s: print(con.execute(s).fetchdf().to_string())
q("""with d as (select p.bin_id, p.contig_id, p.gene_index, p.strand from proteins p join annotations a using(protein_id)
               where a.source='pfam' and a.name='DUF2158'),
     n as (select d.gene_index g0, d.strand s0, p.gene_index - d.gene_index as dd, p.strand, p.sequence_length as ln,
                  string_agg(distinct a.name, ';') as nm
           from d join proteins p on p.contig_id=d.contig_id and p.bin_id=d.bin_id and abs(p.gene_index-d.gene_index) between 1 and 2
           left join annotations a on a.protein_id=p.protein_id and a.source='pfam'
           group by 1,2,3,4,5,p.protein_id)
     select dd, (strand=s0) as same_strand, coalesce(nm,'(none)') as pf, count(*) as n, median(ln) as ln from n
     group by 1,2,3 order by n desc limit 25""")

q("""with d as (select p.bin_id, p.contig_id, p.gene_index, p.strand from proteins p join annotations a using(protein_id)
               where a.source='pfam' and a.name='Peptidase_S58'),
     n as (select d.gene_index g0, d.strand s0, p.gene_index - d.gene_index as dd, p.strand, p.sequence_length as ln,
                  string_agg(distinct a.name, ';') as nm
           from d join proteins p on p.contig_id=d.contig_id and p.bin_id=d.bin_id and abs(p.gene_index-d.gene_index) between 1 and 2
           left join annotations a on a.protein_id=p.protein_id and a.source='pfam'
           group by 1,2,3,4,5,p.protein_id)
     select dd, (strand=s0) as same_strand, coalesce(nm,'(none)') as pf, count(*) as n, median(ln) as ln from n
     group by 1,2,3 order by n desc limit 25""")
