#!/usr/bin/env python3
"""Locus context, CDD best hit and own annotations for chosen example genes; CDD make-up of over-extended labels."""
import sys, duckdb, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 70)
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute("CREATE VIEW pass AS SELECT * FROM read_parquet('pass/dpann_slot_rescue_pass.parquet')")
con.execute("CREATE VIEW ver AS SELECT * FROM read_parquet('verify/verification_per_protein.parquet')")
def locus(pid, w=4):
    r = con.execute("select contig_id, gene_index, bin_id from d.proteins where protein_id=?", [pid]).fetchone()
    if r is None: print("  missing", pid); return
    c, gi, b = r
    tax = con.execute("select taxonomy from d.bins where bin_id=?", [b]).fetchone()
    print(f"  genome {b} | {tax[0] if tax else ''}")
    q = """
    WITH win AS (SELECT protein_id, gene_index, strand, sequence_length FROM d.proteins WHERE contig_id=? AND gene_index BETWEEN ? AND ?),
    pf AS (SELECT protein_id, string_agg(DISTINCT name, ',') pfam FROM d.annotations WHERE source='pfam' AND protein_id IN (SELECT protein_id FROM win) GROUP BY 1),
    ko AS (SELECT protein_id, string_agg(DISTINCT accession || ' ' || coalesce(description, name), '; ') ko FROM d.annotations WHERE source='kofam' AND protein_id IN (SELECT protein_id FROM win) GROUP BY 1),
    cd AS (SELECT protein_id, arg_min(name, evalue) cdd, min(evalue) cdd_e FROM d.protein_resemblance WHERE protein_id IN (SELECT protein_id FROM win) GROUP BY 1)
    SELECT w.gene_index - ? AS off, w.protein_id, w.strand s, w.sequence_length len, pf.pfam, left(ko.ko, 60) ko, cd.cdd, cd.cdd_e,
           p.proposed_label_name proposed, p.votes_for vf, p.supporting_genomes sg, v.tier
    FROM win w LEFT JOIN pf USING (protein_id) LEFT JOIN ko USING (protein_id) LEFT JOIN cd USING (protein_id)
    LEFT JOIN pass p USING (protein_id) LEFT JOIN ver v USING (protein_id) ORDER BY w.gene_index
    """
    print(con.execute(q, [c, gi - w, gi + w, gi]).df().to_string(index=False))
for arg in sys.argv[1:]:
    if arg.startswith("label:"):
        lab = arg[6:]
        print(f"\n##### label {lab}: CDD best hits of the labeled genes")
        print(con.execute("""SELECT coalesce(r.cn,'(no CDD hit)') cdd, any_value(left(r.cdesc,70)) descr, count(*) n, median(pr.sequence_length) med_len
            FROM pass p JOIN d.proteins pr USING (protein_id)
            LEFT JOIN (SELECT protein_id, arg_min("name", evalue) cn, arg_min(description, evalue) cdesc FROM d.protein_resemblance GROUP BY 1) r USING (protein_id)
            WHERE p.proposed_label_name=? GROUP BY 1 ORDER BY n DESC LIMIT 8""", [lab]).df().to_string(index=False))
        print(f"  annotated carriers of {lab} in DPANN:")
        print(con.execute("""SELECT a.protein_id, a.evalue, a.score, pr.sequence_length len, a.start_aa, a.end_aa,
            (SELECT arg_min(name, evalue) FROM d.protein_resemblance r WHERE r.protein_id=a.protein_id) cdd
            FROM d.annotations a JOIN d.proteins pr USING (protein_id) WHERE a.source='pfam' AND a.name=? ORDER BY a.evalue LIMIT 6""", [lab]).df().to_string(index=False))
    else:
        print(f"\n##### {arg}")
        locus(arg)
