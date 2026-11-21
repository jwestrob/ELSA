#!/usr/bin/env python
"""Print the locus context of a protein: Pfam, KO, CDD best hit, DefenseFinder hits, system membership, slot label.
usage: python ctx.py PROTEIN_ID [half_width]"""
import sys

import duckdb
import pandas as pd

pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 45)
pd.set_option('display.max_rows', 200)
R = '/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign'
con = duckdb.connect()
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
con.execute(f"CREATE VIEW pass AS SELECT * FROM read_parquet('{R}/slot_rescue/pass/dpann_slot_rescue_pass.parquet')")
con.execute(f"CREATE VIEW ver AS SELECT * FROM read_parquet('{R}/slot_rescue/verify/verification_per_protein.parquet')")


def locus(pid, w=5):
    c, gi, b = con.execute("select contig_id, gene_index, bin_id from d.proteins where protein_id=?", [pid]).fetchone()
    tax, comp = con.execute("select taxonomy, completeness from d.bins where bin_id=?", [b]).fetchone()
    print(f'## {pid} genome {b} completeness {comp} | {tax}')
    q = """
    WITH win AS (SELECT protein_id, gene_index, strand, sequence_length FROM d.proteins
                 WHERE contig_id=? AND gene_index BETWEEN ? AND ?),
    pf AS (SELECT protein_id, string_agg(DISTINCT name, ',') pfam FROM d.annotations
           WHERE source='pfam' AND protein_id IN (SELECT protein_id FROM win) GROUP BY 1),
    ko AS (SELECT protein_id, string_agg(DISTINCT accession, ',') ko FROM d.annotations
           WHERE source='kofam' AND protein_id IN (SELECT protein_id FROM win) GROUP BY 1),
    df AS (SELECT protein_id, string_agg(DISTINCT accession, ',') dfhit FROM d.annotations
           WHERE source IN ('defensefinder','txsscan') AND protein_id IN (SELECT protein_id FROM win) GROUP BY 1),
    sy AS (SELECT protein_id, string_agg(DISTINCT system_id || ':' || profile_name, ',') sysm FROM d.system_proteins
           WHERE protein_id IN (SELECT protein_id FROM win) GROUP BY 1),
    cd AS (SELECT protein_id, arg_min("name", evalue) cdd, min(evalue) cdd_e FROM d.protein_resemblance
           WHERE protein_id IN (SELECT protein_id FROM win) GROUP BY 1)
    SELECT w.gene_index - ? AS o, w.protein_id, w.strand s, w.sequence_length len, left(pf.pfam, 40) pfam, ko.ko,
           left(cd.cdd, 25) cdd, cd.cdd_e, left(df.dfhit, 40) df, left(sy.sysm, 50) sysm,
           p.proposed_label_name slot, p.votes_for vf, p.supporting_genomes sg, v.tier, p.alignment_lists_same_label al
    FROM win w LEFT JOIN pf USING (protein_id) LEFT JOIN ko USING (protein_id) LEFT JOIN cd USING (protein_id)
    LEFT JOIN df USING (protein_id) LEFT JOIN sy USING (protein_id)
    LEFT JOIN pass p USING (protein_id) LEFT JOIN ver v USING (protein_id) ORDER BY w.gene_index
    """
    print(con.execute(q, [c, gi - w, gi + w, gi]).df().to_string(index=False))


if __name__ == '__main__':
    w = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 5
    for a in sys.argv[1:]:
        if not a.isdigit():
            locus(a, w)
