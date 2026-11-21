#!/usr/bin/env python
"""Per-gene annotation table for every DPANN protein, from the Sharur DPANN database (read-only) and the TMHMM table.
Columns are OBSERVATIONS (raw hits) except cur_systems, which lists curated caller calls (DefenseFinder systems,
TXSScan systems, CCTyper crispr_cas / cas, HydDB classes).
usage: annot_export.py OUT.parquet
"""
import sys
import duckdb

DB = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
TM = "/Users/jacob/Documents/Berkeley/Sharur/tmhmm_dpann_2026-11-11/out/tmhmm_dpann.parquet"
TAGS = ["ribosomal_protein", "rna_polymerase", "atp_synthesis", "translation_factor", "trna_synthetase",
        "mobile_element", "transposase", "integrase", "phage_related", "recombinase",
        "carbohydrate_active", "glycosyltransferase", "transporter", "abc_transporter", "secretion_component",
        "secretion_system", "pilus", "defense_component", "toxin_domain", "antitoxin_domain", "toxin",
        "electron_transport", "energy_metabolism", "hydrogen_metabolism", "cofactor_biosynthesis",
        "amino_acid_biosynthesis", "lipid_metabolism", "nucleotide_metabolism", "dna_repair", "replication",
        "translation", "cell_surface", "adhesin", "membrane", "regulator", "transcription_factor", "radical_sam",
        "iron_sulfur", "protease", "kinase", "isoprenoid_biosynthesis", "sugar_metabolism", "cell_division",
        "chaperone", "rna_processing", "trna_modification", "restriction_modification", "sec_pathway",
        "signal_recognition", "dna_polymerase", "transcription", "rrna_modification", "divisome", "topoisomerase"]


def main(out):
    c = duckdb.connect(config={"threads": 4, "memory_limit": "12GB"})
    c.execute(f"ATTACH '{DB}' AS d (READ_ONLY)")
    tags = ",".join(f"'{t}'" for t in TAGS)
    c.execute(f"""
    COPY (
    WITH p AS (
      SELECT protein_id, bin_id, contig_id, gene_index, strand, sequence_length AS aa,
             length(regexp_replace(upper(sequence), '[^ILVFMAWC]', '', 'g'))::DOUBLE / greatest(length(sequence), 1) AS hyd
      FROM d.proteins),
    pf AS (SELECT protein_id, string_agg(name, ';' ORDER BY start_aa) AS pfam,
                  string_agg(DISTINCT accession, ';') AS pfam_acc
           FROM d.annotations WHERE source = 'pfam' GROUP BY 1),
    ko AS (SELECT protein_id, arg_min(accession, evalue) AS ko, arg_min(coalesce(description, name), evalue) AS ko_name
           FROM d.annotations WHERE source IN ('kofam', 'kegg') GROUP BY 1),
    df AS (SELECT protein_id, string_agg(DISTINCT name, ';') AS defensefinder
           FROM d.annotations WHERE source = 'defensefinder' GROUP BY 1),
    tx AS (SELECT protein_id, string_agg(DISTINCT name, ';') AS txsscan
           FROM d.annotations WHERE source = 'txsscan' GROUP BY 1),
    anyrow AS (SELECT DISTINCT protein_id FROM d.annotations),
    cdd AS (SELECT protein_id, arg_min(accession, evalue) AS cdd_acc,
                   arg_min(name || ' ' || coalesce(description, ''), evalue) AS cdd_desc, min(evalue) AS cdd_e
            FROM d.protein_resemblance GROUP BY 1),
    fm AS (SELECT protein_id, family_id FROM d.protein_families WHERE clustering_run = 'permissive-e3c50s75'),
    fhp AS (SELECT family_id, arg_max(hit_name, probability) AS fam_hh_pfam, max(probability) AS fam_hh_pfam_prob
            FROM d.family_annotations WHERE source = 'hhsearch_pfam' AND probability >= 90 GROUP BY 1),
    fhd AS (SELECT family_id, arg_max(hit_description, probability) AS fam_hh_pdb, max(probability) AS fam_hh_pdb_prob
            FROM d.family_annotations WHERE source = 'hhblits_pdb' AND probability >= 90 GROUP BY 1),
    sysd AS (SELECT sp.protein_id, 'DF:' || ds.system_type AS st FROM d.system_proteins sp
             JOIN d.defense_systems ds ON sp.system_id = ds.system_id WHERE sp.system_source = 'defensefinder_system'),
    syst AS (SELECT sp.protein_id, 'TXSS:' || ss.system_type AS st FROM d.system_proteins sp
             JOIN d.secretion_systems ss ON sp.system_id = ss.system_id WHERE sp.system_source = 'txsscan_system'),
    sysc AS (SELECT sp.protein_id, 'CRISPR-Cas:' || cs.best_type AS st FROM d.system_proteins sp
             JOIN d.crispr_cas_systems cs ON sp.system_id = cs.system_id
             WHERE sp.system_source = 'cctyper' AND cs.status IN ('crispr_cas', 'cas')),
    sysh AS (SELECT protein_id, 'HydDB:' || reference_class AS st FROM d.hydrogenase_classifications
             WHERE outcome = 'assigned'),
    sy AS (SELECT protein_id, string_agg(DISTINCT st, ';') AS cur_systems
           FROM (SELECT * FROM sysd UNION ALL SELECT * FROM syst UNION ALL SELECT * FROM sysc UNION ALL SELECT * FROM sysh)
           GROUP BY 1),
    pr AS (SELECT protein_id, array_to_string(list_filter(predicates, x -> list_contains([{tags}], x)), ';') AS tags
           FROM d.protein_predicates),
    tm AS (SELECT protein_id, n_tm FROM read_parquet('{TM}'))
    SELECT p.*, pf.pfam, pf.pfam_acc, ko.ko, ko.ko_name, df.defensefinder, tx.txsscan, cdd.cdd_acc, cdd.cdd_desc, cdd.cdd_e,
           fm.family_id AS perm_family, fhp.fam_hh_pfam, fhp.fam_hh_pfam_prob, fhd.fam_hh_pdb, fhd.fam_hh_pdb_prob,
           sy.cur_systems, pr.tags, tm.n_tm,
           (anyrow.protein_id IS NOT NULL) AS any_annot_row
    FROM p LEFT JOIN pf USING (protein_id) LEFT JOIN ko USING (protein_id) LEFT JOIN df USING (protein_id)
    LEFT JOIN tx USING (protein_id) LEFT JOIN cdd USING (protein_id) LEFT JOIN fm USING (protein_id)
    LEFT JOIN fhp ON fm.family_id = fhp.family_id LEFT JOIN fhd ON fm.family_id = fhd.family_id
    LEFT JOIN sy USING (protein_id) LEFT JOIN pr USING (protein_id) LEFT JOIN tm USING (protein_id)
    LEFT JOIN anyrow USING (protein_id)
    ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")
    print(c.execute(f"SELECT count(*), count(pfam), count(ko), count(cdd_acc), count(cur_systems), count(fam_hh_pfam), "
                    f"sum(any_annot_row::INT) FROM read_parquet('{out}')").fetchall())


if __name__ == "__main__":
    main(sys.argv[1])
