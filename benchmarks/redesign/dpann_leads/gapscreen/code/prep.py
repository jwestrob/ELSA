#!/usr/bin/env python3
"""gapscreen step 2 (Mac, read-only): genome-level annotation tables, clades, labeled-gene table, KEGG reference parse.

Writes data/: genome_ko.parquet (bin_id, ko; sources kofam + kegg), genome_pfam.parquet (bin_id, pf = accession or
name-form row), bins.parquet (completeness, contamination, phylum / class / order / family from bins.taxonomy),
labeled.parquet (pass row + verification tier + composition), kegg_ko.tsv (KO, name, EC list, modules),
kegg_modules.tsv copy of the parsed module definitions (module, name, class, definition).
KEGG source: Sharur's local KEGG snapshot (data/reference/kegg; ko 2026/09/24, module 2026/09/11), built from KEGG by
`sharur setup-kegg`; read only.
"""
import re
from pathlib import Path

import duckdb
import pandas as pd

H = Path(__file__).resolve().parents[1]
R = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/slot_rescue")
KEGG = Path("/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/reference/kegg")
con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.execute("ATTACH '/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb' AS d (READ_ONLY)")
D = H / "data"
gk = con.execute("""SELECT DISTINCT p.bin_id, a.accession AS ko FROM d.annotations a JOIN d.proteins p USING (protein_id)
                    WHERE a.source IN ('kofam','kegg') AND regexp_matches(a.accession, '^K[0-9]{5}$')""").df()
gk.to_parquet(D / "genome_ko.parquet", index=False)
gp = con.execute("""SELECT DISTINCT p.bin_id, split_part(a.accession, '.', 1) AS pf, a."name" AS pfname FROM d.annotations a
                    JOIN d.proteins p USING (protein_id) WHERE a.source = 'pfam'""").df()
gp.to_parquet(D / "genome_pfam.parquet", index=False)
b = con.execute("SELECT bin_id, completeness, contamination, taxonomy FROM d.bins").df()
for i, rk in enumerate(["domain", "phylum", "class", "order", "family", "genus"]):
    b[rk] = b.taxonomy.str.split(";").str[i].str.replace(r"^[a-z]__", "", regex=True).fillna("")
b.to_parquet(D / "bins.parquet", index=False)
lab = con.execute(f"""SELECT p.*, v.tier, v.vfrac, v.amp, v.annotated, v.proposed_evalue, v.best_other_label, pr.bin_id,
        length(regexp_replace(pr.sequence, '[^ILVFMAWC]', '', 'g'))::DOUBLE / pr.sequence_length AS ilvfmawc
    FROM read_parquet('{R}/pass/dpann_slot_rescue_pass.parquet') p
    JOIN read_parquet('{R}/verify/verification_per_protein.parquet') v USING (protein_id)
    JOIN d.proteins pr USING (protein_id)""").df()
assert len(lab) == 175729
lab.to_parquet(D / "labeled.parquet", index=False)
rows = []
for line in open(KEGG / "kegg_evidence_snapshot.tsv"):
    if line.startswith("#"):
        continue
    t = line.rstrip("\n").split("\t")
    ec = re.findall(r"EC:([0-9.\- ]+)\]", t[2])
    ecs = sorted({e for grp in ec for e in grp.split()})
    rows.append(dict(ko=t[0], symbol=t[1], name=t[2], ec=";".join(ecs), modules=t[4] if len(t) > 4 else ""))
pd.DataFrame(rows).to_csv(D / "kegg_ko.tsv", sep="\t", index=False)
mods = pd.read_csv(KEGG / "kegg_modules.tsv", sep="\t", comment="#", header=None, names=["module", "name", "class", "definition"])
mods.to_csv(D / "kegg_modules.tsv", sep="\t", index=False)
print(len(gk), "genome-KO;", len(gp), "genome-Pfam;", len(b), "bins;", len(lab), "labeled;", len(rows), "KOs;", len(mods), "modules")
