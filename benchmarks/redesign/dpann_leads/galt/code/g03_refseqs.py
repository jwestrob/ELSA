#!/usr/bin/env python
"""galt step 3: reference sequence panel for the phylogeny and the profile classification.

Characterized / reviewed members of every GalT-like branch (UniProt records in ref/uniprot_json, plus Q8ZXN7, the
Pyrobaculum aerophilum GalT characterized in PMID 31693208 with crystal structures 6K5Z / 6K9Z), and, to give the single
characterized ADP-glucose phosphorylase (Q9FK51) a branch, one unreviewed IPR053177 (PANTHER PTHR42763) member per
phylum (length 320-360 aa; first listed by UniProt), flagged evidence='unreviewed_branch_context'.
Writes ref/refs.faa and data/ref_panel.tsv (accession, label, branch, evidence)."""
import json, re, urllib.request
from pathlib import Path
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
R = pd.read_csv(H / "data/ref_records.tsv", sep="\t", keep_default_na=False).set_index("accession")
PANEL = {  # accession: (short label, branch)
    "P09148": ("GalT_Ecoli", "GalT_I"), "P07902": ("GalT_human", "GalT_I"), "P08431": ("GalT_yeast", "GalT_I"),
    "P13212": ("GalT_Strlivid", "GalT_I"), "P31764": ("GalT_Hinflu", "GalT_I"), "Q7RYE7": ("GalT_Neurospora", "GalT_I"),
    "O33836": ("GalT_Tmaritima", "GalT_I"), "Q8ZXN7": ("GalT_Pyrobaculum", "GalT_I"), "Q9VMA2": ("GalT_Dmel", "GalT_I"),
    "E8MF11": ("GalTII_Bifido", "GalT_II"), "P39575": ("GalTII_Bsub", "GalT_II"), "Q9S6S3": ("GalTII_Llactis", "GalT_II"),
    "Q97EZ4": ("GalTII_Cacetob", "GalT_II"), "P45981": ("GalTII_Butyrivib", "GalT_II"),
    "Q9FK51": ("ADPGP_Athal", "ADPGP"),
    "P16550": ("Apa1_yeast", "Ap4A_phosphorylase"), "P22108": ("Apa2_yeast", "Ap4A_phosphorylase"),
    "P49348": ("Apa2_Klactis", "Ap4A_phosphorylase"),
    "Q6ZNW5": ("GDPGP1_human", "GDPhexose_phosphorylase"), "Q5ZR76": ("GDPGP1_Celegans", "GDPhexose_phosphorylase"),
    "Q8RWE8": ("VTC2_Athal", "GDPhexose_phosphorylase"), "Q9FLP9": ("VTC5_Athal", "GDPhexose_phosphorylase"),
    "Q08CA1": ("GDPGP1_zebrafish", "GDPhexose_phosphorylase"),
}
q8 = json.load(open(H / "ref/uniprot_json/Q8ZXN7.json"))
seqs = {a: R.loc[a, "sequence"] for a in PANEL if a in R.index}
seqs["Q8ZXN7"] = q8["sequence"]["value"]
rows = []
for a, (lab, br) in PANEL.items():
    ev = "experimental_reaction" if a in R.index and R.loc[a, "experimental_reactions"] else "reviewed_by_similarity"
    if a == "Q8ZXN7":
        ev = "characterized_literature_PMID31693208"
    rows.append(dict(accession=a, label=lab, branch=br, evidence=ev, length=len(seqs[a])))
I = pd.read_csv(H / "ref/uniprot_query/ipr053177.tsv", sep="\t")
I = I[(I.Reviewed == "unreviewed") & I.Length.between(320, 360)].copy()
I["phylum"] = I["Taxonomic lineage"].str.split(",").str[3].str.strip().str.replace(r" \(.*", "", regex=True)
I["domain"] = I["Taxonomic lineage"].str.split(",").str[1].str.strip().str.replace(r" \(.*", "", regex=True)
pick = I.groupby("phylum").head(1)
pick = pick[pick.domain != "Eukaryota"].head(14)
for r in pick.itertuples():
    t = urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{r.Entry}.fasta", timeout=60).read().decode()
    s = "".join(t.splitlines()[1:])
    org = re.sub(r"[^A-Za-z]", "", r.Organism.split()[0])[:10]
    seqs[r.Entry] = s
    rows.append(dict(accession=r.Entry, label=f"ADPGPlike_{org}_{r.phylum.split()[0][:12]}", branch="ADPGP_IPR053177",
                     evidence="unreviewed_branch_context", length=len(s)))
T = pd.DataFrame(rows)
T.to_csv(H / "data/ref_panel.tsv", sep="\t", index=False)
with open(H / "ref/refs.faa", "w") as fh:
    for r in T.itertuples():
        fh.write(f">REF|{r.branch}|{r.label}|{r.accession}\n{seqs[r.accession]}\n")
print(T.to_string(index=False))
