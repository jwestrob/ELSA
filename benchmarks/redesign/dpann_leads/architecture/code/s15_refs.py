#!/usr/bin/env python3
"""Step 15: download public reference data by accession (UniProt records with residue features and sequences; RCSB PDB
entries). Nothing is sent but accessions. Writes refs/uniprot_features.tsv, refs/ref.faa, refs/pdb/<id>.pdb and
refs/pdb_titles.tsv (refs/ lives in the scratchpad; the tables are copied to data/)."""
import json, os, urllib.request, shutil
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
SP = os.environ.get("SP", "/private/tmp/claude-501/-Users-jacob-Documents-Sandbox-elsa-test-ELSA/4a153818-3d15-49e9-b3de-41cb21532c73/scratchpad")
RD = f"{SP}/ref"; os.makedirs(f"{RD}/pdb", exist_ok=True)
UNIPROT = {  # accession: role
    "P0A720": "Tmk E. coli (dTMP kinase, reviewed)",
    "P23919": "TMPK human (reviewed)",
    "Q9XBQ8": "KamA Clostridium subterminale (lysine 2,3-aminomutase, reviewed)",
    "Q6LYX4": "AblA Methanococcus maripaludis (lysine 2,3-aminomutase, reviewed)",
    "P0A3Y5": "APH(3')-IIIa Enterococcus faecalis (aminoglycoside phosphotransferase, reviewed)",
    "P35790": "CHKA human (choline kinase alpha, reviewed)",
    "P32173": "MobA E. coli (MobA-like NTP transferase, reviewed)",
    "Q7WYX5": "LicCA Treponema denticola (CTP:phosphocholine cytidylyltransferase/choline kinase, unreviewed)",
    "A0A0H2UQB5": "LicC Streptococcus pneumoniae TIGR4 (unreviewed)",
    "P39762": "AmpS Bacillus subtilis (M29 aminopeptidase, reviewed)",
    "P07874": "AlgA Pseudomonas aeruginosa (PMI/GMP bifunctional, reviewed)",
}
PDB = ["1JYK", "1JYL", "4R77", "4R78", "1J7L", "2CKQ", "4TMK", "1E2D", "2A5H", "1ZJC", "4ICQ", "1DZR", "1J58"]
rows, fa = [], open(f"{RD}/ref.faa", "w")
for acc, role in UNIPROT.items():
    d = json.load(urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{acc}.json"))
    seq = d["sequence"]["value"]; fa.write(f">{acc} {role}\n{seq}\n")
    for f in d.get("features", []):
        if f["type"] in ("Active site", "Binding site", "Site", "Metal binding", "Modified residue", "Motif", "Region", "Domain"):
            lig = (f.get("ligand") or {}).get("name", "")
            rows.append(dict(acc=acc, role=role, type=f["type"], start=f["location"]["start"]["value"], end=f["location"]["end"]["value"],
                             ligand=lig, note=f.get("description", ""), evidence=";".join(e.get("evidenceCode", "") for e in f.get("evidences", []))))
fa.close()
import pandas as pd
F = pd.DataFrame(rows); F.to_csv(f"{RD}/uniprot_features.tsv", sep="\t", index=False); shutil.copy(f"{RD}/uniprot_features.tsv", "data/uniprot_features.tsv")
tit = []
for p in PDB:
    fn = f"{RD}/pdb/{p}.pdb"
    if not os.path.exists(fn):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{p}.pdb", fn)
    title = " ".join(l[10:].strip() for l in open(fn) if l.startswith("TITLE"))
    tit.append(dict(pdb=p, title=title))
pd.DataFrame(tit).to_csv("data/pdb_titles.tsv", sep="\t", index=False)
print(F.groupby(["acc", "type"]).size().to_string())
print(pd.DataFrame(tit).to_string(index=False))
