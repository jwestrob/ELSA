#!/usr/bin/env python
"""galt step 2: curated reference table for the GalT-like branch of the HIT superfamily.

Input: reviewed UniProt records (ref/uniprot_json/*.json), fetched by accession from rest.uniprot.org after queries
xref:pfam-PF01087 / PF02744 / PF19327 / PF09830 and xref:interpro-IPR026506 / IPR001937 / IPR009163 (reviewed:true),
plus HIT / FHIT outgroups (P49773, P49789, P0ACE7, P49775).
Per record: EC, catalytic-activity reactions with their evidence codes (ECO:0000269 = experimental) and PubMed IDs,
active-site and binding-site features, PDB cross references, Pfam / InterPro.
Branch = InterPro family assignment (class I GalT IPR001937 without IPR053177; class II GalT IPR000766; ADP-glucose
phosphorylase IPR053177; Ap4A phosphorylase IPR009163; GDP-hexose phosphorylase IPR026506; HIT / FHIT outgroup).
Writes data/ref_records.tsv and data/ref_features.tsv.
"""
import json
from pathlib import Path
import pandas as pd

H = Path("/Users/jacob/Documents/Sandbox/elsa_test/ELSA/benchmarks/redesign/dpann_leads/galt")
rows, feats = [], []
for p in sorted((H / "ref/uniprot_json").glob("*.json")):
    j = json.load(open(p))
    acc = j["primaryAccession"]
    ipr = [x["id"] for x in j.get("uniProtKBCrossReferences", []) if x["database"] == "InterPro"]
    pf = [x["id"] for x in j.get("uniProtKBCrossReferences", []) if x["database"] == "Pfam"]
    pdb = [x["id"] for x in j.get("uniProtKBCrossReferences", []) if x["database"] == "PDB"]
    if "IPR053177" in ipr:
        br = "ADP-glucose phosphorylase (IPR053177)"
    elif "IPR009163" in ipr:
        br = "Ap4A phosphorylase (IPR009163)"
    elif "IPR026506" in ipr:
        br = "GDP-hexose phosphorylase (IPR026506)"
    elif "IPR001937" in ipr:
        br = "GalT class I (IPR001937)"
    elif "IPR000766" in ipr:
        br = "GalT class II (IPR000766)"
    else:
        br = "outgroup HIT/FHIT"
    rx, exp_rx, pmids = [], [], set()
    for c in j.get("comments", []):
        if c["commentType"] == "CATALYTIC ACTIVITY":
            r = c["reaction"]
            ev = [e.get("evidenceCode") for e in r.get("evidences", [])]
            pm = [e.get("id") for e in r.get("evidences", []) if e.get("source") == "PubMed"]
            rx.append(r["name"])
            if "ECO:0000269" in ev:
                exp_rx.append(r["name"])
                pmids.update(pm)
    func = " ".join(t["value"] for c in j.get("comments", []) if c["commentType"] == "FUNCTION" for t in c.get("texts", []))
    sub = " ".join(t["value"] for c in j.get("comments", []) if c["commentType"] == "SUBUNIT" for t in c.get("texts", []))
    ec = ";".join(e["value"] for e in j.get("proteinDescription", {}).get("recommendedName", {}).get("ecNumbers", []))
    for f in j.get("features", []):
        if f["type"] in ("Active site", "Binding site", "Site"):
            lig = f.get("ligand", {}).get("name", "")
            s, e = f["location"]["start"]["value"], f["location"]["end"]["value"]
            ev = ",".join(sorted({x.get("evidenceCode", "") for x in f.get("evidences", [])}))
            feats.append(dict(accession=acc, type=f["type"], start=s, end=e, ligand=lig, description=f.get("description", ""),
                              evidence=ev, residues=j["sequence"]["value"][s - 1:e]))
    rows.append(dict(accession=acc, entry=j["uniProtkbId"], branch=br, organism=j["organism"]["scientificName"],
                     length=j["sequence"]["length"], ec=ec, n_reactions=len(rx), experimental_reactions=" | ".join(exp_rx),
                     reactions=" | ".join(rx), exp_pubmed=";".join(sorted(pmids)), pdb=";".join(pdb), pfam=";".join(pf),
                     interpro=";".join(ipr), subunit=sub[:300], function=func[:400], sequence=j["sequence"]["value"]))
R = pd.DataFrame(rows)
R.to_csv(H / "data/ref_records.tsv", sep="\t", index=False)
pd.DataFrame(feats).to_csv(H / "data/ref_features.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print(R.groupby("branch").agg(n=("accession", "size"), with_exp_rx=("experimental_reactions", lambda s: (s != "").sum()),
                             with_pdb=("pdb", lambda s: (s != "").sum())))
print(R[R.experimental_reactions != ""][["accession", "entry", "branch", "length", "experimental_reactions", "exp_pubmed", "pdb"]].to_string())
