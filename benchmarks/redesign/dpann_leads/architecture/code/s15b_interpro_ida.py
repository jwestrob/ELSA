#!/usr/bin/env python3
"""Step 15b: curated-record rate of each lead architecture: InterPro domain architectures (IDA) of the label family,
fetched by Pfam accession; counts of proteins whose IDA contains the partner family (or the family twice)."""
import json, urllib.request
def allida(pf):
    url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{pf}/?ida&page_size=200"; out = []
    while url:
        d = json.load(urllib.request.urlopen(url)); out += d["results"]; url = d.get("next")
    return out
for pf, partner in [("PF14907", "PF02223"), ("PF01050", "PF01050"), ("PF12804", "PF01633"), ("PF12804", "PF01636"), ("PF01050", "PF22640")]:
    r = allida(pf); tot = sum(x["unique_proteins"] for x in r)
    hit = [(x["unique_proteins"], x["ida"]) for x in r if partner in x["ida"]]
    if partner == pf: hit = [(n, i) for n, i in hit if i.count(pf) >= 2]
    print(pf, "proteins over IDAs", tot, "| with", partner, sum(n for n, _ in hit), hit[:6])
