#!/usr/bin/env python3
"""Fetch PDB entry titles by 4-character ID from the RCSB GraphQL API (IDs only; no sequence leaves the machine).
Usage: python3 sc2_pdb_titles.py IDS.txt OUT.tsv  (IDs one per line; existing rows in OUT.tsv are kept)."""
import json
import os
import sys
import time
import urllib.request

ids = sorted({l.strip()[:4].lower() for l in open(sys.argv[1]) if l.strip()})
have = {}
if os.path.exists(sys.argv[2]):
    for l in open(sys.argv[2]):
        k, _, v = l.rstrip("\n").partition("\t")
        have[k] = v
todo = [i for i in ids if i not in have]
for k in range(0, len(todo), 200):
    chunk = todo[k:k + 200]
    q = "{ entries(entry_ids: %s) { rcsb_id struct { title } } }" % json.dumps([c.upper() for c in chunk])
    req = urllib.request.Request("https://data.rcsb.org/graphql", data=json.dumps({"query": q}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "elsa-showcase2/0.1"})
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    for e in d["data"]["entries"] or []:
        if e:
            have[e["rcsb_id"].lower()] = ((e.get("struct") or {}).get("title") or "").replace("\t", " ")
    time.sleep(0.5)
with open(sys.argv[2], "w") as f:
    for k in sorted(have):
        f.write(f"{k}\t{have[k]}\n")
print(len(ids), "ids;", len(todo), "fetched;", len(have), "titles total")
