#!/usr/bin/env python3
"""Model metadata tables for the annotation Parquets (single process, login node).

pfam_meta.tsv : name, accession, description  (NAME / ACC / DESC lines of the Pfam-A.hmm that Aksha's PFAM uses)
kofam_meta.tsv: name (KO), accession (KO), threshold, score_type, definition  (ko_list beside the KOFAM database)
Usage: build_meta.py PFAM_A_HMM KO_LIST OUTDIR
"""
import csv
import hashlib
import json
import os
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    pfam_hmm, ko_list, outdir = sys.argv[1:4]
    os.makedirs(outdir, exist_ok=True)
    proc = subprocess.run(
        ["grep", "-E", "^(NAME|ACC|DESC) ", pfam_hmm],
        check=True, capture_output=True, text=True, env={"LC_ALL": "C"},
    )
    entries, cur = [], {}
    for line in proc.stdout.splitlines():
        tag, val = line[:4], line[4:].strip()
        if tag == "NAME":
            if cur:
                entries.append(cur)
            cur = {"name": val}
        elif tag == "ACC ":
            cur["accession"] = val
        elif tag == "DESC":
            cur["description"] = val
    if cur:
        entries.append(cur)
    names = [e["name"] for e in entries]
    assert len(names) == len(set(names)), "duplicate Pfam NAME"
    with open(os.path.join(outdir, "pfam_meta.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["name", "accession", "description"])
        for e in entries:
            w.writerow([e["name"], e.get("accession", ""), e.get("description", "")])
    kos = []
    with open(ko_list) as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for r in rd:
            kos.append(r)
    with open(os.path.join(outdir, "kofam_meta.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["name", "accession", "threshold", "score_type", "definition"])
        for r in kos:
            w.writerow([r["knum"], r["knum"], r["threshold"], r["score_type"], r["definition"]])
    info = {
        "pfam_hmm": pfam_hmm,
        "pfam_hmm_sha256": sha256(pfam_hmm),
        "pfam_models": len(entries),
        "ko_list": ko_list,
        "ko_list_sha256": sha256(ko_list),
        "ko_entries": len(kos),
        "ko_without_threshold": sum(1 for r in kos if r["threshold"] in ("-", "")),
    }
    with open(os.path.join(outdir, "meta_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
