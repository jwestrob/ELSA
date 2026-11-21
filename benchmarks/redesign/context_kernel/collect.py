#!/usr/bin/env python
"""Collect engine results of the context-kernel arms (and reference arms) into one TSV.

Reads <run>/compare_vs_base.txt (paired against the MMseqs2 base) and <run>/summary.json for each
run directory given, plus the token manifest of the arm. Login-node safe (a few small files).
usage: collect.py out.tsv <run dir> [<run dir> ...]
"""
import json
import re
import sys
from pathlib import Path

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
num = r"([+-]?\d+\.\d+)"
pat_c = re.compile(rf"^\s+(all|tune \(odd\)|test \(even\))\s+(DefenseFinder|TXSScan|CRISPRCasTyper)\s+n=\s*([\d,]+)\s+F1 A {num} B {num} A-B {num} \[{num}, {num}\]\s+excess A {num} B {num} A-B {num} \[{num}, {num}\]")
pat_s = re.compile(rf"^\s+(all|tune \(odd\)|test \(even\))\s+DefenseFinder\+TXSScan summed excess: A {num} B {num}\s+A-B {num} \[{num}, {num}\]")
SPL = {"all": "all", "tune (odd)": "tune", "test (even)": "test"}


def parse(run: Path):
    row = {"run": run.name}
    txt = (run / "compare_vs_base.txt").read_text().splitlines()
    for line in txt:
        m = pat_c.match(line)
        if m:
            sp, c = SPL[m.group(1)], {"DefenseFinder": "DF", "TXSScan": "TXSS", "CRISPRCasTyper": "CCT"}[m.group(2)]
            g = m.groups()
            row[f"{sp}_{c}_F1_A"] = float(g[3])
            row[f"{sp}_{c}_dF1"] = f"{g[5]} [{g[6]}, {g[7]}]"
            row[f"{sp}_{c}_exA"] = float(g[8])
            row[f"{sp}_{c}_dex"] = float(g[10])
            row[f"{sp}_{c}_dex_ci"] = f"[{g[11]}, {g[12]}]"
            continue
        m = pat_s.match(line)
        if m:
            sp = SPL[m.group(1)]
            row[f"{sp}_sum_A"] = float(m.group(2))
            row[f"{sp}_sum_d"] = float(m.group(4))
            row[f"{sp}_sum_d_ci"] = f"[{m.group(5)}, {m.group(6)}]"
    s = json.loads((run / "summary.json").read_text())
    row.update(tokens=s["tokens"], significant_pairs=s["significant_pairs"], engine_fdr=round(s["fdr"], 5),
               genes_covered_frac=round(s["genes_covered_frac"], 4), loci_conserved=s["loci_all_genomes"]["conserved"])
    return row


def main():
    out = Path(sys.argv[1])
    rows = [parse(Path(d)) for d in sys.argv[2:] if (Path(d) / "compare_vs_base.txt").exists()]
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")
    short = ["run", "tokens", "significant_pairs", "engine_fdr", "tune_sum_d", "test_DF_dex", "test_DF_dex_ci", "test_TXSS_dex",
             "test_TXSS_dex_ci", "test_sum_d", "test_sum_d_ci", "test_CCT_dex"]
    for r in rows:
        print("\t".join(str(r.get(k, "")) for k in short))


if __name__ == "__main__":
    main()
