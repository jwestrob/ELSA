#!/usr/bin/env python
"""Task 1 table: per arm, even-genome (test) F1 and excess of the three callers and the summed DefenseFinder +
TXSScan excess difference to the base with its genome-bootstrap interval (parsed from compare_vs_base.txt), the
odd-genome summed difference, token count and the ribosomal guard."""
import json
import re
from pathlib import Path

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
H = RD / "engine/results_dpann_hier"
ARMS = [("V0 r19000 s17 (context_kernel)", RD / "engine/results_dpann_ctx/dpann__ctx__esmpp_small__sim_all_r19000__w3",
         RD / "hierarchy/ribo/ctx_r19000" if (RD / "hierarchy/ribo/ctx_r19000").exists() else H / "hier__single__r19000__w3"),
        ("V0 r19000 s18", H / "hier__t1__v0_r19000_s18__w3", None), ("V0 r19000 s19", H / "hier__t1__v0_r19000_s19__w3", None),
        ("V0 r19000 s20", H / "hier__t1__v0_r19000_s20__w3", None),
        ("V0 r13000 s17", H / "hier__t1__v0_r13000_s17__w3", None), ("V0 r13500 s17", H / "hier__t1__v0_r13500_s17__w3", None),
        ("V0 r11000 s17", RD / "engine/results_dpann_ctx/dpann__ctx__esmpp_small__sim_all_r11000__w3", RD / "hierarchy/ribo/ctx_r11000"),
        ("V0 r15000 s17", RD / "engine/results_dpann_ctx/dpann__ctx__esmpp_small__sim_all_r15000__w3", RD / "hierarchy/ribo/ctx_r15000"),
        ("G1 r3000", H / "hier__t1__g1_r3000_s17__w3", None), ("G1 r19000", H / "hier__t1__g1_r19000_s17__w3", None),
        ("G1 r30000", H / "hier__t1__g1_r30000_s17__w3", None),
        ("G3 r19000", H / "hier__t1__mutual_r19000_s17__w3", None), ("G3 r30000", H / "hier__t1__mutual_r30000_s17__w3", None)]
PAT = re.compile(r"(tune \(odd\)|test \(even\))\s+(\w+)\s+n=\s*[\d,]+\s+F1 A ([\d.]+) B ([\d.]+) A-B ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]"
                 r"\s+excess A ([+-][\d.]+) B ([+-][\d.]+) A-B ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]")
SUM = re.compile(r"(tune \(odd\)|test \(even\))\s+DefenseFinder\+TXSScan summed excess: A ([+-][\d.]+) B ([+-][\d.]+)\s+A-B ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]")


def main():
    rows = []
    for name, d, rib in ARMS:
        f = d / "compare_vs_base.txt"
        if not f.exists():
            continue
        txt = f.read_text()
        r = dict(arm=name, dir=d.name)
        for m in PAT.finditer(txt):
            if m.group(1).startswith("test"):
                c = m.group(2)
                r[f"{c}_F1"] = float(m.group(3))
                r[f"{c}_dF1"] = float(m.group(5))
                r[f"{c}_excess"] = float(m.group(8))
                r[f"{c}_dex"] = f"{m.group(10)} [{m.group(11)}, {m.group(12)}]"
        for m in SUM.finditer(txt):
            key = "test" if m.group(1).startswith("test") else "tune"
            r[f"{key}_summed_d"] = f"{m.group(4)} [{m.group(5)}, {m.group(6)}]"
        s = json.load(open(d / "summary.json"))
        r["tokens"] = s.get("tokens")
        g = (rib or d) / "ribosomal_check.json"
        if g.exists():
            p = json.load(open(g))
            pr = p["primary"]
            r["rib"] = f"{pr['recovered']}/1686 Parv {pr['parvarchaeaceae']} res@2 {pr['resolved']} unit {p['results']['g2_m16']['unit@0.5']['resolved']}"
            r["admissible"] = pr["recovered"] >= 1650 and int(str(pr["parvarchaeaceae"]).split("/")[0]) >= 110
        rows.append(r)
    for r in rows:
        print(json.dumps(r))
    json.dump(rows, open(RD / "hierarchy/t1_table.json", "w"), indent=1)


if __name__ == "__main__":
    main()
