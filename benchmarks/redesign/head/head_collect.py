#!/usr/bin/env python
"""Collect the head engine arms: compare_vs_base.txt (vs dpann__mmseqs__seq__e3c50s75__w3), summary.json, ribosomal check,
admissibility (>= 1,650 of 1,686 runs and >= 110 of 114 Parvarchaeaceae recovered) and the predeclared choice (highest
odd-genome summed excess difference among admissible arms). Login-node safe (small files).
usage: head_collect.py
"""
import json
import re
import sys
from pathlib import Path

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
sys.path.insert(0, str(RD / "context_kernel/code"))
from collect import parse  # noqa: E402

pat_r = re.compile(r"runs=(\d+) recovered=(\d+) resolved=(\d+).*top_module_purity=([\d.]+) Parvarchaeaceae=(\d+)/(\d+)"
                   r".*resolved conserved@80=(\d+) unit@0.5=(\d+)")


def main():
    runs = sorted((RD / "engine/results_dpann_head").glob("dpann__head__*__w3"))
    rows = []
    for r in runs:
        if not (r / "compare_vs_base.txt").exists():
            continue
        row = parse(r)
        t = (r / "ribosomal_check.txt").read_text() if (r / "ribosomal_check.txt").exists() else ""
        m = pat_r.search(t)
        if m:
            row.update(ribo_runs=int(m.group(1)), ribo_recovered=int(m.group(2)), ribo_resolved=int(m.group(3)),
                       ribo_purity=float(m.group(4)), parv=int(m.group(5)), ribo_resolved80=int(m.group(7)),
                       ribo_unit=int(m.group(8)))
            row["admissible"] = bool(row["ribo_recovered"] >= 1650 and row["parv"] >= 110)
        else:
            row["admissible"] = None
        arm = r.name[len("dpann__"):-len("__w3")]
        man = json.loads((RD / "engine/dpann_tokens" / f"{arm}.json").read_text())
        row["resolution"] = man["resolution"]
        rows.append(row)
    rows.sort(key=lambda x: x["resolution"])
    adm = [x for x in rows if x.get("admissible")]
    choice = max(adm, key=lambda x: x["tune_sum_d"]) if adm else None
    out = dict(rows=rows, admissible=[x["run"] for x in adm], choice=choice["run"] if choice else None)
    if choice:
        d = choice["test_sum_d"]
        out["H2"] = dict(test_sum_d=d, ci=choice["test_sum_d_ci"], met=bool(d >= -0.01), beats=bool(d >= 0.02))
    else:
        out["H2"] = dict(met=False, reason="no admissible arm")
    (RD / "head/engine/engine_results.json").write_text(json.dumps(out, indent=1) + "\n")
    cols = ["run", "resolution", "tokens", "engine_fdr", "tune_sum_d", "test_sum_d", "test_sum_d_ci", "test_DF_F1_A", "test_DF_dex",
            "test_TXSS_F1_A", "test_TXSS_dex", "test_CCT_F1_A", "test_CCT_dex", "ribo_recovered", "parv", "ribo_resolved",
            "admissible"]
    with open(RD / "head/engine/engine_results.tsv", "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for x in rows:
            fh.write("\t".join(str(x.get(k, "")) for k in cols) + "\n")
    print((RD / "head/engine/engine_results.tsv").read_text())
    print("choice", out["choice"], json.dumps(out["H2"]))


if __name__ == "__main__":
    main()
