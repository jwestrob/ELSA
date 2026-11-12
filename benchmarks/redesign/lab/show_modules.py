#!/usr/bin/env python
"""Module quality of engine runs on the DPANN gold loci, from the scorer's instance table.

usage: show_modules.py <result dir> [<result dir> ...]
For each run and caller: F1, excess, share of instances whose best locus has a module, number of
distinct modules, AMI / homogeneity / completeness of that module against the system type, and
the scorer's instance-weighted plurality consistency with its control.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_completeness_v_measure

for d in map(Path, sys.argv[1:]):
    r = json.load(open(d / "summary.json"))
    s = json.load(open(d / "score_min2" / "summary.json"))
    m = r["modules"]
    print(f"{d.parent.name}/{d.name}: engine {r.get('engine')} modules {m.get('modules'):,} largest {m.get('largest'):,} "
          f"method {m.get('method', 'uf')} | significant pairs {r['significant_pairs']:,} | loci {r.get('loci_all_genomes')}")
    t = pd.read_parquet(d / "score_min2" / "instance_scores.parquet")
    for c in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
        v = s["by_caller"][c]
        k = t[t.caller == c]
        x = k[k.best_module.notna()]
        ty, mo = x.system_type.astype(str), x.best_module.astype(str)
        ami = adjusted_mutual_info_score(ty, mo) if len(x) > 1 else float("nan")
        h, cp, _ = homogeneity_completeness_v_measure(ty, mo) if len(x) > 1 else (0.0, 0.0, 0.0)
        cons = v.get("module_consistency_weighted", float("nan"))
        ctrl = v.get("control_module_consistency_weighted", float("nan"))
        print(f"   {c:<15} F1 {v['mean_best_f1']:.3f} excess {v['f1_minus_control']:+.3f} | best locus has a module: "
              f"{len(x):,}/{len(k):,}, distinct {x.best_module.nunique()}, AMI {ami:.3f} hom {h:.2f} comp {cp:.2f} | "
              f"plurality consistency {cons:.3f} (control {ctrl:.3f})")
