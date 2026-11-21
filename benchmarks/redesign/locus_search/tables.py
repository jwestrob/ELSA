#!/usr/bin/env python
"""Print report tables (Markdown) from mirrored small outputs: phase0_headroom.json, best_configs.json,
summary.json, phase2_summary.json. usage: tables.py <dir with the mirrored files>"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])


def ci(t, k=3):
    return f"{t[0]:.{k}f} [{t[1]:.{k}f}, {t[2]:.{k}f}]"


def sci(t, k=3):
    return f"{t[0]:+.{k}f} [{t[1]:+.{k}f}, {t[2]:+.{k}f}]"


if (d / "phase0_headroom.json").exists():
    h = json.load(open(d / "phase0_headroom.json"))
    print("random-pair cosine quantiles", h["random_pair_cosine_quantiles"])
    print("\n| Caller | Ordered pairs | Types | f_tok mean | tok-visible (>=0.5) | tok zero | f_cos >=0.5 | f_rank >=0.5 | emb (cos or rank) >=0.5 | union >=0.5 | tok-invisible & emb-visible |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
        x = h[c]
        print(f"| {c} | {x['ordered_pairs']:,} | {x['types']} | {x['f_tok']['mean']:.3f} | {x['f_tok']['frac_ge_half']:.3f} | "
              f"{x['f_tok']['frac_zero']:.3f} | {x['f_cos']['frac_ge_half']:.3f} | {x['f_rank']['frac_ge_half']:.3f} | "
              f"{x['f_emb']['frac_ge_half']:.3f} | {x['union_ge_half']:.3f} | {x['token_invisible_and_f_emb_ge_half']:.3f} |")
    for c in ("DefenseFinder",):
        print(f"\n{c} macro over types:", {k: round(v, 3) for k, v in h[c]["macro_over_types"].items()})
        print("| Type | Pairs | f_tok | tok-visible | f_emb | emb-visible | union-visible |\n|---|---|---|---|---|---|---|")
        for r in h[c]["per_type_top"]:
            print(f"| {r['type']} | {r['pairs']:,} | {r['f_tok']:.3f} | {r['vis_tok']:.3f} | {r['f_emb']:.3f} | {r['vis_emb']:.3f} | {r['vis_union']:.3f} |")

if (d / "best_configs.json").exists():
    b = json.load(open(d / "best_configs.json"))
    print("\n| Caller | Method | mode | w | W extra | tau | odd MAP | odd MAP strict |\n|---|---|---|---|---|---|---|---|")
    for c, mm in b.items():
        for m, x in mm.items():
            print(f"| {c} | {m} | {x['mode']} | {x['w']:g} | {x['wx']} | {x['tau']:g} | {x['odd_map']:.4f} | {x['odd_map_strict']:.4f} |")

for sf in sorted(d.glob("summary*.json")):
    s = json.load(open(sf))
    print("\n#####", sf.name)
    if "selection" not in s or "verdict" not in s.get("selection", {}):
        continue
    print("\nselection:", json.dumps(s["selection"]["odd_map"]), "->", s["selection"]["selected"])
    print("verdict:", json.dumps(s["selection"]["verdict"]))
    for key in sorted(s["tables"]):
        tab = s["tables"][key]
        print(f"\n### {key}\n| Method | n | MAP | MAP strict | recall E<=0.01 | recall E<=1 | recall inv E<=0.01 | recall inv E<=1 | "
              f"micro inv E<=0.01 | unlabelled/q E<=0.01 | unlabelled/q E<=1 | wrong/q E<=0.01 | precision E<=0.01 | strict prec E<=0.01 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for m in sorted(tab):
            x = tab[m]
            print(f"| {m} | {x['queries']} | {ci(x['ap'])} | {ci(x['ap_strict'])} | {ci(x['recall_E0.01'])} | {ci(x['recall_E1'])} | "
                  f"{ci(x['recall_inv_E0.01'])} | {ci(x['recall_inv_E1'])} | {x['micro_recall_inv_E0.01']:.3f} | "
                  f"{ci(x['n_unlabelled_E0.01'], 2)} | {ci(x['n_unlabelled_E1'], 2)} | {ci(x['n_wrong_E0.01'], 2)} | "
                  f"{ci(x['precision_E0.01'])} | {ci(x['precision_strict_E0.01'])} |")
        pr = s["paired"].get(key, {})
        if pr:
            print(f"\npaired vs M0 ({key})\n| Method | dMAP | dMAP strict | d recall E<=0.01 | d recall E<=1 | d recall inv E<=0.01 | d recall inv E<=1 | d unlabelled E<=0.01 |")
            print("|---|---|---|---|---|---|---|---|")
            for m in sorted(pr):
                x = pr[m]
                print(f"| {m} | {sci(x['ap'])} | {sci(x['ap_strict'])} | {sci(x['recall_E0.01'])} | {sci(x['recall_E1'])} | "
                      f"{sci(x['recall_inv_E0.01'])} | {sci(x['recall_inv_E1'])} | {sci(x['n_unlabelled_E0.01'], 2)} |")
    print("\ntiming (per query, one CPU):")
    for k, v in sorted(s["timing"].items()):
        print(f"  {k}: mean {v['mean_seconds']:.2f}s median {v['median_seconds']:.2f}s -> {v['queries_per_cpu_second']:.2f} queries/CPU-s")

for pf in sorted(d.glob("phase2_summary*.json")):
    p = json.load(open(pf))
    print("\n#####", pf.name)
    print("\nphase 2:", {k: p[k] for k in ("modules", "tokens", "tokens_with_centroid", "K", "R", "tau", "w", "even_genomes",
                                         "even_genes", "engine_full_loci_on_even")})
    for mode, r in p["runs"].items():
        print(f"\n{mode}: wall {r['wall_seconds']}s, CPU {r['cpu_seconds']}s, {r['even_genomes_per_wall_second']:.2f} genomes/s, hits saved {r.get('hits_saved', r.get('hits_E1')):,}")
        for thr in [k for k in r if isinstance(r[k], dict)]:
            x = r[thr]
            print(f"  {thr}: loci {x['loci']:,}, span sum {x.get('span_sum', x.get('genes_covered')):,}, module consistency {x['module_consistency']}")
            print("  | Caller | n | F1 profile | F1 engine | dF1 | excess profile | excess engine | d excess | covered50 P/E |\n  |---|---|---|---|---|---|---|---|---|")
            for y in x["paired_vs_engine"]:
                print(f"  | {y['caller']} | {y['instances']} | {y['f1']['A']:.3f} | {y['f1']['B']:.3f} | {sci(y['f1']['diff'])} | "
                      f"{y['excess']['A']:.3f} | {y['excess']['B']:.3f} | {sci(y['excess']['diff'])} | "
                      f"{y['covered50']['A']:.3f} / {y['covered50']['B']:.3f} |")
