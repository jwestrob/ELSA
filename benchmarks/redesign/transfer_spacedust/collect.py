#!/usr/bin/env python
"""Tables for REPORT.txt from results/*.json (workstream transfer_spacedust). Writes results/table.txt; prints it."""
import json
import sys
from pathlib import Path

TS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111/transfer_spacedust")
R = TS / "results"


def j(p):
    p = R / p
    return json.load(open(p)) if p.exists() else None


def f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}".replace("0.", ".", 1) if abs(x) < 1 else f"{x:.{d}f}"


def ci(v):
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"


lines = []
ref = j("refcheck.json")
rows = []
if ref:
    for nm, r in (("engine reference (best of 42, hybrid w8)", ref["engine_reference"]),
                  ("engine single reading (conserved level 2)", ref["engine_single_reading"]),
                  ("native Spacedust (all blocks)", ref["spacedust"])):
        rows.append((nm, r, None))
for nm, fn in (("engine, rescue3 tokens, w3 (union)", "ref_engine_r3_w3.json"), ("engine, rescue3 tokens, w8 (union)", "ref_engine_r3_w8.json")):
    x = j(fn)
    if x:
        rows.append((nm, x["union"], None))
        rows.append((nm.replace("(union)", "(single, level 2)"), x["single"], None))
scores = {}
for tag in ("P2", "P1", "P3"):
    s = j(f"score_{tag}.json")
    if s:
        scores[tag] = s
        for rd, it in s["readings"].items():
            lab = f"{tag} {rd}" + (" PRIMARY" if rd == s["primary"] else (" secondary (CC baseline)" if rd.startswith("CC") else " secondary reading"))
            rows.append((lab, it["score"], it))
lines.append("source207, matched protocol (F1 / control / excess [95%] / median boundary precision / median coverage / query genes in loci / intervals)")
lines.append(f"{'output':52s} {'F1':>6s} {'ctrl':>6s} {'excess [95%]':>24s} {'bprec':>6s} {'cov':>5s} {'qgenes':>6s} {'n_int':>7s}")
for nm, r, it in rows:
    lines.append(f"{nm:52s} {f(r['f1_observed']):>6s} {f(r['f1_random']):>6s} {r['f1_excess']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                 f"{f(r.get('median_boundary_precision')):>6s} {f(r.get('median_coverage'), 2):>5s} {f(r.get('query_gene_coverage')):>6s} {r['n_intervals']:>7d}")
lines.append("")
lines.append("paired differences (A - reference; F1 and excess, 95% region bootstrap; verdict on excess)")
for tag, s in scores.items():
    for rd, it in s["readings"].items():
        parts = []
        for k, v in it.items():
            if k.startswith("vs_"):
                parts.append(f"{k[3:]}: F1 {ci(v['paired']['f1'])} exc {ci(v['paired']['excess'])} {v['verdict']}")
        lines.append(f"{tag} {rd}")
        for p in parts:
            lines.append("    " + p)
lines.append("")
lines.append("by gold-region size (F1 / control / excess; median best-interval span vs median region size)")
for nm, r, it in rows:
    bs = r.get("by_size")
    if not bs:
        continue
    lines.append(f"{nm:52s} " + "  ".join(f"{k}: {f(v['f1'])}/{f(v['control'])}/{v['excess']:+.3f} span {v['best_span_q50']:.0f}/{v['size_q50']:.0f}"
                                         for k, v in bs.items()))
for fn in ("ladder_syn.json", "ladder_group_genome.json"):
    x = j(fn)
    if x:
        lines.append("")
        for spec, res in x["results"].items():
            lines.append(f"tau ladder {spec}: tau* = {res['tau_star']}")
            for L in res["ladder"]:
                lines.append(f"   tau {L['tau']:>5g}  obs types {L['obs_types']:>7d}  null mean {L['null_types_mean']:>9.1f}  FDR {L['fdr_types']:.5f}"
                             f"  genes in words {L['genes_frac']:.3f}  rounds {L['rounds']}  {L['obs_seconds']:.0f} s  {L['obs_rss_gb']:.1f} GB")
for tag, s in scores.items():
    lines.append("")
    v = s["vocab"]
    lines.append(f"{tag} vocabulary: types {v['word_types']:,}, instances {v['instances']:,}, top-level {v['top_instances']:,} "
                 f"({v['top_types']:,} types), genes in top-level words {v['genes_in_top_frac']:.3f} (slots {v['slot_genes']:,}, "
                 f"inserted {v['inserted_genes']:,}); top span q10/q25/q50/q75/q90/q99/max {v['top_span_q10_q25_q50_q75_q90_q99_max']}; "
                 f"on query genomes {v['query_top_instances']:,} instances, span q50/q90/q99/max {v['top_span_query_q50_q90_q99_max']}; "
                 f"max slots {v['max_slots']}, depth {v['types_by_depth']}; Lambda q {[round(x, 1) for x in v['lam_q01_q10_q50_q90_q99']]}; "
                 f"units per word q10/q50/q90/q99 {v['n_units_q10_q50_q90_q99']}; rounds {v['rounds']}")
    lines.append(f"{tag} gold member genes outside every top-level word: {s['gold_member_genes_outside_words']:.3f}")
    lines.append(f"{tag} phrases: {json.dumps(s['info']['phrases'])}")
    for rd, it in s["readings"].items():
        lines.append(f"   {rd:10s} loci {it['loci']['loci']:>9,}  genes in loci {it['loci']['genes_in_loci_frac']:.3f}  query genes {it['loci']['query_genes_in_loci_frac']:.3f}"
                     f"  span q10/q50/q90/q99/max {it['loci']['span_q10_q50_q90_q99_max']}  query span {it['loci']['query_span_q10_q50_q90_q99_max']}")
g = j("../groups/groups.json") if (TS / "groups/groups.json").exists() else None
if (TS / "groups/groups.json").exists():
    g = json.load(open(TS / "groups/groups.json"))
    lines.append("")
    lines.append(f"genome groups at 0.75: {g['groups']} groups (largest {g['largest']}, singletons {g['singletons']}, size q50/q90/q99 {g['size_q50_q90_q99']}); "
                 f"emb_hom pairs q50/q90/q99/q999 {[round(x, 3) for x in g['pair_q50_q90_q99_q999']]}; nn q10/q50/q90 {[round(x, 3) for x in g['nn_q10_q50_q90']]}; "
                 f"AMI GTDB family {g['ami_family_ref']:.3f}, genus {g['ami_genus_ref']:.3f}; query genome groups {g['query_genome_groups']} sizes {g['query_group_sizes']}")
st = j("stress.json")
if st:
    lines.append("")
    lines.append(f"stress test: clade {st['clade']}")
    for k, m in st["runs"].items():
        lines.append(f"   {k:10s} " + json.dumps(m))
out = "\n".join(lines) + "\n"
(R / "table.txt").write_text(out)
print(out)
