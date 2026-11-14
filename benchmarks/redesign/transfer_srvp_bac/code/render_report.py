#!/usr/bin/env python
"""Render the descriptive tables of REPORT.txt from the package receipts (describe.json, universe.json,
strandqc.json, gold.json, taxonomy.json, split.json, token JSONs). Prints plain text; no method output is read.

usage: render_report.py RECEIPTS_DIR [TOKENS_DIR]
"""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1])
TK = Path(sys.argv[2]) if len(sys.argv) > 2 else None
d = json.loads((R / "describe.json").read_text())
f = lambda x: f"{int(round(x)):,}" if isinstance(x, (int, float)) else str(x)
out = []
u = d["universe"]
out += ["UNIVERSE",
        f"  genomes {f(u['genomes'])}; contigs {f(u['contigs'])} ({f(u['circular_contigs'])} circular); proteins {f(u['proteins'])}"
        f" (+ {f(u['strand_plus'])} / - {f(u['strand_minus'])}); partial flags {u['partial']}; truncated {f(u['truncated'])}",
        "  contig length in genes, quantiles 0/1/5/25/50/75/95/99/100%: "
        + " / ".join(f(v) for v in u["contig_genes"].values()),
        "  contig length in genes, bins: " + ", ".join(f"{k} {v}" for k, v in u["contig_genes_hist"].items()),
        "  contig length in nt, median " + f(u["contig_nt"]["0.5"]) + ", range " + f(u["contig_nt"]["0"]) + "-" + f(u["contig_nt"]["1"]),
        "  protein length aa, median " + f(u["protein_length_aa"]["0.5"]) + ", 99% " + f(u["protein_length_aa"]["0.99"])
        + ", max " + f(u["protein_length_aa"]["1"]),
        f"  order-trusted genomes {u['order_trusted_genomes']} of {u['genomes']}; flags {u['order_flags']}"]
t = d["taxonomy"]
out += ["TAXONOMY (" + ", ".join(f"{k} {v}" for k, v in t["tax_source"].items()) + ")",
        "  distinct: " + ", ".join(f"{r} {t[r]['distinct']} ({t[r]['genomes_named']} genomes named)"
                                   for r in ("phylum", "class", "ord", "family", "genus", "species")),
        "  genomes per phylum: " + ", ".join(f"{k} {v}" for k, v in t["genomes_per_phylum"].items()),
        "  completeness q0/5/50/95/100: " + " / ".join(f"{v:.1f}" for v in t["completeness"].values())
        + "; contamination: " + " / ".join(f"{v:.2f}" for v in t["contamination"].values())]
s = d["split"]
out += [f"SPLIT (test = even genome_code): genomes {s['genomes']}, genes {s['genes']}, phyla {s['phyla']}"]
out += ["GOLD (evaluation set = resolved, order trusted, >= 2 resolved members; span = first-to-last member, genes)",
        "  caller | instances | resolved | single-member | >=2 members | types | genomes | test/tune | span q50/q90/q99/max | members q50/max | contiguous"]
for c, g in d["gold"].items():
    sp = g["eval_span_genes"]
    out.append(f"  {c} | {f(g['instances'])} | {f(g['resolved'])} | {f(g['resolved_single_member'])} | {f(g['eval_set_ge2_members'])}"
               f" | {g['eval_set_system_types']} | {g['eval_set_genomes']} | {g['eval_set_by_half'].get('test', 0)}/{g['eval_set_by_half'].get('tune', 0)}"
               f" | {sp.get('0.5', 0):g}/{sp.get('0.9', 0):g}/{sp.get('0.99', 0):g}/{sp.get('1', 0):g}"
               f" | {g['eval_members'].get('0.5', 0):g}/{g['eval_members'].get('1', 0):g} | {g['eval_contiguous']}")
for c, g in d["gold"].items():
    out.append(f"  {c}: size bins {g['eval_size_bins']}; top types {g['eval_top_types']}" + (f"; status {g['status']}" if g["status"] else ""))
if "ribosomal" in d:
    r = d["ribosomal"]
    out += [f"RIBOSOMAL RUNS (Pfam ^ribosom(al)?_ markers: {f(r['marker_hits'])} hits on {f(r['marked_genes'])} genes, {r['marker_names']} names)"]
    for k, v in r["sets"].items():
        out.append(f"  {k}: runs {v['runs']}, genomes {v['genomes_with_run']} (>= 2 runs: {v['genomes_with_2plus_runs']}),"
                   f" genes q0/10/25/50/75/90/100 {'/'.join(f'{x:g}' for x in v['n_genes_quantiles'])}, at contig edge {v['at_contig_edge']}")
    out.append(f"  g2_m16: phyla {r['g2_m16_phyla']}, genera {r['g2_m16_genera']}, by half {r['g2_m16_by_half']}")
out += ["FAMILIES"]
for k, v in d["families"].items():
    out += [f"  {k}: tokens {f(v['tokens'])}, singletons {f(v['singletons'])}, largest {f(v['largest'])},"
            f" size q50/q90/q99/q99.9 {'/'.join(f'{x:g}' for x in list(v['size_q'].values())[:4])};"
            f" tokens in >= 2 / 5 / 50 / 500 genomes {f(v['tokens_in_ge2_genomes'])} / {f(v['tokens_in_ge5_genomes'])} /"
            f" {f(v['tokens_in_ge50_genomes'])} / {f(v['tokens_in_ge500_genomes'])}",
            "    tokens by size: " + ", ".join(f"{a} {f(b)}" for a, b in v["size_hist"].items()),
            "    genes by family size: " + ", ".join(f"{a} {f(b)}" for a, b in v["genes_by_family_size"].items())]
if "embedding_run" in d:
    e = d["embedding_run"]
    out += [f"EMBEDDING RUN {e['name']}: rows {f(e['rows'])}, unique {f(e['unique_sequences'])}, residues unique {f(e['residues_unique'])},"
            f" shards {e['shards']}, > 2,046 aa {e['unique_longer_than_2046']}, substituted {e['substituted_symbols']}"]
    if "final" in e:
        fi = e["final"]
        out.append(f"  finalized {fi['finalized']}: unique failed {fi['unique_failed']}, rows without vector {fi['rows_without_vector']},"
                   f" GPU seconds {fi['embedding_seconds']}, shard audit min cos {fi['shard_audit']['min_cosine']}, assembly {fi['assembly']['passed']}")
if "zca10" in d:
    z = d["zca10"]
    out.append(f"ZCA10: unique {f(z['unique'])} (valid {f(z['valid'])}), global mean norm {z['transform'].get('global_mean_norm'):.3f},"
               f" participation ratio {z['transform'].get('participation_ratio_before'):.1f} -> {z['participation_ratio_after']:.1f},"
               f" check {z['check']}")
print("\n".join(out))
