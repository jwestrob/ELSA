#!/usr/bin/env python
"""Print compact tables from round-4 result JSONs: collect4.py results/odd_B.json [more.json ...] [--sort]"""
import json
import sys

C3 = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")


def row(name, it, split):
    sp = "odd" if split == "odd" else "even"
    prim = it.get("primary", "DH")
    r = it["readings"][prim]
    s = r["summary"][sp]
    rb = r["ribo"]["scored"]
    sc = it["score"]
    m = sc["margins"]
    f1 = "/".join(f"{s[c]['f1']:.3f}" for c in C3)
    bp = "/".join(("%.2f" % s[c]["median_bprec_covered"]) if s[c].get("median_bprec_covered") is not None else "-" for c in C3)
    c50 = "/".join(f"{s[c]['covered50']:.3f}" for c in C3)
    ms = it.get("member_support", {}).get(sp, {})
    cas = ms.get("CRISPRCasTyper")
    lab = it.get("labels", {})
    nl = lab.get("modules", lab.get("nontrivial", ""))
    out = (f"{name:34s} {prim:3s} short {sc['shortfall']:5.2f} sum {s['summed_DF_TXSS_excess']:.4f} F1 {f1} "
           f"ribo {rb['recovered']}/{rb['resolved']}/{rb['parvarchaeaceae']}/{rb.get('parvarchaeaceae_resolved')} "
           f"m[sum {m['summed']:+.2f} DF {m['DefenseFinder']:+.2f} TX {m['TXSScan']:+.2f} CC {m['CRISPRCasTyper']:+.2f} "
           f"res {m['resolved']:+.2f}] r2c {''.join('Y' if x else 'n' for x in it['r2_conditions'])} bprec {bp} c50 {c50} "
           f"cas {cas if cas is None else round(cas, 3)} lab {nl} genes {r['loci']['genes_covered']}")
    if "critic" in it:
        c = it["critic"]
        ag = c.get("added_genes", {})
        out += (f"\n{'':34s}     critic added bnd {c['boundaries_added']} ext {c['extensions']} bri {c['bridges']}; genes added "
                f"(split) {ag.get('split_genomes')} singleton {ag.get('singleton_family_split')} gold share "
                f"{ag.get('gold_member_share_split', 0):.3f} (base {ag.get('base_rate_gold_member_split', 0):.3f}) "
                f"singleton gold {ag.get('gold_member_share_singleton_split', 0):.3f} CCT members {ag.get('members_CRISPRCasTyper_split')}")
    if "label_diag" in it:
        d = it["label_diag"]
        rb_ = d["ribosomal"]
        tx = d.get("TXSScan", {})
        out += (f"\n{'':34s}     labels: ribo one {rb_['one_label']} 2+ {rb_['two_plus']} none {rb_['no_label']} Parv one "
                f"{rb_['parv_one_label']}/{rb_['parv_runs']} distinct {rb_.get('distinct_labels_all_runs')} | TXSS one "
                f"{tx.get('one_label', 0):.3f} 2+ {tx.get('two_plus', 0):.3f} edge {tx.get('edge_change_both_sides', 0):.3f} | "
                f"labelled bnd {d['labelled_boundaries']:.3f}")
    for k, v in it.get("checks", {}).items():
        out += f"\n{'':34s}     check {k}: n {v['n_common']}/{v['n_here']} f1 maxdiff {v['f1_maxdiff']:.2e} ctl {v['control_maxdiff']:.2e}"
    pk = [k for k in it.get("paired", {}) if k.startswith(f"{sp}:{prim}_vs_r2sel")]
    for k in pk:
        p = it["paired"][k]
        out += (f"\n{'':34s}     vs r2sel {prim}: summed {p['summed'][0]:+.4f} [{p['summed'][1]:+.4f},{p['summed'][2]:+.4f}] dF1 "
                + " ".join(f"{c[:4]} {p[c]['d_f1'][0]:+.4f}" for c in C3))
    k = f"{sp}:{prim}_vs_tok_DH"
    if k in it.get("paired", {}):
        p = it["paired"][k]
        out += (f"\n{'':34s}     vs tok DH: summed {p['summed'][0]:+.4f} [{p['summed'][1]:+.4f},{p['summed'][2]:+.4f}] dF1 "
                + " ".join(f"{c[:4]} {p[c]['d_f1'][0]:+.4f} [{p[c]['d_f1'][1]:+.3f},{p[c]['d_f1'][2]:+.3f}]" for c in C3))
    if "verdict_vs_tok_DH" in it:
        v = it["verdict_vs_tok_DH"]
        out += f"\n{'':34s}     VERDICT matches {v['matches']} beats {v['beats']} blocking {v['blocking_matches']}"
    return out


def main():
    files = [x for x in sys.argv[1:] if not x.startswith("--")]
    for fn in files:
        d = json.load(open(fn))
        split = "odd" if "/odd_" in fn or fn.startswith("odd_") else "all"
        ref = d.get("_refs", {})
        if ref:
            t = ref["tok_DH"]
            print(f"== {fn}  token DH ({ref['split']}): summed {t['summed_DF_TXSS_excess']:.4f} F1 "
                  + "/".join(f"{t[c]['f1']:.3f}" for c in C3) + f" ribo {ref['tok_DH_ribo']}")
        items = [(k, v) for k, v in d.items() if not k.startswith("_")]
        if "--sort" in sys.argv:
            items.sort(key=lambda kv: (-kv[1]["score"]["score"][0], -kv[1]["score"]["score"][1]))
        for k, v in items:
            try:
                print(row(k, v, split))
            except Exception as e:  # noqa: BLE001
                print(k, "ERR", e)


if __name__ == "__main__":
    main()
