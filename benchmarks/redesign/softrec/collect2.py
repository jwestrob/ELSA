#!/usr/bin/env python
"""softrec2: tables from readings2.py result files and the predeclared odd-genome selection score.

usage: collect2.py odd TAG [TAG...]      odd-genome table (tune) with conditions against the token reading on odd genomes
       collect2.py even TAG [TAG...]     even-genome table with the predeclared matches / beats verdict against token DH
Conditions on odd (predeclared2.json): o1 ribosomal on odd runs (recovered >= 830, Parvarchaeaceae recovered >= 54 of 55,
DH also resolved >= 705); o2 summed DF + TXSS excess >= token - 0.01; o3 each caller F1 >= token - 0.02.
Score = (conditions met, odd summed excess).
"""
import json
import sys
from pathlib import Path

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
RES = RD / "softrec/r2/results"
CALLERS = ("DefenseFinder", "TXSScan", "CRISPRCasTyper")


def parv(rib, key="parvarchaeaceae"):
    return int(str(rib.get(key, "0/0")).split("/")[0])


def odd_rows(tags):
    rows = []
    for tag in tags:
        d = json.loads((RES / f"odd_{tag}.json").read_text())
        for key, it in d.items():
            for rn, r in it["readings"].items():
                ref = "tok_DH" if rn in ("DH", "H", "C") else "tok_R1"
                p = it.get("paired", {}).get(f"odd:{rn}_vs_{ref}")
                s = r["summary"]["odd"]
                rib = r["ribo"]["scored"]
                o1 = rib["recovered"] >= 830 and parv(rib) >= 54 and (rib["resolved"] >= 705 if rn == "DH" else True)
                o2 = o3 = None
                if p:
                    o2 = p["summed_A"] >= p["summed_B"] - 0.01
                    o3 = all(p[c]["f1_A"] >= p[c]["f1_B"] - 0.02 for c in CALLERS)
                cond = [o1, o2, o3]
                rows.append(dict(tag=tag, key=key, reading=rn, ref=ref, f1=[s[c]["f1"] for c in CALLERS],
                                 excess=[s[c]["excess"] for c in CALLERS], summed=s["summed_DF_TXSS_excess"],
                                 ref_summed=(p["summed_B"] if p else None), ref_f1=([p[c]["f1_B"] for c in CALLERS] if p else None),
                                 d_summed=(p["summed"] if p else None), ribo=(rib["recovered"], rib["resolved"], rib["parvarchaeaceae"],
                                                                             rib.get("parvarchaeaceae_resolved")),
                                 cond=cond, score=(sum(bool(x) for x in cond), s["summed_DF_TXSS_excess"]),
                                 loci=r["loci"], member=it["member_support"].get("odd"), modules=it.get("modules")))
    return rows


def fmt(r):
    f = "/".join(f"{x:.3f}" for x in r["f1"])
    rf = "/".join(f"{x:.3f}" for x in r["ref_f1"]) if r["ref_f1"] else "-"
    ds = f"{r['d_summed'][0]:+.3f} [{r['d_summed'][1]:+.3f},{r['d_summed'][2]:+.3f}]" if r["d_summed"] else "-"
    cc = "".join("Y" if x else ("n" if x is not None else "?") for x in r["cond"])
    m = r["member"] or {}
    mod = r["modules"] or {}
    return (f"{r['key']:<34} {r['reading']:<3} F1 {f} (ref {rf}) sum {r['summed']:.3f} d {ds} ribo {r['ribo'][0]}/{r['ribo'][1]}/"
            f"{r['ribo'][2]}/{r['ribo'][3]} cond {cc} loci {r['loci']['loci']} genes {r['loci']['genes_covered']} "
            f"CCTtouch {m.get('CRISPRCasTyper', float('nan')):.3f} mods {mod.get('modules', '-')}")


def main():
    mode, tags = sys.argv[1], sys.argv[2:]
    if mode == "odd":
        rows = odd_rows(tags)
        for rn in ("R1", "A", "C", "H", "DH"):
            rr = sorted([r for r in rows if r["reading"] == rn], key=lambda r: r["score"], reverse=True)
            if rr:
                print(f"== {rn} (odd; ref {rr[0]['ref']})")
                for r in rr:
                    print(fmt(r))
    else:
        for tag in tags:
            d = json.loads((RES / f"all_{tag}.json").read_text())
            for key, it in d.items():
                ms = it["member_support"].get("even", {})
                print(f"===== {key}  rule {it['land'].get('rule')}  modules {(it.get('modules') or {}).get('modules', '-')}  "
                      f"member support even DF/TXSS/CCT {ms.get('DefenseFinder', float('nan')):.3f}/{ms.get('TXSScan', float('nan')):.3f}/"
                      f"{ms.get('CRISPRCasTyper', float('nan')):.3f}")
                for rn, r in it["readings"].items():
                    s_ = r["summary"]["even"]
                    rib = r["ribo"]["scored"]
                    print("  %-3s even F1 %s ctl %s exc %s sum %.4f cov50 %s bprec %s | ribo %d/%d Parv %s res %s | loci %d genes %d span %.0f" % (
                        rn, "/".join("%.3f" % s_[c]["f1"] for c in CALLERS), "/".join("%.3f" % s_[c]["control"] for c in CALLERS),
                        "/".join("%.3f" % s_[c]["excess"] for c in CALLERS), s_["summed_DF_TXSS_excess"],
                        "/".join("%.3f" % s_[c]["covered50"] for c in CALLERS),
                        "/".join("%.3f" % (s_[c]["median_bprec_covered"] or 0) for c in CALLERS),
                        rib["recovered"], rib["resolved"], rib["parvarchaeaceae"], rib.get("parvarchaeaceae_resolved"),
                        r["loci"]["loci"], r["loci"]["genes_covered"], r["loci"]["median_span"]))
                    ref = "tok_DH" if rn in ("DH", "H", "C") else "tok_R1"
                    for rr in ((ref,) if ref == "tok_R1" else ("tok_DH", "tok_R1")):
                        p = it["paired"].get(f"even:{rn}_vs_{rr}")
                        if not p:
                            continue
                        line = "      vs %-6s summed %+.4f [%+.4f, %+.4f] (A %.4f B %.4f) dF1 %s" % (
                            rr, *p["summed"], p["summed_A"], p["summed_B"],
                            " ".join("%s %+.3f [%+.3f,%+.3f]" % (c[:4], *p[c]["d_f1"]) for c in CALLERS))
                        if rn == "DH" and rr == "tok_DH":
                            parv_rec = parv(rib)
                            ribo_ok = rib["recovered"] >= 1650 and parv_rec >= 110 and rib["resolved"] >= 1400
                            m = (p["summed"][0] >= -0.01 and all(p[c]["d_f1"][0] >= -0.02 for c in CALLERS) and ribo_ok)
                            b = (p["summed"][0] >= 0.02 and p["summed"][1] > 0 and all(p[c]["d_f1"][0] >= -0.01 for c in CALLERS) and ribo_ok)
                            blk = [n for n, ok in (("summed", p["summed"][0] >= -0.01),
                                                   *((f"F1 {c}", p[c]["d_f1"][0] >= -0.02) for c in CALLERS),
                                                   ("ribosomal recovered", rib["recovered"] >= 1650), ("Parv recovered", parv_rec >= 110),
                                                   ("ribosomal resolved", rib["resolved"] >= 1400)) if not ok]
                            line += f"\n      VERDICT matches={m} beats={b} blocking(matches)={blk}"
                        print(line)


if __name__ == "__main__":
    main()
