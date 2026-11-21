#!/usr/bin/env python
"""Token-free runs against the genus run of the same code (lineagefree/softrec/genus) and against stored round 2.
usage: sr_compare.py tag1 tag2 ...   (reads $RD/lineagefree/scores/sr_<tag>__{R1,DH}.parquet and softrec/<tag>/result.json)"""
import json
import sys
from pathlib import Path

import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
sys.path.insert(0, str(RD / "boundary/code"))
import blib as B  # noqa: E402
import rules as BR  # noqa: E402

LF = RD / "lineagefree"
C = BR.CALLERS
gcode = B.odd_even_code()


def load(fn):
    t = pd.read_parquet(fn)
    t["excess"] = t.f1 - t.control_mean_f1
    return t.set_index("instance_id")


out = {}
for tag in sys.argv[1:]:
    res = json.load(open(LF / "softrec" / tag / "result.json"))
    item = dict(lineages=res["lineages"], rule=res["candidate"]["rule"], fdr=res["candidate"]["fdr"],
                significant_instances=res["candidate"]["significant_instances"], genes_covered=res["candidate"]["genes_covered"],
                families=res["families"], modules=res["modules"].get("modules"), refcheck=res.get("refcheck"), timing=res["timing"])
    for rn in ("R1", "DH"):
        a = load(LF / "scores" / f"sr_{tag}__{rn}.parquet")
        g = load(LF / "scores" / f"sr_genus__{rn}.parquet")
        pr = BR.paired(a, g, gcode, "even")
        rib = res["readings"][rn]["ribo"]["scored"]
        rib_g = json.load(open(LF / "softrec/genus/result.json"))["readings"][rn]["ribo"]["scored"]
        ev = res["readings"][rn]["summary"]["even"]
        parv = int(str(rib["parvarchaeaceae"]).split("/")[0])
        cond = dict(summed_not_below_m0p01=pr["summed"][0] >= -0.01, f1=all(pr[c]["d_f1"][0] >= -0.01 for c in C),
                    ribosomal=(rib["recovered"] >= 1650 and parv >= 110 and abs(rib["resolved"] - rib_g["resolved"]) <= 0.03 * rib_g["resolved"]))
        cond["replaces"] = all(cond.values())
        item[rn] = dict(summed=ev["summed_DF_TXSS_excess"], f1={c: ev[c]["f1"] for c in C}, vs_genus_rerun=pr,
                        vs_round2_stored=res["paired_vs_genus"].get(f"even:{rn}"), ribo=rib, ribo_genus_rerun=rib_g, conditions=cond)
        print(f"{tag:<16} {rn} summed {ev['summed_DF_TXSS_excess']:.4f} d {pr['summed'][0]:+.4f} [{pr['summed'][1]:+.4f},{pr['summed'][2]:+.4f}] "
              + " ".join(f"{c[:4]} {ev[c]['f1']:.3f} ({pr[c]['d_f1'][0]:+.3f} [{pr[c]['d_f1'][1]:+.3f},{pr[c]['d_f1'][2]:+.3f}])" for c in C)
              + f" | ribo {rib['recovered']}/{rib['resolved']} Parv {rib['parvarchaeaceae']} res {rib.get('parvarchaeaceae_resolved')} (genus {rib_g['resolved']}) | {cond}")
    out[tag] = item
    print(f"{tag:<16} units {res['lineages']} rule {item['rule']} fdr {item['fdr']:.4f} sig inst {item['significant_instances']:,} "
          f"genes {item['genes_covered']:,} families {item['families']} modules {item['modules']} refcheck {item['refcheck']}")
fn = LF / "results" / "softrec_compare.json"
prev = json.loads(fn.read_text()) if fn.exists() else {}
prev.update(out)
fn.write_text(json.dumps(prev, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")
