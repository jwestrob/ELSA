#!/usr/bin/env python
"""softrec3 readings: the round-2 code (readings2.py) pointed at round-3 candidates ($RD/softrec/r3), plus a gold-free
Leiden resolution pick.

usage
  readings3.py <readings2 command> ...       families | tune | test | famcmp | diag | refcheck, as readings2.py, on r3/
  readings3.py rescount --cand A,B --threads T
      for every fam_l<res>.npy of each candidate: families with >= 2 genes on genes whose sequence has a validated edge,
      against rescue3's families with >= 2 genes on the same genes; writes cand/<c>/rescount.json with the resolution
      closest to 1x in |log ratio| (the predeclared gold-free pick) and whether connected components are non-degenerate.
"""
import json
import sys
from pathlib import Path

import numpy as np

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
sys.path.insert(0, str(RD / "softrec/code"))
import readings2 as R2m  # noqa: E402

R3 = RD / "softrec/r3"
R2m.R2 = R3


def ge2(tok):
    _, inv, cnt = np.unique(tok, return_inverse=True, return_counts=True)
    return int((cnt >= 2).sum())


def cmd_rescount(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    uq = np.load(R2m.EMB / "uid_of_row.npy").astype(np.int64)
    import pandas as pd
    t = pd.read_parquet(RD / "engine/dpann_tokens/mmseqs__seq__e3c50s75_rescue3__dpann.parquet")
    ref = np.full(len(uq), -1, np.int64)
    ref[t.gid.to_numpy()] = t.token.to_numpy()
    U = int(uq.max()) + 1
    for cand in a.cand.split(","):
        cd = R3 / "cand" / cand
        ea, eb = np.load(cd / "edges_a.npy"), np.load(cd / "edges_b.npy")
        deg = np.bincount(ea, minlength=U) + np.bincount(eb, minlength=U)
        has = deg[uq] > 0
        target = ge2(ref[has])
        rows = []
        for fn in sorted(cd.glob("fam_l*.npy")):
            res = float(fn.stem[5:])
            memb = np.load(fn).astype(np.int64)
            n2 = ge2(memb[uq][has])
            rows.append(dict(fam=fn.stem[4:], resolution=res, families_ge2_on_genes_with_edge=n2, ratio=n2 / target,
                             abs_log_ratio=abs(float(np.log(n2 / target)))))
        rows.sort(key=lambda r: r["resolution"])
        pick = min(rows, key=lambda r: r["abs_log_ratio"]) if rows else None
        cc = np.load(cd / "fam_cc.npy")
        sz = np.bincount(cc)
        nodes_with_edge = int((deg > 0).sum())
        out = dict(cand=cand, genes_with_edge=int(has.sum()), rescue3_families_ge2_on_those_genes=target, scan=rows,
                   pick=pick, cc_largest=int(sz.max()), cc_largest_share=float(sz.max() / max(1, nodes_with_edge)),
                   cc_nondegenerate=bool(sz.max() < 0.5 * nodes_with_edge))
        R2m.jdump(cd / "rescount.json", out)
        print(json.dumps(out), flush=True)


def cmd_stage2(argv):
    """odd-genome DH (with R1, C, H) for each candidate at its gold-free Leiden pick (and connected components when
    non-degenerate); configs written to cand/<c>/stage2_configs.txt."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--tag", default="stage2")
    ap.add_argument("--res", default="0.3")
    ap.add_argument("--extra", default="", help="additional configs CAND:FAM[:RES], comma-separated")
    a = ap.parse_args(argv)
    cfgs = []
    for cand in a.cand.split(","):
        rc = json.loads((R3 / "cand" / cand / "rescount.json").read_text())
        cfgs.append(f"{cand}:{rc['pick']['fam']}:{a.res}")
        if rc["cc_nondegenerate"]:
            cfgs.append(f"{cand}:cc:{a.res}")
    cfgs += [x for x in a.extra.split(",") if x]
    print("stage2 configs", cfgs, flush=True)
    ns = argparse.Namespace(configs=",".join(cfgs), threads=a.threads, readings="R1,C,H,DH", tag=a.tag, redo=False)
    R2m.run_configs(ns, "odd")


def expand_auto(argv):
    """--configs auto:<prefix>[:<fam>[:<res>]] -> every candidate directory with a cand.json whose name starts with prefix."""
    out = list(argv)
    for i, x in enumerate(out):
        if x == "--configs" and i + 1 < len(out) and out[i + 1].startswith("auto:"):
            p = out[i + 1].split(":")
            pre, suf = p[1], ":".join(p[2:])
            names = sorted(d.name for d in (R3 / "cand").iterdir() if d.name.startswith(pre) and (d / "cand.json").exists())
            out[i + 1] = ",".join(n + (":" + suf if suf else "") for n in names)
            print("configs", out[i + 1], flush=True)
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rescount":
        cmd_rescount(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "stage2":
        cmd_stage2(sys.argv[2:])
    else:
        sys.argv = expand_auto(sys.argv)
        R2m.main()
