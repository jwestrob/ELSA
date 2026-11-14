#!/usr/bin/env python
"""ws_showcase2 disorder-zipping and low-complexity checks for interfaces passing both replicates (census rule,
unchanged): for interface (i, j) and replicate, the residues of chain i (and j) in an i-j contact in the cognate
complex; the fraction of them with pLDDT < 60 in the chain's monomer fold (sc2m_<rank>_r<rep>_<chain>); flagged if
>= 50 % for either chain, or if either chain's mean pLDDT in the complex is < 60. Low complexity: SEG-trigger fraction
of each chain (12-residue windows with Shannon entropy < 2.2 bits; screen filter 0.145) and of its interface residues.
Usage: python sc2_zipping.py FOLD_DIR [...]; writes analysis/zipping.tsv."""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
A = HERE / "analysis"
W = HERE / "work"
AB = "ABCDEFGH"


def lc_mask(seq, w=12, k1=2.2):
    n = len(seq)
    cov = np.zeros(n, bool)
    if n < w:
        return cov
    c = Counter(seq[:w])
    for i in range(n - w + 1):
        if i:
            o, inn = seq[i - 1], seq[i + w - 1]
            c[o] -= 1
            if c[o] == 0:
                del c[o]
            c[inn] += 1
        if -sum((v / w) * math.log2(v / w) for v in c.values()) < k1:
            cov[i:i + w] = True
    return cov


def main(dirs):
    met = {}
    for d in map(Path, dirs):
        p = d / "metrics.jsonl"
        if p.exists():
            for line in open(p):
                m = json.loads(line)
                met[m["fold_id"]] = m
    seqs = {}
    for p in W.glob("jobs*.jsonl"):
        for line in open(p):
            j = json.loads(line)
            seqs[j["fold_id"]] = [c["seq"] for c in j["chains"]]
    calls = pd.read_csv(A / "calls.tsv", sep="\t")
    calls = calls[calls.pass_both]
    rows = []
    for _, r in calls.iterrows():
        rank, a, b = int(r["rank"]), int(r.i), int(r.j)
        for rep in (1, 2):
            fid = f"sc2c_{rank:03d}_r{rep}"
            m = met.get(fid)
            if m is None:
                continue
            pp = m["pairs"][f"{AB[a]}{AB[b]}"]
            out = dict(rank=rank, rep=rep, i=a, j=b)
            flag = False
            for lab, ch, res in (("i", a, pp["res_i"]), ("j", b, pp["res_j"])):
                cpl = m["chain_plddt"][ch]
                out[f"cplx_plddt_{lab}"] = cpl
                s = seqs[fid][ch]
                lm = lc_mask(s)
                out[f"lc_frac_{lab}"] = round(float(lm.mean()), 3)
                out[f"lc_iface_frac_{lab}"] = round(float(lm[res].mean()), 3) if len(res) else np.nan
                mono = met.get(f"sc2m_{rank:03d}_r{rep}_{AB[ch]}")
                if mono is None:
                    out[f"mono_lowconf_frac_{lab}"] = np.nan
                    flag |= cpl < 60
                    continue
                mp = np.array(mono["plddt_per_res"])
                assert len(mp) == m["lengths"][ch]
                frac = float((mp[res] < 60).mean()) if len(res) else np.nan
                out[f"mono_plddt_{lab}"] = round(float(mp.mean()), 1)
                out[f"iface_n_{lab}"] = len(res)
                out[f"mono_lowconf_frac_{lab}"] = round(frac, 3)
                flag |= (frac >= 0.5) or (cpl < 60)
            out["zipping_flag"] = bool(flag)
            out["low_complexity_flag"] = bool(max(out["lc_frac_i"], out["lc_frac_j"]) > 0.145)
            out["monomers_present"] = all(f"sc2m_{rank:03d}_r{rep}_{AB[c]}" in met for c in (a, b))
            rows.append(out)
    z = pd.DataFrame(rows)
    z.to_csv(A / "zipping.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(z.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
