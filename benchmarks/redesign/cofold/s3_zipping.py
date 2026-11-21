"""Disorder-zipping check for screen-3 interfaces passing both replicates (predeclared_criteria_screen3.json):
for interface (i, j) and replicate, the residues of chain i (and j) in an i-j contact in the cognate complex; the
fraction of them with pLDDT < 60 in the chain's monomer fold (mono3_<rank>_r<rep>_<chain>); flagged if >= 50 % for
either chain, or if either chain's mean pLDDT in the complex is < 60.
Usage: python s3_zipping.py FOLD_DIR [...]; writes work/s3/s3_zipping.tsv."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W3 = HERE / "work" / "s3"
AB = "ABCDEFGH"


def main(dirs):
    met = {}
    for d in map(Path, dirs):
        p = d / "metrics.jsonl"
        if p.exists():
            for line in open(p):
                m = json.loads(line)
                met[m["fold_id"]] = m
    calls = pd.read_csv(W3 / "s3_calls.tsv", sep="\t")
    calls = calls[calls.pass_both]
    rows = []
    for _, r in calls.iterrows():
        rank, a, b = int(r.s3_rank), int(r.i), int(r.j)
        for rep in (1, 2):
            m = met.get(f"cog3_{rank:03d}_r{rep}")
            if m is None:
                continue
            pp = m["pairs"][f"{AB[a]}{AB[b]}"]
            out = dict(s3_rank=rank, rep=rep, i=a, j=b)
            flag = False
            for lab, ch, res in (("i", a, pp["res_i"]), ("j", b, pp["res_j"])):
                cpl = m["chain_plddt"][ch]
                out[f"cplx_plddt_{lab}"] = cpl
                mono = met.get(f"mono3_{rank:03d}_r{rep}_{AB[ch]}")
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
            out["monomers_present"] = all(f"mono3_{rank:03d}_r{rep}_{AB[c]}" in met for c in (a, b))
            rows.append(out)
    z = pd.DataFrame(rows)
    z.to_csv(W3 / "s3_zipping.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(z.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
