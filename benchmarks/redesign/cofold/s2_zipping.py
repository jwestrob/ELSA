"""Disorder-zipping check for screen-2 interface hits (predeclared_criteria_screen2.json, refutations):
for each interface (i, j) and replicate, the residues of chain i (and of chain j) in contact across i-j
(CB-CB <= 8 A) in the complex; the fraction of them with pLDDT < 60 in the chain's single-chain fold
(mono2_<rank>_r<rep>_<chain>); flagged if >= 50 % for either chain, or if either chain's mean pLDDT in the
complex is < 60. Usage: python s2_zipping.py COMPLEX_DIR [...] -- MONO_DIR [...]; writes work/s2/s2_zipping.tsv."""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from s2_analyze_folds import read_pdb  # noqa: E402

W2 = HERE / "work" / "s2"


def main(cdirs, mdirs):
    mono = {}
    for d in map(Path, mdirs):
        for line in open(d / "metrics.jsonl"):
            m = json.loads(line)
            mono[m["fold_id"]] = np.array(m["plddt_per_res"])
    calls = pd.read_csv(W2 / "s2_calls.tsv", sep="\t")
    calls = calls[(calls["mode"] == "single") & calls.pass_both]
    rows = []
    for (rank,), c in calls.groupby(["s2_rank"]):
        for rep in (1, 2):
            fid = f"cog2_{rank:03d}_r{rep}"
            pdb = next((Path(d) / "pdb" / f"{fid}.pdb" for d in cdirs if (Path(d) / "pdb" / f"{fid}.pdb").exists()), None)
            if pdb is None:
                continue
            ch, xyz, pl = read_pdb(pdb)
            ids = sorted(set(ch), key=lambda x: "ABCDEFGH".index(x))
            d = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
            for _, r in c.iterrows():
                a, b = int(r.i), int(r.j)
                ia, ib = ch == ids[a], ch == ids[b]
                cm = d[np.ix_(ia, ib)] <= 8.0
                out = dict(s2_rank=rank, rep=rep, i=a, j=b)
                flag = False
                for lab, sel, mask in (("i", ia, cm.any(1)), ("j", ib, cm.any(0))):
                    cid = ids[a] if lab == "i" else ids[b]
                    mp = mono.get(f"mono2_{rank:03d}_r{rep}_{cid}")
                    cpl = float(pl[sel].mean())
                    out[f"cplx_plddt_{lab}"] = round(cpl, 1)
                    if mp is None:
                        out[f"mono_lowconf_frac_{lab}"] = np.nan
                        continue
                    assert len(mp) == int(sel.sum())
                    frac = float((mp[mask] < 60).mean()) if mask.any() else np.nan
                    out[f"mono_plddt_{lab}"] = round(float(mp.mean()), 1)
                    out[f"iface_n_{lab}"] = int(mask.sum())
                    out[f"mono_lowconf_frac_{lab}"] = round(frac, 3)
                    flag |= (frac >= 0.5) or (cpl < 60)
                out["zipping_flag"] = bool(flag)
                rows.append(out)
    z = pd.DataFrame(rows)
    z.to_csv(W2 / "s2_zipping.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(z.to_string())


if __name__ == "__main__":
    k = sys.argv.index("--")
    main(sys.argv[1:k], sys.argv[k + 1:])
