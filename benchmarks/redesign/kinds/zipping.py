#!/usr/bin/env python3
"""Disorder-zipping rule (screen 1, predeclared): a passing pair is flagged when >= 50% of a chain's interface
residues have pLDDT < 60 in that chain's monomer fold, or the chain's mean pLDDT in the complex is < 60."""
import json
from pathlib import Path

import pandas as pd

W = Path(__file__).resolve().parent / "work"


def load(d):
    return {m["fold_id"]: m for m in map(json.loads, open(W / d / "metrics.jsonl")) if m}


def main():
    cx = load("folds_main")
    mono = load("folds_mono")
    if (W / "folds_mono2/metrics.jsonl").exists():
        mono.update(load("folds_mono2"))      # resubmission of 18 items whose API worker died
    f = pd.read_csv(W / "folds_kinds.tsv", sep="\t")
    rows = []
    for fid in f[f.pass_c123].fold_id:
        m = cx[fid]
        la, lb = m["lengths"]
        mask = m["iface_mask"]
        r = dict(fold_id=fid)
        flag = False
        for ch, (s, e) in (("A", (0, la)), ("B", (la, la + lb))):
            mm = mono.get(f"kM_{fid}_{ch}")
            ires = [i - s for i in range(s, e) if mask[i]]
            cpl = m["chain_plddt"][0 if ch == "A" else 1]
            r[f"complex_plddt_{ch}"] = cpl
            if mm is None:
                r[f"mono_{ch}"] = None
                continue
            pl = mm["plddt_per_res"]
            low = sum(pl[i] < 60 for i in ires) / max(len(ires), 1)
            r[f"mono_plddt_{ch}"] = round(sum(pl) / len(pl), 1)
            r[f"iface_low_frac_{ch}"] = round(low, 3)
            flag |= (low >= 0.5) or (cpl < 60)
        r["zipping_flag"] = bool(flag)
        r["monomers_complete"] = all(f"kM_{fid}_{c}" in mono for c in "AB")
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(W / "zipping.tsv", sep="\t", index=False)
    print(t.to_string(index=False))
    print("flagged", int(t.zipping_flag.sum()), "of", len(t), "complete", int(t.monomers_complete.sum()))


if __name__ == "__main__":
    main()
