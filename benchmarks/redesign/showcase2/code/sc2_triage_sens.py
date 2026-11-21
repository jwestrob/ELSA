#!/usr/bin/env python3
"""ws_showcase2 triage wording sensitivity (amendment written before any fold): a family whose 'characterised' call rests
only on HHsearch hits to PDB entries from structural-genomics consortia or de novo designs (titles matching
STRUCT_GEN below), with no Pfam or CDD characterised hit and no Pfam GA hit, is re-classed 'unknown_function' in a
sensitivity column; blocks that gain an unrecognised position this way are listed as a secondary fold set.
Writes data/triage_family_sens.tsv and data/triage_blocks_sens.tsv."""
import re
from pathlib import Path

import pandas as pd

import sc2_triage_parse as P

S = P.S
STRUCT_GEN = re.compile(r"structural genomics|NESG|MCSG|JCSG|NYSGRC|SGC|CSGID|RIKEN Structural|de novo designed|"
                        r"\bprotein [A-Z]{1,4}[_-]?\d{2,5}[A-Z]?\b|\b[A-Z]{1,4}[_-]?\d{2,5}[A-Z]? protein\b", re.I)


def main():
    titles = {}
    for p in (S / "data/pdb_titles_s2.tsv", S / "data/pdb_titles_sc2.tsv"):
        if p.exists():
            t = pd.read_csv(p, sep="\t", header=None, names=["id", "title"], dtype=str).fillna("")
            titles.update(dict(zip(t.id.str.lower(), t.title)))
    F = pd.read_csv(S / "data/triage_family.tsv", sep="\t")
    rows = []
    for _, f in F.iterrows():
        sens = f.cls
        basis = ""
        if f.cls == "characterised" and f.get("source", "") == "hhsearch":
            d = S / "triage/out" / str(int(f.token))
            char = []
            for src in ("pfam_enr", "pfam_raw", "pdb_enr", "pdb_raw"):
                for h in P.parse_hhr(d / f"{src}.hhr") or []:
                    if h["prob"] >= 90 and h["cols"] >= 30:
                        name = h["name"]
                        if src.startswith("pdb"):
                            name = f"{name} | {titles.get(name.split()[0][:4].lower(), '')}"
                        if not P.UNKNOWN.search(name):
                            char.append((src, name))
            if char and all(s.startswith("pdb") and STRUCT_GEN.search(n) for s, n in char):
                sens = "unknown_function"
                basis = " || ".join(n[:120] for _, n in char[:3])
        rows.append(dict(token=int(f.token), cls=f.cls, cls_sens=sens, sens_basis=basis))
    Fs = pd.DataFrame(rows)
    Fs.to_csv(S / "data/triage_family_sens.tsv", sep="\t", index=False)
    cls = dict(zip(Fs.token, Fs.cls_sens))
    B = pd.read_csv(S / "data/triage_blocks.tsv", sep="\t")
    reps = pd.read_csv(S / "data/reps.tsv", sep="\t")
    out = []
    for _, b in B.iterrows():
        rr = reps[reps.cand == b.cand].sort_values("rep")
        t1 = [int(x) for x in rr.tokens.iloc[0].split(";")]
        t2 = [int(x) for x in rr.tokens.iloc[1].split(";")]
        pos = [max((cls[a], cls[c]), key=lambda x: P.ORDER[x]) for a, c in zip(t1, t2)]
        n_unrec = sum(p != "characterised" for p in pos)
        out.append(dict(cand=b.cand, rank=b["rank"], fold_primary=b.fold, position_cls_sens=";".join(pos),
                        fold_sens=n_unrec >= 1, all_unrecognised_sens=n_unrec == len(pos)))
    Bs = pd.DataFrame(out)
    Bs["secondary_set"] = Bs.fold_sens & ~Bs.fold_primary
    Bs.to_csv(S / "data/triage_blocks_sens.tsv", sep="\t", index=False)
    print(Fs[Fs.cls != Fs.cls_sens].to_string(index=False))
    print("secondary set:", Bs[Bs.secondary_set].cand.tolist())


if __name__ == "__main__":
    main()
