#!/usr/bin/env python
"""Write one ChimeraX command file per passing block (format of cofold/view_candidate1_abc.cxc): both replicate
complexes, chains coloured by chain ID, side chains within 4 A across each hit interface as sticks, tiled; never
opened by this workstream. Usage: python sc2_cxc.py RANK [RANK ...] (reads analysis/blocks.tsv, results/candidates/
reps.tsv, candidates.tsv; structures under work/folds_main/pdb)."""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent
AB = "ABCDEFGH"


def main(ranks):
    bl = pd.read_csv(HERE / "analysis/blocks.tsv", sep="\t").set_index("rank")
    cand = pd.read_csv(HERE / "results/candidates/candidates.tsv", sep="\t").set_index("rank")
    reps = pd.read_csv(HERE / "results/candidates/reps.tsv", sep="\t")
    for rk in map(int, ranks):
        c = cand.loc[rk]
        b = bl.loc[rk]
        rr = reps[reps["rank"] == rk].sort_values("rep")
        pairs = [p for p in str(b.hit_pairs).split(";") if p and p != "nan"] or \
                [p for p in str(b.pass_both_pairs).split(";") if p and p != "nan"]
        lines = [f"# ws_showcase2 rank {rk} ({c.cand}): {c.k} genes, median lengths {c.len_med} aa, {c.groups} genome groups, "
                 f"{c.famsets} family sets, {c.phyla} phyla.",
                 "# ESMFold2 single-sequence complexes of the two replicates (different genome groups); chains coloured by chain ID;",
                 f"# side chains within 4 A across the interface(s) {', '.join(pairs)} shown as sticks.",
                 "# Colour by confidence instead:  color bfactor palette alphafold",
                 "# Superpose replicate 2 on replicate 1:  matchmaker #2 to #1",
                 "set bgColor white", ""]
        for m, (_, r) in enumerate(rr.iterrows(), 1):
            pdb = HERE / "work/folds_main/pdb" / f"sc2c_{rk:03d}_r{int(r.rep)}.pdb"
            lines.append(f"open {pdb}")
            lines.append(f'rename #{m} "rank {rk} replicate {int(r.rep)} | {r.phylum} ({r.ord}) {r.genome} | '
                         f'{r.protein_ids.replace(";", " / ")} | complex ipTM {b[f"complex_iptm_r{int(r.rep)}"]:.2f}"')
        lines += ["", "hide atoms", "show cartoons", "color bychain"]
        for m in (1, 2):
            for p in pairs:
                x, y = p[0], p[1]
                lines.append(f"show (#{m}/{x} & (#{m}/{y} :< 4)) | (#{m}/{y} & (#{m}/{x} :< 4)) atoms")
        lines += ["style stick", "color byhetero", "lighting soft", "graphics silhouettes true", "tile", "view", ""]
        out = HERE / f"view_rank{rk:03d}_{c.cand}.cxc"
        out.write_text("\n".join(lines))
        print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
