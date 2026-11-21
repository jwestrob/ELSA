#!/usr/bin/env python
"""ws_showcase2: assemble the ranked candidate table (candidates + triage + folds + checks) for the report.
Reads results/candidates/*.tsv, results/triage/*.tsv, analysis/*.tsv (whatever exists). Writes
results/candidate_table.tsv and prints a fixed-width version."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
AB = "ABCDEFGH"


def opt(p):
    p = HERE / p
    return pd.read_csv(p, sep="\t") if p.exists() else None


def main():
    C = pd.read_csv(HERE / "results/candidates/candidates.tsv", sep="\t")
    R = pd.read_csv(HERE / "results/candidates/reps.tsv", sep="\t")
    T = opt("results/triage/triage_blocks.tsv")
    TS = opt("results/triage/triage_blocks_sens.tsv")
    B = opt("analysis/blocks.tsv")
    K = opt("analysis/calls.tsv")
    rows = []
    for _, c in C.sort_values("rank").iterrows():
        rk = int(c["rank"])
        rr = R[R["rank"] == rk].sort_values("rep")
        row = dict(rank=rk, cand=c.cand, source=c.source, k=int(c.k), groups=int(c.groups), famsets=int(c.famsets),
                   phyla=int(c.phyla), genomes=int(c.genomes), len_med=c.len_med, short=bool(c.short),
                   rep_identity=c.rep_identity, pair_tier=c.pair_tier,
                   rep1=f"{rr.phylum.iloc[0]}/{rr.ord.iloc[0]} {rr.protein_ids.iloc[0]}",
                   rep2=f"{rr.phylum.iloc[1]}/{rr.ord.iloc[1]} {rr.protein_ids.iloc[1]}")
        if T is not None:
            t = T[T.cand == c.cand]
            if len(t):
                row.update(triage_positions=t.position_cls.iloc[0], folded=bool(t.fold.iloc[0]),
                           all_unrecognised=bool(t.all_unrecognised.iloc[0]))
            if TS is not None:
                ts = TS[TS.cand == c.cand].iloc[0]
                row["triage_positions_sens"] = ts.position_cls_sens
                row["fold_set"] = "primary" if ts.fold_primary else ("secondary" if ts.fold_sens else "not folded")
                row["folded"] = bool(ts.fold_sens)
        if B is not None and row.get("folded"):
            b = B[B["rank"] == rk]
            if len(b):
                b = b.iloc[0]
                row.update(complex_iptm=f"{b.complex_iptm_r1:.2f}/{b.complex_iptm_r2:.2f}",
                           best_min_pair_iptm=round(float(b.best_min_pair_iptm), 3),
                           pass_both=b.pass_both_pairs if isinstance(b.pass_both_pairs, str) else "",
                           interface_hits=b.hit_pairs if isinstance(b.hit_pairs, str) else "",
                           controls_pending=int(b.controls_pending))
            if K is not None:
                kk = K[K["rank"] == rk].sort_values("min_pair_iptm", ascending=False)
                if len(kk):
                    x = kk.iloc[0]
                    row["best_pair"] = (f"{AB[int(x.i)]}{AB[int(x.j)]} ipTM {x.r1_pair_iptm:.2f}/{x.r2_pair_iptm:.2f} "
                                        f"PAE {x.r1_iface_pae:.1f}/{x.r2_iface_pae:.1f} "
                                        f"ctrl {x.get('r1_ctrl_iptm', np.nan):.2f}/{x.get('r2_ctrl_iptm', np.nan):.2f}")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results/candidate_table.tsv", sep="\t", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 70)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
