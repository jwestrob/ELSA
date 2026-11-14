"""Screen 2, step 1: candidate pool of dark conserved blocks of 2-8 adjacent genes.

Reuses build_candidates.py unchanged (dark genes / dark families / significant-pair links / runs /
windows / per-type aggregation / low complexity), with two changes made by monkeypatching:
  KMAX = 8   (windows of up to 8 genes inside a dark-link run; screen 1 stopped at 4)
  W    = work/s2  (screen-1 outputs in work/ stay untouched; genes_dark.parquet is symlinked)
Screen-2 filters (all others exactly as screen 1):
  pass_len2  every family median 60-1,200 aa and median summed length <= 2,500 aa (API 4,096 tokens)
  hyd <= 0.45, low complexity <= 0.145, copies <= 1.5, geometry, recurrence (>= 5 genomes, >= 2 orders)
Pool (what screen 2 adds to screen 1):
  multi    block types of k >= 3 genes passing every filter
  longpair pairs passing every filter with one family median > 700 aa (screen 1 capped at 700)
Non-redundancy among multi blocks: a type whose token set is contained in the token set of a
better-ranked selected type is dropped (keeps the largest recurrent window of a run); ranking:
orders, phyla, then k (larger first), genomes, block integrity.
Writes work/s2/candidates_all.tsv, block_instances.parquet (from build_candidates) and
work/s2/pool.tsv (selected screen-2 candidates, with pool label and s2_rank).
"""
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_candidates as bc  # noqa: E402

W2 = HERE / "work" / "s2"


def build():
    W2.mkdir(parents=True, exist_ok=True)
    link = W2 / "genes_dark.parquet"
    if not link.exists():
        os.symlink(HERE / "work" / "genes_dark.parquet", link)
    bc.W = W2
    bc.KMAX = 8
    bc.main()


def select():
    c = pd.read_csv(W2 / "candidates_all.tsv", sep="\t")
    c["pass_len2"] = (c.fam_len_med_min >= 60) & (c.fam_len_med_max <= 1200) & (c.len_sum_med <= 2500)
    other = ["pass_dark", "pass_hyd", "pass_lowcomplex", "pass_copies", "pass_geometry", "pass_recurrence"]
    c["pass_all2"] = c.pass_len2 & c[other].all(axis=1)
    c["toks"] = c.tokens.map(lambda s: [int(x) for x in str(s).split(";")])
    ok = c[c.pass_all2].copy()
    ok = ok.sort_values(["n_ord", "n_phylum", "k", "n_genomes", "block_integrity"],
                        ascending=[False, False, False, False, False])
    multi = ok[ok.k >= 3]
    sel, sets = [], []
    for key, tt in zip(multi.type_key, multi.toks):
        s = set(tt)
        if any(s <= t for t in sets):
            continue
        sel.append(key)
        sets.append(s)
    longpair = ok[(ok.k == 2) & (ok.fam_len_med_max > 700)].type_key.tolist()
    pool = ok[ok.type_key.isin(sel + longpair)].copy()
    pool["pool"] = ["multi" if k >= 3 else "longpair" for k in pool.k]
    pool["s2_rank"] = range(1, len(pool) + 1)
    pool = pool.drop(columns=["toks"])
    pool.to_csv(W2 / "pool.tsv", sep="\t", index=False)
    summ = dict(types_ge3_genomes=len(c), pass_len2=int(c.pass_len2.sum()), pass_all2=int(c.pass_all2.sum()),
                pass_all2_by_k=c[c.pass_all2].k.value_counts().sort_index().to_dict(),
                multi_types=int((ok.k >= 3).sum()), multi_nonredundant=len(sel), longpair=len(longpair),
                pool=len(pool))
    print(summ)
    pd.Series(summ).to_json(W2 / "pool_summary.json", indent=1)
    cols = ["s2_rank", "pool", "type_key", "k", "n_genomes", "n_ord", "n_phylum", "len_med", "len_sum_med"]
    print(pool[cols].to_string())


if __name__ == "__main__":
    if "--select-only" not in sys.argv:
        build()
    select()
