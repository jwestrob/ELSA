#!/usr/bin/env python
"""Stage-1 diagnosis: why the vocabulary loses the ribosomal run and CRISPR-Cas (gold used for evaluation only).

For a run (arm, tau):
 1. gene status in ribosomal runs and in gold members: background (family in < 2 groups), free terminal (family in
    >= 2 groups, in no word), word slot, inserted gene; by Parvarchaeaceae / other runs and by caller.
 2. junctions between adjacent top-level words (<= 2 genes apart): the family pair across the junction counted at token
    level (n groups, E, Lambda as in round 0) against the word-type pair (n groups). A junction whose family pair has
    Lambda >= tau but whose type pair recurs in < 2 groups is a fragmentation boundary (variants split the count).
 3. ribosomal runs: top-level words per run, distinct top types, best single-word coverage.
usage: diag1.py --arm main --tau 10 --threads N
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
VD = RD / "vocab"
sys.path.insert(0, str(VD / "code"))
import vocab as V  # noqa: E402

PKG = RD / "ws_dpann/package"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="main")
    ap.add_argument("--tau", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    cc, gc, pos = V.load_universe()
    n = len(cc)
    tok = V.load_tokens(n)
    grp = V.load_groups(a.arm, int(gc.max()) + 1)
    ng = int(grp.max()) + 1
    d = VD / "runs" / a.arm / f"tau{a.tau:g}" / "obs"
    live = dict(np.load(d / "live.npz"))
    inst = dict(np.load(d / "inst.npz"))
    T = int(tok.max()) + 1
    tg = np.unique(tok * ng + grp[gc])
    tok_groups = np.bincount(tg // ng, minlength=T)
    bg = tok_groups[tok] < 2
    # status per gene: 0 background, 1 free terminal, 2 slot, 3 inserted
    status = np.where(bg, 0, 1).astype(np.int8)
    top = ~live["term"]
    lens = live["ne"][top] - live["ns"][top] + 1
    idx = np.repeat(np.arange(top.sum()), lens)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    genes_in = live["ns"][top][idx] + (np.arange(len(idx)) - off[idx])
    status[genes_in] = 3
    # slot genes = terminal children of any instance
    for c in ("a", "b"):
        ch = inst[c]
        status[-ch[ch < 0] - 1] = 2
    top_of_gene = np.full(n, -1, dtype=np.int64)
    top_of_gene[genes_in] = idx
    tsym = live["sym"][top]
    out = dict(arm=a.arm, tau=a.tau, genes=dict(zip(["background", "free", "slot", "inserted"],
                                                    np.bincount(status, minlength=4).tolist())))
    # ---- ribosomal runs
    runs = pd.read_parquet(RD / "ribosomal/gold/runs.parquet")
    runs = runs[(runs.run_set == "g2") & (runs.n_marked >= 16)].reset_index(drop=True)
    rows = []
    for rs, re_, fam in zip(runs.gid_start.to_numpy(), runs.gid_end.to_numpy(), runs.family.to_numpy()):
        st = status[rs:re_ + 1]
        tops = top_of_gene[rs:re_ + 1]
        tt = tops[tops >= 0]
        cnt = np.bincount(tt - tt.min()) if len(tt) else np.zeros(1, int)
        rows.append(dict(parv=fam == "Parvarchaeaceae", L=re_ - rs + 1, bg=float((st == 0).mean()), free=float((st == 1).mean()),
                         slot=float((st == 2).mean()), ins=float((st == 3).mean()), n_top=int(len(np.unique(tt))),
                         n_types=int(len(np.unique(tsym[tt]))) if len(tt) else 0, best=float(cnt.max() / (re_ - rs + 1))))
    rr = pd.DataFrame(rows)
    out["ribosomal"] = {k: dict(runs=int(len(g)), **{c: float(g[c].mean()) for c in ("bg", "free", "slot", "ins", "best")},
                                median_n_top=float(g.n_top.median()), median_n_types=float(g.n_types.median()),
                                best_ge_0p5=int((g.best >= 0.5).sum()))
                        for k, g in (("parv", rr[rr.parv]), ("other", rr[~rr.parv]))}
    # ---- gold members by caller (members of instances with >= 2 members)
    mem = pd.read_parquet(PKG / "gold/members.parquet")
    k = mem.groupby("instance_id").gid.transform("size")
    mem = mem[k >= 2]
    out["gold_members"] = {}
    for c, g in mem.groupby("caller"):
        st = status[g.gid.to_numpy()]
        out["gold_members"][c] = dict(n=int(len(g)), **{nm: float((st == i).mean()) for i, nm in
                                                         enumerate(["background", "free", "slot", "inserted"])},
                                      family_groups_q=np.quantile(tok_groups[tok[g.gid.to_numpy()]], [.1, .25, .5, .75]).tolist())
    # ---- junctions between adjacent top-level words
    ts, te = live["ns"][top], live["ne"][top]
    j = np.flatnonzero((cc[ts[1:]] == cc[te[:-1]]) & (ts[1:] - te[:-1] - 1 <= 2))
    # are all genes between them terminals (not inside another word)? top words are contiguous in live order only if
    # there are no other top words between; check via top_of_gene of the gap genes (gap genes are terminals by construction)
    left_gene, right_gene = te[j], ts[j + 1]
    ta, tb = tok[left_gene], tok[right_gene]
    ok = (ta != tb) & ~bg[left_gene] & ~bg[right_gene]
    j, ta, tb, left_gene = j[ok], ta[ok], tb[ok], left_gene[ok]
    # token-level n groups for the family pair across the junction (within 3 genes, round-0 definition)
    key_j = np.minimum(ta, tb) * T + np.maximum(ta, tb)
    sym = np.where(bg, -1, tok)
    ks, gs = [], []
    for dd in (1, 2, 3):
        okk = (cc[:-dd] == cc[dd:]) & (sym[:-dd] >= 0) & (sym[dd:] >= 0) & (sym[:-dd] != sym[dd:])
        kk = np.flatnonzero(okk)
        ks.append(np.minimum(sym[kk], sym[kk + dd]) * T + np.maximum(sym[kk], sym[kk + dd]))
        gs.append(grp[gc[kk]])
    kall, gall = np.concatenate(ks), np.concatenate(gs)
    uk, inv = np.unique(kall, return_inverse=True)
    kg = np.unique(inv.astype(np.int64) * ng + gall)
    nG_tok = np.bincount(kg // ng, minlength=len(uk))
    p = np.searchsorted(uk, key_j)
    found = (p < len(uk)) & (uk[np.minimum(p, len(uk) - 1)] == key_j)
    n_tok_pair = np.where(found, nG_tok[np.minimum(p, len(uk) - 1)], 0)
    # word-type pair n groups (adjacent top words with these two types)
    sa, sb = tsym[j], tsym[j + 1]
    wkey = np.minimum(sa, sb) * (int(tsym.max()) + 1) + np.maximum(sa, sb)
    wg = np.unique(np.c_[wkey, grp[gc[ts[j]]]], axis=0)
    wu, wc = np.unique(wg[:, 0], return_counts=True)
    n_type_pair = wc[np.searchsorted(wu, wkey)]
    frag = (n_tok_pair >= 10) & (n_type_pair < 2)
    out["junctions"] = dict(n=int(len(j)), family_pair_groups_q=np.quantile(n_tok_pair, [.25, .5, .75, .9]).tolist(),
                            type_pair_groups_q=np.quantile(n_type_pair, [.25, .5, .75, .9]).tolist(),
                            fam_ge10_type_lt2=int(frag.sum()), fam_ge10=int((n_tok_pair >= 10).sum()))
    # junctions inside ribosomal runs
    inr = np.zeros(n, dtype=bool)
    for rs, re_ in zip(runs.gid_start.to_numpy(), runs.gid_end.to_numpy()):
        inr[rs:re_ + 1] = True
    m = inr[left_gene]
    out["junctions_in_ribosomal_runs"] = dict(n=int(m.sum()), family_pair_groups_q=np.quantile(n_tok_pair[m], [.25, .5, .75]).tolist(),
                                              type_pair_groups_q=np.quantile(n_type_pair[m], [.25, .5, .75]).tolist(),
                                              fam_ge10_type_lt2=int((frag & m).sum()))
    # composition-canonical types: multiset of slot-gene families of each top instance (order-free hash)
    def mix(x):
        x = (x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15))
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))
    slot = np.flatnonzero(status == 2)
    slot = slot[top_of_gene[slot] >= 0]
    comp = np.zeros(int(top.sum()), dtype=np.uint64)
    np.add.at(comp, top_of_gene[slot], mix(tok[slot]))
    out["composition"] = dict(top_types=int(len(np.unique(tsym))), compositions=int(len(np.unique(comp))))
    ca, cb = comp[j], comp[j + 1]
    ckey = np.c_[np.minimum(ca, cb), np.maximum(ca, cb), grp[gc[ts[j]]].astype(np.uint64)]
    cg = np.unique(ckey, axis=0)
    cu, cc_ = np.unique(cg[:, :2], axis=0, return_counts=True)
    pk = np.c_[np.minimum(ca, cb), np.maximum(ca, cb)]
    lut = {(int(x), int(y)): int(c) for (x, y), c in zip(cu, cc_)}
    n_comp_pair = np.array([lut[(int(x), int(y))] for x, y in pk])
    out["junctions"]["comp_pair_groups_q"] = np.quantile(n_comp_pair, [.25, .5, .75, .9]).tolist()
    out["junctions"]["fam_ge10_comp_lt2"] = int(((n_tok_pair >= 10) & (n_comp_pair < 2)).sum())
    out["junctions_in_ribosomal_runs"]["comp_pair_groups_q"] = np.quantile(n_comp_pair[m], [.25, .5, .75]).tolist()
    out["junctions_in_ribosomal_runs"]["fam_ge10_comp_lt2"] = int(((n_tok_pair >= 10) & (n_comp_pair < 2) & m).sum())
    fn = VD / "results" / f"diag1_{a.arm}_tau{a.tau:g}.json"
    with open(fn, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps(out, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


if __name__ == "__main__":
    main()
