#!/usr/bin/env python
"""Search by parsing (round 2A, descriptive; predeclared2a.json 'search').

Queries: DefenseFinder gold instances with >= 2 members and >= 1 target. Targets: same-type DefenseFinder instances with
>= 2 members in other genomes (bench.py). The query's word = the top-level word holding the plurality of the query's
members (>= 50% of them). Retrieved = every top-level instance in another genome with the query word's label (class),
ranked exact type first, then the rest of the class; ties in random order (40 draws, seed 17 + query index; AP averaged).
Hit labels as locus_search.label_hits: correct = covers >= 50% of the members of a same-type target (first hit per target
counts), wrong = overlaps a member of a different-type DefenseFinder instance, otherwise ignored. AP = sum over new
correct hits of precision (correct + wrong only) / number of targets. Queries without a word get AP 0 (no retrieval).
usage: search2a.py --src runs/main --tau 10 [--src2 runs2a/<tag> --tau2 T] --threads N
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
VD = RD / "vocab"
PKG = RD / "ws_dpann/package"
sys.path.insert(0, str(VD / "code"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_vocab(src, tau):
    d = VD / src / f"tau{tau:g}" / "obs"
    inst = dict(np.load(d / "inst.npz"))
    live = dict(np.load(d / "live.npz"))
    top = ~live["term"]
    ti = live["ninst"][top]
    return dict(s=live["ns"][top], e=live["ne"][top], label=live["sym"][top], exact=inst["sym"][ti])


def evaluate(src, tau, ngenes, gcode, inst, mem):
    v = load_vocab(src, tau)
    nt = len(v["s"])
    lens = v["e"] - v["s"] + 1
    tidx = np.repeat(np.arange(nt), lens)
    off = np.r_[0, np.cumsum(lens)[:-1]]
    genes = v["s"][tidx] + (np.arange(len(tidx)) - off[tidx])
    top_of_gene = np.full(ngenes, -1, dtype=np.int64)
    top_of_gene[genes] = tidx
    gen_of_top = gcode[v["s"]]
    # overlaps of each top instance with DefenseFinder instances (member counts)
    m = mem[["instance_id", "gid"]].copy()
    m["top"] = top_of_gene[m.gid.to_numpy()]
    ov = m[m.top >= 0].groupby(["top", "instance_id"]).size().rename("k").reset_index()
    ov = ov.merge(inst[["instance_id", "system_type", "n_mem"]], on="instance_id")
    ov["cov"] = ov.k / ov.n_mem
    ov_by_top = {t: g for t, g in ov.groupby("top")}
    # queries
    q = inst[inst.n_mem >= 2].copy()
    tgt = q.groupby("system_type").instance_id.apply(list).to_dict()
    gen_of_inst = dict(zip(inst.instance_id, inst.genome_code))
    by_label = pd.Series(np.arange(nt)).groupby(v["label"]).apply(np.asarray).to_dict()
    rows = []
    for qi, (iid, typ, g) in enumerate(zip(q.instance_id, q.system_type, q.genome_code)):
        targets = [t for t in tgt[typ] if gen_of_inst[t] != g]
        if not targets:
            continue
        mg = mem.gid[mem.instance_id == iid].to_numpy()
        tops = top_of_gene[mg]
        tops = tops[tops >= 0]
        r = dict(instance_id=iid, genome_code=int(g), n_rel=len(targets), has_word=False, ap=0.0, recall=0.0,
                 n_hits=0, correct=0, wrong=0)
        if len(tops):
            u, c = np.unique(tops, return_counts=True)
            best = u[np.argmax(c)]
            if c.max() / len(mg) >= 0.5:
                r["has_word"] = True
                cand = by_label[v["label"][best]]
                cand = cand[gen_of_top[cand] != g]
                exact = v["exact"][cand] == v["exact"][best]
                lab = np.zeros(len(cand), np.int8)
                match = np.full(len(cand), -1, np.int64)
                tset = set(targets)
                for j, t in enumerate(cand):
                    o = ov_by_top.get(t)
                    if o is None:
                        continue
                    same = o[(o.system_type == typ) & (o.n_mem >= 2) & (o["cov"] >= 0.5) & o.instance_id.isin(tset)]
                    if len(same):
                        lab[j] = 1
                        match[j] = int(same.sort_values(["cov", "instance_id"], ascending=[False, True]).instance_id.iloc[0])
                    elif (o.system_type != typ).any():
                        lab[j] = 2
                rng = np.random.default_rng(17 + qi)
                aps, recs = [], []
                for _ in range(40):
                    o_ = np.lexsort((rng.random(len(cand)), ~exact))
                    L, M = lab[o_], match[o_]
                    new = np.zeros(len(L), bool)
                    ic = np.flatnonzero(L == 1)
                    if len(ic):
                        _, first = np.unique(M[ic], return_index=True)
                        new[ic[first]] = True
                    cons = new | (L == 2)
                    k = np.cumsum(cons)
                    tp = np.cumsum(new)
                    aps.append(float((tp[new] / k[new]).sum() / len(targets)))
                    recs.append(float(new.sum() / len(targets)))
                r.update(ap=float(np.mean(aps)), recall=float(np.mean(recs)), n_hits=int(len(cand)),
                         correct=int((lab == 1).sum()), wrong=int((lab == 2).sum()))
        rows.append(r)
    R = pd.DataFrame(rows)
    out = {}
    for name, sub in (("all", R), ("even", R[R.genome_code % 2 == 0])):
        w = sub[sub.has_word]
        out[name] = dict(queries=int(len(sub)), with_word=int(len(w)), map=float(sub.ap.mean()),
                         map_with_word=float(w.ap.mean()) if len(w) else None, recall=float(sub.recall.mean()),
                         hits_per_query=float(w.n_hits.mean()) if len(w) else None,
                         wrong_per_query=float(w.wrong.mean()) if len(w) else None,
                         precision_labelled=float(w.correct.sum() / max(w.correct.sum() + w.wrong.sum(), 1)))
    return out, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--tau", action="append", type=float, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    genes = pd.read_parquet(PKG / "universe/genes.parquet", columns=["gid", "genome_code"])
    gcode = genes.genome_code.to_numpy()
    inst = pd.read_parquet(PKG / "gold/instances.parquet", columns=["instance_id", "caller", "system_type", "genome"])
    inst = inst[inst.caller == "DefenseFinder"].copy()
    gmap = pd.read_parquet(PKG / "universe/genomes.parquet", columns=["genome", "genome_code"])
    inst["genome_code"] = inst.genome.map(dict(zip(gmap.genome, gmap.genome_code)))
    mem = pd.read_parquet(PKG / "gold/members.parquet")
    mem = mem[mem.caller == "DefenseFinder"]
    inst["n_mem"] = inst.instance_id.map(mem.groupby("instance_id").size()).fillna(0).astype(int)
    res = {}
    for src, tau in zip(a.src, a.tau):
        t0 = time.time()
        out, R = evaluate(src, tau, len(gcode), gcode, inst, mem)
        R.to_parquet(VD / "results" / f"search2a_{src.replace('/', '_')}_tau{tau:g}.parquet", index=False)
        out["seconds"] = round(time.time() - t0, 1)
        res[f"{src}@{tau:g}"] = out
        log(src, tau, json.dumps(out))
    with open(VD / "results" / f"search2a{('_' + a.tag) if a.tag else ''}.json", "w") as fh:
        json.dump(dict(results=res, reference=dict(M1z_cv=0.796, S1_cv=0.610, tokens_M0=0.369),
                       written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), fh, indent=1)


if __name__ == "__main__":
    main()
