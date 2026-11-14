"""Screen 2, MSA mode (orchestrator request 12 Nov: every fold with summed length >= 450 aa is folded both
single-sequence and with a block-paired MSA, candidates, controls and positives alike).

Block-paired MSA, one definition for every class: the rows are the genomes in which the same ordered token
tuple occurs as consecutive genes on one contig (either orientation), one window per genome (the folded
window's own genome first, then by genome completeness), at most 256 rows. Each chain position is aligned
separately (mafft --auto); row i of every chain comes from the same genome. For a candidate this is the
block's recurrence; an uncalled window or a positive gets whatever recurrence its token tuple has (often
only itself, i.e. no MSA beyond the query). Non-cognate swap controls: kept chains keep the receiving block's
rows, the replaced chain takes the donor block's rows (the donor's own replicate window as row 0), all
truncated to the shallowest chain, so pairing is broken as in screen 1.

  prep RANKS|all   write work/s2/msa/fa/<tuple>__c<i>.faa for every needed (tuple, chain) and
                   work/s2/msa/rows.json (per fold: tuple key, anchor row key per chain)
  jobs             after alignment (s2_msa_align.sbatch on biotite, results in work/s2/msa/aln/): project
                   every row onto its query and write work/s2/jobs_msa_<tag>.jsonl (fold ids msa2_<fold_id>)
Usage: python s2_msa_build.py prep 12,17,24   |   python s2_msa_build.py jobs TAG
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
W = HERE / "work"
W2 = W / "s2"
M = W2 / "msa"
D = W / "data"
SHARUR = "/Users/jacob/Documents/Sandbox/Sharur/Sharur/data/dpann/sharur.duckdb"
MAXROWS = 256
MINLEN = 450


def windows_for(tup, tok, cc):
    """Start indices (and orientation) of consecutive windows whose tokens equal tup in order or reversed."""
    k = len(tup)
    n = len(tok)
    out = []
    for orient, t in ((1, tup), (-1, tup[::-1])):
        m = np.ones(n - k + 1, bool)
        for j in range(k):
            m &= tok[j:n - k + 1 + j] == t[j]
            if j:
                m &= cc[j:n - k + 1 + j] == cc[:n - k + 1]
        for s in np.flatnonzero(m):
            gl = list(range(s, s + k))
            out.append(gl if orient == 1 else gl[::-1])
        if k == 1 or tup == tup[::-1]:
            break
    return out


def prep(ranks):
    man = pd.read_csv(W2 / "fold_manifest2.tsv", sep="\t")
    g = pd.read_parquet(W / "genes_dark.parquet", columns=["gid", "protein_id", "genome_code", "contig_code", "token"])
    gm = pd.read_parquet(D / "genomes.parquet").set_index("genome_code")
    tok = g.token.to_numpy()
    cc = g.contig_code.to_numpy()
    gc = g.genome_code.to_numpy()
    comp = gm.completeness.reindex(np.arange(gc.max() + 1)).fillna(0).to_numpy()
    sel = man[(man.total_aa >= MINLEN) & ((man.cls == "positive") | man.s2_rank.isin(ranks))]
    cog = man[man.cls == "cognate"].set_index("fold_id")
    rows, tuples = {}, {}

    def tuple_rows(tup, anchors):
        key = "-".join(map(str, tup))
        if key not in tuples:
            tuples[key] = dict(tup=tup, wins=windows_for(list(tup), tok, cc), anchors=set())
        tuples[key]["anchors"] |= {tuple(a) for a in anchors}
        return key

    for _, r in sel.iterrows():
        gl = [int(x) for x in r.gids.split(";")]
        tt = tuple(int(tok[x]) for x in gl)
        if r.cls == "noncognate":
            s = int(r.slot)
            base = cog.loc[f"cog2_{r.s2_rank:03d}_r{r.rep}"]
            bgl = [int(x) for x in base.gids.split(";")]
            btt = tuple(int(tok[x]) for x in bgl)
            donor_rank = int(r.match_level.split(";")[0].split("=")[1])
            dgl = [int(x) for x in cog.loc[f"cog2_{donor_rank:03d}_r{r.rep}"].gids.split(";")]
            dtt = tuple(int(tok[x]) for x in dgl)
            kb = tuple_rows(btt, [bgl])
            kd = tuple_rows(dtt, [dgl])
            rows[r.fold_id] = dict(cls=r.cls, chains=[(kd if c == s else kb, dgl.index(gl[s]) if c == s else c,
                                                       dgl if c == s else bgl) for c in range(len(gl))])
        else:
            kk = tuple_rows(tt, [gl])
            rows[r.fold_id] = dict(cls=r.cls, chains=[(kk, c, gl) for c in range(len(gl))])
    # choose rows per tuple: anchors' genomes first, then completeness; one window per genome
    need = set()
    for key, t in tuples.items():
        wins = t["wins"]
        anchors = [list(a) for a in t["anchors"]]
        for a in anchors:
            if a not in wins:
                wins.append(a)
        agen = {gc[a[0]] for a in anchors}
        wins.sort(key=lambda w: (0 if list(w) in anchors else 1, 0 if gc[w[0]] in agen else 1, -comp[gc[w[0]]]))
        keep, seen = [], set()
        for w in wins:
            if list(w) in anchors or gc[w[0]] not in seen:
                keep.append(list(map(int, w)))
                seen.add(gc[w[0]])
            if len(keep) >= MAXROWS + len(anchors):
                break
        t["keep"] = keep
        need |= {x for w in keep for x in w}
    sh = duckdb.connect(SHARUR, read_only=True)
    sh.execute("SET threads=1")
    pid = g.protein_id.to_numpy()
    sh.register("p", pd.DataFrame({"protein_id": pid[sorted(need)]}))
    seq = dict(sh.execute("SELECT q.protein_id, rtrim(q.sequence,'*') FROM proteins q JOIN p USING (protein_id)").fetchall())
    (M / "fa").mkdir(parents=True, exist_ok=True)
    nfa = 0
    for key, t in tuples.items():
        for c in range(len(t["tup"])):
            p = M / "fa" / f"{key}__c{c}.faa"
            with open(p, "w") as f:
                for w in t["keep"]:
                    f.write(f">{'_'.join(map(str, w))}\n{seq[pid[w[c]]]}\n")
            nfa += 1
    meta = dict(rows=rows, tuples={k: dict(tup=list(v["tup"]), keep=v["keep"]) for k, v in tuples.items()})
    json.dump(meta, open(M / "rows.json", "w"))
    depth = pd.Series({k: len(v["keep"]) for k, v in tuples.items()})
    print(len(rows), "folds;", len(tuples), "tuples;", nfa, "chain FASTAs; depth quantiles",
          depth.quantile([0, .25, .5, .75, 1]).to_dict(), "; by class", pd.Series([v["cls"] for v in rows.values()]).value_counts().to_dict())


def read_fa(p):
    out, h = {}, None
    for line in open(p):
        line = line.strip()
        if line.startswith(">"):
            h = line[1:].split()[0]
            out[h] = []
        elif line:
            out[h].append(line)
    return {k: "".join(v).upper() for k, v in out.items()}


def jobs(tag):
    meta = json.load(open(M / "rows.json"))
    cache = {}

    def aln(key, c):
        if (key, c) not in cache:
            p = M / "aln" / f"{key}__c{c}.aln.faa"
            cache[(key, c)] = read_fa(p) if p.exists() else read_fa(M / "fa" / f"{key}__c{c}.faa")
        return cache[(key, c)]

    have = set()
    for p in (W2).glob("folds_s2*/state.json"):
        have |= set(json.load(open(p))["tasks"])
    n = 0
    with open(W2 / f"jobs_msa_{tag}.jsonl", "w") as f:
        for fid, r in meta["rows"].items():
            if f"msa2_{fid}" in have:
                continue
            chains, msas = [], []
            for c, (key, ci, anchor) in enumerate(r["chains"]):
                a = aln(key, ci)
                qk = "_".join(map(str, anchor))
                q = a[qk]
                cols = [i for i, ch in enumerate(q) if ch != "-"]
                order = [qk] + [k for k in a if k != qk]
                rows_ = ["".join(a[k][i] for i in cols) for k in order]
                qseq = rows_[0]
                assert "-" not in qseq
                msas.append(rows_)
                chains.append(dict(id="ABCDEFGH"[c], seq=qseq, gid=anchor[ci]))
            depth = min(len(m) for m in msas) if r["cls"] == "noncognate" else None
            for ch, m in zip(chains, msas):
                mm = m[:depth] if depth else m
                ch["msa"] = mm if len(mm) > 1 else None
                if ch["msa"] is None:
                    del ch["msa"]
            f.write(json.dumps(dict(fold_id=f"msa2_{fid}", cls=f"msa_{r['cls']}", chains=chains)) + "\n")
            n += 1
    print(n, "MSA jobs ->", W2 / f"jobs_msa_{tag}.jsonl")


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep([int(x) for x in sys.argv[2].split(",")] if sys.argv[2] != "all" else list(range(1, 62)))
    else:
        jobs(sys.argv[2])
