#!/usr/bin/env python3
"""Compare a CPU and a GPU Aksha hit table (<DB>_hits_df.tsv) row by row.

Usage: compare.py LABEL CPU.tsv GPU.tsv OUT.json
Levels: whole-file bytes; raw row order; multiset of full rows (every column as printed);
rows keyed by (protein, model, env/ali/hmm coordinates) with the printed scores compared;
(protein, model) pairs. Single process (login node; small files).
"""
import collections
import hashlib
import json
import sys

COORDS = ("env_from", "env_to", "ali_from", "ali_to", "hmm_from", "hmm_to")
SCORES = ("bitscore", "evalue", "c_evalue", "i_evalue", "dom_bitscore")


def load(path):
    raw = open(path, "rb").read()
    lines = raw.decode().rstrip("\n").split("\n")
    header = lines[0].split("\t")
    rows = [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]
    return raw, header, lines[1:], rows


def main():
    label, cpu_path, gpu_path, out = sys.argv[1:5]
    craw, ch, clines, crows = load(cpu_path)
    graw, gh, glines, grows = load(gpu_path)
    res = {
        "label": label,
        "cpu_file": cpu_path,
        "gpu_file": gpu_path,
        "header_equal": ch == gh,
        "cpu_sha256": hashlib.sha256(craw).hexdigest(),
        "gpu_sha256": hashlib.sha256(graw).hexdigest(),
        "bytes_identical": craw == graw,
        "cpu_rows": len(crows),
        "gpu_rows": len(grows),
        "raw_order_identical": clines == glines,
    }
    cm, gm = collections.Counter(clines), collections.Counter(glines)
    res["full_rows_only_cpu"] = sum((cm - gm).values())
    res["full_rows_only_gpu"] = sum((gm - cm).values())
    res["full_row_multisets_identical"] = cm == gm

    def key(r):
        return (r["sequence_id"], r["hmm_name"]) + tuple(r[c] for c in COORDS)

    ck = collections.defaultdict(list)
    gk = collections.defaultdict(list)
    for r in crows:
        ck[key(r)].append(r)
    for r in grows:
        gk[key(r)].append(r)
    shared = set(ck) & set(gk)
    res["keyed_rows_only_cpu"] = sum(len(ck[k]) for k in set(ck) - set(gk))
    res["keyed_rows_only_gpu"] = sum(len(gk[k]) for k in set(gk) - set(ck))
    res["keyed_rows_shared"] = len(shared)
    diff = collections.Counter()
    examples = []
    for k in sorted(shared):
        a = sorted(tuple(r[s] for s in SCORES) for r in ck[k])
        b = sorted(tuple(r[s] for s in SCORES) for r in gk[k])
        if a != b:
            for s_i, s in enumerate(SCORES):
                if [x[s_i] for x in a] != [x[s_i] for x in b]:
                    diff[s] += 1
            if len(examples) < 20:
                examples.append({"key": k, "cpu": a, "gpu": b})
    res["shared_keys_with_score_difference"] = dict(diff)
    res["score_difference_examples"] = examples
    only_c = sorted(set(ck) - set(gk))[:20]
    only_g = sorted(set(gk) - set(ck))[:20]
    res["only_cpu_examples"] = [ck[k][0] for k in only_c]
    res["only_gpu_examples"] = [gk[k][0] for k in only_g]
    cp = {(r["sequence_id"], r["hmm_name"]) for r in crows}
    gp = {(r["sequence_id"], r["hmm_name"]) for r in grows}
    res["pairs_cpu"] = len(cp)
    res["pairs_gpu"] = len(gp)
    res["pairs_only_cpu"] = len(cp - gp)
    res["pairs_only_gpu"] = len(gp - cp)
    res["proteins_cpu"] = len({p for p, _ in cp})
    res["proteins_gpu"] = len({p for p, _ in gp})
    res["identical_hit_sets"] = (
        res["full_row_multisets_identical"] and res["header_equal"]
    )
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    brief = {k: v for k, v in res.items() if not k.endswith("examples")}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
