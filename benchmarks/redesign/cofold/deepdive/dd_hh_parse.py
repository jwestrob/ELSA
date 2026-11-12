"""Parse the candidate-1 HHblits / HHsearch results (local, seconds).

For each .hhr: every hit with its probability, E-value, aligned columns, query and template ranges and the full
template description (from the alignment section). UniRef30 hits: taxonomy from 'Tax=' in the representative's
description. Fused-ORF scan: (i) in the concatenated A|B search (c1AB), any template whose alignment covers
query columns of both A (1-60) and B (61-130) by >= 20 columns each; (ii) any UniRef30 cluster hit by both the
A search and the B search (Prob >= 50) in collinear, non-overlapping template segments.
Writes work/hh_hits.tsv and prints a summary.
"""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
H = HERE / "work/hh"


def parse_hhr(p):
    lines = open(p).read().split("\n")
    hits, desc = [], {}
    i = 0
    while i < len(lines) and not lines[i].startswith(" No Hit"):
        i += 1
    i += 1
    pat = re.compile(r"^\s*(\d+)\s+(.{30})\s+([\d.]+)\s+(\S+)\s+(\S+)\s+([\d.-]+)\s+([\d.-]+)\s+(\d+)\s+(\d+)-(\d+)\s+(\d+)-(\d+)\s*\((\d+)\)")
    while i < len(lines) and lines[i].strip():
        m = pat.match(lines[i])
        if m:
            hits.append(dict(no=int(m.group(1)), prob=float(m.group(3)), E=float(m.group(4)), cols=int(m.group(8)),
                             qs=int(m.group(9)), qe=int(m.group(10)), ts=int(m.group(11)), te=int(m.group(12)),
                             tlen=int(m.group(13))))
        i += 1
    cur = None
    for line in lines[i:]:
        if line.startswith("No "):
            cur = int(line.split()[1])
        elif line.startswith(">") and cur is not None:
            desc[cur] = line[1:].strip()
    for h in hits:
        h["desc"] = desc.get(h["no"], "")
        h["id"] = h["desc"].split()[0] if h["desc"] else ""
        t = re.search(r"Tax=(.*?) TaxID=(\d+)", h["desc"])
        h["tax"] = t.group(1) if t else ""
        h["taxid"] = int(t.group(2)) if t else None
    return hits


def main():
    rows = []
    for p in sorted(H.glob("*.hhr")):
        q, mode = p.name.split(".")[0], ".".join(p.name.split(".")[1:-1])
        for h in parse_hhr(p):
            rows.append(dict(query=q, search=mode, **h))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "work/hh_hits.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    u = d[d.search == "u"]
    for q, x in u.groupby("query"):
        s = x[x.prob >= 95]
        print(f"{q}: UniRef30 hits {len(x)}, Prob>=95 {len(s)}, E<=1e-3 {(x.E <= 1e-3).sum()}")
        print("   taxa (Prob>=95):", s.tax.value_counts().head(25).to_dict())
    for (q, mode), x in d[d.search != "u"].groupby(["query", "search"]):
        t = x.sort_values("prob", ascending=False).head(2)
        print(q, mode, [(r.id, r.desc[:60], r.prob, r.E, r.cols) for r in t.itertuples()])
    ab = u[u["query"] == "c1AB"]
    span = ab[(ab.qs <= 40) & (ab.qe >= 81)]
    print("c1AB hits spanning both halves (qs<=40, qe>=81):", len(span))
    print(span[["id", "prob", "E", "qs", "qe", "ts", "te", "tlen", "tax"]].to_string())
    a = u[(u["query"] == "c1A") & (u.prob >= 50)].set_index("id")
    b = u[(u["query"] == "c1B") & (u.prob >= 50)].set_index("id")
    both = sorted(set(a.index) & set(b.index))
    out = []
    for k in both:
        ra, rb = a.loc[[k]].iloc[0], b.loc[[k]].iloc[0]
        ov = max(0, min(ra.te, rb.te) - max(ra.ts, rb.ts) + 1)
        out.append((k, ra.ts, ra.te, rb.ts, rb.te, ra.tlen, ov, ra.tax))
    print("UniRef30 clusters hit by both A and B searches:", len(out))
    for o in out:
        print("  ", o)


if __name__ == "__main__":
    main()
