#!/usr/bin/env python3
"""ws_showcase2: parse the triage (sc2_triage.py) with the screen-2 family rule (predeclared_criteria_screen2.json
'triage_before_folding', unchanged) and decide which candidate blocks are folded.

Per family (rescue3 token):
  characterised   HHsearch probability >= 90 % over >= 30 aligned columns against Pfam 39 or PDB100 (enriched or
                  DPANN-only alignment) to an entry without unknown-function wording [the task's rule]; or the same
                  CDD model at E <= 0.01 in >= 2 RPS-BLAST instances; or a Pfam GA hit in >= 2 members (>= 10 % when
                  the family has > 20 triage members); or a curated Pfam hit with characterised wording on the
                  replicate gene itself (pfam_annotated.tsv; no HH search run)
  unknown_function  the same thresholds met only by unknown-function entries (DUF, UPF, uncharacterised, ...)
  weak            best HHsearch probability 50-90 %, a CDD model at E <= 0.01 in one instance, or at 0.01 < E <= 1 in
                  >= 2, or hmmsearch E <= 0.01 below GA
  none            otherwise
Block rule (task): fold a block when at least one position has no characterised hit in the families of both
replicates (position status = the more recognised of its two replicate families).
Writes data/triage_family.tsv, data/triage_hits.tsv (every HHsearch hit with probability >= 50 %, every CDD hit at
E <= 0.01, every Pfam hmmsearch hit at E <= 0.01), data/triage_blocks.tsv.
"""
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
S = RD / "showcase2"
TR = S / "triage"
UNKNOWN = re.compile(r"DUF\d|UPF\d|unknown function|uncharacteri|hypothetical|function unknown|unknown protein", re.I)
ORDER = {"none": 0, "weak": 1, "unknown_function": 2, "characterised": 3}


def parse_hhr(path):
    if not path.exists():
        return None
    hits, cur = [], None
    for line in open(path, errors="replace"):
        if line.startswith("No ") and line[3:].strip().isdigit():
            cur = dict(rank=int(line[3:]))
            hits.append(cur)
        elif cur is not None and line.startswith(">") and "name" not in cur:
            cur["name"] = line[1:].strip()
        elif cur is not None and line.startswith("Probab="):
            kv = dict(x.split("=") for x in line.split())
            cur.update(prob=float(kv["Probab"]), evalue=float(kv["E-value"]), cols=int(kv["Aligned_cols"]))
    return [h for h in hits if "prob" in h]


def main():
    pdb_title = {}
    for p in (S / "data/pdb_titles_s2.tsv", S / "data/pdb_titles_sc2.tsv"):     # screen-2 table + RCSB titles by ID
        if p.exists():
            tt = pd.read_csv(p, sep="\t", header=None, names=["id", "title"], dtype=str).fillna("")
            pdb_title.update(dict(zip(tt.id.str.lower(), tt.title)))
    nmem = {}
    for fa in (TR / "fasta").glob("fam_*.faa"):
        nmem[int(fa.stem[4:])] = sum(1 for l in open(fa) if l.startswith(">"))
    # CDD
    cdd = pd.read_csv(TR / "cdd_hits.tsv", sep="\t", header=None,
                      names=["q", "s", "qlen", "qs", "qe", "evalue", "bits", "title"]) if (TR / "cdd_hits.tsv").exists() else pd.DataFrame()
    cdd_char, cdd_unk, cdd_weak = {}, {}, defaultdict(list)
    if len(cdd):
        cdd["token"] = cdd.q.str.split("|").str[0].astype(int)
        cdd["inst"] = cdd.q.str.split("|").str[1]
        cdd["model"] = cdd.title.str.split(",").str[0]
        best = cdd.sort_values("evalue").drop_duplicates(["token", "inst", "model"])
        for (t, m), d in best.groupby(["token", "model"]):
            n01 = int((d.evalue <= 0.01).sum())
            n1 = int((d.evalue <= 1).sum())
            title = d.title.iloc[0][:120]
            if n01 >= 2:
                (cdd_unk if UNKNOWN.search(title) else cdd_char).setdefault(t, []).append(f"CDD {title} (E<=0.01 in {n01})")
            elif n01 == 1 or n1 >= 2:
                cdd_weak[t].append(f"CDD {title} ({n01}/{n1})")
    # Pfam hmmsearch
    ga, e01 = defaultdict(set), defaultdict(set)
    for tag, store in (("ga", ga), ("e01", e01)):
        f = TR / f"pfam39_hmmsearch_{tag}.domtbl"
        if f.exists():
            for line in open(f):
                if line.startswith("#"):
                    continue
                x = line.split()
                tok, mem = x[0].split("|")
                store[(int(tok), x[3], " ".join(x[22:])[:80])].add(mem)
    pann = pd.read_csv(TR / "pfam_annotated.tsv", sep="\t") if (TR / "pfam_annotated.tsv").exists() else pd.DataFrame(columns=["token", "query_gid", "pfam"])
    rows, hitrows = [], []
    toks = sorted(set(nmem) | set(pann.token))
    for t in toks:
        r = dict(token=t, n_triage_members=nmem.get(t, 0))
        if t in set(pann.token):
            r.update(cls="characterised", source="pfam_ga_on_replicate_gene",
                     char_hit="Pfam GA on replicate gene: " + pann[pann.token == t].pfam.iloc[0], complete=True)
            rows.append(r)
            continue
        d = TR / "out" / str(t)
        best_char = best_unk = best_any = None
        for src in ("pfam_enr", "pfam_raw", "pdb_enr", "pdb_raw"):
            for h in parse_hhr(d / f"{src}.hhr") or []:
                name = h["name"]
                if src.startswith("pdb"):
                    name = f"{name} | {pdb_title.get(name.split()[0][:4].lower(), '')}"
                h2 = dict(src=src, name=name[:160], prob=h["prob"], cols=h["cols"], evalue=h["evalue"])
                if h["prob"] >= 50:
                    hitrows.append(dict(token=t, kind="hhsearch", **h2))
                if best_any is None or h["prob"] > best_any["prob"]:
                    best_any = h2
                if h["prob"] >= 90 and h["cols"] >= 30:
                    if UNKNOWN.search(name):
                        if best_unk is None or h["prob"] > best_unk["prob"]:
                            best_unk = h2
                    elif best_char is None or h["prob"] > best_char["prob"]:
                        best_char = h2
        u = parse_hhr(d / "uniref.hhr") or []
        r["uniref_p95"] = sum(h["prob"] >= 95 for h in u)
        r["uniref_top"] = u[0]["name"][:100] if u else ""
        n = nmem.get(t, 0)
        need = 2 if n <= 20 else max(2, int(0.1 * n + 0.999))
        ga_char = [f"Pfam GA {acc} {nm} ({len(m)} members)" for (tt, acc, nm), m in ga.items()
                   if tt == t and len(m) >= need and not UNKNOWN.search(f"{acc} {nm}")]
        ga_unk = [f"Pfam GA {acc} {nm} ({len(m)})" for (tt, acc, nm), m in ga.items()
                  if tt == t and len(m) >= need and UNKNOWN.search(f"{acc} {nm}")]
        e_weak = [f"Pfam E<=0.01 {acc} ({len(m)})" for (tt, acc, nm), m in e01.items() if tt == t]
        for (tt, acc, nm), m in e01.items():
            if tt == t:
                hitrows.append(dict(token=t, kind="hmmsearch_e01", src="pfam39", name=f"{acc} {nm}", prob=None, cols=None,
                                    evalue=None, members=len(m)))
        for lab, lst in (("cdd_char", cdd_char.get(t, [])), ("cdd_unk", cdd_unk.get(t, [])), ("cdd_weak", cdd_weak.get(t, []))):
            for x in lst:
                hitrows.append(dict(token=t, kind=lab, src="cdd", name=x, prob=None, cols=None, evalue=None))
        hh_char = best_char is not None
        r["hh_char_hit"] = "" if best_char is None else f"{best_char['src']}: {best_char['name']} (P {best_char['prob']:.1f}, {best_char['cols']} cols)"
        r["hh_unk_hit"] = "" if best_unk is None else f"{best_unk['src']}: {best_unk['name']} (P {best_unk['prob']:.1f}, {best_unk['cols']} cols)"
        r["best_hit"] = "" if best_any is None else f"{best_any['src']}: {best_any['name']} (P {best_any['prob']:.1f}, {best_any['cols']} cols)"
        r["best_prob"] = None if best_any is None else best_any["prob"]
        r["cdd_char"] = "; ".join(cdd_char.get(t, []))[:300]
        r["pfam_ga_char"] = "; ".join(ga_char)[:300]
        r["hh_rule_characterised"] = hh_char
        if hh_char or cdd_char.get(t) or ga_char:
            r["cls"] = "characterised"
            r["source"] = ";".join(x for x, f in (("hhsearch", hh_char), ("cdd", bool(cdd_char.get(t))), ("pfam_ga", bool(ga_char))) if f)
        elif best_unk is not None or cdd_unk.get(t) or ga_unk:
            r["cls"] = "unknown_function"
            r["source"] = ";".join(x for x, f in (("hhsearch", best_unk is not None), ("cdd", bool(cdd_unk.get(t))), ("pfam_ga", bool(ga_unk))) if f)
        elif (best_any and best_any["prob"] >= 50) or cdd_weak.get(t) or e_weak:
            r["cls"] = "weak"
            r["source"] = ""
        else:
            r["cls"] = "none"
            r["source"] = ""
        r["complete"] = all((d / f"{x}.hhr").exists() for x in ("pfam_enr", "pfam_raw", "pdb_enr", "pdb_raw"))
        rows.append(r)
    F = pd.DataFrame(rows)
    F.to_csv(S / "data/triage_family.tsv", sep="\t", index=False)
    pd.DataFrame(hitrows).to_csv(S / "data/triage_hits.tsv", sep="\t", index=False)
    cls = dict(zip(F.token, F.cls))
    cand = pd.read_csv(S / "data/candidates.tsv", sep="\t")
    reps = pd.read_csv(S / "data/reps.tsv", sep="\t")
    out = []
    for _, c in cand.iterrows():
        rr = reps[reps.cand == c.cand].sort_values("rep")
        t1 = [int(x) for x in rr.tokens.iloc[0].split(";")]
        t2 = [int(x) for x in rr.tokens.iloc[1].split(";")]
        pos = []
        for a, b in zip(t1, t2):
            ca, cb = cls.get(a, "missing"), cls.get(b, "missing")
            pos.append(max((ca, cb), key=lambda x: ORDER.get(x, 9)))
        n_unrec = sum(p in ("none", "weak", "unknown_function") for p in pos)
        out.append(dict(cand=c.cand, rank=int(c["rank"]), k=int(c.k), position_cls=";".join(pos),
                        rep1_cls=";".join(cls.get(a, "missing") for a in t1), rep2_cls=";".join(cls.get(b, "missing") for b in t2),
                        unrecognised_positions=n_unrec, all_unrecognised=n_unrec == len(pos),
                        missing=any(p == "missing" for p in pos), fold=n_unrec >= 1 and not any(p == "missing" for p in pos)))
    B = pd.DataFrame(out)
    B.to_csv(S / "data/triage_blocks.tsv", sep="\t", index=False)
    print(F.cls.value_counts().to_dict(), "incomplete", int((~F.complete.astype(bool)).sum()))
    print("blocks", len(B), "fold", int(B.fold.sum()), "all positions unrecognised", int(B.all_unrecognised.sum()),
          "missing", int(B.missing.sum()))


if __name__ == "__main__":
    main()
