"""Parse the screen-3 enriched triage (s3_triage.py families -> triage/out/<token>/*.hhr, copied to work/s3/triage/out)
with the screen-2 HHsearch rule (predeclared_criteria_screen2.json 'triage_before_folding'):
characterised = HHsearch probability >= 90 % over >= 30 aligned columns against Pfam 39 or PDB100 (enriched or
DPANN-only alignment) to an entry without unknown-function wording; unknown_function = the same met only by such
entries; weak = best probability 50-90 %; none otherwise. UniRef30 hits (HHblits) are counted as context
(probability >= 95 %, names are automatic). Writes work/s3/triage_family_s3.tsv."""
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / "work" / "s3" / "triage" / "out"
UNKNOWN = re.compile(r"DUF\d|UPF\d|unknown function|uncharacteri|hypothetical|function unknown|unknown protein", re.I)


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
    ttl = {}
    p = HERE / "work" / "s2" / "pdb_titles.tsv"
    if p.exists():
        tt = pd.read_csv(p, sep="\t", header=None, names=["id", "title"], dtype=str).fillna("")
        ttl = dict(zip(tt.id, tt.title))
    rows = []
    for d in sorted(R.iterdir()):
        r = dict(token=int(d.name))
        best_char = best_unk = best_any = None
        for src in ("pfam_enr", "pfam_raw", "pdb_enr", "pdb_raw"):
            hs = parse_hhr(d / f"{src}.hhr") or []
            for h in hs:
                name = h["name"]
                if src.startswith("pdb"):
                    code = name.split()[0][:4].lower()
                    name = f"{name[:40]} | {ttl.get(code, '')[:80]}"
                h2 = dict(src=src, name=name[:140], prob=h["prob"], cols=h["cols"], evalue=h["evalue"])
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
        r["uniref_top"] = (u[0]["name"][:100] if u else "")
        for lab, h in (("char", best_char), ("unk", best_unk), ("best", best_any)):
            r[f"{lab}_hit"] = "" if h is None else f"{h['src']}: {h['name']} (P {h['prob']:.1f}, {h['cols']} cols)"
            r[f"{lab}_prob"] = None if h is None else h["prob"]
        r["cls"] = ("characterised" if best_char else "unknown_function" if best_unk else
                    "weak" if best_any and best_any["prob"] >= 50 else "none")
        r["complete"] = all((d / f"{x}.hhr").exists() for x in ("pfam_enr", "pfam_raw", "pdb_enr", "pdb_raw"))
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "work" / "s3" / "triage_family_s3.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 110)
    print(out[["token", "cls", "complete", "uniref_p95", "char_hit", "unk_hit", "best_hit"]].to_string(index=False))


if __name__ == "__main__":
    main()
