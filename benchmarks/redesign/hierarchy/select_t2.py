#!/usr/bin/env python
"""Task 2 tables, ribosomal guard and selection (light: instance_scores.parquet + ribosomal_check.json per run).

For each run: mean best-locus F1 and excess per caller and split (odd = tune, even = test), the summed
DefenseFinder + TXSScan excess, and the ribosomal guard (conserved@2 recovered of 1,686 and Parvarchaeaceae
recovered of 114; admissible = >= 1,650 and >= 110; resolved at conserved@2, conserved@80 and unit@0.5).
Selection on odd genomes within each predeclared family, guarded (primary, amendment 1) and unguarded (original).
  select_t2.py                 table + selection; writes sets_B.tsv (B1 / B2 / nested E) and t2_select.json
  select_t2.py --final         also runs dpann_compare.py for chosen multi-level vs comparator (even-genome CI)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from levels import LEVEL_SETS, by_tokens, spec  # noqa: E402

RD = Path("/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111")
H = RD / "engine/results_dpann_hier"
CTX = RD / "engine/results_dpann_ctx"
BASE = RD / "engine/results_dpann_v2/dpann__mmseqs__seq__e3c50s75__w3"
PY = "/home/jwestrob/miniconda3/envs/elsa/bin/python"
OUT = RD / "hierarchy"


def run_dir(name):
    if name == "base":
        return BASE
    if name.startswith("ctx:"):
        return CTX / f"dpann__ctx__esmpp_small__sim_all_{name[4:]}__w3"
    return H / f"{name}__w3"


def guard_file(name):
    if name.startswith("ctx:"):
        return OUT / f"ribo/ctx_{name[4:]}/ribosomal_check.json"
    if name == "base":
        return H / "hier__single__e3c50s75__w3/ribosomal_check.json"
    return run_dir(name) / "ribosomal_check.json"


SINGLE_E = {"r1000": "hier__single__r1000", "r3000": "hier__single__r3000", "r7500": "hier__single__r7500",
            "r11000": "ctx:r11000", "r15000": "ctx:r15000", "r19000": "ctx:r19000", "r23000": "ctx:r23000",
            "r30000": "ctx:r30000"}
SINGLE_S = {"id30c80": "hier__single__id30c80", "e3c50s75": "hier__single__e3c50s75", "e3c30s75": "hier__single__e3c30s75"}
FAMILIES = {"a_embedding": [f"hier__ml__E{i}" for i in range(1, 7)],
            "b_sequence": ["hier__ml__S1", "hier__ml__S2a", "hier__ml__S2b", "hier__ml__S2c"],
            "c_both": ["hier__ml__B1", "hier__ml__B1u", "hier__ml__B2"]}
POOLS = {"a_embedding": SINGLE_E, "b_sequence": SINGLE_S, "c_both": {**SINGLE_E, **SINGLE_S}}
CODE = None


def stats(d):
    global CODE
    if CODE is None:
        g = pd.read_parquet(RD / "ws_dpann/package/universe/genomes.parquet", columns=["genome", "genome_code"])
        CODE = dict(zip(g.genome, g.genome_code))
    f = run_dir(d) / "score_min2/instance_scores.parquet"
    if not f.exists():
        return None
    t = pd.read_parquet(f)
    t["excess"] = t.f1 - t.control_mean_f1
    t["odd"] = t.genome.map(CODE).to_numpy() % 2 == 1
    out = {}
    for split, m in (("odd", t.odd), ("even", ~t.odd)):
        for c in ("DefenseFinder", "TXSScan", "CRISPRCasTyper"):
            s = t[m & (t.caller == c)]
            out[f"{split}_{c}_f1"] = float(s.f1.mean())
            out[f"{split}_{c}_excess"] = float(s.excess.mean())
        out[f"{split}_summed"] = out[f"{split}_DefenseFinder_excess"] + out[f"{split}_TXSScan_excess"]
    g = guard_file(d)
    if g.exists():
        r = json.load(open(g))
        p = r["primary"]
        res = r["results"]["g2_m16"]
        out.update(rib_recovered=int(p["recovered"]), rib_parv=int(str(p["parvarchaeaceae"]).split("/")[0]),
                   rib_resolved_c2=int(p["resolved"]), rib_resolved_c80=res.get("ladder_conserved@80", {}).get("resolved"),
                   rib_resolved_unit=int(res["unit@0.5"]["resolved"]), rib_recovered_unit=int(res["unit@0.5"]["recovered"]))
        out["admissible"] = bool(out["rib_recovered"] >= 1650 and out["rib_parv"] >= 110)
    sm = json.load(open(run_dir(d) / "summary.json")) if (run_dir(d) / "summary.json").exists() else {}
    out["significant_pairs"] = sm.get("significant_pairs")
    out["rule"] = sm.get("rule")
    if "level_of_significant_pairs" in sm:
        out["level_of_significant_pairs"] = json.dumps(sm["level_of_significant_pairs"])
    return out


def levels_of(run):
    s = run.split("__")[-1]
    return LEVEL_SETS[s]


def choose(T, cands, singles, guarded):
    def adm(n):
        x = T.loc[n].get("admissible")
        return isinstance(x, (bool, np.bool_)) and bool(x)
    ok = (lambda n: n in T.index and (not guarded or adm(n)))
    have = [c for c in cands if ok(c)]
    sing = {k: v for k, v in singles.items() if ok(v)}
    d = {}
    if sing:
        bs = max(sing, key=lambda k: T.loc[sing[k], "odd_summed"])
        d.update(best_single=bs, best_single_run=sing[bs], best_single_odd=float(T.loc[sing[bs], "odd_summed"]))
    else:
        d.update(best_single="base (no admissible single level)", best_single_run="base",
                 best_single_odd=float(T.loc["base", "odd_summed"]))
    if have:
        ch = max(have, key=lambda c: T.loc[c, "odd_summed"])
        d.update(chosen=ch, chosen_odd=float(T.loc[ch, "odd_summed"]),
                 candidates_odd={c: round(float(T.loc[c, "odd_summed"]), 4) for c in have})
    return d


def compare(a, b):
    out = subprocess.run([PY, str(RD / "engine/code/dpann_compare.py"), str(run_dir(a)), str(run_dir(b))],
                         capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if "test (even)" in l or "tune (odd)" in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args()
    names = (["base"] + list(SINGLE_E.values()) + list(SINGLE_S.values()) + [x for v in FAMILIES.values() for x in v]
             + ["hier__ml__Enest", "hier__ml__Enestu"])
    rows = []
    for n in dict.fromkeys(names):
        s = stats(n)
        if s is not None:
            rows.append(dict(run=n, **s))
    T = pd.DataFrame(rows).set_index("run")
    T.to_csv(OUT / "t2_table.tsv", sep="\t", float_format="%.4f")
    cols = ["odd_summed", "even_summed", "even_DefenseFinder_f1", "even_TXSScan_f1", "even_CRISPRCasTyper_f1",
            "rib_recovered", "rib_parv", "rib_resolved_c2", "rib_resolved_unit", "admissible"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(T[[c for c in cols if c in T.columns]].round(4))
    sel = {}
    for fam, cands in FAMILIES.items():
        for guarded in (True, False):
            sel[f"{fam}|{'guarded' if guarded else 'unguarded'}"] = choose(T, cands, POOLS[fam], guarded)
    for k, v in sel.items():
        print(k, json.dumps(v))
    # sets for the 'both' family and the nested E runs
    lines = []
    eg, sg = sel["a_embedding|guarded"].get("chosen"), sel["b_sequence|guarded"].get("chosen")
    eu, su = sel["a_embedding|unguarded"].get("chosen"), sel["b_sequence|unguarded"].get("chosen")
    if eg and sg:
        lines.append(("hier__ml__B1", by_tokens(levels_of(eg) + levels_of(sg))))
    if eu and su and (eu, su) != (eg, sg):
        lines.append(("hier__ml__B1u", by_tokens(levels_of(eu) + levels_of(su))))
    lines.append(("hier__ml__B2", by_tokens(LEVEL_SETS["E4"] + LEVEL_SETS["S1"])))
    nest = []
    for tag, e in (("Enest", eg), ("Enestu", eu if eu != eg else None)):
        if e:
            s = e.split("__")[-1]
            nest.append(s)
            lines.append((f"hier__ml__{tag}", [f"nested__{s}__{x}" for x in LEVEL_SETS[s]]))
    with open(OUT / "code/sets_B.tsv", "w") as fh:
        for n, lv in lines:
            fh.write(f"{n}\t{spec(lv)}\n")
    (OUT / "code/nest_sets.txt").write_text(" ".join(dict.fromkeys(nest)) + "\n")
    print("sets_B:", [n for n, _ in lines], "nest:", nest)
    if a.final:
        cmp = {}
        for k, v in sel.items():
            if "chosen" in v:
                cmp[k] = dict(chosen=v["chosen"], comparator=v["best_single_run"], lines=compare(v["chosen"], v["best_single_run"]))
                print(k, v["chosen"], "vs", v["best_single_run"])
                print("\n".join(cmp[k]["lines"]))
        # hypothesis reading: each admissible multi-level arm against its finest level alone
        fin = {}
        for fam in ("a_embedding", "b_sequence"):
            for c in FAMILIES[fam]:
                if c in T.index:
                    f = levels_of(c)[0]
                    sr = POOLS[fam][f]
                    fin[c] = dict(finest=f, finest_run=sr, lines=compare(c, sr))
        sel["compare"] = cmp
        sel["vs_finest"] = fin
    json.dump(sel, open(OUT / "t2_select.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
