#!/usr/bin/env python
"""Writes the round-4 config files (JSON lines) for readings4.py."""
import json
import sys
from pathlib import Path

RD = "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/ELSA/redesign_20261111"
OUT = Path(__file__).resolve().parent / "r4configs"
VF = "vf:rw1000_e50:l2000:0.3"
R2ALL = RD + "/softrec/r2/scores/all/selected__rw1000_e50__l2000__0.3__{}.parquet"
BSC = RD + "/boundary/scores/rescue3/{}.parquet"


def r2sel(**kw):
    c = dict(name="r2sel", land="rw1000_e50", lab=VF, crit="none", f=0.5, keep_per=True, diag=True, save_per=True,
             check_against={r: R2ALL.format(r) for r in ("R1", "C", "H", "DH")})
    c.update(kw)
    return c


def write(name, rows):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(name, len(rows))


def tuneB():
    rows = [r2sel(),
            dict(name="tokDH_repro", land="tok", lab="tokmod:0.3", f=0.5, diag=True,
                 check_against={"R1": BSC.format("R1_conserved2"), "C": BSC.format("C_res0.3"), "H": BSC.format("H_res0.3_f0.5"),
                                "DH": BSC.format("DH_coreH_res0.3_f0.5")}),
            dict(name="tokor", land="tokor", lab="tokmod:0.3", f=0.5, diag=True, save_per=True)]
    for q in (0.99, 0.995, 0.999, 0.9999):
        for m in (1, 2):
            for x in ("x", "b"):
                rows.append(dict(name=f"L3_q{q:g}m{m}{x}", land="rw1000_e50", lab=VF, crit=f"q{q:g}m{m}{x}", f=0.5))
    write("tune_B.jsonl", rows)


def tuneC():
    rows = [r2sel(diag=False, save_per=False)]
    for res in (0.1, 0.2, 0.3, 0.5, 1, 3):
        for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
            rows.append(dict(name=f"L4_r{res:g}_f{f:g}", land="rw1000_e50", lab=f"vf:rw1000_e50:l2000:{res:g}", f=f,
                             readings=["H", "DH"]))
    write("tune_C.jsonl", rows)


def tuneL1(land="rw1000_e50", k=5, wos=(0, 0.3, 1), gammas=(0.3, 1, 3, 10, 30, 100, 1000), fname="tune_L1.jsonl", extra=None):
    rows = [r2sel(diag=False, save_per=False)]
    for wo in wos:
        tags = [f"k{k}o{wo:g}_cc"] + [f"k{k}o{wo:g}_g{g:g}" for g in gammas]
        if wo == 1:
            tags = tags[1:]           # components of the augmented graph do not depend on w_o
        for t in tags:
            rows.append(dict(name=f"L1_{land}_{t}", land=land, lab=f"ic:{t}", f=0.5, diag=True, readings=["R1", "C", "H", "DH"]))
    rows += extra or []
    write(fname, rows)


def tuneL2():
    rows = [r2sel(diag=False, save_per=False)]
    for mode in (1, 2):
        rows.append(dict(name=f"L2_o{mode}", land=f"o{mode}rw1000_e50", lab=VF, f=0.5, diag=True))
        rows.append(dict(name=f"L2_o{mode}_rw200", land=f"o{mode}rw200_e50", lab="none", f=0.5, readings=["R1"]))
    write("tune_L2.jsonl", rows)


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        globals()[fn]()


def tuneComb(land, wo, gammas, fs, k=5, fname="tune_comb.jsonl"):
    """Joint odd grid of the combination: instance-community labels of `land` (k, w_o, gamma) x trimming factor."""
    rows = [r2sel(diag=False, save_per=False)]
    for g in gammas:
        for f in fs:
            t = f"k{k}o{float(wo):g}_g{float(g):g}"
            rows.append(dict(name=f"comb_{land}_{t}_f{float(f):g}", land=land, lab=f"ic:{t}", f=float(f), diag=True))
    write(fname, rows)


def testAll(comb, fname="test_main.jsonl"):
    """Even-genome test: the round-4 configuration (verdict) first after r2sel, then each lever alone, token control."""
    rows = [r2sel(diag=False), dict(comb, name="ROUND4_" + comb["name"], save_per=True, diag=True),
            dict(name="L1_sel", land="rw1000_e50", lab="ic:k5o0.3_g1", f=0.5, save_per=True, diag=True),
            dict(name="L2_sel", land="o2rw1000_e50", lab=VF, f=0.5, save_per=True, diag=True),
            dict(name="L3_sel", land="rw1000_e50", lab=VF, crit="q0.995m1x", f=0.5, save_per=True),
            dict(name="L4_sel", land="rw1000_e50", lab="vf:rw1000_e50:l2000:0.5", f=0.8, save_per=True),
            dict(name="tokor", land="tokor", lab="tokmod:0.3", f=0.5, save_per=True, diag=True)]
    write(fname, rows)
